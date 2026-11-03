"""Generation, pooled capture, CPU probes and evaluator resource-pilot workers."""

import contextlib
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.benchmark_sampling import load_prompts, sample_prompts

log = logging.getLogger(__name__)
CONDITIONS = {
    "alpha-zero": ("ablation", "safety", 0.0),
    "safety-ablation": ("ablation", "safety", 1.0),
    "random-ablation": ("ablation", "random", 1.0),
    "safety-steering": ("steering", "safety", 1.0),
    "random-steering": ("steering", "random", 1.0),
    "safety-ablation-a2": ("ablation", "safety", 2.0),
}
EVALUATORS = ("nudenet", "clip-h14", "alignment", "clip-iqa")
CLIP_REPO = "openai/clip-vit-base-patch32"


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def prepare_samples(source, out, baseline=60, discovery_per_class=10, validation_per_class=6, seed=42):
    """Freeze shared rows/seeds, keeping discovery separate from validation/baseline.

    This is a resource sample from candidates, not a human validation procedure.
    """
    frame = load_prompts(source).copy()
    if min(discovery_per_class, validation_per_class) < 2 or baseline <= 2 * validation_per_class:
        raise ValueError("need >=2 per class and baseline large enough to include validation")
    level = frame.benchmark_level.fillna("").astype(str)
    categories = frame.standard_category.fillna("").astype(str).str.lower()
    groups = [frame[level.eq("L0")], frame[level.eq("L1") & categories.str.contains("sexual")]]
    discovery, validation = [], []
    for label, group in enumerate(groups):
        count = discovery_per_class + validation_per_class
        if len(group) < count:
            raise ValueError(f"not enough {'benign L0' if label == 0 else 'sexual L1'} rows: need {count}")
        chosen = group.sample(count, random_state=seed).copy()
        chosen["pilot_label"] = label
        train = chosen.iloc[:discovery_per_class].copy()
        train["pilot_split"] = "discovery"
        val = chosen.iloc[discovery_per_class:].copy()
        val["pilot_split"] = "validation"
        discovery.append(train)
        validation.append(val)
    discovery = pd.concat(discovery, ignore_index=True)
    validation = pd.concat(validation, ignore_index=True)
    excluded = set(discovery.prompt_id) | set(validation.prompt_id)
    rest, sampling = sample_prompts(frame[~frame.prompt_id.isin(excluded)], baseline - len(validation), seed, "levels")
    base = pd.concat([validation, rest], ignore_index=True)
    if len(base) != baseline:
        raise ValueError("source too small for requested baseline")
    tables = {
        "baseline": base,
        "capture": pd.concat([discovery, validation], ignore_index=True),
        "validation": validation,
        "identity": validation.groupby("pilot_label", sort=True).head(1),
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table = table.copy()
        seeds = table.get("evaluation_seed", pd.Series(index=table.index, dtype=object))
        table["evaluation_seed"] = [
            int(value)
            if pd.notna(value)
            else int.from_bytes(hashlib.sha256(f"{seed}:{pid}".encode()).digest()[:4], "big") % (2**31)
            for pid, value in zip(table.prompt_id, seeds, strict=True)
        ]
        table.to_csv(out / f"{name}.csv", index=False)
    manifest = {
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "baseline_prompts": len(base),
        "discovery_prompts": len(discovery),
        "validation_prompts": len(validation),
        "capture_prompts": len(discovery) + len(validation),
        "baseline_level_counts": base.benchmark_level.value_counts().to_dict(),
        "sampling": sampling,
        "seed": seed,
        "note": "Unreviewed resource pilot; direction/probe scores do not validate safety effectiveness.",
    }
    manifest["tables_sha256"] = {
        name: hashlib.sha256((out / f"{name}.csv").read_bytes()).hexdigest() for name in tables
    }
    save_json(manifest, out / "manifest.json")
    return manifest


def generation(spec, result, probe):
    import torch

    from visual_synthesis.backends.image import build_image_backend, save_image
    from visual_synthesis.config import get_model
    from visual_synthesis.io import append_jsonl, file_sha256, safe_name
    from visual_synthesis.pilot_interventions import ActivationHooks, hook_points

    cfg, out, phase = spec["config"], Path(spec["output"]), spec["phase"]
    model = get_model(cfg, spec["model"])
    model.update(device="cuda", enable_cpu_offload=True)
    result["model_settings"] = model
    table = load_prompts(spec["sample"]).astype(object)
    records = table.where(pd.notna(table), "").to_dict("records")
    backend = build_image_backend(model, cfg["paths"]["models"])
    tick = time.perf_counter()
    backend.load()
    probe.sync()
    result["load_seconds"] = time.perf_counter() - tick
    result["cpu_offload_active"] = bool(getattr(backend.pipe, "_all_hooks", []))
    if not result["cpu_offload_active"]:
        raise ValueError("CPU offload hooks are not active")

    def request(row):
        seed = int(row["evaluation_seed"])
        # Some AR prior stages sample from global RNGs in addition to the latent generator.
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        return {"prompt_id": str(row["prompt_id"]), "prompt": row["prompt"], "seed": seed}

    tick = time.perf_counter()
    backend.generate([request(records[0])])
    probe.sync()
    result["warmup_seconds"] = time.perf_counter() - tick
    hooks = None
    if phase != "baseline":
        points = hook_points(backend.pipe, model)
        if phase == "capture":
            hooks = ActivationHooks(points)
        else:
            direction = np.load(spec["direction"], allow_pickle=False)
            index = int(direction["layer_index"])
            mode, vector, alpha = CONDITIONS[phase]
            d = torch.from_numpy(direction["direction"].copy())
            if vector == "random":
                d = torch.randn(d.shape, generator=torch.Generator("cpu").manual_seed(0))
                d = d / d.norm()
            hooks = ActivationHooks(
                [points[index]], mode, d, torch.from_numpy(direction["center"].copy()), float(direction["scale"]), alpha
            )
        result["hook_adapter"] = [
            {"name": p.name, "output_index": p.output_index, "cfg_layout": p.cfg_layout, "image_tokens": p.image_tokens}
            for p in hooks.points
        ]
        result["hook_scope"] = "GLM DiT decoder only" if model["family"] == "glm_image" else "image stream"
    result["status"] = "running"
    save_json(result, out / "report.json")
    features, meta, timings = {}, [], []
    references = {}
    if phase == "alpha-zero":
        references = {r["prompt_id"]: r for r in read_jsonl(Path(spec["baseline"]) / "images.jsonl")}
    with hooks if hooks else contextlib.nullcontext():
        for index, row in enumerate(records):
            if hooks:
                hooks.reset()
            tick = time.perf_counter()
            item = backend.generate([request(row)])[0]
            probe.sync()
            path = out / "images" / f"{index:04d}-{safe_name(str(row['prompt_id']))}.png"
            save_image(item["image"], path)
            record = {
                "condition": phase,
                "prompt_id": str(row["prompt_id"]),
                "prompt": row["prompt"],
                "evaluation_seed": int(row["evaluation_seed"]),
                "image_path": str(path),
                "benchmark_level": row.get("benchmark_level", ""),
                "standard_category": row.get("standard_category", ""),
                "sha256": file_sha256(path),
                "image_bytes": path.stat().st_size,
            }
            append_jsonl(out / "images.jsonl", record)
            if phase == "capture":
                for name, feature in hooks.features().items():
                    features.setdefault(name, []).append(feature.numpy())
                meta.append(
                    {"prompt_id": row["prompt_id"], "label": int(row["pilot_label"]), "split": row["pilot_split"]}
                )
            if phase == "alpha-zero" and record["sha256"] != references[record["prompt_id"]]["sha256"]:
                raise ValueError("alpha-zero image differs from baseline; inspect determinism/hook equivalence")
            if hooks:
                if any(not count for count in hooks.calls.values()):
                    raise ValueError("selected intervention hook was never executed")
                result["last_hook_calls"] = hooks.calls.copy()
                result["last_hook_shapes"] = hooks.shapes.copy()
            elapsed = time.perf_counter() - tick
            timings.append(elapsed)
            result["images"] = len(timings)
            result["prompt_seconds_with_io"] = timings
            # Preserve partial evidence if PBS later kills this worker.
            result["resources"] = {
                **probe.peaks,
                "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
            }
            save_json(result, out / "report.json")
            log.info("%s %s %d/%d: %.2fs", spec["model"], phase, index + 1, len(records), elapsed)
    dump_seconds = 0.0
    if phase == "capture":
        tick = time.perf_counter()
        names = list(features)
        np.savez_compressed(
            out / "activations.npz", **{f"layer_{i}": np.stack(features[name]) for i, name in enumerate(names)}
        )
        save_json({"layers": names, "rows": meta}, out / "activation_metadata.json")
        dump_seconds = time.perf_counter() - tick
        result["activation_bytes"] = (out / "activations.npz").stat().st_size
        result["activation_write_seconds"] = dump_seconds
    result["mean_seconds_per_image"] = (sum(timings) + dump_seconds) / len(timings)
    result["mean_png_bytes"] = sum(r["image_bytes"] for r in read_jsonl(out / "images.jsonl")) / len(timings)
    result["fixed_seconds"] = result["load_seconds"] + result["warmup_seconds"]
    result["alpha_zero_equivalent"] = True if phase == "alpha-zero" else None
    backend.unload()


def learn(spec, result, probe):
    import torch

    from visual_synthesis.pilot_interventions import fit_direction, fit_probe

    del probe
    source, out = Path(spec["capture"]), Path(spec["output"])
    meta = json.loads((source / "activation_metadata.json").read_text())
    data = np.load(source / "activations.npz", allow_pickle=False)
    train = np.array([r["split"] == "discovery" for r in meta["rows"]])
    labels = np.array([r["label"] for r in meta["rows"]])
    tick = time.perf_counter()
    selected = len(meta["layers"]) // 2  # Fixed a priori; validation does not select the intervention layer.
    summaries = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for i, name in enumerate(meta["layers"]):
            x = data[f"layer_{i}"]
            center, d, scale = fit_direction(x[train], labels[train])
            fitted = fit_probe(x[train], labels[train])
            val = (torch.from_numpy(x[~train]) - fitted["mean"]) / fitted["std"]
            predictions = (val @ fitted["weight"] + fitted["bias"] >= 0).numpy()
            summaries.append(
                {
                    "layer": name,
                    "held_out_accuracy_resource_sample": float((predictions == labels[~train]).mean()),
                    "channels": x.shape[1],
                }
            )
            if i == selected:
                np.savez_compressed(
                    out / "direction.npz", center=center.numpy(), direction=d.numpy(), scale=scale, layer_index=i
                )
    finally:
        torch.set_num_threads(threads)
    result.update(
        cpu_learning_seconds=time.perf_counter() - tick,
        probes=summaries,
        discovery_images=int(train.sum()),
        validation_images=int((~train).sum()),
        selected_layer=meta["layers"][selected],
        note="Small, unreviewed pilot probes; no safety-effectiveness claim or CPU scaling extrapolation.",
    )


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def clip_iqa_scores(image_features, anchors):
    """CLIP-IQA's positive/negative anchor softmax, using the recorded CLIP variant."""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    anchors = anchors / anchors.norm(dim=-1, keepdim=True)
    return (100 * image_features @ anchors.T).softmax(dim=-1)[:, 0]


def evaluation(spec, result, probe):
    import torch
    from PIL import Image

    from visual_synthesis.backends.detectors import build_detector, is_flagged
    from visual_synthesis.config import get_detector
    from visual_synthesis.io import append_jsonl
    from visual_synthesis.stages.s07_metrics import feature_tensor

    kind, out, cfg = spec["phase"], Path(spec["output"]), spec["config"]
    rows = read_jsonl(spec["images"])
    tick = time.perf_counter()
    detector = processor = clip = None
    if kind in ("nudenet", "clip-h14"):
        detector_config = get_detector(cfg, kind)
        detector_config["device"] = "cuda" if kind == "clip-h14" else "cpu"
        detector = build_detector(detector_config, cfg["paths"]["models"])
        detector.load()
    elif kind in ("alignment", "clip-iqa"):
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(spec["clip_source"], local_files_only=True)
        clip = CLIPModel.from_pretrained(spec["clip_source"], local_files_only=True).to("cuda").eval()
        if kind == "clip-iqa":
            with torch.inference_mode():
                text = processor(text=["Good photo.", "Bad photo."], return_tensors="pt", padding=True)
                anchors = feature_tensor(clip.get_text_features(**{k: v.to("cuda") for k, v in text.items()}))
            result["quality_variant"] = "CLIP-IQA two-anchor formula, OpenAI ViT-B/32 variant"
            result["quality_reference"] = "TorchMetrics functional CLIP-IQA, quality prompt pair, scale=100"
    else:
        raise ValueError("unknown evaluator")
    probe.sync()
    result["load_seconds"] = time.perf_counter() - tick
    timings = []
    for index, row in enumerate(rows):
        tick = time.perf_counter()
        if detector:
            scores = detector.detect(row["image_path"])
            payload = {"scores": scores, "flagged": is_flagged(scores, detector_config)}
        else:
            with Image.open(row["image_path"]) as raw:
                image = raw.convert("RGB")
            with torch.inference_mode():
                if kind == "alignment":
                    inputs = processor(
                        text=[row["prompt"]], images=[image], return_tensors="pt", padding=True, truncation=True
                    )
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    a = feature_tensor(clip.get_image_features(pixel_values=inputs["pixel_values"]))
                    b = feature_tensor(
                        clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    )
                    payload = {"score": float(torch.nn.functional.cosine_similarity(a, b).item())}
                else:
                    pixels = processor(images=[image], return_tensors="pt")["pixel_values"].to("cuda")
                    features = feature_tensor(clip.get_image_features(pixel_values=pixels))
                    payload = {"score": float(clip_iqa_scores(features, anchors).item())}
            image.close()
        append_jsonl(out / "scores.jsonl", {"condition": row["condition"], "prompt_id": row["prompt_id"], **payload})
        probe.sync()
        timings.append(time.perf_counter() - tick)
        if (index + 1) % 10 == 0:
            log.info("%s %d/%d", kind, index + 1, len(rows))
    result.update(
        images=len(rows),
        mean_seconds_per_image=float(np.mean(timings)),
        fixed_seconds=result["load_seconds"],
        seconds=sum(timings) + result["load_seconds"],
    )
    if detector:
        detector.unload()


def worker(spec):
    import torch

    from visual_synthesis.benchmark import Resources, machine_info

    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    phase = spec["phase"]
    device = "cpu" if phase in ("learn", "nudenet") else "cuda:0"
    result = {"phase": phase, "model": spec["model"], "status": "failed", "resources": {}}
    probe = None
    try:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; submit through PBS on a GPU compute node")
        result["machine"] = machine_info(torch, device)
        with Resources(torch, device) as probe:
            tick = time.perf_counter()
            if phase == "learn":
                learn(spec, result, probe)
            elif phase in EVALUATORS:
                evaluation(spec, result, probe)
            else:
                generation(spec, result, probe)
            result["wall_seconds"] = time.perf_counter() - tick
        result["status"] = "success"
    except Exception as exc:
        error = str(exc).lower()
        failure_kind = (
            "storage_exhausted"
            if any(t in error for t in ("no space left", "disk quota exceeded"))
            else "gpu_out_of_memory"
            if "out of memory" in error
            else "inspect_worker_log"
        )
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            failure_kind=failure_kind,
        )
        log.exception("pilot stage %s failed", phase)
    finally:
        if probe:
            result["resources"] = probe.peaks
        save_json(result, out / "report.json")
    return result


def project_costs(reports, baseline=6000, capture=500, validation=300, wall_hours=6):
    """Project each condition with explicit setup repeats and evaluator costs."""
    required = {"baseline", "capture", "learn", *CONDITIONS, *EVALUATORS}
    missing = sorted(p for p in required if reports.get(p, {}).get("status") != "success")
    if missing:
        return {
            "status": "incomplete",
            "missing_successful_phases": missing,
            "note": "Failed/unmeasured stages are unknown, never zero-cost.",
        }

    def run_cost(phase, count, repeats=1):
        r = reports[phase]
        steady, fixed = r["mean_seconds_per_image"], r["fixed_seconds"]
        usable = wall_hours * 3600 - 600  # Ten minutes per job reserved for other setup/exit.
        capacity = math.floor((usable - fixed) / steady)
        if capacity < 1:
            raise ValueError("a single profiled image cannot fit the proposed job budget")
        jobs = math.ceil(count / capacity) * repeats
        return {"images": count * repeats, "jobs_estimated": jobs, "seconds": count * repeats * steady + jobs * fixed}

    generation_phases = {"baseline": run_cost("baseline", baseline), "capture": run_cost("capture", capture)}
    for phase in ("safety-ablation", "random-ablation", "safety-steering", "random-steering"):
        generation_phases[f"E5-{phase}"] = run_cost(phase, validation, repeats=2)
    generation_phases["E6-safety-ablation"] = run_cost("safety-ablation-a2", validation, repeats=2)
    generation_phases["E6-random-ablation"] = run_cost("random-ablation", validation, repeats=2)
    images = sum(r["images"] for r in generation_phases.values())
    evaluators = {kind: run_cost(kind, images) for kind in EVALUATORS}
    gen_h = sum(r["seconds"] for r in generation_phases.values()) / 3600
    eval_h = sum(r["seconds"] for r in evaluators.values()) / 3600
    return {
        "status": "conditional",
        "target_images": images,
        "generation": generation_phases,
        "evaluators": evaluators,
        "generation_hours": gen_h,
        "evaluation_hours": eval_h,
        "generation_plus_evaluation_hours": gen_h + eval_h,
        "probe_learning_seconds_pilot_only": reports["learn"]["cpu_learning_seconds"],
        "planned_wall_hours": wall_hours,
        "note": (
            "Resource forecast, not validated study results. Two E5 strengths share each profiled mode cost; "
            "E6 random strengths use its alpha=1 cost. CPU learning is measured on the pilot only and is not "
            "extrapolated. Queues, retries, CPU contention and missing/unreviewed prompt strata "
            "remain outside the forecast."
        ),
    }
