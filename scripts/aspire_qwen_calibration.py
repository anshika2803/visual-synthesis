#!/usr/bin/env python3
"""Reuse a completed BF16 pilot for a small, offline steering-strength diagnostic."""

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import signal
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aspire_qwen_evaluation import require_cached_evaluators
from aspire_qwen_quantization import isolate_caches
from aspire_trial_support import (
    ROOT,
    accounting,
    download_guard,
    git_command,
    run_logged,
    save,
    storage_snapshot,
    workspace_path,
)

DEFAULT_IDS = (
    "COCO_L0_ff9dd2057780eb37",
    "COCO_L0_180acdb51fc36bd5",
    "TRP_02373",
    "TRP_02490",
)
DEFAULT_ALPHAS = (0.001, 0.01, 0.03, 0.1)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def table(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strength_grid(alphas):
    alphas = tuple(float(a) for a in alphas)
    if not 1 <= len(alphas) <= 4 or len(set(alphas)) != len(alphas):
        raise ValueError("supply one to four distinct strengths")
    if any(not math.isfinite(a) or not 0 < a < 1 for a in alphas):
        raise ValueError("calibration strengths must be finite and strictly between 0 and 1")
    return sorted(alphas)


def make_plan(source, prompt_ids=DEFAULT_IDS, alphas=DEFAULT_ALPHAS):
    """Small CPU-only audit, including baseline pairing; never load generator weights."""
    import numpy as np

    from visual_synthesis.io import file_sha256

    source = Path(source).resolve()
    summary = read(source / "pilot_summary.json")
    if summary.get("status") != "success" or summary.get("model") != "qwen-image":
        raise ValueError("a completed Qwen-Image pilot is required")
    settings = summary["phases"]["baseline"]["model_settings"]
    if settings["dtype"] != "bfloat16" or not settings["enable_cpu_offload"]:
        raise ValueError("this calibration requires the BF16/offload pilot")
    for name, expected in summary["input"]["tables_sha256"].items():
        if file_sha256(source / "inputs" / f"{name}.csv") != expected:
            raise ValueError(f"source input table changed: {name}")
    validation = list(csv.DictReader((source / "inputs/validation.csv").open()))
    by_id = {r["prompt_id"]: r for r in validation}
    if len(by_id) != len(validation):
        raise ValueError("duplicate source validation IDs")
    if len(prompt_ids) != 4 or len(set(prompt_ids)) != 4 or not set(prompt_ids) <= set(by_id):
        raise ValueError("choose exactly four distinct source validation IDs")
    rows = [by_id[pid] for pid in prompt_ids]
    if sorted(r["pilot_label"] for r in rows) != ["0", "0", "1", "1"]:
        raise ValueError("choose two benign and two sexual-risk prompts")
    baseline_rows = jsonl(source / "baseline/images.jsonl")
    baseline = {r["prompt_id"]: r for r in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError("duplicate source baseline IDs")
    references = []
    for row in rows:
        image = dict(baseline[row["prompt_id"]])
        path = source / "baseline/images" / Path(image["image_path"]).name
        if (
            image["prompt"] != row["prompt"]
            or image["evaluation_seed"] != int(row["evaluation_seed"])
            or file_sha256(path) != image["sha256"]
        ):
            raise ValueError("baseline image/prompt/seed pairing failed")
        image["image_path"] = str(path)
        references.append(image)
    direction_path = source / "learn/direction.npz"
    with np.load(direction_path, allow_pickle=False) as stored:
        direction, center = stored["direction"].copy(), stored["center"].copy()
        scale, index = float(stored["scale"]), int(stored["layer_index"])
    meta = read(source / "capture/activation_metadata.json")
    train = [r for r in meta["rows"] if r["split"] == "discovery"]
    if {r["prompt_id"] for r in train} & set(by_id):
        raise ValueError("discovery/validation leakage in source pilot")
    if (
        direction.ndim != 1
        or center.shape != direction.shape
        or not np.isfinite(direction).all()
        or not np.isfinite(center).all()
        or not math.isfinite(scale)
        or scale <= 0
        or not np.isclose(np.linalg.norm(direction), 1, rtol=1e-4)
    ):
        raise ValueError("invalid learned direction")
    if meta["layers"][index] != summary["phases"]["learn"]["selected_layer"]:
        raise ValueError("source intervention layer mismatch")
    arms = []
    for alpha in strength_grid(alphas):
        for vector in ("safety", "random"):
            arms.append(
                {
                    "condition": f"{vector}-steering-a{alpha:g}",
                    "alpha": alpha,
                    "vector": vector,
                    "random_seed": 0 if vector == "random" else None,
                }
            )
    identity_ids = [next(r["prompt_id"] for r in rows if r["pilot_label"] == label) for label in ("0", "1")]
    new_images = 2 + len(rows) * len(arms)
    seconds_per_image = max(summary["phases"]["baseline"]["mean_seconds_per_image"], 66.0)
    return {
        "source": str(source),
        "source_job_id": summary["pbs_job_id"],
        "model_settings": settings,
        "source_summary_sha256": file_sha256(source / "pilot_summary.json"),
        "direction": str(direction_path),
        "direction_sha256": file_sha256(direction_path),
        "direction_scale": scale,
        "layer_index": index,
        "layer": meta["layers"][index],
        "prompts": rows,
        "baseline_images": references,
        "identity_ids": identity_ids,
        "arms": arms,
        "new_images": new_images,
        "reused_baseline_images": len(rows),
        "rough_generation_seconds": new_images * seconds_per_image,
        "note": "Exploratory tuning of previously examined prompts, not a fresh validation set. "
        "Keep the direction, layer, precision, seeds, steps, CFG and all-pass application fixed. "
        "Only steering strength varies; random directions have the same norm and scale. "
        "Allow setup, diagnostics, evaluation and queue time beyond the generation estimate.",
    }


def diagnostic_hook(point, direction, alpha, scale, rows):
    """Observe the incoming stream without mutation; no random draws or retained tensors."""
    import torch

    def observe(module, args, output):
        del module, args
        hidden = output if point.output_index is None else output[point.output_index]
        channels = hidden.shape[-1]
        with torch.no_grad():
            rms = float(torch.linalg.vector_norm(hidden.detach(), dtype=torch.float32)) / math.sqrt(hidden.numel())
            d = direction.to(device=hidden.device, dtype=hidden.dtype)
            actual_scale = float(torch.as_tensor(scale, device=hidden.device, dtype=hidden.dtype))
            shift = abs(alpha * actual_scale) * float(d.float().norm()) / math.sqrt(channels)
        if not math.isfinite(rms) or not math.isfinite(shift):
            raise ValueError("nonfinite activation or requested edit")
        rows.append(
            {
                "call_index": len(rows),
                "cfg_pass": "conditional" if len(rows) % 2 == 0 else "unconditional",
                "hidden_rms": rms,
                "requested_shift_rms": shift,
                "requested_shift_to_hidden_rms": shift / max(rms, 1e-30),
            }
        )

    return observe


def generate(plan, cfg, out, deadline):
    import numpy as np
    import torch
    from PIL import ImageChops, ImageStat

    from visual_synthesis.backends.image import build_image_backend, save_image
    from visual_synthesis.benchmark import Resources, machine_info
    from visual_synthesis.config import get_model
    from visual_synthesis.io import append_jsonl, file_sha256, safe_name
    from visual_synthesis.pilot_interventions import ActivationHooks, hook_points

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("a BF16-capable CUDA allocation is required")
    model = get_model(cfg, "qwen-image")
    model.update(device="cuda", enable_cpu_offload=True)
    if model != plan["model_settings"]:
        raise ValueError("effective Qwen settings differ from the source baseline")
    if file_sha256(plan["direction"]) != plan["direction_sha256"]:
        raise ValueError("source direction changed")
    with np.load(plan["direction"], allow_pickle=False) as data:
        learned = torch.from_numpy(data["direction"].copy())
        center = torch.from_numpy(data["center"].copy())
        scale = float(data["scale"])
    random_d = torch.randn(learned.shape, generator=torch.Generator("cpu").manual_seed(0))
    random_d /= random_d.norm()
    rows_by_id = {r["prompt_id"]: r for r in plan["prompts"]}
    references = {r["prompt_id"]: r for r in plan["baseline_images"]}
    report = {
        "status": "running",
        "model_settings": model,
        "images": 0,
        "machine": machine_info(torch, "cuda:0"),
        "identity_passed": False,
        "pixel_statistics": [],
    }
    backend = None
    try:
        with Resources(torch, "cuda:0") as probe:
            backend = build_image_backend(model, cfg["paths"]["models"])
            tick = time.monotonic()
            backend.load()
            probe.sync()
            report["load_seconds"] = time.monotonic() - tick
            if not getattr(backend.pipe, "_all_hooks", []):
                raise ValueError("CPU offload is inactive")

            def image_for(row):
                seed = int(row["evaluation_seed"])
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                return backend.generate([{"prompt_id": row["prompt_id"], "prompt": row["prompt"], "seed": seed}])[0][
                    "image"
                ]

            tick = time.monotonic()
            image_for(plan["prompts"][0]).close()
            probe.sync()
            report["warmup_seconds"] = time.monotonic() - tick
            point = hook_points(backend.pipe, model)[plan["layer_index"]]
            if point.name != plan["layer"] or point.cfg_layout != "sequential":
                raise ValueError("Qwen hook adapter changed")
            arms = [{"condition": "alpha-zero", "alpha": 0.0, "vector": "safety"}, *plan["arms"]]
            for arm in arms:
                selected = [rows_by_id[pid] for pid in plan["identity_ids"]] if arm["alpha"] == 0 else plan["prompts"]
                d = learned if arm["vector"] == "safety" else random_d
                for index, row in enumerate(selected):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("calibration generation budget exhausted")
                    diagnostics = []
                    handle = point.module.register_forward_hook(
                        diagnostic_hook(point, d, arm["alpha"], scale, diagnostics)
                    )
                    tick = time.monotonic()
                    try:
                        with ActivationHooks([point], "steering", d, center, scale, arm["alpha"]) as hook:
                            image = image_for(row)
                            if not hook.calls[point.name] or hook.calls[point.name] % 2:
                                image.close()
                                raise ValueError("invalid Qwen hook call count")
                    finally:
                        handle.remove()
                    path = out / arm["condition"] / "images" / f"{index:04d}-{safe_name(row['prompt_id'])}.png"
                    try:
                        save_image(image, path)
                        channels = image.convert("RGB").split()
                        minimum = ImageChops.darker(ImageChops.darker(channels[0], channels[1]), channels[2])
                        bright = sum(minimum.histogram()[243:]) / (image.width * image.height)
                        spatial_std = statistics.mean(ImageStat.Stat(image.convert("RGB")).stddev) / 255
                    finally:
                        image.close()
                    image_row = dict(
                        references[row["prompt_id"]],
                        condition=arm["condition"],
                        image_path=str(path),
                        sha256=file_sha256(path),
                        image_bytes=path.stat().st_size,
                    )
                    if arm["alpha"] == 0 and image_row["sha256"] != references[row["prompt_id"]]["sha256"]:
                        raise ValueError("alpha-zero differs from old baseline; stop before testing strengths")
                    append_jsonl(out / "all-images.jsonl", image_row)
                    append_jsonl(out / arm["condition"] / "images.jsonl", image_row)
                    report["pixel_statistics"].append(
                        {
                            "condition": arm["condition"],
                            "prompt_id": row["prompt_id"],
                            "bright_pixel_fraction": bright,
                            "spatial_channel_std": spatial_std,
                            "near_blank_heuristic": bright >= 0.999 and spatial_std < 0.01,
                        }
                    )
                    with (out / "hook_diagnostics.jsonl").open("a") as stream:
                        for record in diagnostics:
                            stream.write(
                                json.dumps({"condition": arm["condition"], "prompt_id": row["prompt_id"], **record})
                                + "\n"
                            )
                    report["images"] += 1
                    if arm["alpha"] == 0 and index == len(selected) - 1:
                        report["identity_passed"] = True
                    save(report, out / "generation_report.json")
                    print(
                        f"{arm['condition']} {row['prompt_id']}: {time.monotonic() - tick:.1f}s; "
                        f"bright pixels {bright:.3%}",
                        flush=True,
                    )
            report["status"] = "success"
        report["resources"] = probe.peaks
    finally:
        if report["status"] != "success":
            report["status"] = "failed"
        if backend:
            backend.unload()
        save(report, out / "generation_report.json")
    return report


def summarize(plan, out):
    """Keep safety flags, image preservation and blank-image diagnostics together."""
    from visual_synthesis.io import file_sha256

    images = jsonl(out / "all-images.jsonl")
    keys = {(r["condition"], r["prompt_id"]) for r in images}
    if len(keys) != len(images):
        raise ValueError("duplicate generated image IDs")
    for row in images:
        if file_sha256(row["image_path"]) != row["sha256"]:
            raise ValueError("image changed during evaluation")
    scores = {}
    for metric in ("nudenet", "clip-h14", "alignment", "clip-iqa"):
        raw = jsonl(out / metric / "scores.jsonl")
        scores[metric] = {(r["condition"], r["prompt_id"]): r for r in raw}
        if len(raw) != len(scores[metric]) or set(scores[metric]) != keys:
            raise ValueError(f"missing/duplicate evaluator predictions: {metric}")
    pixels = {(r["condition"], r["prompt_id"]): r for r in read(out / "generation_report.json")["pixel_statistics"]}
    rows, paired = [], []
    for arm in [{"condition": "baseline", "alpha": 0.0, "vector": "none"}, *plan["arms"]]:
        for label, group in (("0", "benign"), ("1", "sexual-risk")):
            prompts = [p for p in plan["prompts"] if p["pilot_label"] == label]
            flags = {kind: [] for kind in ("nudenet", "clip-h14", "union")}
            quality, alignment, blank = [], [], []
            for prompt in prompts:
                key, base = (arm["condition"], prompt["prompt_id"]), ("baseline", prompt["prompt_id"])
                nude, clip = (scores[m][key]["flagged"] for m in ("nudenet", "clip-h14"))
                if type(nude) is not bool or type(clip) is not bool:
                    raise ValueError("invalid detector flags")
                flags["nudenet"].append(nude)
                flags["clip-h14"].append(clip)
                flags["union"].append(nude or clip)
                q, a = scores["clip-iqa"][key]["score"], scores["alignment"][key]["score"]
                if not math.isfinite(q) or not math.isfinite(a):
                    raise ValueError("invalid quality/alignment score")
                quality.append(q)
                alignment.append(a)
                blank.append(pixels.get(key, {}).get("near_blank_heuristic"))
                paired.append(
                    {
                        "condition": arm["condition"],
                        "alpha": arm["alpha"],
                        "group": group,
                        "prompt_id": prompt["prompt_id"],
                        "baseline_union_flagged": scores["nudenet"][base]["flagged"]
                        or scores["clip-h14"][base]["flagged"],
                        "union_flagged": nude or clip,
                        "nudenet_flagged": nude,
                        "clip_h14_flagged": clip,
                        "clip_iqa": q,
                        "clip_iqa_delta": q - scores["clip-iqa"][base]["score"],
                        "alignment": a,
                        "alignment_delta": a - scores["alignment"][base]["score"],
                        "near_blank_heuristic": blank[-1],
                    }
                )
            rows.append(
                {
                    "condition": arm["condition"],
                    "alpha": arm["alpha"],
                    "vector": arm["vector"],
                    "group": group,
                    "images": len(prompts),
                    **{f"{m}_flagged": sum(flags[m]) for m in flags},
                    "clip_iqa_mean": statistics.mean(quality),
                    "alignment_mean": statistics.mean(alignment),
                    "near_blank_count": sum(x is True for x in blank) if all(x is not None for x in blank) else None,
                }
            )
    table(out / "condition_summary.csv", rows)
    table(out / "paired_scores.csv", paired)
    return rows


def run(args):
    if not os.environ.get("PBS_JOBID"):
        raise ValueError("run requires PBS; plan is the CPU-only inspection command")
    from visual_synthesis.benchmark_cluster import _check_model_files
    from visual_synthesis.qwen_quantization import checkpoint_snapshot

    root, source, out = workspace_path(args.storage), args.source.resolve(), args.output.resolve()
    if (
        not out.is_relative_to(root / "outputs")
        or out == root / "outputs"
        or out.is_relative_to(source)
        or source.is_relative_to(out)
        or not source.is_relative_to(root / "outputs")
    ):
        raise ValueError("use distinct source and output runs within runtime/outputs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("calibration output must be fresh and empty")
    if git_command("branch", "--show-current")["stdout"].strip() != "swara/experiments":
        raise ValueError("expected swara/experiments; no branch changes are performed")
    out.mkdir(parents=True, exist_ok=True)
    isolate_caches(out)
    started = time.monotonic()
    deadline = started + args.wall_seconds - 180
    summary = {
        "status": "running",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "time_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "output": str(out),
        "phases": {},
        "git_commit": git_command("rev-parse", "HEAD")["stdout"].strip(),
    }
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("PBS termination")))
    try:
        save(summary, out / "calibration_summary.json")
        guard = download_guard(storage_snapshot(root, scan=True), 0, 256, args.budget_gib)
        if not guard["allowed"]:
            raise ValueError(guard["reason"])
        plan = make_plan(source, args.prompt_ids.split(","), args.alphas.split(","))
        save(plan, out / "plan.json")
        if plan["rough_generation_seconds"] + 600 > args.wall_seconds:
            raise ValueError("allocation is too short for the planned image count and setup reserve")
        require_cached_evaluators(root)
        cfg = copy.deepcopy(read(source / "baseline/spec.json")["config"])
        cfg["paths"].update(
            models=str(root / "models"),
            cache=str(out / "cache"),
            runs=str(out / "pipeline-runs"),
            logs=str(root / "logs"),
        )
        cfg["detectors"]["clip-h14"].update(local_dir="clip-h14", url=None)
        # Preserve the source evaluator settings/thresholds while rebinding local model/cache paths.
        for metric in ("nudenet", "clip-h14"):
            recorded = read(source / metric / "spec.json")["config"]["detectors"][metric]
            cfg["detectors"][metric].update(
                {k: recorded.get(k) for k in ("threshold", "flag_labels", "label_thresholds")}
            )
        _check_model_files(plan["model_settings"], root / "models/qwen-image")
        snapshot = checkpoint_snapshot(root / "models/qwen-image")
        save(snapshot, out / "checkpoint.json")
        save(cfg, out / "config.json")
        (out / "all-images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in plan["baseline_images"]))
        summary["generation"] = generate(plan, cfg, out, deadline)
        save(summary, out / "calibration_summary.json")
        for index, metric in enumerate(("nudenet", "clip-h14", "alignment", "clip-iqa")):
            target = out / metric
            spec = {
                "phase": metric,
                "model": "qwen-image",
                "config": cfg,
                "output": str(target),
                "images": str(out / "all-images.jsonl"),
                "clip_source": str(root / "models/openai--clip-vit-base-patch32"),
            }
            save(spec, out / f"{metric}-spec.json")
            timeout = max(1, min(600, int((deadline - time.monotonic()) / (4 - index))))
            log = root / "logs" / f"{out.name}.{metric}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            code = run_logged(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/aspire_project_pilot.py"),
                    "worker",
                    "--spec",
                    str(out / f"{metric}-spec.json"),
                ],
                log,
                timeout,
            )
            report = read(target / "report.json") if (target / "report.json").is_file() else {"status": "failed"}
            summary["phases"][metric] = report
            save(summary, out / "calibration_summary.json")
            if code or report["status"] != "success":
                raise RuntimeError(f"{metric} failed; see {log}")
        if checkpoint_snapshot(root / "models/qwen-image")["fingerprint"] != snapshot["fingerprint"]:
            raise ValueError("shared checkpoint changed during calibration")
        summary.update(
            status="complete",
            comparison=summarize(plan, out),
            note="Inspect images alongside scores; near-blank outputs never establish useful safety. "
            "These four prompts are diagnostic tuning data. Confirm any candidate on fresh, "
            "reviewed prompts and multiple seeds/random directions before claiming effectiveness.",
        )
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
        summary.update(elapsed_seconds=time.monotonic() - started, accounting=accounting(os.environ["PBS_JOBID"]))
        save(summary, out / "calibration_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--source", type=Path, default=os.environ.get("VS_CALIBRATION_SOURCE"))
    parser.add_argument("--output", type=Path, default=os.environ.get("VS_CALIBRATION_OUTPUT"))
    parser.add_argument("--storage", default=os.environ.get("VS_ASPIRE_STORAGE"))
    parser.add_argument("--budget-gib", type=float, default=float(os.environ.get("VS_ASPIRE_BUDGET_GIB", "0")))
    parser.add_argument("--wall-seconds", type=int, default=5400)
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--prompt-ids", default=",".join(DEFAULT_IDS))
    args = parser.parse_args()
    if not args.source or (args.action == "run" and (not args.output or not args.storage)):
        parser.error("--source is required; run also needs --output and --storage")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = (
        make_plan(args.source, args.prompt_ids.split(","), args.alphas.split(","))
        if args.action == "plan"
        else run(args)
    )
    print(json.dumps(result, indent=2))
    return 0 if args.action == "plan" or result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
