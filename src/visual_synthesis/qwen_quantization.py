"""Paired Qwen precision trials; cached weights only and one fresh process per arm."""

import hashlib
import html
import importlib.metadata
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.benchmark_sampling import load_prompts, sample_prompts
from visual_synthesis.io import append_jsonl, file_sha256
from visual_synthesis.project_pilot import read_jsonl, save_json

log = logging.getLogger(__name__)
BNB_VERSION = "0.48.2"
ARMS = ("bf16", "nf4")


def prepare_pairs(source, output, count=12, seed=42):
    frame, sampling = sample_prompts(load_prompts(source), count, seed, "auto")
    if len(frame) != count:
        raise ValueError("input cannot supply the exact requested sample size")
    seeds = frame.get("evaluation_seed", pd.Series(index=frame.index, dtype=object))
    frame["evaluation_seed"] = [
        int(value) if pd.notna(value) else
        int.from_bytes(hashlib.sha256(f"{seed}:{pid}".encode()).digest()[:4], "big") % (2**31)
        for pid, value in zip(frame.prompt_id, seeds, strict=True)
    ]
    if any(not 0 <= value < 2**63 for value in frame.evaluation_seed):
        raise ValueError("evaluation_seed must be between 0 and 2**63-1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "prompts.csv", index=False)
    manifest = {"source": str(Path(source).resolve()), "source_sha256": file_sha256(source),
                "sample_sha256": file_sha256(output / "prompts.csv"), "images_per_arm": len(frame),
                "sampling": sampling, "sampling_seed": seed,
                "seed_policy": "preserve evaluation_seed; stable SHA256 fallback per prompt ID"}
    save_json(manifest, output / "input_manifest.json")
    return manifest


def checkpoint_snapshot(source):
    """Detect changes without reading/hash-copying 54 GiB of checkpoint tensors."""
    source = Path(source).resolve()
    entries = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        stat = path.stat()
        entries[str(relative)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix in (".json", ".txt"):
            entries[str(relative)]["sha256"] = file_sha256(path)
    if "model_index.json" not in entries:
        raise ValueError("Qwen's cached model_index.json is missing; no model downloads are performed")
    return {"path": str(source), "files": entries,
            "logical_bytes": sum(item["bytes"] for item in entries.values()),
            "fingerprint": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
            "note": "Metadata/config fingerprint; tensor contents are not fully hashed."}


def quantization_config():
    import torch
    from diffusers import BitsAndBytesConfig, PipelineQuantizationConfig

    # Granular mapping avoids the differing config signatures in pinned
    # Diffusers 0.41 / Transformers 5.20. Quantize only the diffusion transformer.
    return PipelineQuantizationConfig(quant_mapping={
        "transformer": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    })


def gpu_preflight(output):
    import bitsandbytes as bnb
    import torch

    from visual_synthesis.benchmark import machine_info

    if not os.environ.get("PBS_JOBID") or not torch.cuda.is_available():
        raise ValueError("NF4 preflight requires a PBS GPU allocation")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("allocated GPU does not support BF16")
    if importlib.metadata.version("bitsandbytes") != BNB_VERSION:
        raise ValueError(f"expected the isolated bitsandbytes=={BNB_VERSION} installation")
    quantization_config()
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                             quant_type="nf4", compress_statistics=True).to("cuda")
    with torch.inference_mode():
        value = layer(torch.ones((1, 64), device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    if not torch.isfinite(value).all():
        raise ValueError("NF4 CUDA kernel returned nonfinite values")
    info = machine_info(torch, "cuda:0")
    info.update(status="ok", bitsandbytes=BNB_VERSION, nf4_kernel="ok")
    save_json(info, output)
    return info


def build_backend(spec, arm):
    import torch
    from diffusers import QwenImagePipeline

    from visual_synthesis.backends.image import DiffusersBackend

    if arm not in ARMS:
        raise ValueError("unknown precision arm")
    model = spec["model"]
    options = {"dtype": torch.bfloat16, "local_files_only": True}
    if arm != "bf16":
        options["quantization_config"] = quantization_config()
    pipe = QwenImagePipeline.from_pretrained(spec["source"], **options)
    pipe.enable_model_cpu_offload(device=torch.device("cuda:0"))
    if not getattr(pipe, "_all_hooks", []):
        raise ValueError("requested CPU offload hooks are not active")
    if model["enable_attention_slicing"]:
        pipe.enable_attention_slicing()
    if model["enable_xformers"]:
        raise ValueError("this comparison requires the standard SDPA attention backend")
    if model["scheduler"]:
        raise ValueError("custom scheduler overrides are unsupported in this paired trial")
    pipe.set_progress_bar_config(disable=True)
    backend = DiffusersBackend(model)
    backend.pipe = pipe
    return backend


def generation_worker(spec):
    import torch

    from visual_synthesis.backends.image import save_image
    from visual_synthesis.benchmark import Resources, machine_info

    out, arm = Path(spec["output"]), spec["arm"]
    report = {"status": "running", "arm": arm, "sample_sha256": spec["sample_sha256"],
              "settings_sha256": spec["settings_sha256"], "checkpoint_fingerprint": spec["checkpoint_fingerprint"],
              "model_settings": spec["model"], "images": 0, "timings": [], "offload_policy": "model_cpu_offload",
              "quantized_components": [] if arm == "bf16" else ["transformer"]}
    out.mkdir(parents=True, exist_ok=True)
    save_json(report, out / "report.json")
    probe = backend = None
    try:
        if not os.environ.get("PBS_JOBID") or not torch.cuda.is_available():
            raise ValueError("generation requires a PBS GPU allocation")
        report["machine"] = machine_info(torch, "cuda:0")
        report["machine"]["packages"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        if file_sha256(spec["sample"]) != spec["sample_sha256"]:
            raise ValueError("paired prompt file changed")
        rows = load_prompts(spec["sample"]).astype(object)
        rows = rows.where(pd.notna(rows), "").to_dict("records")
        with Resources(torch, "cuda:0") as probe:
            tick = time.perf_counter()
            backend = build_backend(spec, arm)
            probe.sync()
            report["load_seconds"] = time.perf_counter() - tick
            import bitsandbytes as bnb

            quantized = [module for module in backend.pipe.transformer.modules()
                         if isinstance(module, bnb.nn.Linear4bit)]
            report["nf4_linear_count"] = len(quantized)
            if arm == "bf16" and quantized:
                raise ValueError("BF16 arm unexpectedly contains quantized layers")
            if arm != "bf16" and (not quantized or any(
                getattr(module.weight, "quant_type", None) != "nf4" for module in quantized
            )):
                raise ValueError("NF4 transformer replacement did not produce NF4 layers")

            def generate(row):
                seed = int(row["evaluation_seed"])
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                return backend.generate([{"prompt_id": str(row["prompt_id"]),
                                          "prompt": row["prompt"], "seed": seed}])[0]["image"]

            tick = time.perf_counter()
            image = generate(rows[0])
            probe.sync()
            image.close()
            report["warmup_seconds"] = time.perf_counter() - tick
            save_json(report, out / "report.json")
            for index, row in enumerate(rows):
                probe.sync()
                tick = time.perf_counter()
                image = generate(row)
                probe.sync()
                inference = time.perf_counter() - tick
                write_tick = time.perf_counter()
                path = out / "images" / f"{index:04d}.png"
                save_image(image, path)
                image.close()
                write_seconds = time.perf_counter() - write_tick
                append_jsonl(out / "images.jsonl", {
                    "condition": arm, "prompt_id": str(row["prompt_id"]), "prompt": row["prompt"],
                    "evaluation_seed": int(row["evaluation_seed"]), "image_path": str(path),
                    "sha256": file_sha256(path), "image_bytes": path.stat().st_size,
                    "inference_seconds": inference, "png_write_seconds": write_seconds,
                })
                report["timings"].append({"inference_seconds": inference, "png_write_seconds": write_seconds})
                report["images"] = index + 1
                report["resources"] = {**probe.peaks,
                                       "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                                       "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved()}
                save_json(report, out / "report.json")
                log.info("%s %d/%d: inference %.2fs, PNG %.2fs", arm, index + 1, len(rows), inference, write_seconds)
            report["mean_inference_seconds"] = float(np.mean([r["inference_seconds"] for r in report["timings"]]))
            report["mean_inference_and_png_seconds"] = float(np.mean([
                sum(r.values()) for r in report["timings"]
            ]))
            report["generation_hours_6000_excluding_reloads"] = (
                report["load_seconds"] + report["warmup_seconds"] +
                6000 * report["mean_inference_and_png_seconds"]
            ) / 3600
            report["status"] = "success"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        log.exception("%s failed", arm)
    finally:
        if probe:
            report["resources"] = probe.peaks
        save_json(report, out / "report.json")
        if backend:
            backend.unload()
    return report


def compare_reports(reports, expected_images):
    """Only complete, matched protocols yield a speedup; failures stay unknown."""
    result = {"status": "incomplete", "arms": reports,
              "note": "NF4 can change outputs. Pixel differences do not measure quality or safety. "
                      "6000-image forecasts exclude queueing, reloads across PBS chunks, evaluation and ablations."}
    a, b = reports.get("bf16", {}), reports.get("nf4", {})
    if any(r.get("status") != "success" or r.get("images") != expected_images for r in (a, b)):
        return result
    for key in ("sample_sha256", "settings_sha256", "checkpoint_fingerprint", "offload_policy"):
        if not a.get(key) or a[key] != b.get(key):
            raise ValueError(f"cannot compare mismatched {key}")
    if a.get("nf4_linear_count") != 0 or b.get("nf4_linear_count", 0) <= 0:
        raise ValueError("BF16/NF4 layer evidence is missing")
    result.update(status="complete", inference_speedup_bf16_over_nf4=
                  a["mean_inference_seconds"] / b["mean_inference_seconds"],
                  inference_and_png_speedup_bf16_over_nf4=
                  a["mean_inference_and_png_seconds"] / b["mean_inference_and_png_seconds"])
    return result


def paired_outputs(output):
    """Join image pairs by identity and seed; export a local HTML comparison."""
    from PIL import Image

    output = Path(output)
    rows = {}
    for arm in ARMS:
        path = output / arm / "images.jsonl"
        if not path.is_file():
            return []
        records = read_jsonl(path)
        rows[arm] = {(row["prompt_id"], row["evaluation_seed"]): row for row in records}
        if len(rows[arm]) != len(records):
            raise ValueError("duplicate image pair identity")
    if rows["bf16"].keys() != rows["nf4"].keys():
        raise ValueError("BF16/NF4 image IDs or seeds differ")
    pairs, sections = [], []
    for key, a in rows["bf16"].items():
        b = rows["nf4"][key]
        if a["prompt"] != b["prompt"]:
            raise ValueError("paired prompt text differs")
        with Image.open(a["image_path"]) as first, Image.open(b["image_path"]) as second:
            x = np.asarray(first.convert("RGB"), dtype=np.float32)
            y = np.asarray(second.convert("RGB"), dtype=np.float32)
        if x.shape != y.shape:
            raise ValueError("paired image dimensions differ")
        mse = float(np.mean((x - y)**2))
        pairs.append({"prompt_id": key[0], "evaluation_seed": key[1], "prompt": a["prompt"],
                      "bf16_seconds": a["inference_seconds"], "nf4_seconds": b["inference_seconds"],
                      "speedup": a["inference_seconds"] / b["inference_seconds"],
                      "pixel_mae_0_255": float(np.mean(np.abs(x - y))),
                      "psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
                      "identical_pixels": mse == 0})
        images = "".join(f'<figure><figcaption>{arm.upper()}</figcaption><img loading="lazy" src="'
                         f'{html.escape(str(Path(row["image_path"]).relative_to(output)), quote=True)}"></figure>'
                         for arm, row in (("bf16", a), ("nf4", b)))
        sections.append(f'<section><h2>{html.escape(key[0])} · seed {key[1]}</h2>'
                        f'<p>{html.escape(a["prompt"])}</p><div>{images}</div></section>')
    pd.DataFrame(pairs).to_csv(output / "paired_images.csv", index=False)
    (output / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Qwen BF16 vs NF4</title>'
        '<style>body{font:16px system-ui;max-width:1300px;margin:2rem auto;padding:1rem}'
        'section{border-top:1px solid #bbb;padding:1rem 0}section div{display:flex;gap:1rem}'
        'figure{margin:0;flex:1;min-width:0}img{width:100%}figcaption{font-weight:bold}</style>'
        '<h1>Qwen BF16 vs NF4</h1><p>Identical prompts/settings/seeds. '
        'Visual and pixel differences are not quality or safety scores.</p>' + "".join(sections), encoding="utf-8"
    )
    return pairs
