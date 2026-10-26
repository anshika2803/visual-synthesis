"""Small, hardware-specific pilots of the existing pretrained evaluation pipeline.

Each model runs in a fresh process, using production requests and image/record
writers. One model copy, sequential prompts; no synthetic training estimate.
"""

import copy
import importlib.metadata
import json
import logging
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.backends.image import build_image_backend
from visual_synthesis.config import get_detector, get_model
from visual_synthesis.io import append_jsonl, file_sha256, find_table, read_table, safe_name, select_rows, write_json
from visual_synthesis.pipeline import find_input, run_stage
from visual_synthesis.runs import Run, git_commit
from visual_synthesis.stages import s02_standardize, s03_unify, s04_postprocess, s05_generate, s06_nsfw, s07_metrics

log = logging.getLogger(__name__)


class Resources:
    """Process RSS and accelerator memory, sampled every 100 ms.

    CUDA allocator peaks are exact for this process (not total device usage).
    MPS values are sampled, share system RAM, and may miss short-lived peaks.
    """

    def __init__(self, torch=None, device="cpu"):
        self.torch, self.device = torch, device
        self.peaks = {}
        self.stop = threading.Event()
        self.thread = None
        try:
            import psutil

            self.process = psutil.Process()
        except ImportError:
            self.process = None

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def sample(self):
        if self.process:
            self.peaks["rss_peak_sampled_bytes"] = max(
                self.peaks.get("rss_peak_sampled_bytes", 0), self.process.memory_info().rss
            )
        if self.device == "mps":
            for label, method in (("mps_allocated_peak_sampled_bytes", "current_allocated_memory"),
                                  ("mps_driver_peak_sampled_bytes", "driver_allocated_memory")):
                if hasattr(self.torch.mps, method):
                    self.peaks[label] = max(self.peaks.get(label, 0), getattr(self.torch.mps, method)())

    def _poll(self):
        while not self.stop.wait(0.1):
            self.sample()

    def __enter__(self):
        self.sync()
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        try:
            self.sync()
        finally:
            self.stop.set()
            self.thread.join()
            self.sample()
            # Unlike sampled RSS, this high-water mark includes short-lived
            # allocations but is for the entire fresh worker's lifetime.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peaks["rss_process_lifetime_peak_bytes"] = int(rss * (1 if sys.platform == "darwin" else 1024))
            if self.device.startswith("cuda"):
                self.peaks["cuda_allocated_peak_bytes"] = self.torch.cuda.max_memory_allocated(self.device)
                self.peaks["cuda_reserved_peak_bytes"] = self.torch.cuda.max_memory_reserved(self.device)


def machine_info(torch=None, device="cpu"):
    info = {
        "hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(), "device": device,
        "packages": {},
    }
    for name in ("torch", "diffusers", "transformers", "accelerate", "pandas", "nudenet"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import psutil

        info["system_ram_bytes"] = psutil.virtual_memory().total
        info["system_available_ram_bytes_at_start"] = psutil.virtual_memory().available
    except ImportError:
        pass
    if device.startswith("cuda"):
        info.update(gpu=torch.cuda.get_device_name(device),
                    gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                    cuda_runtime=torch.version.cuda)
    elif device == "mps":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=False).stdout.strip()
        info.update(gpu=f"Apple Metal: {chip or 'unknown chip'} (unified memory)")
    return info


def summarize_model(result, target_prompts, passes):
    """Extrapolate only complete pilots; failures never become cheap successes."""
    if result["status"] != "success":
        return None
    times = result["generation"]["prompt_seconds_with_io"]
    mean = float(np.mean(times))
    target_images = target_prompts * passes * result["images_per_prompt"]
    # Model load and warmup occur once per pass, rather than once per prompt.
    fixed = result["load_seconds"] + result["warmup_seconds"] + result["generation"]["table_write_seconds_mean"]
    generation = passes * (fixed + target_prompts * mean)
    evaluation = result.get("evaluation")
    evaluation_seconds = None
    if evaluation and evaluation["status"] == "success":
        # Full-stage wall time includes evaluator load/table/plot setup. Linear
        # scaling is intentionally conservative, not a steady-state measurement.
        evaluation_seconds = target_images * evaluation["seconds"] / evaluation["images"]
    return {
        "target_prompts_per_model_per_pass": target_prompts, "passes": passes, "target_images": target_images,
        "mean_seconds_per_prompt_with_io": mean,
        "p50_seconds_per_prompt_with_io": float(np.percentile(times, 50)),
        "p95_seconds_per_prompt_with_io": float(np.percentile(times, 95)),
        "generation_hours": generation / 3600,
        "evaluation_hours_conservative": evaluation_seconds / 3600 if evaluation_seconds is not None else None,
        "pipeline_hours_conservative": (generation + evaluation_seconds) / 3600
        if evaluation_seconds is not None else None,
        "image_storage_gib": target_images * result["generation"]["mean_image_bytes"] / 2**30,
        "training_hours_in_current_scope": 0,
        "training_status": "out_of_scope: pretrained evaluation; no training stage implemented",
    }


def evaluate(cfg, generation_run, resources):
    """Time the actual detection and metrics stages on the final pilot repeat."""
    result = {"status": "failed", "stages": []}
    cfg["metrics"]["generations_input"] = str(generation_run.dir)
    source = generation_run.dir
    started = time.perf_counter()
    for stage in (s06_nsfw, s07_metrics):
        resources.sync()
        tick = time.perf_counter()
        stage_result = run_stage(cfg, stage, run_name="evaluation", input_path=source)
        resources.sync()
        result["stages"].append({**stage_result, "wall_seconds": time.perf_counter() - tick})
        if stage_result["status"] != "success" or stage_result.get("errors", 0):
            return result
        source = Path(stage_result["run_dir"])
    payload = json.loads((source / cfg["metrics"]["json_name"]).read_text())
    unavailable = {name: payload[name] for name in ("alignment", "fid")
                   if cfg["metrics"][name]["enabled"] and payload[name]["status"] != "ok"}
    if unavailable:
        return {**result, "unavailable_evaluators": unavailable}
    return {**result, "status": "success", "seconds": time.perf_counter() - started,
            "images": read_table(generation_run.path(cfg["generate"]["output_name"])).shape[0]}


def benchmark_worker(spec):
    """One loaded model across warmup and repeats; executed in its own process."""
    cfg = copy.deepcopy(spec["config"])
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["runs"], cfg["paths"]["logs"] = str(out / "runs"), str(out / "logs")
    cfg["generate"]["model"] = spec["model"]
    cfg["run"].update(dry_run=False, limit=None, overwrite=False, resume_latest=False)
    cfg["stages"] = {}
    model = get_model(cfg)
    result = {"model": model["id"], "status": "failed", "backend": model["backend"],
              "synthetic": model["backend"] == "fake", "model_settings": model,
              "images_per_prompt": model["num_images_per_prompt"], "workers": 1,
              "batching": "sequential prompts, matching the existing image backend",
              "load_may_include_download": spec["allow_download"], "resources": {}}
    backend = None
    torch = None
    device = "cpu"
    probe = None
    try:
        if model["backend"] != "fake":
            import torch

            from visual_synthesis.backends.device import pick_device, pick_dtype

            device = pick_device(model["device"], model["device_index"])
            result["resolved_dtype"] = str(pick_dtype(model["dtype"], device))
        result["machine"] = machine_info(torch, device)
        sample = read_table(spec["sample"])
        records = sample.to_dict("records")
        result["sample_prompts"] = len(records)
        backend = build_image_backend(model, cfg["paths"]["models"])
        with Resources(torch, device) as probe:
            tick = time.perf_counter()
            backend.load()
            probe.sync()
            result["load_seconds"] = time.perf_counter() - tick
            pipe = getattr(backend, "pipe", None)
            result["loaded_pipeline_class"] = type(pipe).__name__
            result["cpu_offload_hooks_active"] = bool(getattr(pipe, "_all_hooks", []))
            weights_source = getattr(backend, "weights_source", lambda: None)()
            if weights_source:
                weights_path = Path(weights_source)
                if not weights_path.is_dir():
                    from huggingface_hub import snapshot_download

                    try:
                        weights_path = Path(snapshot_download(model["repo_id"], revision=model["revision"],
                                                              local_files_only=True))
                    except OSError:
                        weights_path = None
                if weights_path:
                    result["local_weight_files_logical_bytes"] = sum(
                        p.stat().st_size for p in weights_path.rglob("*") if p.is_file()
                    )
            result["output_disk_free_bytes_at_start"] = shutil.disk_usage(out).free
            tick = time.perf_counter()
            for index in range(spec["warmup"]):
                request = s05_generate.make_request(
                    records[index % len(records)], spec["seed"] + index, cfg["generate"]
                )
                images = backend.generate([request])
                probe.sync()
                del images
            result["warmup_seconds"] = time.perf_counter() - tick
            timings, inference_timings, image_sizes, table_times, repeat_seconds = [], [], [], [], []
            for repeat in range(spec["repeats"]):
                run = Run(cfg, s05_generate.SLUG, run_name=f"repeat-{repeat + 1}", mode="full")
                repeat_start = time.perf_counter()
                for position, row in enumerate(records):
                    request = s05_generate.make_request(row, spec["seed"] + position + repeat * len(records),
                                                        cfg["generate"])
                    probe.sync()
                    tick = time.perf_counter()
                    images = backend.generate([request])
                    probe.sync()
                    inference_timings.append(time.perf_counter() - tick)
                    if len(images) != model["num_images_per_prompt"]:
                        raise RuntimeError(f"expected {model['num_images_per_prompt']} images, got {len(images)}")
                    for image in images:
                        record = s05_generate.save_record(image, row, model["id"], cfg["generate"], run)
                        append_jsonl(run.path("generations.jsonl"), record)
                        image_sizes.append(Path(record["image_path"]).stat().st_size)
                    timings.append(time.perf_counter() - tick)
                    del images
                    log.info("%s repeat %d/%d prompt %d/%d: %.2fs", model["id"], repeat + 1,
                             spec["repeats"], position + 1, len(records), timings[-1])
                tick = time.perf_counter()
                s05_generate.write_records_table(run.path("generations.jsonl"), run, cfg["generate"])
                table_times.append(time.perf_counter() - tick)
                repeat_seconds.append(time.perf_counter() - repeat_start)
                run.finish("success", {"rows_out": len(records) * model["num_images_per_prompt"]})
            result["generation"] = {
                "prompt_seconds_with_io": timings, "prompt_seconds_inference_only": inference_timings,
                "repeat_seconds": repeat_seconds, "table_write_seconds_mean": float(np.mean(table_times)),
                "mean_image_bytes": float(np.mean(image_sizes)), "images_written": len(image_sizes),
            }
            # The existing pipeline unloads generation before loading evaluators.
            backend.unload()
            backend = None
            if spec["evaluate"]:
                result["evaluation"] = evaluate(cfg, run, probe)
            result["status"] = "success"
    except Exception as exc:
        log.exception("pilot failed for %s", model["id"])
        result.update(error=f"{type(exc).__name__}: {exc}")
    finally:
        if probe is not None:
            result["resources"] = probe.peaks
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                log.exception("model cleanup failed")
    result["estimate"] = summarize_model(result, spec["target_prompts"], spec["passes"])
    write_json(result, out / "report.json")
    return result


def project_summary(results, preprocessing_seconds=0):
    generated = [r for r in results if not r.get("synthetic") and r.get("estimate")]
    complete = [r for r in results if not r.get("synthetic") and r.get("estimate")
                and r["estimate"]["pipeline_hours_conservative"] is not None]
    summary = {
        "requested_models": len(results), "measured_complete_models": len(complete),
        "status": "complete" if len(complete) == len(results) else "incomplete",
        "generation_hours_measured_models": sum(r["estimate"]["generation_hours"] for r in generated),
        "pipeline_hours_conservative_measured_models": sum(
            r["estimate"]["pipeline_hours_conservative"] for r in complete
        ),
        "image_storage_gib_measured_models": sum(r["estimate"]["image_storage_gib"] for r in complete),
        "preprocessing_seconds_once": preprocessing_seconds,
    }
    summary["full_project_hours_conservative"] = (
        summary["pipeline_hours_conservative_measured_models"] + preprocessing_seconds / 3600
        if summary["status"] == "complete" else None
    )
    # Two models per person: on equivalent dedicated machines, pair slowest with
    # fastest to minimize the longest pair. Never divide serial time by four on
    # a shared GPU, nor recommend pairings across different measured hardware.
    hardware = {json.dumps({k: r["machine"].get(k) for k in
                           ("device", "gpu", "gpu_memory_bytes", "machine", "cpu_count",
                            "system_ram_bytes", "packages")},
                          sort_keys=True) for r in complete}
    if len(complete) == len(results) == 8 and len(hardware) == 1:
        ordered = sorted(complete, key=lambda r: r["estimate"]["pipeline_hours_conservative"])
        pairs = []
        for i in range(4):
            pair = [ordered[i], ordered[-1 - i]]
            pairs.append({"models": [r["model"] for r in pair],
                          "hours": sum(r["estimate"]["pipeline_hours_conservative"] for r in pair)})
        summary["suggested_pairs_if_four_equivalent_dedicated_devices"] = pairs
        summary["four_device_wall_hours_conservative"] = max(p["hours"] for p in pairs)
    return summary


def write_reports(payload, output):
    output = Path(output)
    write_json(payload, output / "report.json")
    rows = []
    for result in payload["models"]:
        estimate = result.get("estimate") or {}
        memory = result.get("resources", {})
        rows.append({
            "model": result["model"], "status": result["status"], "synthetic": result.get("synthetic", False),
            "device": result.get("machine", {}).get("device"), "gpu": result.get("machine", {}).get("gpu"),
            "load_seconds": result.get("load_seconds"),
            "mean_seconds_per_prompt": estimate.get("mean_seconds_per_prompt_with_io"),
            "generation_hours": estimate.get("generation_hours"),
            "evaluation_hours_conservative": estimate.get("evaluation_hours_conservative"),
            "pipeline_hours_conservative": estimate.get("pipeline_hours_conservative"),
            "image_storage_gib": estimate.get("image_storage_gib"),
            "rss_peak_gib": memory.get("rss_process_lifetime_peak_bytes", 0) / 2**30,
            "cuda_allocated_peak_gib": memory.get("cuda_allocated_peak_bytes", 0) / 2**30
            if "cuda_allocated_peak_bytes" in memory else None,
            "error": result.get("error"),
        })
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)


def run_benchmark(cfg, model_ids, output, input_path=None, sample_size=60, warmup=2, repeats=3,
                  target_prompts=6000, passes=1, seed=42, evaluate_models=True, allow_download=False,
                  device=None, cpu_offload=False, prepare=False, timeout=3600):
    """Run fresh subprocesses sequentially, retaining failures and partial reports."""
    if min(sample_size, repeats, target_prompts, passes, timeout) < 1 or warmup < 0:
        raise ValueError("sample size, repeats, target prompts, passes and timeout must be positive; warmup >= 0")
    model_ids = list(dict.fromkeys(model_ids))
    if not model_ids:
        raise ValueError("select at least one model")
    for name in model_ids:
        get_model(cfg, name)
    if cfg["generate"]["workers"] != 1 or cfg["nsfw"]["workers"] != 1:
        raise ValueError("this pilot profiles one model/detector copy; set generate.workers=1 and nsfw.workers=1")
    if cpu_offload and device != "cuda":
        raise ValueError("--cpu-offload requires --device cuda")
    if evaluate_models and get_detector(cfg) is None:
        raise ValueError("evaluation requires a detector; select one or pass --generation-only")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new --output so measurements cannot resume or be overwritten")
    output.mkdir(parents=True)
    local_cfg = copy.deepcopy(cfg)
    if not allow_download:
        # The CLIP-H/14 head uses a direct URL, outside Hugging Face's offline
        # controls. Missing heads should fail clearly instead of being fetched.
        for detector in local_cfg["detectors"].values():
            detector["url"] = None
    preprocessing = []
    source = find_input(cfg, s05_generate, input_path) if not prepare else None
    if prepare:
        if input_path:
            raise ValueError("--prepare uses paths.raw; omit --input")
        prepare_cfg = copy.deepcopy(cfg)
        prepare_cfg["paths"].update(runs=str(output / "preprocess"), logs=str(output / "preprocess-logs"))
        prepare_cfg["run"].update(dry_run=False, limit=None, resume_latest=False)
        prepare_cfg["stages"] = {}
        for stage in (s02_standardize, s03_unify, s04_postprocess):
            tick = time.perf_counter()
            stage_result = run_stage(prepare_cfg, stage, run_name="prepare", input_path=source)
            preprocessing.append({**stage_result, "wall_seconds": time.perf_counter() - tick})
            if stage_result["status"] != "success":
                raise RuntimeError(f"preprocessing failed: {stage_result}")
            source = stage_result["run_dir"]
    source_table = find_table(source, cfg["dedup"]["output_name"])
    frame = read_table(source_table)
    required = {"prompt_id", "prompt"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("input must be a nonempty prompt table with prompt_id and prompt")
    if frame[list(required)].isna().any().any() or frame["prompt_id"].astype(str).duplicated().any():
        raise ValueError("prompt IDs must be unique and prompt/ID values must be non-null")
    sample = select_rows(frame, sample_size, "stratified" if "standard_category" in frame else "random",
                         "standard_category", seed)
    sample_path = output / "sample.parquet"
    sample.to_parquet(sample_path, index=False)
    if device:
        for model in local_cfg["models"].values():
            model["device"] = device
        for model in local_cfg["metric_models"].values():
            model["device"] = device
        # NudeNet is CPU/ONNX regardless of this preference.
        for detector in local_cfg["detectors"].values():
            detector["device"] = device
    if cpu_offload:
        for name in model_ids:
            local_cfg["models"][name]["enable_cpu_offload"] = True
    payload = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
        "source_table": str(source_table), "source_sha256": file_sha256(source_table), "dataset_rows": len(frame),
        "sample_sha256": file_sha256(sample_path), "sample_size": len(sample), "seed": seed,
        "warmup": warmup, "repeats": repeats, "target_prompts": target_prompts, "passes": passes,
        "evaluation_protocol": {"enabled": evaluate_models, "nsfw": cfg["nsfw"], "metrics": cfg["metrics"]},
        "preprocessing": preprocessing, "models": [],
        "assumptions": [
            "Pretrained evaluation only; no training or learned intervention exists in the repository.",
            "One model copy per device; timings apply only to the recorded hardware and exact settings.",
            "Generation includes prompt encoding, sampling, decoding, image writing and JSONL fsync.",
            "Evaluation is conservatively scaled from whole-stage wall time, including model load and plots.",
            "Downloads (unless allowed), queue time, manual analysis, retries and extra experiments excluded.",
            "FID is timed only if enabled with reference images; small-pilot FID is not a scientific score.",
            "MPS memory shares system RAM; do not add MPS allocation to RSS. Memory does not scale with prompt count.",
            "Synthetic backends validate the harness and never contribute to project resource totals.",
        ],
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                            environment.get("PYTHONPATH")]))
    if not allow_download:
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in model_ids:
        model_output = output / safe_name(name)
        model_output.mkdir()
        spec = {"config": local_cfg, "model": name, "output": str(model_output), "sample": str(sample_path),
                "warmup": warmup, "repeats": repeats, "seed": seed, "target_prompts": target_prompts,
                "passes": passes, "evaluate": evaluate_models, "allow_download": allow_download}
        log.info("benchmarking %s; live log: %s", name, model_output / "worker.log")
        with tempfile.TemporaryDirectory(prefix="vs-benchmark-") as scratch:
            spec_path = Path(scratch) / "spec.json"
            spec_path.write_text(json.dumps(spec))
            with (model_output / "worker.log").open("w") as handle:
                try:
                    process = subprocess.run([sys.executable, "-m", "visual_synthesis.benchmark", str(spec_path)],
                                             stdout=handle, stderr=subprocess.STDOUT, env=environment,
                                             timeout=timeout, check=False)
                    report_path = model_output / "report.json"
                    if process.returncode == 0 and report_path.exists():
                        result = json.loads(report_path.read_text())
                    else:
                        result = {"model": name, "status": "failed", "estimate": None,
                                  "error": f"worker exited {process.returncode}; see {model_output / 'worker.log'}"}
                except subprocess.TimeoutExpired:
                    result = {"model": name, "status": "failed", "estimate": None,
                              "error": f"pilot exceeded {timeout}s; see {model_output / 'worker.log'}"}
        payload["models"].append(result)
        payload["project"] = project_summary(payload["models"], sum(r["wall_seconds"] for r in preprocessing))
        payload["project"]["requested_models"] = len(model_ids)
        if len(payload["models"]) != len(model_ids):
            payload["project"]["status"] = "incomplete"
            payload["project"]["full_project_hours_conservative"] = None
        write_reports(payload, output)
        log.info("%s: %s", name, result["status"])
    return payload


def merge_reports(paths, output):
    """Combine members' reports; reject incompatible workloads and duplicates."""
    reports = [json.loads(Path(path).read_text()) for path in paths]
    if not reports:
        raise ValueError("supply at least one report.json")
    reference = reports[0]
    for report in reports[1:]:
        for key in ("source_sha256", "sample_sha256", "seed", "target_prompts", "passes", "evaluation_protocol"):
            if report[key] != reference[key]:
                raise ValueError(f"reports disagree on {key}; use the same input, pilot sample and target workload")
    models = [r for report in reports for r in report["models"]]
    if len({r["model"] for r in models}) != len(models):
        raise ValueError("duplicate model results; select one hardware/config experiment per model")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = {**reference, "models": models, "source_reports": [str(p) for p in paths],
               "project": project_summary(models, sum(r["wall_seconds"] for r in reference["preprocessing"]))}
    write_reports(payload, output)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    benchmark_worker(json.loads(Path(sys.argv[1]).read_text()))
