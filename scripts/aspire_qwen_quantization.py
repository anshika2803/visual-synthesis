#!/usr/bin/env python3
"""Short paired Qwen trial, independent of the long project-pilot cache lock."""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aspire_trial_support import (
    ROOT,
    accounting,
    classify,
    configs,
    disk_bytes,
    download_guard,
    git_command,
    run_logged,
    save,
    storage_snapshot,
    workspace_path,
)


def isolate_caches(output):
    """Read shared model weights, but never share mutable HF/compile/pip caches."""
    cache = Path(output) / "cache"
    os.environ.update(
        HF_HOME=str(cache / "hf-home"), HF_HUB_CACHE=str(cache / "hf-home/hub"),
        HF_TOKEN_PATH=str(cache / "no-token"), HF_TOKEN="", HUGGING_FACE_HUB_TOKEN="",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        HF_XET_CACHE=str(cache / "xet"), XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache / "torch"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"), PIP_CACHE_DIR=str(cache / "pip"), TMPDIR=str(cache / "tmp"),
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
    )
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "tmp").mkdir(exist_ok=True)


def write_table(output, reports):
    rows = []
    for arm, report in reports.items():
        memory = report.get("resources", {})
        rows.append({"arm": arm, "status": report["status"], "images": report.get("images", 0),
                     "load_seconds_including_quantization": report.get("load_seconds"),
                     "warmup_seconds": report.get("warmup_seconds"),
                     "mean_inference_seconds": report.get("mean_inference_seconds"),
                     "mean_inference_and_png_seconds": report.get("mean_inference_and_png_seconds"),
                     "gpu_allocated_peak_gib": memory.get("cuda_allocated_peak_bytes", 0) / 2**30
                     if "cuda_allocated_peak_bytes" in memory else None,
                     "gpu_reserved_peak_gib": memory.get("cuda_reserved_peak_bytes", 0) / 2**30
                     if "cuda_reserved_peak_bytes" in memory else None,
                     "ram_peak_gib": memory.get("rss_process_lifetime_peak_bytes", 0) / 2**30
                     if "rss_process_lifetime_peak_bytes" in memory else None,
                     "nf4_linear_count": report.get("nf4_linear_count"),
                     "generation_hours_6000_excluding_reloads":
                     report.get("generation_hours_6000_excluding_reloads"), "error": report.get("error")})
    if rows:
        with (Path(output) / "comparison_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def run(args):
    if not os.environ.get("PBS_JOBID"):
        raise ValueError("submit via PBS; generation and dependency preparation are forbidden on the login node")
    from visual_synthesis.benchmark_cluster import _check_model_files
    from visual_synthesis.config import get_model
    from visual_synthesis.qwen_quantization import (
        BNB_VERSION,
        checkpoint_snapshot,
        compare_reports,
        paired_outputs,
        prepare_pairs,
    )

    root, out = workspace_path(args.storage), Path(args.output).resolve()
    if not out.is_relative_to(root / "outputs") or out == root / "outputs":
        raise ValueError("comparison output must be a dedicated run inside runtime/outputs")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "comparison_summary.json").exists():
        raise ValueError("comparison output already contains a run; submit a fresh run instead")
    isolate_caches(out)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    def log_path(phase):
        return logs / f"{out.name}.{phase}.log"

    summary = {"status": "running", "pbs_job_id": os.environ["PBS_JOBID"], "output": str(out),
               "time_utc": datetime.now(UTC).isoformat(), "arms": {},
               "git_commit": git_command("rev-parse", "HEAD")["stdout"].strip(),
               "protocol": "BF16 versus transformer-only NF4, BF16 compute, identical model CPU offload",
               "checkpoint_downloads": False, "checkpoint_saves": False}
    started = time.monotonic()
    deadline = started + args.wall_seconds - 180
    # Different from .aspire-trial.lock: only this comparison's writes are locked.
    lock = (out / ".comparison.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("PBS termination")))
    try:
        save(summary, out / "comparison_summary.json")
        before = storage_snapshot(root, scan=True)
        save(before, out / "storage-before.json")
        guard = download_guard(before, 512 * 1024**2, 256, args.budget_gib)
        if not guard["allowed"]:
            raise ValueError(guard["reason"])
        model = get_model(configs(), "qwen-image")
        model.update(device="cuda", dtype="bfloat16", enable_cpu_offload=True)
        source = Path(args.source or root / "models/qwen-image").resolve()
        _check_model_files(model, source)
        snapshot = checkpoint_snapshot(source)
        save(snapshot, out / "checkpoint.json")
        manifest = prepare_pairs(args.input, out, args.sample_size, args.seed)
        model_hash = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        order = ["nf4", "bf16"] if args.order == "nf4-first" else ["bf16", "nf4"]
        summary.update(input=manifest, source=str(source), checkpoint_fingerprint=snapshot["fingerprint"],
                       settings_sha256=model_hash, arm_order=order)
        summary["current_phase"] = "dependency-preparation"
        save(summary, out / "comparison_summary.json")
        dependencies = out / "dependencies"
        if dependencies.exists():
            raise ValueError("dependency directory already exists; submit a fresh run instead of overwriting it")
        # --no-deps/--target prevents pip from changing the active shared Conda environment.
        pip = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--no-cache-dir",
               "--disable-pip-version-check", "--only-binary=:all:", "--target", str(dependencies),
               f"bitsandbytes=={BNB_VERSION}"]
        print(f"Preparing isolated bitsandbytes; log: {log_path('dependencies')}", flush=True)
        if run_logged(pip, log_path("dependencies"), 600):
            raise RuntimeError("isolated bitsandbytes installation failed; inspect the dependencies log")
        os.environ["PYTHONPATH"] = str(dependencies) + os.pathsep + os.environ.get("PYTHONPATH", "")
        script = str(Path(__file__).resolve())
        summary["current_phase"] = "nf4-preflight"
        save(summary, out / "comparison_summary.json")
        if run_logged([sys.executable, script, "check", "--output", str(out / "environment-preflight.json")],
                      log_path("preflight"), 180):
            raise RuntimeError("NF4 API/CUDA preflight failed; inspect the preflight log")
        for index, arm in enumerate(summary["arm_order"]):
            if checkpoint_snapshot(source)["fingerprint"] != snapshot["fingerprint"]:
                raise ValueError("shared checkpoint changed; cannot run a paired comparison")
            summary["current_phase"] = arm
            save(summary, out / "comparison_summary.json")
            spec = {"source": str(source), "arm": arm, "model": model, "sample": str(out / "prompts.csv"),
                    "sample_sha256": manifest["sample_sha256"], "settings_sha256": model_hash,
                    "checkpoint_fingerprint": snapshot["fingerprint"], "output": str(out / arm)}
            save(spec, out / f"{arm}-spec.json")
            timeout = max(1, int((deadline - time.monotonic()) / (2 - index)))
            print(f"Starting {arm} (time budget {timeout}s); log: {log_path(arm)}", flush=True)
            code = run_logged([sys.executable, "-u", script, "worker", "--spec", str(out / f"{arm}-spec.json")],
                              log_path(arm), timeout)
            report_path = out / arm / "report.json"
            report = json.loads(report_path.read_text()) if report_path.is_file() else {"status": "failed"}
            if code or report.get("status") != "success":
                error = report.get("error") or f"worker exit={code}; inspect {log_path(arm)}"
                report.update(status="failed", error=error,
                              failure_kind="timeout" if code == 124 else classify(error))
            summary["arms"][arm] = report
            save(summary, out / "comparison_summary.json")
            write_table(out, summary["arms"])
            print(f"{arm}: {report['status']}", flush=True)
        if checkpoint_snapshot(source)["fingerprint"] != snapshot["fingerprint"]:
            raise ValueError("shared checkpoint changed during comparison")
        comparison = compare_reports(summary["arms"], args.sample_size)
        if comparison["status"] == "complete":
            pairs = paired_outputs(out)
            if len(pairs) != args.sample_size:
                raise ValueError("paired image count differs from the requested sample size")
            summary.update({key: value for key, value in comparison.items() if key != "arms"})
        else:
            summary.update(status="incomplete", note=comparison["note"])
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(summary["error"], flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
        summary.update(elapsed_seconds=time.monotonic() - started, output_disk_usage=disk_bytes(out),
                       accounting=accounting(os.environ["PBS_JOBID"]),
                       accounting_note="Collect after job exit for final PBS state and board-level GPU peak.")
        save(summary, out / "comparison_summary.json")
        write_table(out, summary["arms"])
        lock.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "check", "collect"))
    parser.add_argument("--storage", default=os.environ.get("VS_ASPIRE_STORAGE"))
    parser.add_argument("--output", type=Path, default=os.environ.get("VS_QWEN_OUTPUT"))
    parser.add_argument("--source", default=os.environ.get("VS_QWEN_SOURCE"))
    parser.add_argument("--input", type=Path, default=os.environ.get(
        "VS_QWEN_INPUT", ROOT / "data/processed/unified/unified_prompt_candidates.csv"))
    parser.add_argument("--sample-size", type=int, default=int(os.environ.get("VS_QWEN_SAMPLE_SIZE", "12")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", choices=("bf16-first", "nf4-first"),
                        default=os.environ.get("VS_QWEN_ORDER", "bf16-first"))
    parser.add_argument("--wall-seconds", type=int, default=int(os.environ.get("VS_QWEN_WALL_SECONDS", "6300")))
    parser.add_argument("--budget-gib", type=float, default=float(os.environ.get("VS_ASPIRE_BUDGET_GIB", "0")))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.action == "collect":
            path = args.output / "comparison_summary.json"
            report = json.loads(path.read_text())
            if args.job_id != report["pbs_job_id"]:
                raise ValueError("job ID does not match the comparison output")
            result = accounting(args.job_id)
            save(result, args.output / "accounting-final.json")
        else:
            if not os.environ.get("PBS_JOBID"):
                raise ValueError("GPU comparison/check requires PBS; use the submission script")
            from visual_synthesis.qwen_quantization import generation_worker, gpu_preflight

            if args.action == "check":
                result = gpu_preflight(args.output)
            elif args.action == "worker":
                result = generation_worker(json.loads(args.spec.read_text()))
            else:
                if not 1 <= args.sample_size <= 24 or not 600 <= args.wall_seconds <= 7200:
                    raise ValueError("use 1-24 prompts and a 10-minute to 2-hour allocation for this short trial")
                result = run(args)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["status"] in ("complete", "success", "ok") else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
