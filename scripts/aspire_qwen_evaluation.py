"""Evaluate original paired PNGs with the existing project-pilot evaluator workers."""

import fcntl
import importlib.util
import json
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aspire_qwen_quantization import isolate_caches
from aspire_trial_support import (
    ROOT,
    accounting,
    classify,
    configs,
    disk_bytes,
    download_guard,
    run_logged,
    save,
    storage_snapshot,
    workspace_path,
)


def require_cached_evaluators(root):
    """No checkpoint downloads, including the detector head's URL fallback."""
    for folder, required in (
        (root / "models/clip-h14", ("config.json", "preprocessor_config.json", "h14_nsfw.pth")),
        (root / "models/openai--clip-vit-base-patch32",
         ("config.json", "preprocessor_config.json", "vocab.json", "merges.txt")),
    ):
        if not all((folder / name).is_file() and (folder / name).stat().st_size for name in required):
            raise ValueError(f"missing cached evaluator files in {folder}; no downloads are performed by this job")
        if not any((folder / name).is_file() and (folder / name).stat().st_size
                   for name in ("model.safetensors", "pytorch_model.bin")):
            raise ValueError(f"missing cached evaluator weights in {folder}")
    package = importlib.util.find_spec("nudenet")
    if not package or not package.origin or not (Path(package.origin).parent / "320n.onnx").is_file():
        raise ValueError("NudeNet's bundled 320n.onnx is required in the existing environment")


def evaluate(args):
    if not os.environ.get("PBS_JOBID"):
        raise ValueError("evaluation requires a PBS allocation; do not score images on the login node")
    from visual_synthesis.config import get_detector
    from visual_synthesis.project_pilot import EVALUATORS
    from visual_synthesis.qwen_safety_comparison import aggregate_scores, prepare_images

    root, out = workspace_path(args.storage), Path(args.output).resolve()
    comparison = Path(args.comparison).resolve()
    if not all(path.is_relative_to(root / "outputs") and path != root / "outputs" for path in (out, comparison)):
        raise ValueError("comparison and evaluation outputs must be inside the runtime/outputs directory")
    if (out.is_relative_to(comparison) or comparison.is_relative_to(out) or
            (out / "evaluation_summary.json").exists()):
        raise ValueError("use a fresh evaluation output directory; original comparison outputs are preserved")
    out.mkdir(parents=True, exist_ok=True)
    isolate_caches(out)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    lock = (out / ".evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary = {"status": "running", "pbs_job_id": os.environ["PBS_JOBID"], "comparison": str(comparison),
               "output": str(out), "time_utc": datetime.now(UTC).isoformat(), "phases": {},
               "protocol": "Full-precision evaluators, identical thresholds, original BF16/NF4 images only"}
    started = time.monotonic()
    deadline = started + args.wall_seconds - 120
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("PBS termination")))
    try:
        save(summary, out / "evaluation_summary.json")
        before = storage_snapshot(root, scan=True)
        save(before, out / "storage-before.json")
        guard = download_guard(before, 0, 128, args.budget_gib)
        if not guard["allowed"]:
            raise ValueError(guard["reason"])
        require_cached_evaluators(root)
        pairs, images = prepare_images(comparison, out)
        summary.update(pairs=len(pairs), images=len(images))
        cfg = configs()
        cfg["paths"].update(models=str(root / "models"), cache=str(out / "cache"),
                            runs=str(out / "pipeline-runs"), logs=str(logs))
        cfg["detectors"]["clip-h14"].update(local_dir="clip-h14", url=None)
        settings = {kind: {key: get_detector(cfg, kind)[key]
                           for key in ("threshold", "label_thresholds", "flag_labels")}
                    for kind in ("nudenet", "clip-h14")}
        summary["detector_settings"] = settings
        save(cfg, out / "evaluator-config.json")
        for index, kind in enumerate(EVALUATORS):
            target = out / kind
            spec = {"config": cfg, "model": "qwen-image", "phase": kind, "output": str(target),
                    "images": str(out / "all-images.jsonl"),
                    "clip_source": str(root / "models/openai--clip-vit-base-patch32")}
            save(spec, out / f"{kind}-spec.json")
            summary["current_phase"] = kind
            save(summary, out / "evaluation_summary.json")
            stage_log = logs / f"{out.name}.{kind}.log"
            print(f"Evaluating {kind}; log: {stage_log}", flush=True)
            timeout = max(1, min(600, int((deadline - time.monotonic()) / (len(EVALUATORS) - index))))
            code = run_logged([sys.executable, "-u", str(ROOT / "scripts/aspire_project_pilot.py"),
                               "worker", "--spec", str(out / f"{kind}-spec.json")], stage_log, timeout)
            report_file = target / "report.json"
            report = json.loads(report_file.read_text()) if report_file.is_file() else {"status": "failed"}
            if code or report.get("status") != "success":
                error = report.get("error") or f"evaluator exit={code}; inspect {stage_log}"
                report.update(status="failed", error=error,
                              failure_kind="timeout" if code == 124 else classify(error))
            summary["phases"][kind] = report
            save(summary, out / "evaluation_summary.json")
        # Recheck original image hashes after scoring; no source data should have changed.
        prepare_images(comparison, out)
        summary.update(aggregate_scores(out, pairs, images, summary["phases"], settings))
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(summary["error"], flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
        summary.update(elapsed_seconds=time.monotonic() - started, output_disk_usage=disk_bytes(out),
                       accounting=accounting(os.environ["PBS_JOBID"]),
                       accounting_note="Collect after exit for final PBS state; thresholds are unchanged between arms.")
        save(summary, out / "evaluation_summary.json")
        lock.close()
    return summary
