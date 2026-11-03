#!/usr/bin/env python3
"""PBS-only project pilots; collect/aggregate are lightweight login-node commands."""

import argparse
import contextlib
import csv
import fcntl
import fnmatch
import json
import os
import shutil
import signal
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from aspire_trial_support import (
    CONFIGS,
    GIB,
    ORDER,
    ROOT,
    accounting,
    classify,
    configs,
    configure_storage,
    download_guard,
    git_command,
    model_plan,
    run_logged,
    save,
    storage_snapshot,
    tail,
    workspace_path,
)


def stage_log(root, out, phase):
    """One shared log directory; a unique run prefix prevents overwrites."""
    return root / "logs" / f"{out.name}.{phase}.log"


def select_evaluator_files(siblings, patterns, required):
    """Select flat Transformers CLIP repositories, not Diffusers pipelines.

    Keep the generator's model_index.json validation in model_plan unchanged.
    Both CLIP evaluators use root config/processor files and one root weight
    file. Prefer safetensors without downloading a duplicate binary checkpoint.
    """
    files = {
        item.rfilename: item.size
        for item in siblings
        if any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns)
    }
    if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name for name in files):
        raise ValueError("unsafe evaluator repository file path")
    files = {name: size for name, size in files.items() if "/" not in name}
    weight = next((name for name in ("model.safetensors", "pytorch_model.bin") if name in files), None)
    if weight is None:
        raise ValueError("evaluator metadata has no supported root weight file")
    missing = sorted(set(required) - set(files))
    if missing:
        raise ValueError(f"evaluator metadata is missing required files: {missing}")
    files = {
        name: size
        for name, size in files.items()
        if name not in ("model.safetensors", "pytorch_model.bin") or name == weight
    }
    if any(not isinstance(size, int) or size < 0 for size in files.values()):
        raise ValueError("evaluator file sizes are unavailable; cannot authorize a quota-checked download")
    if any(files[name] == 0 for name in (*required, weight)):
        raise ValueError("evaluator metadata contains an empty required file")
    return files


def prepare_weights(model_id, cfg, root, out, budget, allow_download):
    from huggingface_hub import HfApi, snapshot_download

    from visual_synthesis.benchmark_cluster import _check_model_files
    from visual_synthesis.config import get_model
    from visual_synthesis.project_pilot import CLIP_REPO

    plans = []

    def guarded(plan):
        check = download_guard(storage_snapshot(root, scan=True), plan["selected_bytes"], len(plan["files"]), budget)
        plans.append({**plan, "guard": check})
        save(plans, out / "download-plans.json")
        if not check["allowed"]:
            raise ValueError(check["reason"])
        if not allow_download:
            raise ValueError("missing cached resources; set VS_PILOT_ALLOW_DOWNLOAD=1 to prepare them on this GPU job")

    model = get_model(cfg, model_id)
    try:
        _check_model_files(model, Path(cfg["paths"]["models"]) / (model["local_dir"] or model_id))
    except ValueError:
        plan = model_plan(model_id, cfg)
        guarded(plan)
        overlay = {"models": {model_id: {"revision": plan["revision"], "allow_patterns": list(plan["files"])}}}
        overlay_path = out / "model-download-selection.json"
        save(overlay, overlay_path)
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmark_cluster.py"),
            "prefetch",
            *[v for c in CONFIGS for v in ("-c", str(c))],
            "-c",
            str(overlay_path),
            "--model",
            model_id,
            "--input",
            str(out / "inputs/baseline.csv"),
            "--generation-only",
        ]
        if run_logged(command, stage_log(root, out, "generator-prefetch"), 5400, offline=False):
            raise RuntimeError("generator prefetch failed; inspect the generator-prefetch log") from None

    model_root = Path(cfg["paths"]["models"])
    for repo, destination, patterns in (
        (
            CLIP_REPO,
            model_root / "openai--clip-vit-base-patch32",
            ["*.json", "*.txt", "model.safetensors", "pytorch_model.bin"],
        ),
        (
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            model_root / "clip-h14",
            ["config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"],
        ),
    ):
        required = ["config.json", "preprocessor_config.json"]
        if repo == CLIP_REPO:
            required += ["vocab.json", "merges.txt"]
        if all((destination / name).is_file() for name in required) and any(
            (destination / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
        ):
            continue
        info = HfApi().model_info(repo, files_metadata=True)
        try:
            if not info.sha:
                raise ValueError("no immutable repository revision")
            files = select_evaluator_files(info.siblings, patterns, required)
        except ValueError as exc:
            raise ValueError(f"evaluator {repo}: {exc}") from exc
        plan = {"repo_id": repo, "revision": info.sha, "files": files, "selected_bytes": sum(files.values())}
        guarded(plan)
        print(f"Preparing evaluator {repo} -> {destination}", flush=True)
        snapshot_download(repo, revision=info.sha, local_dir=destination, allow_patterns=list(files))
    head = model_root / "clip-h14/h14_nsfw.pth"
    if not head.is_file():
        guarded({"repo_id": "LAION H14 head", "files": {"h14_nsfw.pth": 32 * 1024**2}, "selected_bytes": 32 * 1024**2})
        source = ROOT / "models/clip-h14/h14_nsfw.pth"
        if source.is_file():
            head.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, head)
        else:
            from visual_synthesis.download import download_url

            download_url(cfg["detectors"]["clip-h14"]["url"], head.parent)
        if not head.is_file():
            raise RuntimeError("H14 head preparation failed")
    return str(model_root / "openai--clip-vit-base-patch32")


def write_tables(summary, out):
    from visual_synthesis.project_pilot import EVALUATORS, read_jsonl

    def gib(memory, key):
        return memory[key] / GIB if key in memory else None

    rows = []
    for name, report in summary["phases"].items():
        memory = report.get("resources", {})

        rows.append(
            {
                "phase": name,
                "status": report["status"],
                "images": report.get("images"),
                "load_seconds": report.get("load_seconds"),
                "seconds_per_image": report.get("mean_seconds_per_image"),
                "gpu_allocated_gib": gib(memory, "cuda_allocated_peak_bytes"),
                "gpu_reserved_gib": gib(memory, "cuda_reserved_peak_bytes"),
                "ram_peak_gib": gib(memory, "rss_process_lifetime_peak_bytes"),
                "error": report.get("error"),
            }
        )
    if rows:
        with (out / "phase_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    if not (out / "all-images.jsonl").is_file():
        return
    scores = {}
    for kind in EVALUATORS:
        path = out / kind / "scores.jsonl"
        if path.is_file():
            scores[kind] = {(r["condition"], r["prompt_id"]): r for r in read_jsonl(path)}
    metrics = []
    for image in read_jsonl(out / "all-images.jsonl"):
        key = image["condition"], image["prompt_id"]
        nude, h14 = (scores.get(kind, {}).get(key, {}).get("flagged") for kind in ("nudenet", "clip-h14"))
        metrics.append(
            {
                "condition": key[0],
                "prompt_id": key[1],
                "benchmark_level": image["benchmark_level"],
                "standard_category": image["standard_category"],
                "nudenet_flagged": nude,
                "clip_h14_flagged": h14,
                "union_flagged": (nude or h14) if None not in (nude, h14) else None,
                "alignment": scores.get("alignment", {}).get(key, {}).get("score"),
                "clip_iqa_b32": scores.get("clip-iqa", {}).get(key, {}).get("score"),
            }
        )
    if metrics:
        with (out / "image_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=metrics[0])
            writer.writeheader()
            writer.writerows(metrics)


def run(args):
    if not os.environ.get("PBS_JOBID"):
        raise ValueError("run needs a PBS GPU job; use submit_aspire_project_pilot.sh")
    branch = git_command("branch", "--show-current")
    if branch["returncode"] or branch["stdout"].strip() != "swara/experiments":
        raise ValueError("expected swara/experiments; no branch changes are performed")
    root = workspace_path(args.storage)
    configure_storage(root)
    # Explicit file credentials take precedence over stale tokens loaded from .env.
    # Empty values also prevent child config imports from restoring those entries.
    if Path(os.environ.get("HF_TOKEN_PATH", root / "hf-home/token")).is_file():
        os.environ["HF_TOKEN"] = ""
        os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".aspire-trial.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another fit/pilot uses this workspace; wait or submit with VS_PILOT_AFTER.") from None
        return run_locked(args, root)


def run_locked(args, root):
    from visual_synthesis.project_pilot import CONDITIONS, EVALUATORS, prepare_samples, project_costs, read_jsonl

    runs = root / "outputs"
    runs.mkdir(exist_ok=True)
    if os.environ.get("VS_PILOT_OUTPUT"):
        out = Path(os.environ["VS_PILOT_OUTPUT"]).resolve()
        # Also accept paths already submitted by the earlier launcher.
        allowed = (runs.resolve(), (root / "runs").resolve())
        if not any(out != parent and out.is_relative_to(parent) for parent in allowed):
            raise ValueError("VS_PILOT_OUTPUT must be a run directory inside runtime/outputs")
        out.mkdir(exist_ok=True)
        if (out / "pilot_summary.json").exists():
            raise ValueError("run already exists; submit a new run instead of overwriting it")
    else:
        out = Path(tempfile.mkdtemp(prefix=f"{args.model}-", dir=runs))
    logs, outputs = root / "logs", out
    logs.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    def event(message):
        print(message, flush=True)
        with stage_log(root, out, "pilot").open("a") as stream:
            stream.write(f"{datetime.now(UTC).isoformat()} {message}\n")

    event(f"Pilot artifacts: {out}")
    summary = {
        "status": "running",
        "model": args.model,
        "pbs_job_id": os.environ["PBS_JOBID"],
        "time_utc": datetime.now(UTC).isoformat(),
        "phases": {},
        "output": str(out),
        "log_directory": str(logs),
        "log_prefix": out.name,
        "git_commit": git_command("rev-parse", "HEAD")["stdout"].strip(),
        "scope": "Resource pilot on candidates; no validated safety claims.",
        "projection": {"status": "incomplete"},
    }
    save(summary, out / "pilot_summary.json")
    save(accounting(os.environ["PBS_JOBID"]), out / "pbs-allocation.json")
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("PBS termination")))
    try:
        summary["input"] = prepare_samples(args.input, out / "inputs", args.baseline, args.discovery, args.validation)
        summary["current_phase"] = "preflight"
        save(summary, out / "pilot_summary.json")
        before = storage_snapshot(root, scan=True)
        save(before, out / "storage-before.json")
        guard = download_guard(before, 0, 1000, args.budget_gib)
        if not guard["allowed"]:
            raise ValueError(guard["reason"])
        preflight = [
            sys.executable,
            str(ROOT / "scripts/benchmark_cluster.py"),
            "check",
            *[v for c in CONFIGS for v in ("-c", str(c))],
            "--model",
            args.model,
            "--device",
            "cuda",
            "--sample-size",
            "2",
            "--input",
            str(args.input),
        ]
        if run_logged(preflight, stage_log(root, out, "preflight"), 300):
            raise RuntimeError("environment preflight failed; inspect the preflight log and SETUP.md")
        cfg = configs()
        summary["current_phase"] = "prepare-resources"
        save(summary, out / "pilot_summary.json")
        event(f"Preparing missing cached resources; see {stage_log(root, out, 'preparation')}")
        with stage_log(root, out, "preparation").open("w") as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                clip_source = prepare_weights(args.model, cfg, root, out, args.budget_gib, args.allow_download)
        base = {"config": cfg, "model": args.model, "clip_source": clip_source}

        def phase(name, **extras):
            capacity = download_guard(storage_snapshot(root, scan=True), 0, 0, args.budget_gib)
            if not capacity["allowed"]:
                raise RuntimeError("storage headroom exhausted; stopping before next phase")
            target = outputs / name
            target.mkdir(exist_ok=True)
            spec = {**base, "phase": name, "output": str(target), **extras}
            save(spec, target / "spec.json")
            summary["current_phase"] = name
            save(summary, out / "pilot_summary.json")
            event(f"Starting {name}; log: {stage_log(root, out, name)}")
            code = run_logged(
                [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--spec", str(target / "spec.json")],
                stage_log(root, out, name),
                args.timeout,
            )
            path = target / "report.json"
            report = json.loads(path.read_text()) if path.is_file() else {"phase": name, "status": "failed"}
            if code or report["status"] != "success":
                error = report.get("error") or f"worker exit={code}; {tail(stage_log(root, out, name), limit=3000)}"
                report.update(status="failed", error=error)
                report.setdefault("failure_kind", "timeout" if code == 124 else classify(report["error"]))
            summary["phases"][name] = report
            save(summary, out / "pilot_summary.json")
            event(f"{name}: {report['status']}")
            if report.get("failure_kind") == "storage_exhausted":
                raise RuntimeError("storage exhausted; stopping pilot")
            return report["status"] == "success"

        if phase("baseline", sample=str(out / "inputs/baseline.csv")):
            if phase("capture", sample=str(out / "inputs/capture.csv")):
                if phase("learn", capture=str(outputs / "capture")):
                    if phase(
                        "alpha-zero",
                        sample=str(out / "inputs/identity.csv"),
                        direction=str(outputs / "learn/direction.npz"),
                        baseline=str(outputs / "baseline"),
                    ):
                        for name in CONDITIONS:
                            if name != "alpha-zero":
                                phase(
                                    name,
                                    sample=str(out / "inputs/validation.csv"),
                                    direction=str(outputs / "learn/direction.npz"),
                                )
        # Evaluate completed/partial image outputs even if Qwen capture/ablation ran out of memory.
        images = []
        for name in ("baseline", "capture", *CONDITIONS):
            path = outputs / name / "images.jsonl"
            if path.is_file():
                images.extend(read_jsonl(path))
        if images:
            (out / "all-images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in images))
            for kind in EVALUATORS:
                phase(kind, images=str(out / "all-images.jsonl"))
        summary["projection"] = project_costs(summary["phases"])
        summary["core_projection"] = project_costs(summary["phases"], baseline=1200)
        summary["reduced_core_projection"] = project_costs(summary["phases"], baseline=600, capture=250, validation=150)
        summary["status"] = "success" if summary["projection"]["status"] == "conditional" else "incomplete"
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
        summary["accounting"] = accounting(os.environ["PBS_JOBID"])
        save(storage_snapshot(root, scan=True), out / "storage-after.json")
        summary["accounting_note"] = "In-job accounting may say R; collect after exit."
        save(summary, out / "pilot_summary.json")
        write_tables(summary, out)
    event(f"Pilot status: {summary['status']}; summary: {out / 'pilot_summary.json'}")
    return summary


def aggregate(root, output):
    newest = {}
    candidates = []
    paths = list((root / "outputs").glob("*/pilot_summary.json"))
    paths += list((root / "runs").glob("*/pilot_summary.json"))  # Retain earlier pilot reports.
    for path in paths:
        report = json.loads(path.read_text())
        created = report.get("time_utc", datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat())
        candidates.append((created, report))
    for _, report in sorted(candidates, key=lambda row: row[0]):
        newest[report["model"]] = report
    missing = sorted(set(ORDER) - set(newest))
    incomplete = sorted(k for k, r in newest.items() if r["status"] != "success")
    comparable = (
        len({json.dumps(r.get("input", {}).get("tables_sha256"), sort_keys=True) for r in newest.values()}) <= 1
    )
    result = {
        "status": "conditional" if not missing and not incomplete else "incomplete",
        "models": {
            k: {"status": r["status"], "output": r["output"], "projection": r["projection"]} for k, r in newest.items()
        },
        "missing_models": missing,
        "incomplete_models": incomplete,
        "shared_prompt_tables": comparable,
        "note": (
            "Newest run per model, including failures; no automatic fallback to old successes. "
            "Projections remain conditional resource forecasts."
        ),
    }
    if not missing and not incomplete:
        result["generation_plus_evaluation_hours"] = sum(
            r["projection"]["generation_plus_evaluation_hours"] for r in newest.values()
        )
        heavy = {"qwen-image", "glm-image", "flux-dev", "sd-3.5-large"}
        result["scenarios"] = {}
        for scope in ("extended_6000", "full_core", "mixed_core"):
            projections = [
                r["projection"]
                if scope == "extended_6000"
                else r["reduced_core_projection"]
                if scope == "mixed_core" and model in heavy
                else r["core_projection"]
                for model, r in newest.items()
            ]
            result["scenarios"][scope] = {
                "images": sum(p["target_images"] for p in projections),
                "generation_hours": sum(p["generation_hours"] for p in projections),
                "evaluation_hours": sum(p["evaluation_hours"] for p in projections),
                "generation_plus_evaluation_hours": sum(p["generation_plus_evaluation_hours"] for p in projections),
            }
    save(result, output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "worker", "collect", "aggregate"])
    parser.add_argument("--storage", type=Path, default=os.environ.get("VS_ASPIRE_STORAGE"))
    parser.add_argument("--budget-gib", type=float, default=os.environ.get("VS_ASPIRE_BUDGET_GIB", "500"))
    parser.add_argument("--model", choices=ORDER, default=os.environ.get("VS_PILOT_MODEL", "qwen-image"))
    parser.add_argument(
        "--input",
        type=Path,
        default=os.environ.get("VS_ASPIRE_INPUT", str(ROOT / "data/processed/unified/unified_prompt_candidates.csv")),
    )
    parser.add_argument("--baseline", type=int, default=int(os.environ.get("VS_PILOT_BASELINE", "60")))
    parser.add_argument("--discovery", type=int, default=int(os.environ.get("VS_PILOT_DISCOVERY", "10")))
    parser.add_argument("--validation", type=int, default=int(os.environ.get("VS_PILOT_VALIDATION", "6")))
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--allow-download", action="store_true", default=os.environ.get("VS_PILOT_ALLOW_DOWNLOAD") == "1"
    )
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args(argv)
    try:
        if args.action == "worker":
            if not os.environ.get("PBS_JOBID"):
                raise ValueError("worker requires an allocated PBS GPU job")
            from visual_synthesis.project_pilot import worker

            result = worker(json.loads(args.spec.read_text()))
        elif args.action == "run":
            if not args.storage:
                raise ValueError("set VS_ASPIRE_STORAGE")
            result = run(args)
        elif args.action == "aggregate":
            if not args.storage or not args.output:
                raise ValueError("aggregate needs --storage and --output")
            result = aggregate(workspace_path(args.storage), args.output)
        else:
            if not args.output or not args.job_id or not (args.output / "pilot_summary.json").is_file():
                raise ValueError("collect needs --job-id and --output pointing to the pilot directory")
            summary = json.loads((args.output / "pilot_summary.json").read_text())
            if args.job_id.split(".")[0] != summary["pbs_job_id"].split(".")[0]:
                raise ValueError("job ID does not match this pilot directory")
            result = accounting(args.job_id)
            save(result, args.output / "pbs-final.json")
            summary["accounting"] = result
            save(summary, args.output / "pilot_summary.json")
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") in ("success", "conditional") else 1
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
