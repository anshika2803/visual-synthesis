#!/usr/bin/env python3
"""Heavy-model-first ASPIRE PBS trials using the existing benchmark engine."""

import argparse
import errno
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Reuse tested filesystem, atomic-JSON and weight-selection utilities, not TC2 accounting.
from tc2_trial_support import GIB, ROOT, command, disk_bytes, existing_parent, save, select_files, storage_changes

ORDER = ("qwen-image", "glm-image", "flux-dev", "sd-3.5-large",
         "z-image-turbo", "pixart-sigma", "sd-xl", "sd-1.5")
CONFIGS = [ROOT / "config" / name for name in ("pipeline.json", "benchmark.models.json", "benchmark.hpc.json")]
sys.path.insert(0, str(ROOT / "src"))
os.environ["VISUAL_SYNTHESIS_ROOT"] = str(ROOT)


def configs():
    from visual_synthesis.config import load_config

    return load_config(CONFIGS)


def git_command(*args):
    return command([os.environ.get("VS_ASPIRE_GIT") or "git", "-C", str(ROOT), *args])


def model_plan(model_id, cfg):
    """Fetch file metadata only, pinning the exact configured selection to a commit."""
    from huggingface_hub import HfApi

    from visual_synthesis.config import get_model

    model = get_model(cfg, model_id)
    info = HfApi().model_info(model["repo_id"], revision=model["revision"], files_metadata=True)
    if not info.sha:
        raise ValueError("model metadata has no immutable revision")
    files = select_files(info.siblings, model["allow_patterns"], model["variant"])
    total = sum(files.values())
    return {"model": model_id, "repo_id": model["repo_id"], "revision": info.sha,
            "selected_bytes": total, "selected_gib": total / GIB, "file_count": len(files), "files": files}


def plan(models):
    cfg, rows = configs(), []
    for model_id in models:
        try:
            rows.append({"status": "ok", **model_plan(model_id, cfg)})
        except Exception as exc:
            rows.append({"model": model_id, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
    sizes = [row["selected_bytes"] for row in rows if row["status"] == "ok"]
    return {"time_utc": datetime.now(UTC).isoformat(), "status": "complete" if len(sizes) == len(rows) else "partial",
            "models": rows, "known_selected_bytes": sum(sizes), "known_selected_gib": sum(sizes) / GIB,
            "largest_known_model_gib": max(sizes, default=0) / GIB,
            "note": "Metadata only; no weights downloaded. Partial totals exclude unavailable/gated metadata. "
                    "Disk payload is not GPU memory. Keep extra space for caches, environment and outputs."}


def workspace_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("--storage must be absolute")
    path = path.resolve()
    if path == Path(path.anchor) or path == Path.home().resolve() or path.is_relative_to(Path.home().resolve()):
        raise ValueError("use assigned project storage or resolved scratch, outside your quota-limited home")
    return path


def configure_storage(root):
    from visual_synthesis.config import load_dotenv

    load_dotenv(ROOT)
    # Keep an existing HF login usable when relocating caches; never copy/log the token.
    old_hf_home = Path(os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                                      / "huggingface")).expanduser()
    if not old_hf_home.is_absolute():
        old_hf_home = ROOT / old_hf_home
    if (old_hf_home / "token").is_file():
        os.environ.setdefault("HF_TOKEN_PATH", str(old_hf_home / "token"))
    # Override stale TC2 .env paths consistently, including the Hub cache.
    os.environ.update(VS_MODELS_DIR=str(root / "models"), HF_HOME=str(root / "hf-home"),
                      HF_HUB_CACHE=str(root / "hf-home/hub"), XDG_CACHE_HOME=str(root / "cache"),
                      HF_ASSETS_CACHE=str(root / "hf-home/assets"), HF_XET_CACHE=str(root / "hf-home/xet"),
                      HF_HUB_DISABLE_XET="1",
                      TORCH_HOME=str(root / "torch"), VS_RUNS_DIR=str(root / "pipeline-runs"),
                      VS_LOGS_DIR=str(root / "pipeline-logs"))


def storage_snapshot(root, scan=False):
    paths = {"workspace": root, "models": root / "models", "hf_home": root / "hf-home",
             "trial_outputs": root / "trials"}
    report = {"time_utc": datetime.now(UTC).isoformat(), "paths": {}, "myquota": command(["myquota"]),
              "note": "Filesystem free space is not personal/project quota. Path totals overlap; do not sum them."}
    project = os.environ.get("VS_ASPIRE_PROJECT")
    if project:
        report["project_quota"] = command(["myquota", "-p", project])
    for name, path in paths.items():
        parent = existing_parent(path)
        stat = os.statvfs(parent)
        report["paths"][name] = {"path": str(path), "filesystem_free_bytes": shutil.disk_usage(parent).free,
                                  "filesystem_free_inodes": stat.f_favail,
                                  "usage": disk_bytes(path) if scan else {"scan_skipped": True}}
    return report


def download_guard(snapshot, payload_bytes, file_count, budget_gib):
    if not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError("--budget-gib must be positive and finite")
    entry = snapshot["paths"]["workspace"]
    usage = entry["usage"]
    if usage.get("stat_errors") or usage.get("scan_skipped"):
        raise ValueError("cannot check workspace budget from an incomplete directory scan")
    # Full selected payload even for partial downloads: a same-sized file can have another revision.
    required = math.ceil(payload_bytes * 1.2) + 2 * GIB
    budget_available = max(0, int(budget_gib * GIB) - usage["allocated_bytes"])
    available = min(budget_available, entry["filesystem_free_bytes"])
    allowed = available >= required and entry["filesystem_free_inodes"] >= file_count + 256
    return {"allowed": allowed, "required_bytes_with_headroom": required,
            "workspace_budget_remaining_bytes": budget_available,
            "filesystem_free_bytes": entry["filesystem_free_bytes"],
            "reason": "within supplied workspace budget and filesystem capacity" if allowed else
                      "insufficient workspace budget, filesystem space or inodes",
            "quota_note": "Budget supplied by user, not inferred from df. Inspect the recorded myquota output."}


def accounting(job_id):
    if not re.fullmatch(r"\d+(?:\[\d*\])?(?:\.[A-Za-z0-9_.-]+)?", job_id):
        raise ValueError("invalid PBS job ID")
    result = command(["qstat", "-xf", "-F", "json", job_id])
    allowed = {"job_state", "Exit_status", "resources_used", "Resource_List", "queue", "project",
               "stime", "mtime", "etime", "ctime", "comment", "exec_host", "exec_vnode"}
    try:
        jobs = json.loads(result["stdout"])["Jobs"]
        jobs = {key: {k: v for k, v in value.items() if k in allowed} for key, value in jobs.items()}
    except (ValueError, KeyError, TypeError, AttributeError):
        # Some PBS versions lack JSON; retain selected fields, never Variable_List (may contain secrets).
        result = command(["qstat", "-xf", job_id])
        fields = {}
        for line in result["stdout"].splitlines():
            key, separator, value = line.strip().partition(" = ")
            if separator and key.split(".")[0] in allowed:
                fields[key] = value
        jobs = {job_id: fields} if fields else {}
    return {"status": "success" if jobs else "unavailable",
            "returncode": result["returncode"], "stderr": result["stderr"], "jobs": jobs,
            "note": "resources_used.mem is a job-level host-RAM high-water mark, not per-model GPU memory."}


def run_logged(args, log, timeout, offline=True):
    env = os.environ.copy()
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    with Path(log).open("w") as handle:
        process = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                   cwd=ROOT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                handle.write(f"\nTrial stage exceeded {timeout} seconds.\n")
                return 124
            raise


def tail(path, limit=8000):
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - limit))
        return handle.read().decode(errors="replace")


def classify(error):
    error = error.lower()
    if any(text in error for text in ("disk quota exceeded", "no space left on device")):
        return "storage_exhausted"
    if any(text in error for text in ("cuda out of memory", "outofmemoryerror")):
        return "gpu_out_of_memory"
    if any(text in error for text in ("gatedrepo", "gated repo", "401", "403")):
        return "model_access"
    if "exceeded" in error or "timeout" in error:
        return "timeout"
    return "inspect_logs_and_pbs_accounting"


def trial(model_id, out, root, args):
    from visual_synthesis.benchmark_cluster import _check_model_files
    from visual_synthesis.config import get_model

    out.mkdir()
    state = {"model": model_id, "status": "running", "pbs_job_id": os.environ["PBS_JOBID"],
             "stage": "preflight", "cpu_offload_requested": args.cpu_offload}
    save(state, out / "trial_summary.json")
    before = storage_snapshot(root, scan=True)
    save(before, out / "storage-before.json")
    config_flags = [value for path in CONFIGS for value in ("-c", str(path))]
    base = [sys.executable, str(ROOT / "scripts/benchmark_cluster.py")]
    try:
        check = base + ["check", *config_flags, "--model", model_id, "--input", str(args.input),
                        "--device", "cuda", "--generation-only", "--sample-size", str(args.sample_size),
                        "--output", str(out / "environment-preflight.json")]
        if run_logged(check, out / "preflight.log", 240):
            raise RuntimeError("preflight failed; inspect environment-preflight.json and preflight.log")
        cfg = configs()
        model = get_model(cfg, model_id)
        destination = Path(cfg["paths"]["models"]) / (model["local_dir"] or model_id)
        try:
            _check_model_files(model, destination)
            state["weights"] = "cached"
        except ValueError:
            state["stage"] = "download_preparation"
            if not args.allow_download:
                state["status"] = "blocked"
                raise RuntimeError(
                    "model is not fully cached; set VS_ASPIRE_ALLOW_DOWNLOAD=1 to permit this model"
                ) from None
            selection = model_plan(model_id, cfg)
            guard = download_guard(before, selection["selected_bytes"], selection["file_count"], args.budget_gib)
            save({**selection, "guard": guard}, out / "download-plan.json")
            if not guard["allowed"]:
                state["status"] = "blocked"
                raise RuntimeError(guard["reason"]) from None
            overlay = {"models": {model_id: {"revision": selection["revision"],
                                           "allow_patterns": sorted(selection["files"])}}}
            save(overlay, out / "download-selection.json")
            state["stage"] = "prefetch"
            save(state, out / "trial_summary.json")
            tick = time.monotonic()
            code = run_logged(base + ["prefetch", *config_flags, "-c", str(out / "download-selection.json"),
                                      "--model", model_id, "--input", str(args.input), "--generation-only"],
                              out / "prefetch.log", args.download_timeout, offline=False)
            state["download_seconds"] = time.monotonic() - tick
            if code:
                raise RuntimeError(f"prefetch exited {code}; {tail(out / 'prefetch.log')}") from None
            state["weights"] = "downloaded"
        state["stage"] = "benchmark"
        save(state, out / "trial_summary.json")
        extra = ["--cpu-offload"] if args.cpu_offload else []
        code = run_logged([sys.executable, str(ROOT / "scripts/benchmark_resources.py"), *config_flags,
                           "--model", model_id, "--input", str(args.input), "--output", str(out / "benchmark"),
                           "--device", "cuda", "--sample-size", str(args.sample_size), "--warmup", "1",
                           "--repeats", "1", "--target-prompts", "6000", "--passes", "1", "--generation-only",
                           "--timeout", str(args.timeout), *extra], out / "benchmark.log", args.timeout + 120)
        # The worker can preserve its exact error/peaks even when the parent only reports a failed exit.
        worker = out / "benchmark" / model_id / "report.json"
        report = out / "benchmark/report.json"
        measured = json.loads(worker.read_text()) if worker.is_file() else (
            json.loads(report.read_text())["models"][0] if report.is_file() else {})
        state["measurement"] = measured
        state["status"] = "success" if not code and measured.get("status") == "success" else "failed"
        if state["status"] != "success":
            raise RuntimeError(measured.get("error") or f"benchmark exited {code}; {tail(out / 'benchmark.log')}")
        state["stage"] = "completed"
    except Exception as exc:
        state.update(status="blocked" if state["status"] == "blocked" else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        state["failure_kind"] = classify(state["error"])
    finally:
        after = storage_snapshot(root, scan=True)
        save(after, out / "storage-after.json")
        changes, warnings = storage_changes(before, after)
        state.update(storage_allocated_byte_changes=changes, storage_comparison_warnings=warnings)
        save(state, out / "trial_summary.json")
    return state


def run(args):
    if not os.environ.get("PBS_JOBID"):
        raise ValueError("run must execute inside a PBS compute job; submit aspire_resource_trial.sh with qsub")
    branch = git_command("branch", "--show-current")
    if branch["returncode"] != 0:
        raise ValueError(f"Git branch check failed: {branch['stderr']}")
    if branch["stdout"].strip() != "swara/experiments":
        raise ValueError("expected swara/experiments; no checkout or merge will be performed")
    root = workspace_path(args.storage)
    if args.budget_gib is None or not math.isfinite(args.budget_gib) or args.budget_gib <= 0:
        raise ValueError("supply a positive verified workspace capacity with --budget-gib")
    configure_storage(root)
    root.mkdir(parents=True, exist_ok=True)
    # Prevent simultaneous trial jobs from racing the capacity check and downloads.
    with (root / ".aspire-trial.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another trial is using this storage directory; wait for it to finish") from None
        return run_locked(args, root)


def run_locked(args, root):
    import tempfile

    out = Path(tempfile.mkdtemp(prefix=f"pbs-{os.environ['PBS_JOBID']}-", dir=make_trials(root)))
    print(f"Trial artifacts: {out}", flush=True)
    summary = {"status": "running", "pbs_job_id": os.environ["PBS_JOBID"], "order": args.models,
               "git_commit": git_command("rev-parse", "HEAD")["stdout"].strip(),
               "git_status": git_command("status", "--short")["stdout"],
               "scope": "pretrained generation fit only; no evaluators, activation hooks or ablations",
               "models": [], "not_run": list(args.models)}
    save(summary, out / "run_summary.json")
    save(accounting(os.environ["PBS_JOBID"]), out / "pbs-allocation.json")

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"PBS signal {signum}; collect final accounting after job exit")

    old = signal.signal(signal.SIGTERM, interrupt)
    try:
        for model_id in args.models:
            print(f"Starting {model_id}: one warmup + {args.sample_size} measured prompts", flush=True)
            result = trial(model_id, out / model_id, root, args)
            summary["models"].append(result)
            summary["not_run"].remove(model_id)
            save(summary, out / "run_summary.json")
            print(f"{model_id}: {result['status']}", flush=True)
            if result.get("failure_kind") == "storage_exhausted":
                break
        summary["status"] = "success" if all(row["status"] == "success" for row in summary["models"]) and not (
            summary["not_run"]) else "incomplete"
    except KeyboardInterrupt as exc:
        summary.update(status="interrupted", error=str(exc))
    finally:
        signal.signal(signal.SIGTERM, old)
        summary["accounting"] = accounting(os.environ["PBS_JOBID"])
        summary["accounting_note"] = (
            "In-job accounting may say R; run collect after completion for final PBS limits/status."
        )
        save(summary, out / "run_summary.json")
    return summary


def make_trials(root):
    path = root / "trials"
    path.mkdir(exist_ok=True)
    return path


def collect(out, report):
    """Recover interrupted model records and attach final job-level resource limits."""
    path = out / "run_summary.json"
    summary = json.loads(path.read_text())
    jobs = report.get("jobs", {}).values()
    failed = any(str(job.get("Exit_status", "0")) != "0" for job in jobs)
    ended = failed or any(job.get("job_state") in ("F", "X") for job in report.get("jobs", {}).values())
    rows = {row["model"]: row for row in summary["models"]}
    if ended:
        for model_id in summary["order"]:
            partial = out / model_id / "trial_summary.json"
            if partial.is_file():
                row = json.loads(partial.read_text())
                if row["status"] == "running":
                    row.update(status="interrupted", failure_kind="inspect_pbs_resource_limits",
                               error="Job ended before the model summary completed; inspect final PBS accounting.")
                    save(row, partial)
                rows[model_id] = row
        summary["models"] = [rows[model] for model in summary["order"] if model in rows]
        summary["not_run"] = [model for model in summary["order"] if model not in rows]
        if failed or summary["not_run"] or any(row["status"] != "success" for row in rows.values()):
            summary["status"] = "failed" if failed else "incomplete"
    summary["final_accounting"] = report
    save(report, out / "pbs-accounting-final.json")
    save(summary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "storage", "run", "collect"])
    parser.add_argument("--storage", default=os.environ.get("VS_ASPIRE_STORAGE"))
    parser.add_argument("--budget-gib", type=float, default=os.environ.get("VS_ASPIRE_BUDGET_GIB"))
    parser.add_argument("--model", action="append", choices=ORDER, dest="models")
    parser.add_argument("--input", type=Path,
                        default=os.environ.get("VS_ASPIRE_INPUT",
                                               ROOT / "data/processed/unified/unified_prompt_candidates.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-id", default=os.environ.get("PBS_JOBID", ""))
    parser.add_argument("--stdout-only", action="store_true", help="collect without file writes")
    parser.add_argument("--allow-download", action="store_true",
                        default=os.environ.get("VS_ASPIRE_ALLOW_DOWNLOAD") == "1")
    parser.add_argument("--cpu-offload", action="store_true", default=os.environ.get("VS_ASPIRE_CPU_OFFLOAD") == "1")
    parser.add_argument("--sample-size", type=int, default=os.environ.get("VS_ASPIRE_SAMPLE_SIZE", "2"))
    parser.add_argument("--timeout", type=int, default=os.environ.get("VS_ASPIRE_TIMEOUT", "1800"))
    parser.add_argument("--download-timeout", type=int, default=os.environ.get("VS_ASPIRE_DOWNLOAD_TIMEOUT", "5400"))
    args = parser.parse_args(argv)
    args.models = args.models or os.environ.get("VS_ASPIRE_MODELS", "").split() or list(ORDER)
    if len(args.models) != len(set(args.models)) or any(model not in ORDER for model in args.models):
        parser.error("model list must contain distinct IDs from the configured eight")
    if min(args.timeout, args.download_timeout, args.sample_size) <= 0:
        parser.error("timeouts and sample size must be positive")
    if args.stdout_only and args.action != "collect":
        parser.error("--stdout-only is supported only for collect")
    try:
        if args.action == "plan":
            report = plan(args.models)
        elif args.action == "storage":
            if not args.storage:
                parser.error("storage requires --storage")
            report = storage_snapshot(workspace_path(args.storage))
        elif args.action == "run":
            if not args.storage:
                parser.error("run requires --storage")
            report = run(args)
        else:
            report = accounting(args.job_id)
            if not args.stdout_only:
                if not args.output or not (args.output / "run_summary.json").is_file():
                    parser.error("collect requires an existing --output trial directory, or --stdout-only")
                collect(args.output, report)
        if args.output and args.action in ("plan", "storage"):
            save(report, args.output)
        # File lists remain in the saved plan, keeping terminal output compact.
        printed = {**report, "models": [{k: v for k, v in row.items() if k != "files"}
                                       for row in report["models"]]} if args.action == "plan" else report
        print(json.dumps(printed, indent=2))
        return 0 if report.get("status", "success") in ("success", "complete") else 1
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            if args.action == "collect":
                print(json.dumps(report, indent=2))
            print(f"Storage exhausted; stop downloads. Existing completed summaries remain intact. {exc}",
                  file=sys.stderr)
            return 1
        raise
    except (ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
