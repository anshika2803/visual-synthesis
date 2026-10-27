#!/usr/bin/env python3
"""Storage guards and accounting for one TC2 resource trial. No tensor operations."""

import argparse
import csv
import fnmatch
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

GIB = 2**30
MODELS = ("sd-xl", "sd-3.5-large", "flux-dev", "qwen-image", "z-image-turbo",
          "glm-image", "pixart-sigma", "sd-1.5")
ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def existing_parent(path):
    path = Path(path).expanduser().absolute()
    while not path.exists():
        path = path.parent
    return path


def disk_bytes(path):
    """Count allocated/logical bytes without following links or double-counting hard links."""
    path = Path(path).expanduser()
    if not path.exists():
        return {"present": False, "logical_bytes": 0, "allocated_bytes": 0}
    path = path.resolve()
    seen, logical, allocated = set(), 0, 0
    failures = []
    for directory, _, names in os.walk(path, followlinks=False, onerror=lambda exc: failures.append(str(exc))):
        for name in names:
            try:
                item = (Path(directory) / name).lstat()
                key = (item.st_dev, item.st_ino)
                if key not in seen:
                    seen.add(key)
                    logical += item.st_size
                    allocated += getattr(item, "st_blocks", 0) * 512
            except OSError as exc:
                failures.append(str(exc))
    return {"present": True, "resolved_path": str(path), "logical_bytes": logical,
            "allocated_bytes": allocated, "stat_errors": failures}


def parse_quota(text):
    """Parse numeric Linux `quota -w` rows (block counters are KiB).

    Human-readable `quota -s` output is retained separately and never guessed.
    """
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4 or not (fields[0].startswith("/") or ":/" in fields[0]):
            continue
        numbers = [field.rstrip("*") for field in fields[1:4]]
        if not all(number.isdigit() for number in numbers):
            continue
        used, soft, hard = map(int, numbers)
        limits = [number for number in (soft, hard) if number > 0]
        row = {"filesystem": fields[0], "used_bytes": used * 1024,
               "soft_bytes": soft * 1024, "hard_bytes": hard * 1024,
               "available_bytes": max(0, min(limits) - used) * 1024 if limits else None}
        tail = fields[4:]
        if tail and not tail[0].rstrip("*").isdigit():
            tail = tail[1:]  # optional block grace field
        if len(tail) >= 3 and all(value.rstrip("*").isdigit() for value in tail[:3]):
            files_used, files_soft, files_hard = (int(value.rstrip("*")) for value in tail[:3])
            file_limits = [value for value in (files_soft, files_hard) if value > 0]
            row["available_files"] = max(0, min(file_limits) - files_used) if file_limits else None
        rows.append(row)
    return rows


def storage_path(entry, repo=ROOT):
    """Use the measured canonical path, including for snapshots with relative paths."""
    value = entry.get("usage", {}).get("resolved_path") or entry["path"]
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(repo) / path).resolve()


def storage_changes(before, after):
    """Never subtract sizes of different directories or incomplete scans."""
    changes, warnings = {}, {}
    repo = before["paths"].get("repo", {}).get("path", str(ROOT))
    for name, first in before["paths"].items():
        second = after["paths"].get(name)
        if second is None:
            reason = "after snapshot is missing this path"
        elif storage_path(first, repo) != storage_path(second, repo):
            reason = "before and after snapshots measured different directories"
        elif first["usage"].get("stat_errors") or second["usage"].get("stat_errors"):
            reason = "directory scan reported errors"
        else:
            changes[name] = second["usage"]["allocated_bytes"] - first["usage"]["allocated_bytes"]
            continue
        changes[name] = None
        warnings[name] = {"reason": reason, "before_path": str(storage_path(first, repo)),
                          "after_path": str(storage_path(second, repo)) if second else None}
    return changes, warnings


def storage_snapshot(repo=ROOT, models=None, frozen_paths=None):
    repo = Path(repo).resolve()
    if frozen_paths is None:
        # Match prepare/model loading, which imports config and reads .env.
        # Previously `finish` did neither and silently measured another cache.
        sys.path.insert(0, str(ROOT / "src"))
        from visual_synthesis.config import load_dotenv

        load_dotenv(repo)
        models = models or os.environ.get("VS_MODELS_DIR", repo / "models")
        hf_home = os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                                 / "huggingface")
        hub = os.environ.get("HF_HUB_CACHE", Path(hf_home).expanduser() / "hub")
        paths = {"repo": repo, "models": models, "hf_home": hf_home,
                 "hf_hub": hub, "trial_outputs": repo / "results/benchmarks"}
        paths = {name: storage_path({"path": str(path)}, repo) for name, path in paths.items()}
    else:
        # Preserve physical targets even if .env, Conda or shell variables change.
        paths = {name: storage_path(entry, repo) for name, entry in frozen_paths.items()}
    quota = command(["quota", "-w"])
    report = {"time_utc": datetime.now(UTC).isoformat(), "paths": {}, "quota_raw": quota,
              "quota_human": command(["quota", "-s"]), "mytcinfo": command(["mytcinfo"]),
              "note": "Path totals may overlap; do not sum them. Filesystem free space is not user quota."}
    rows = parse_quota(quota["stdout"])
    for label, path in paths.items():
        parent = existing_parent(path)
        free = shutil.disk_usage(parent).free
        df = command(["df", "-Pk", str(parent)])
        lines = df["stdout"].strip().splitlines()
        fields = lines[-1].split() if len(lines) >= 2 else []
        source, mount = (fields[0], fields[-1]) if len(fields) >= 6 else (None, None)
        matching = [row for row in rows if row["filesystem"] in (source, mount)]
        report["paths"][label] = {"path": str(path), "filesystem_source": source, "mount": mount,
                                   "filesystem_free_bytes": free, "usage": disk_bytes(path),
                                   "quota": matching[0] if len(matching) == 1 else None}
    return report


def download_capacity(storage, needed_bytes, files, budget_gib=None, reserve_gib=1):
    """Require both filesystem space and a known user/assigned-space budget."""
    path = storage["paths"]["models"]
    quota = path.get("quota") or {}
    available = quota.get("available_bytes")
    limits = [path["filesystem_free_bytes"]]
    if budget_gib is not None:
        if not math.isfinite(budget_gib) or budget_gib <= 0:
            return {"allowed": False, "reason": "explicit storage budget must be positive and finite"}
        limits.append(int(budget_gib * GIB))
    if available is not None:
        limits.append(available)
    if budget_gib is None and available is None:
        return {"allowed": False,
                "reason": "user quota is unknown; inspect mytcinfo/quota and provide a verified budget"}
    required = math.ceil(needed_bytes * 1.1) + int(reserve_gib * GIB)
    remaining = min(limits)
    inode_ok = quota.get("available_files") is None or quota["available_files"] >= files + 128
    return {"allowed": remaining >= required and inode_ok, "required_bytes_with_headroom": required,
            "available_budget_bytes": remaining, "quota_files_available": quota.get("available_files"),
            "reason": "sufficient verified space" if remaining >= required and inode_ok else
            "insufficient disk quota/free space or file quota; no weights downloaded"}


def select_files(siblings, patterns, variant=None):
    """Keep configured components, preferring safetensors to duplicate binary weights."""
    files = {item.rfilename: item.size for item in siblings
             if not patterns or any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns)}
    if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name for name in files):
        raise ValueError("unsafe repository file path")
    for name in list(files):
        component = name.split("/", 1)[0]
        safe = [path for path in files if path.split("/", 1)[0] == component and path.endswith(".safetensors")]
        if safe and (name.endswith((".bin", ".pt")) or name.endswith(".bin.index.json")):
            del files[name]
    if variant:
        files = {name: size for name, size in files.items()
                 if not name.endswith(".safetensors") or f".{variant}." in name}
    if any(size is None for size in files.values()):
        raise ValueError("model file sizes are unavailable; cannot authorize a quota-checked download")
    if "model_index.json" not in files or not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("model metadata does not contain a complete configured Diffusers weight selection")
    return files


def load_config():
    sys.path.insert(0, str(ROOT / "src"))
    from visual_synthesis.config import load_config as read_config

    return read_config([ROOT / "config/pipeline.json", ROOT / "config/benchmark.models.json",
                        ROOT / "config/benchmark.hpc.json"])


def prepare(model_id, out, allow_download=False, budget_gib=None):
    state = {"model": model_id, "status": "checking", "downloads_allowed": allow_download}
    save(state, out / "preparation.json")
    try:
        cfg = load_config()
        from visual_synthesis.backends.image import DiffusersBackend
        from visual_synthesis.benchmark_cluster import _check_model_files
        from visual_synthesis.config import get_model

        model = get_model(cfg, model_id)
        storage = storage_snapshot(models=cfg["paths"]["models"])
        save(storage, out / "storage-before.json")
        backend = DiffusersBackend(model, cfg["paths"]["models"])
        source = backend.weights_source()
        cached = Path(source) if Path(source).is_dir() else None
        if cached is None:
            from huggingface_hub import try_to_load_from_cache

            index = try_to_load_from_cache(model["repo_id"], "model_index.json", revision=model["revision"])
            cached = Path(index).parent if isinstance(index, str) else None
        try:
            if cached is None:
                raise ValueError("no cached model index")
            _check_model_files(model, cached)
            state.update(status="cached", source=str(cached))
        except ValueError as exc:
            state["cache_check"] = str(exc)
            if not allow_download:
                raise ValueError("model is not fully cached; downloads are disabled for this trial") from exc
            from huggingface_hub import HfApi

            info = HfApi().model_info(model["repo_id"], revision=model["revision"], files_metadata=True)
            files = select_files(info.siblings, model["allow_patterns"], model["variant"])
            destination = Path(cfg["paths"]["models"]) / (model["local_dir"] or model_id)
            needed = sum(size for name, size in files.items()
                         if not (destination / name).is_file() or (destination / name).stat().st_size != size)
            guard = download_capacity(storage, needed, len(files), budget_gib)
            state.update(download_plan={"repo_id": model["repo_id"], "destination": str(destination),
                                        "remaining_bytes_upper_bound": needed, "file_count": len(files),
                                        "guard": guard})
            if not guard["allowed"]:
                raise ValueError(guard["reason"]) from None
            save({"models": {model_id: {"allow_patterns": sorted(files)}}}, out / "download-selection.json")
            state["status"] = "download_permitted"
        outputs = storage["paths"]["trial_outputs"]
        output_quota = (outputs.get("quota") or {}).get("available_bytes")
        output_available = min(outputs["filesystem_free_bytes"],
                               output_quota if output_quota is not None else float("inf"))
        if output_available < 64 * 2**20:
            raise ValueError("less than 64 MiB verified space for the small trial outputs")
    except Exception as exc:
        state.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
    save(state, out / "preparation.json")
    return state


def collect_accounting(job_id):
    if not str(job_id).isdigit():
        raise ValueError("job ID must be numeric")
    fields = "JobIDRaw,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS,AllocTRES,Reason"
    result = command(["sacct", "-j", str(job_id), "-P", "--units=K", f"--format={fields}"])
    result["rows"] = list(csv.DictReader(result["stdout"].splitlines(), delimiter="|")) if result["stdout"] else []
    return result


def finish(out, exit_code, job_id):
    preparation = json.loads((out / "preparation.json").read_text()) if (out / "preparation.json").exists() else {}
    summary = {"model": preparation.get("model"), "exit_code": exit_code, "slurm_job_id": job_id,
               "status": "failed" if exit_code else "success", "preparation": preparation,
               "allocation": {"account": "stud", "partition": "MGPU-TC2", "qos": "normal",
                              "gpus": 1, "cpus": 10, "host_ram": "30G", "wall_time": "06:00:00"}}
    report_path = out / "benchmark/report.json"
    if report_path.exists():
        benchmark = json.loads(report_path.read_text())
        measured = benchmark["models"][0]
        gen = measured.get("generation", {})
        summary.update(status=measured["status"] if not exit_code else "failed",
                       model_load_seconds=measured.get("load_seconds"),
                       warmup_seconds=measured.get("warmup_seconds"),
                       inference_seconds_per_prompt=gen.get("prompt_seconds_inference_only"),
                       prompt_seconds_with_io=gen.get("prompt_seconds_with_io"),
                       resources=measured.get("resources", {}), machine=measured.get("machine", {}),
                       error=measured.get("error"),
                       cpu_offload_hooks_active=measured.get("cpu_offload_hooks_active"))
    else:
        summary["status"] = "blocked" if preparation.get("status") == "blocked" else "failed"
        summary["error"] = preparation.get(
            "error", "no completed benchmark report; inspect Slurm output and accounting"
        )
        preflight_path = out / "environment-preflight.json"
        if preflight_path.exists():
            preflight = json.loads(preflight_path.read_text())
            failures = [check for check in preflight.get("checks", []) if check["status"] == "error"]
            if failures:
                summary.update(failure_stage="environment_preflight", preflight_failures=failures)
    before = json.loads((out / "storage-before.json").read_text()) if (out / "storage-before.json").exists() else None
    repo = before["paths"].get("repo", {}).get("path", str(ROOT)) if before else ROOT
    after = storage_snapshot(repo=repo, frozen_paths=before["paths"] if before else None)
    save(after, out / "storage-after.json")
    summary["trial_output_usage"] = disk_bytes(out)
    changes, warnings = storage_changes(before, after) if before else ({}, {})
    summary.update(storage_allocated_byte_changes=changes, storage_comparison_warnings=warnings)
    summary["accounting"] = collect_accounting(job_id)
    summary["accounting_note"] = "Run collect after the job ends; in-job accounting may still say RUNNING."
    save(summary, out / "trial_summary.json")
    return summary


def repair_storage(out):
    """Preserve old measurements, invalidate mismatched deltas, and recheck the original paths."""
    before = json.loads((out / "storage-before.json").read_text())
    after = json.loads((out / "storage-after.json").read_text())
    summary_path = out / "trial_summary.json"
    original = summary_path.read_text()
    summary = json.loads(original)
    backup = out / "trial_summary.before-storage-repair.json"
    if not backup.exists():
        with backup.open("x") as handle:
            handle.write(original)
    changes, warnings = storage_changes(before, after)
    repo = before["paths"].get("repo", {}).get("path", str(ROOT))
    rechecked = storage_snapshot(repo=repo, frozen_paths=before["paths"])
    snapshot_name = f"storage-rechecked-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    save(rechecked, out / snapshot_name)
    rechecked_changes, rechecked_warnings = storage_changes(before, rechecked)
    summary.update(storage_allocated_byte_changes=changes, storage_comparison_warnings=warnings,
                   storage_recheck={"snapshot": snapshot_name, "allocated_byte_changes": rechecked_changes,
                                    "warnings": rechecked_warnings,
                                    "note": "Measured after the job; changes may include unrelated later writes."})
    save(summary, summary_path)
    return {"status": "repaired", "original_summary_backup": str(backup),
            "invalidated_deltas": warnings, "recheck": summary["storage_recheck"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["storage", "prepare", "finish", "collect", "repair-storage"])
    parser.add_argument("--model", choices=MODELS, default="sd-xl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    args = parser.parse_args()
    if args.action == "storage":
        report = storage_snapshot()
        save(report, args.output)
    elif args.action == "prepare":
        args.output.mkdir(parents=True, exist_ok=True)
        report = prepare(args.model, args.output, args.allow_download, args.budget_gib)
    elif args.action == "finish":
        report = finish(args.output, args.exit_code, args.job_id)
    elif args.action == "repair-storage":
        report = repair_storage(args.output)
    else:
        if not args.output.is_dir():
            raise ValueError("trial directory does not exist")
        report = collect_accounting(args.job_id)
        save(report, args.output / "accounting-final.json")
        if (args.output / "trial_summary.json").exists():
            summary = json.loads((args.output / "trial_summary.json").read_text())
        else:
            prep_path = args.output / "preparation.json"
            prep = json.loads(prep_path.read_text()) if prep_path.exists() else {}
            summary = {"model": prep.get("model"), "slurm_job_id": args.job_id, "status": "incomplete",
                       "error": "job ended before it could write a trial summary; inspect accounting and logs"}
        summary["accounting"] = report
        states = [row.get("State", "") for row in report["rows"]]
        if any(state.startswith(("OUT_OF_MEMORY", "TIMEOUT", "FAILED", "CANCELLED")) for state in states):
            summary.update(status="blocked" if summary.get("status") == "blocked" else "failed",
                           resource_limit_states=states)
        save(summary, args.output / "trial_summary.json")
    print(json.dumps(report, indent=2))
    return 20 if report.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
