"""Housekeeping: delete caches, old logs, rendered Slurm jobs and old runs.

The pipeline never overwrites its output, so disk use only grows. These helpers
remove what is safe to remove. Nothing is deleted unless ``delete=True``;
otherwise the result only lists what *would* go.
"""

import shutil
from pathlib import Path

TARGETS = ("cache", "logs", "jobs", "runs", "all")


def size_of(path):
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def newest_first(directory):
    """The sub-folders of ``directory``, most recently modified first."""
    if not directory.is_dir():
        return []
    return sorted((p for p in directory.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)


def clean(cfg, what="cache", keep=None, delete=False):
    """Remove one kind of leftover. Returns ``{"removed": [...], "kept": [...], "freed_bytes": n}``.

    cache   everything under paths.cache (re-downloadable)
    logs    all but the newest ``keep`` log folders (default: logging.keep_runs)
    jobs    rendered .sbatch files
    runs    all but the newest run folder of every stage -- pipeline OUTPUT
    all     cache + logs + jobs (never runs)
    """
    if what not in TARGETS:
        raise ValueError(f"unknown clean target {what!r}; choose from {TARGETS}")
    paths = {name: Path(value) for name, value in cfg["paths"].items()}
    keep = cfg["logging"]["keep_runs"] if keep is None else keep
    doomed, kept = [], []

    if what in ("cache", "all") and paths["cache"].is_dir():
        doomed += sorted(paths["cache"].iterdir())
    if what in ("logs", "all"):
        folders = newest_first(paths["logs"])
        kept += folders[:keep]
        doomed += folders[keep:]
    if what in ("jobs", "all") and paths["slurm_jobs"].is_dir():
        doomed += sorted(paths["slurm_jobs"].glob("*.sbatch"))
    if what == "runs":
        for stage_dir in newest_first(paths["runs"]):
            folders = newest_first(stage_dir)
            kept += folders[:1]
            doomed += folders[1:]

    freed = sum(size_of(path) for path in doomed)
    if delete:
        for path in doomed:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return {"removed": doomed, "kept": kept, "freed_bytes": freed}
