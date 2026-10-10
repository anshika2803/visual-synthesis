"""Run directories: where a stage writes, and the rule that nothing is overwritten.

Every stage run gets its own folder, ``data/runs/<stage>/<run_name>/``:

* no ``--run-name``            -> a new folder ``<stage>_<timestamp>``
* ``--run-name x``, x is new   -> create ``x``
* ``--run-name x``, x exists   -> ``--mode incremental`` resumes ``x`` (only the
                                  missing work is done); ``--mode full`` leaves
                                  ``x`` alone and creates ``x_v2``, ``x_v3``, ...

Each folder holds ``_run_manifest.json``: the resolved config, git commit,
host, status, outputs and a summary -- enough to audit or reproduce the run.
"""

import json
import logging
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from visual_synthesis.config import ROOT

log = logging.getLogger(__name__)

MANIFEST = "_run_manifest.json"


def now():
    return datetime.now(UTC)


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=ROOT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def next_free_name(parent, name):
    """``exp`` -> ``exp_v2`` -> ``exp_v3`` ... until the name is unused."""
    base, _, tail = name.rpartition("_v")
    if base and tail.isdigit():
        version = int(tail)
    else:
        base, version = name, 1
    candidate = name
    while (parent / candidate).exists():
        version += 1
        candidate = f"{base}_v{version}"
    return candidate


def read_manifest(directory):
    path = Path(directory) / MANIFEST
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def list_runs(runs_root, stage_slug):
    """Manifests of every run of a stage, newest first. Each has a ``path`` key."""
    parent = Path(runs_root) / stage_slug
    if not parent.is_dir():
        return []
    manifests = []
    for directory in parent.iterdir():
        manifest = read_manifest(directory)
        if manifest:
            manifests.append({**manifest, "path": str(directory)})
    return sorted(manifests, key=lambda m: m.get("created_at", ""), reverse=True)


def latest_run(runs_root, stage_slug, include_failed=True):
    """Folder of the newest run of a stage, or None."""
    for manifest in list_runs(runs_root, stage_slug):
        if include_failed or manifest.get("status") != "failed":
            return Path(manifest["path"])
    return None


class Run:
    """One run of one stage: its folder and its manifest."""

    def __init__(self, cfg, stage_slug, run_name=None, mode="incremental", latest=False, run_id=None):
        parent = Path(cfg["paths"]["runs"]) / stage_slug
        dry_run = cfg["run"]["dry_run"]

        if run_name is None and latest and mode == "incremental":
            newest = latest_run(cfg["paths"]["runs"], stage_slug)
            run_name = newest.name if newest else None

        if run_name is None:
            name = next_free_name(parent, f"{stage_slug}_{datetime.now():%Y%m%d-%H%M%S}")
        elif (parent / run_name).exists() and mode == "full":
            name = next_free_name(parent, run_name)
        else:
            name = run_name

        self.cfg = cfg
        self.stage = stage_slug
        self.name = name
        self.dir = parent / name
        self.mode = mode
        self.resumed = self.dir.exists()
        self.id = run_id or uuid.uuid4().hex[:12]
        self.outputs = {}
        # What this run recorded the last time it ran; stages use it to tell
        # whether their inputs changed since.
        self.previous = read_manifest(self.dir)

        if not dry_run:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.write_manifest("running")

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def add_output(self, path):
        """Record a file this run produced."""
        self.outputs[Path(path).name] = str(path)
        return path

    def write_manifest(self, status, summary=None, error=None):
        manifest = read_manifest(self.dir)
        if "created_at" not in manifest:
            manifest["created_at"] = now().isoformat()
        manifest.update(
            run_id=self.id,
            stage=self.stage,
            run_name=self.name,
            mode=self.mode,
            resumed=self.resumed,
            status=status,
            error=error,
            updated_at=now().isoformat(),
            outputs={**manifest.get("outputs", {}), **self.outputs},
            config=self.cfg,
            environment={
                "git_commit": git_commit(),
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "cwd": os.getcwd(),
            },
        )
        if summary is not None:
            manifest["summary"] = summary
        tmp = self.dir / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, default=str, sort_keys=True), encoding="utf-8")
        tmp.replace(self.dir / MANIFEST)

    def finish(self, status="success", summary=None, error=None):
        if not self.cfg["run"]["dry_run"]:
            self.write_manifest(status, summary, error)
        log.info("run finished: status=%s dir=%s", status, self.dir)
