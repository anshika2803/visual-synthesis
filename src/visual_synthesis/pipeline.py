"""Running stages: one at a time, or chained.

    STAGES                the seven stages, in order
    run_stage(...)        run one stage in its own run folder
    run_chain(...)        run several stages; each feeds the next
    check_stage(...)      run one stage on a sample input and check the result
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from visual_synthesis.config import ConfigError, load_config, stage_setting
from visual_synthesis.io import read_table
from visual_synthesis.log import setup_logging
from visual_synthesis.runs import Run, latest_run
from visual_synthesis.stages import (
    s01_download, s02_standardize, s03_unify, s04_postprocess, s05_generate, s06_nsfw, s07_metrics,
)

log = logging.getLogger(__name__)

STAGES = [s01_download, s02_standardize, s03_unify, s04_postprocess, s05_generate, s06_nsfw, s07_metrics]
STAGE_NAMES = [stage.NAME for stage in STAGES]

# Other names accepted on the command line.
ALIASES = {
    "convert": "standardize",
    "merge": "unify",
    "dedup": "postprocess",
    "deduplicate": "postprocess",
    "images": "generate",
    "detect": "nsfw",
}


def find_stage(name):
    """A stage by name (``unify``), slug (``s03_unify``), number (``3``) or alias (``merge``)."""
    key = ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    for number, stage in enumerate(STAGES, start=1):
        if key in (stage.NAME, stage.SLUG, str(number)):
            return stage
    raise KeyError(f"unknown stage {name!r}; available: {STAGE_NAMES}")


def stages_between(first=None, last=None):
    start = STAGES.index(find_stage(first)) if first else 0
    end = STAGES.index(find_stage(last)) if last else len(STAGES) - 1
    if start > end:
        raise KeyError(f"--from {first} comes after --to {last}")
    return STAGES[start : end + 1]


def find_input(cfg, stage, explicit=None):
    """Where a stage reads from.

    1. ``--input`` (or ``stages.<name>.input`` in the config)
    2. otherwise the newest run of the previous stage that did not fail

    Returns None when there is nothing yet. Download has no input, and
    standardize reads ``paths.raw`` unless told otherwise.
    """
    explicit = explicit or cfg["stages"].get(stage.NAME, {}).get("input")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{stage.NAME}: input does not exist: {path}")
        return path
    position = STAGES.index(stage)
    if position <= 1:
        return None
    return latest_run(cfg["paths"]["runs"], STAGES[position - 1].SLUG, include_failed=False)


def run_stage(cfg, stage, run_name=None, mode=None, latest=False, input_path=None, run_id=None):
    """Run one stage and return its result dict.

    The result is the stage's summary plus ``stage``, ``status``, ``run_id`` and
    ``run_dir``. ``status`` is success, partial (some items failed), failed,
    or disabled.
    """
    if not cfg["stages"].get(stage.NAME, {}).get("enabled", True):
        log.warning("stage %s is disabled in the config (stages.%s.enabled)", stage.NAME, stage.NAME)
        return {"stage": stage.NAME, "status": "disabled", "run_id": run_id, "run_dir": None}

    run = Run(
        cfg,
        stage.SLUG,
        run_name=run_name or stage_setting(cfg, stage.NAME, "run_name"),
        mode=mode or stage_setting(cfg, stage.NAME, "mode"),
        latest=latest or cfg["run"]["resume_latest"],
        run_id=run_id,
    )
    setup_logging(
        level=cfg["logging"]["level"],
        log_dir=None if cfg["run"]["dry_run"] else Path(cfg["paths"]["logs"]) / run.id,
        run_id=run.id,
        stage=stage.SLUG,
        console_json=cfg["logging"]["console_json"],
    )
    log.info("%s run %s (%s, mode=%s)", "resuming" if run.resumed else "starting", run.dir, run.id, run.mode)

    result = {"stage": stage.NAME, "run_id": run.id, "run_dir": str(run.dir)}
    try:
        source = find_input(cfg, stage, input_path)
        summary = stage.run(cfg, run, source)
    except Exception as exc:
        if cfg["run"]["dry_run"] and isinstance(exc, FileNotFoundError):
            log.info("dry run: %s has no input yet (the previous stage has not run)", stage.NAME)
            return {**result, "status": "dry_run"}
        log.exception("stage %s failed", stage.NAME)
        run.finish("failed", error=f"{type(exc).__name__}: {exc}")
        return {**result, "status": "failed", "error": str(exc)}

    status = "partial" if summary.get("failed") else "success"
    run.finish(status, summary)
    return {**result, **summary, "status": "dry_run" if cfg["run"]["dry_run"] else status}


def run_chain(cfg, stages, run_name=None, mode=None, latest=False):
    """Run stages in order under one run id; stop at the first failure.

    Each stage reads the output of the one before it, so no paths need to be
    passed around.
    """
    run_id = uuid.uuid4().hex[:12]
    results = []
    source = None
    for stage in stages:
        result = run_stage(cfg, stage, run_name, mode, latest, input_path=source, run_id=run_id)
        results.append(result)
        if result["status"] == "failed":
            break
        # Standardize always reads paths.raw, never the download run folder.
        produced_data = result["status"] in ("success", "partial") and stage is not s01_download
        source = result["run_dir"] if produced_data else None
    return results


def check_stage(stage, input_path, expect_rows=None, expect_columns=None, expect_duplicates=None,
                config_files=None, keep_dir=None):
    """Run one stage on ``input_path`` in a scratch folder and check what it wrote.

    Model stages use the fake backends, so this needs no GPU and no weights.
    Returns ``(checks, table)`` where checks is a list of ``(name, passed, detail)``.
    """
    scratch = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="vs-check-"))
    try:
        cfg = load_config(config_files, overrides={
            "paths.runs": str(scratch / "runs"),
            "paths.logs": str(scratch / "logs"),
            "models.check-fake": {"backend": "fake"},
            "detectors.check-fake": {"backend": "fake"},
            "generate.model": "check-fake",
            "nsfw.detector": "check-fake",
        })
    except ConfigError as exc:
        return [("config is valid", False, str(exc))], None

    result = run_stage(cfg, stage, run_name="check", input_path=input_path)
    checks = [("stage ran", result["status"] in ("success", "partial"), result.get("error", result["status"]))]
    table = None
    if checks[0][1] and stage is not s01_download:
        outputs = sorted(Path(result["run_dir"]).glob("*.parquet"))
        checks.append(("stage wrote a table", bool(outputs), [p.name for p in outputs]))
        # The stage's main table: the one named in its config section when several exist.
        main = [p for p in outputs if p.name in main_output_names(cfg)] or outputs
        table = read_table(main[0]) if main else None
    if table is not None:
        if expect_rows is not None:
            checks.append(("row count", len(table) == expect_rows, f"expected={expect_rows} actual={len(table)}"))
        if expect_columns:
            missing = [c for c in expect_columns if c not in table.columns]
            checks.append(("required columns", not missing, f"missing={missing}"))
        if expect_duplicates is not None:
            actual = result.get("duplicates")
            checks.append(("duplicate count", actual == expect_duplicates,
                           f"expected={expect_duplicates} actual={actual}"))
    if not keep_dir:
        shutil.rmtree(scratch, ignore_errors=True)
    return checks, table


def main_output_names(cfg):
    """The file name of each stage's main table."""
    return [cfg[section]["output_name"] for section in ("unify", "dedup", "generate", "nsfw", "metrics")]
