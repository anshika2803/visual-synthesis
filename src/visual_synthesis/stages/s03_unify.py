"""Stage 3 -- merge every standardized dataset into one table, ``unified.parquet``.

Incremental: when the run already has the unified table and the input tables
have not changed since (same sha256), nothing is rewritten.
"""

import logging
from pathlib import Path

import pandas as pd

from visual_synthesis.io import file_sha256, list_tables, read_table, write_table
from visual_synthesis.schema import validate_table

log = logging.getLogger(__name__)

NAME = "unify"
SLUG = "s03_unify"
HELP = "Merge the standardized datasets into one unified table."
NEEDS_GPU = False


def run(cfg, run, source=None):
    """``source`` is a standardize run folder (or a single standardized table)."""
    settings = cfg["unify"]
    files = [Path(source)] if source and Path(source).is_file() else list_tables(source) if source else []
    if not files:
        raise FileNotFoundError(f"no standardized tables found in {source}; run `standardize` first")

    output = run.path(settings["output_name"])
    inputs = {path.name: file_sha256(path) for path in files}
    previous = run.previous.get("summary", {})
    if output.exists() and previous.get("inputs") == inputs and not cfg["run"]["overwrite"]:
        log.info("unified table is up to date (%s rows); skipping", previous.get("rows_out"))
        run.add_output(output)
        return {**previous, "skipped": True}

    frames = []
    rejected = []
    for path in files:
        frame = read_table(path)
        report = validate_table(frame, cfg["taxonomy"])
        if not report["ok"] and settings["skip_invalid"]:
            log.error("%s rejected: %s", path.name, report["errors"])
            rejected.append(path.name)
            continue
        log.info("merging %s (%d rows)", path.name, len(frame))
        frames.append(frame)
    if not frames:
        raise ValueError("every input table was rejected; nothing to unify")

    unified = pd.concat(frames, ignore_index=True)
    rows_in = len(unified)

    if settings["require_unique_ids"]:
        repeated = unified.loc[unified["prompt_id"].duplicated(), "prompt_id"]
        if len(repeated):
            raise ValueError(
                f"{len(repeated)} prompt_id value(s) appear in more than one row, e.g. {sorted(repeated)[:5]}; "
                "give each dataset its own id_template prefix, or set unify.require_unique_ids=false"
            )

    sort_by = [column for column in settings["sort_by"] if column in unified.columns]
    if sort_by:
        unified = unified.sort_values(sort_by, kind="stable").reset_index(drop=True)

    report = validate_table(unified, cfg["taxonomy"])
    summary = {
        "rows_in": rows_in,
        "rows_out": len(unified),
        "inputs": inputs,
        "rejected": rejected,
        "per_dataset": {str(k): int(v) for k, v in unified["source_dataset"].value_counts().items()},
        "category_counts": report["category_counts"],
        "risk_counts": report["risk_counts"],
        "duplicate_prompts": report["duplicate_prompts"],
    }
    if cfg["run"]["dry_run"]:
        log.info("dry run: would write %d rows to %s", len(unified), output)
        return summary

    run.add_output(write_table(unified, output))
    return summary
