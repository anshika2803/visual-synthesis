"""Stage 4 -- postprocess the unified table: remove duplicate prompts.

Two prompts are duplicates when they are equal after normalisation (trimmed,
lower-cased, whitespace collapsed -- see ``dedup`` in the config). The first
occurrence is kept. Two tables are written, and together they hold every row:

    unified_deduplicated.parquet   the unique prompts (what `generate` reads)
    duplicates.parquet             the removed rows, each pointing at the row it
                                   duplicates via duplicate_of_prompt_id

Incremental: when both tables exist and the input has not changed since (same
sha256), nothing is rewritten.
"""

import logging

from visual_synthesis.io import file_sha256, find_table, read_table, write_table
from visual_synthesis.schema import normalize_prompt

log = logging.getLogger(__name__)

NAME = "postprocess"
SLUG = "s04_postprocess"
HELP = "Postprocess the unified table (deduplicate prompts)."
NEEDS_GPU = False


def run(cfg, run, source=None):
    """``source`` is a unify run folder, or a unified table."""
    settings = cfg["dedup"]
    table = find_table(source, cfg["unify"]["output_name"])
    output = run.path(settings["output_name"])
    duplicates_output = run.path(settings["duplicates_name"])

    input_sha = file_sha256(table)
    previous = run.previous.get("summary", {})
    up_to_date = output.exists() and duplicates_output.exists() and previous.get("input_sha256") == input_sha
    if up_to_date and not cfg["run"]["overwrite"]:
        log.info("deduplicated table is up to date (%s rows); skipping", previous.get("rows_out"))
        run.add_output(output)
        run.add_output(duplicates_output)
        return {**previous, "skipped": True}

    frame = mark_duplicates(read_table(table), settings)
    unique = frame[~frame["is_duplicate"]].reset_index(drop=True)
    duplicates = frame[frame["is_duplicate"]].reset_index(drop=True)
    log.info("%d rows in: %d unique, %d duplicates (strategy=%s)",
             len(frame), len(unique), len(duplicates), settings["strategy"])

    per_dataset = frame.groupby("source_dataset")["is_duplicate"].agg(["size", "sum"])
    summary = {
        "rows_in": len(frame),
        "rows_out": len(unique),
        "duplicates": len(duplicates),
        "duplicate_groups": int(duplicates["duplicate_group_id"].nunique()),
        "input": str(table),
        "input_sha256": input_sha,
        "per_dataset": {
            str(name): {"rows": int(row["size"]), "duplicates": int(row["sum"])}
            for name, row in per_dataset.iterrows()
        },
    }
    if cfg["run"]["dry_run"]:
        log.info("dry run: would write %d unique and %d duplicate rows", len(unique), len(duplicates))
        return summary

    run.add_output(write_table(unique, output))
    run.add_output(write_table(duplicates, duplicates_output))
    return summary


def mark_duplicates(frame, settings):
    """Fill is_duplicate, duplicate_of_prompt_id and duplicate_group_id."""
    frame = frame.copy()
    frame["is_duplicate"] = False
    frame["duplicate_of_prompt_id"] = None
    frame["duplicate_group_id"] = None
    if settings["strategy"] == "none" or frame.empty:
        return frame

    key = frame["prompt"].astype(str).map(
        lambda text: normalize_prompt(text, settings["case_sensitive"], settings["collapse_whitespace"])
    )
    is_duplicate = key.duplicated(keep="first")
    first_id = frame["prompt_id"].groupby(key).transform("first")
    # Number the groups in the order their first duplicate appears.
    group_number = {k: n for n, k in enumerate(key[is_duplicate].unique(), start=1)}

    frame["is_duplicate"] = is_duplicate
    frame.loc[is_duplicate, "duplicate_of_prompt_id"] = first_id[is_duplicate]
    frame.loc[is_duplicate, "duplicate_group_id"] = key[is_duplicate].map(lambda k: f"dup_{group_number[k]:06d}")
    return frame
