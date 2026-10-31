"""Stage 2 -- convert each raw dataset into the standard schema.

One output table per dataset: ``<dataset id>.parquet``. How a dataset is read
and how its labels map onto the standard categories is described entirely by
its ``datasets.<id>`` entry in the config, so adding a dataset needs no code.

Incremental: a dataset whose output table already exists in the run is skipped.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from visual_synthesis.config import dataset_dir, resolve_path
from visual_synthesis.io import clean, read_table, write_table
from visual_synthesis.schema import SEPARATOR, STANDARD_COLUMNS, content_hash, risk_level, validate_table

log = logging.getLogger(__name__)

NAME = "standardize"
SLUG = "s02_standardize"
HELP = "Convert raw datasets (I2P, T2I-RiskyPrompt, ...) to the standard schema."
NEEDS_GPU = False


def run(cfg, run, source=None):
    """``source`` overrides ``paths.raw`` as the folder holding the dataset folders."""
    report = {}
    for dataset_id, dataset in cfg["datasets"].items():
        if not dataset["enabled"]:
            continue
        folder = Path(source) / (dataset["dir"] or dataset_id) if source else dataset_dir(cfg, dataset_id)
        report[dataset_id] = standardize_dataset(cfg, run, dataset_id, folder)

    return {
        "rows_in": sum(r["rows_in"] for r in report.values()),
        "rows_out": sum(r["rows_out"] for r in report.values()),
        "datasets": report,
        "failed": [name for name, r in report.items() if r["status"] == "failed"],
    }


def standardize_dataset(cfg, run, dataset_id, folder):
    dataset = cfg["datasets"][dataset_id]
    output = run.path(f"{dataset_id}.parquet")
    outcome = {"status": "failed", "rows_in": 0, "rows_out": 0, "output": str(output), "detail": ""}

    files = ([resolve_path(dataset["input_table"])] if dataset["input_table"] else
             sorted(p for p in folder.glob(dataset["input_glob"]) if p.is_file()) if folder.is_dir() else [])
    files = [path for path in files if path.is_file()]
    if not files:
        log.warning("%s: no %s in %s; skipping (run `download` or place the files)",
                    dataset_id, dataset["input_glob"], folder)
        return {**outcome, "status": "missing", "detail": f"no {dataset['input_glob']} in {folder}"}

    if output.exists() and not cfg["run"]["overwrite"]:
        rows = len(pd.read_parquet(output, columns=["prompt_id"]))
        log.info("%s: already standardized (%d rows); skipping", dataset_id, rows)
        run.add_output(output)
        return {**outcome, "status": "skipped", "rows_in": rows, "rows_out": rows, "detail": "output already exists"}

    if cfg["run"]["dry_run"]:
        log.info("dry run: would convert %s from %s", dataset_id, [f.name for f in files])
        return {**outcome, "status": "dry_run"}

    try:
        source_rows = pd.concat([read_table(path) if dataset["input_table"] else read_source(path, dataset)
                                 for path in files], ignore_index=True)
        table = (complete_standard_table(source_rows, run.id) if dataset["input_table"] else
                 convert(source_rows, dataset_id, dataset, cfg["taxonomy"], run.id))
        validation = validate_table(table, cfg["taxonomy"])
        if not validation["ok"]:
            raise ValueError("invalid standardized table: " + "; ".join(validation["errors"]))
    except Exception as exc:  # one bad dataset must not stop the others
        log.error("%s: conversion failed: %s", dataset_id, exc)
        return {**outcome, "detail": str(exc)}

    if table.empty:
        log.warning("%s: no rows left after conversion", dataset_id)
        return {**outcome, "status": "empty", "rows_in": len(source_rows)}

    run.add_output(write_table(table, output))
    log.info("%s: %d source rows -> %d standard rows", dataset_id, len(source_rows), len(table))
    return {**outcome, "status": "converted", "rows_in": len(source_rows), "rows_out": len(table)}


def complete_standard_table(frame, run_id):
    """Fill bookkeeping columns in a previously standardized dataset without changing its labels."""
    required = {"prompt_id", "prompt", "source_dataset", "standard_category", "risk_level",
                "benchmark_level", "transformation"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"standardized input missing columns: {sorted(missing)}")
    table = frame.copy()
    defaults = {
        "source_id": table["prompt_id"], "original_category": "", "split": "unassigned",
        "is_duplicate": False, "duplicate_of_prompt_id": None, "duplicate_group_id": None,
        "parent_prompt_id": table["prompt_id"], "source_metadata": "",
        "content_hash": table["prompt"].map(content_hash), "run_id": run_id,
    }
    for column, value in defaults.items():
        if column not in table:
            table[column] = value
    return table[STANDARD_COLUMNS + [column for column in table if column not in STANDARD_COLUMNS]]


def read_source(path, dataset):
    """Read one raw file as a DataFrame, according to the dataset's ``format``."""
    fmt = dataset["format"]
    if fmt == "auto":
        fmt = {".csv": "csv", ".json": "json", ".parquet": "parquet"}.get(path.suffix.lower(), "csv")
    if fmt == "csv":
        return pd.read_csv(path, sep=dataset["delimiter"])
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        # Either a list of records, or an object keyed by id: {"0": {...}, "1": {...}}.
        if isinstance(data, dict):
            data = list(data.values())
        return pd.DataFrame(data)
    raise ValueError(f"unsupported format {fmt!r}")


def convert(source_rows, dataset_id, dataset, taxonomy, run_id):
    """Turn a dataset's raw rows into standard rows."""
    for column in (dataset["prompt_column"], dataset["category_column"]):
        if column not in source_rows.columns:
            raise KeyError(f"column {column!r} not found; the file has {list(source_rows.columns)}")

    rows = []
    dropped = 0
    for index, source in enumerate(source_rows.to_dict("records")):
        prompt = clean(source[dataset["prompt_column"]]) or ""
        original = split_labels(source[dataset["category_column"]], dataset["category_delimiter"])
        categories = map_categories(original, dataset, taxonomy, dataset_id)
        if not categories or (dataset["drop_empty_prompts"] and not prompt.strip()):
            dropped += 1
            continue

        id_column = dataset["id_column"]
        source_id = str(source[id_column]) if id_column else str(index)
        prompt_id = dataset["id_template"].format(index=index, source_id=source_id)
        metadata = {c: clean(source.get(c)) or "" for c in dataset["metadata_columns"] if c in source}
        rows.append({
            "prompt_id": prompt_id,
            "source_dataset": dataset["label"] or dataset_id,
            "source_id": source_id,
            "prompt": prompt.strip(),
            "original_category": SEPARATOR.join(original),
            "standard_category": SEPARATOR.join(categories),
            "risk_level": risk_level(categories, taxonomy),
            "benchmark_level": dataset["benchmark_level"],
            "transformation": dataset["transformation"],
            "split": "unassigned",
            "is_duplicate": False,
            "duplicate_of_prompt_id": None,
            "duplicate_group_id": None,
            "parent_prompt_id": prompt_id,
            "source_metadata": SEPARATOR.join(f"{k}={v}" for k, v in metadata.items()),
            "content_hash": content_hash(prompt),
            "run_id": run_id,
        })

    if dropped:
        log.warning("%s: dropped %d row(s) with an empty prompt or no known category", dataset_id, dropped)
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def split_labels(value, delimiter):
    """A raw category cell as a list of labels.

    Handles a delimited string ("violence, self-harm"), a list, and a dict
    whose keys are the labels (T2I-RiskyPrompt).
    """
    if isinstance(value, dict):
        value = list(value)
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = clean(value) or ""
    return [part.strip() for part in text.split(delimiter) if part.strip()]


def map_categories(labels, dataset, taxonomy, dataset_id):
    """Map a dataset's own labels onto the standard categories (no repeats, order kept)."""
    mapped = []
    for label in labels:
        category = dataset["category_map"].get(label)
        if category is None:
            policy = dataset["on_unknown_category"]
            if policy == "raise":
                raise ValueError(
                    f"unmapped label {label!r}; add it to datasets.{dataset_id}.category_map "
                    "or set on_unknown_category to 'warn_skip'"
                )
            if policy == "warn_skip":
                log.warning("%s: skipping unmapped label %r", dataset_id, label)
                continue
            category = label  # keep_as_is
        if category in taxonomy and category not in mapped:
            mapped.append(category)
    return mapped
