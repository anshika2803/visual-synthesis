"""Reading and writing tables, and choosing which rows a stage processes.

Parquet is the pipeline's table format. Long-running stages also keep a JSONL
file they append to after every item, which is what makes them resumable.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TABLE_SUFFIXES = (".parquet", ".csv", ".jsonl", ".json")


# ------------------------------------------------------------------- tables


def read_table(path):
    """Read a parquet/csv/jsonl/json table into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported table format {suffix!r}: {path} (expected {TABLE_SUFFIXES})")


def write_table(frame, path):
    """Write a DataFrame to parquet atomically, so a killed job never leaves half a file."""
    path = Path(path).with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    log.info("wrote %s (%d rows, %d columns)", path, len(frame), len(frame.columns))
    return path


def list_tables(directory):
    """Table files directly inside ``directory``. Names starting with ``_`` are bookkeeping."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.startswith("_")
    ]


def find_table(source, name=None):
    """Resolve a stage input to one table file.

    ``source`` may be a table file, or a run directory; in a directory the table
    called ``name`` is preferred, falling back to the only/first table there.
    """
    if source is None:
        raise FileNotFoundError("no input found: run the previous stage first, or pass --input")
    source = Path(source)
    if source.is_file():
        return source
    if name and (source / name).is_file():
        return source / name
    tables = [t for t in list_tables(source) if t.suffix.lower() == ".parquet"] or list_tables(source)
    if not tables:
        raise FileNotFoundError(f"no table found in {source}")
    return tables[0]


# ------------------------------------------------------- incremental records


def append_jsonl(path, record):
    """Append one record and flush it to disk, so at most one item is lost on a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    """All records of a JSONL file; a half-written last line is ignored."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("ignoring a malformed line in %s", path)
    return records


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


# ------------------------------------------------------------ small helpers


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name):
    """Turn an id such as ``stabilityai/sd-turbo`` into a safe file/folder name."""
    name = str(name).strip().replace("/", "__").replace("\\", "__")
    return "".join(char if (char.isalnum() or char in "._-") else "_" for char in name)


def batches(items, size):
    size = max(int(size), 1)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def clean(value):
    """A table cell as a plain value: NaN/None become None, everything else a string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


# ------------------------------------------------------------ row selection


def apply_filters(frame, filters):
    """Keep rows matching ``filters``.

    {"risk_level": "high"}                      exact match
    {"standard_category": ["sexual", "violence"]}  any of
    {"_query": "risk_level == 'high'"}          a pandas query string
    """
    for column, wanted in (filters or {}).items():
        if column == "_query":
            frame = frame.query(wanted)
        elif column not in frame.columns:
            log.warning("filter column %r is not in the table; ignoring it", column)
        elif isinstance(wanted, (list, tuple, set)):
            frame = frame[frame[column].isin(list(wanted))]
        else:
            frame = frame[frame[column] == wanted]
    return frame.reset_index(drop=True)


def select_rows(frame, limit=None, sample="first", stratify_by=None, seed=42, filters=None):
    """Choose the rows a stage should process. Deterministic for a given seed.

    sample: ``first`` / ``last`` keep dataset order, ``random`` shuffles, and
    ``stratified`` keeps each group's share of ``stratify_by`` (so a 60-row
    pilot keeps the benchmark's category mix).
    """
    frame = apply_filters(frame, filters)
    if limit is None or limit >= len(frame):
        return frame
    if sample == "first":
        selected = frame.head(limit)
    elif sample == "last":
        selected = frame.tail(limit)
    elif sample == "random":
        selected = frame.sample(n=limit, random_state=seed)
    elif sample == "stratified":
        if not stratify_by or stratify_by not in frame.columns:
            raise ValueError(f"sample='stratified' needs stratify_by to name a column, got {stratify_by!r}")
        selected = stratified_sample(frame, limit, stratify_by, seed)
    else:
        raise ValueError(f"unknown sample strategy {sample!r}")
    log.info("selected %d of %d rows (sample=%s, seed=%s)", len(selected), len(frame), sample, seed)
    return selected.sort_index().reset_index(drop=True)


def stratified_sample(frame, limit, by, seed):
    """Sample ``limit`` rows, proportionally per group, keeping at least one row per group."""
    sizes = frame.groupby(by, dropna=False).size()
    quota = (sizes / sizes.sum() * limit).round().astype(int).clip(lower=1)
    # Rounding can overshoot; trim the largest groups until the total fits.
    while quota.sum() > limit and quota.max() > 1:
        quota[quota.idxmax()] -= 1
    picked = [
        group.sample(n=min(int(quota[key]), len(group)), random_state=seed)
        for key, group in frame.groupby(by, dropna=False)
    ]
    return pd.concat(picked).head(limit)
