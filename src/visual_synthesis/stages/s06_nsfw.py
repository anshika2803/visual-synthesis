"""Stage 6 -- run an NSFW detector over the generated images.

Which images: every image under the generate run folder (or any folder / single
image passed with ``--input``). Which detector: ``nsfw.detector`` (or
``--detector``; ``--detector none`` skips the stage).

Writes, into the run folder:

    detections.jsonl     one record appended after every image
    detections.parquet   the same records as a table, one row per image
                         (per_class holds the per-label scores as JSON)
    summary.parquet      headline numbers: flagged rate, counts per label/category

When the generation table is found next to the images, every detection also
carries its prompt, category and risk level -- so "do prompts classified as
sexual actually produce NSFW images?" is answerable from one table.

Incremental: an image that already has a successful record is skipped.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import pandas as pd

from visual_synthesis.backends.detectors import build_detector, is_flagged, supports_threads
from visual_synthesis.config import get_detector
from visual_synthesis.io import append_jsonl, clean, read_jsonl, read_table, write_table
from visual_synthesis.progress import Progress

log = logging.getLogger(__name__)

NAME = "nsfw"
SLUG = "s06_nsfw"
HELP = "Detect NSFW content in the generated images."
NEEDS_GPU = False

# Copied from the generation table onto every detection record.
CONTEXT_COLUMNS = ["prompt_id", "prompt", "model", "standard_category", "risk_level"]


def run(cfg, run, source=None):
    """``source`` is a generate run folder, a folder of images, or one image."""
    settings = cfg["nsfw"]
    detector = get_detector(cfg)
    if detector is None:
        log.info("NSFW detection is off (nsfw.detector=%r); nothing to do", settings["detector"])
        return {"rows_in": 0, "rows_out": 0, "detector": None}
    if source is None:
        raise FileNotFoundError("no images found: run `generate` first, or pass --input")
    threshold = detector["threshold"] if settings["threshold"] is None else settings["threshold"]

    images = find_images(Path(source), settings)
    context = load_generation_context(Path(source), cfg) if settings["join_manifest"] else {}

    records_file = run.path("detections.jsonl")
    done = set()
    if not cfg["run"]["overwrite"]:
        done = {r["image_id"] for r in read_jsonl(records_file) if r["status"] == "success"}
    pending = [path for path in images if path.stem not in done]
    log.info("%d images found: %d to check, %d already done (detector=%s, threshold=%s)",
             len(images), len(pending), len(images) - len(pending), detector["id"], threshold)

    worker_count = min(settings["workers"], len(pending))
    if worker_count > 1 and not supports_threads(detector):
        log.warning(
            "detector %s runs torch on Metal, which is not thread-safe; using one worker instead of %d",
            detector["id"], worker_count,
        )
        worker_count = 1

    summary = {
        "rows_in": len(images),
        "rows_out": 0,
        "rows_skipped": len(images) - len(pending),
        "detector": detector["id"],
        "workers": worker_count,
        "threshold": threshold,
    }
    if cfg["run"]["dry_run"]:
        return {**summary, "planned": len(pending)}

    started = time.perf_counter()
    records_lock = Lock()

    def detect_images(assigned_images):
        """Check an image group with one private detector instance."""
        backend = build_detector(detector, cfg["paths"]["models"])
        errors = 0
        try:
            backend.load()
            for path in assigned_images:
                record = {
                    "image_id": path.stem,
                    "image_path": str(path),
                    "detector_id": detector["id"],
                    "threshold": threshold,
                    "detected_at": datetime.now(UTC).isoformat(),
                    "run_id": run.id,
                    **context.get(path.stem, {}),
                }
                try:
                    scores = backend.detect(path)
                    record.update(
                        status="success",
                        flagged=is_flagged(scores, detector, settings["threshold"]),
                        max_score=round(max(scores.values(), default=0.0), 6),
                        labels=sorted(scores, key=scores.get, reverse=True),
                        per_class=scores,
                        error=None,
                    )
                except Exception as exc:  # one unreadable image must not stop the run
                    log.error("detection failed for %s: %s", path.name, exc)
                    errors += 1
                    record.update(
                        status="error", flagged=False, max_score=0.0, labels=[], per_class={}, error=str(exc)
                    )
                # See generate: a lock prevents interleaved JSONL writes while
                # retaining an on-disk record for every completed image.
                with records_lock:
                    append_jsonl(records_file, record)
                progress.advance(1, errors=int(record["status"] == "error"), flagged=int(record["flagged"]))
        finally:
            backend.unload()
        return len(assigned_images), errors

    try:
        checked = errors = 0
        with Progress(len(pending), "nsfw", "img", settings["log_every"]) as progress:
            if worker_count:
                assignments = [pending[index::worker_count] for index in range(worker_count)]
                with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="nsfw") as executor:
                    futures = [executor.submit(detect_images, assignment) for assignment in assignments]
                    for future in as_completed(futures):
                        completed, worker_errors = future.result()
                        checked += completed
                        errors += worker_errors
    finally:
        table = write_detection_tables(records_file, run, settings, detector["id"])

    error_rate = errors / len(pending) if pending else 0.0
    if error_rate > settings["max_error_rate"]:
        log.warning("error rate %.3f is above nsfw.max_error_rate=%s", error_rate, settings["max_error_rate"])
        summary["failed"] = [f"{errors} image(s) could not be checked"]
    flagged = int(table["flagged"].sum()) if len(table) else 0
    return {
        **summary,
        "rows_out": len(pending),
        "errors": errors,
        "flagged": flagged,
        "flagged_rate": round(flagged / len(table), 4) if len(table) else 0.0,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def find_images(source, settings):
    if source.is_file():
        return [source]
    pattern = "**/*" if settings["recursive"] else "*"
    return [p for p in sorted(source.glob(pattern)) if p.is_file() and p.suffix.lower() in settings["extensions"]]


def load_generation_context(source, cfg):
    """``{image file stem: {prompt_id, prompt, ...}}`` from the generation table, if there is one."""
    table = source / cfg["generate"]["output_name"]
    if not source.is_dir() or not table.exists():
        return {}
    context = {}
    for row in read_table(table).to_dict("records"):
        context[Path(row["image_path"]).stem] = {c: clean(row.get(c)) for c in CONTEXT_COLUMNS}
    return context


def write_detection_tables(records_file, run, settings, detector_id):
    """Rebuild ``detections.parquet`` and ``summary.parquet`` from every record written so far."""
    records = read_jsonl(records_file)
    if not records:
        return pd.DataFrame()
    # A re-checked image (after an error, or with --overwrite) replaces its earlier record.
    table = pd.DataFrame(records).drop_duplicates("image_id", keep="last").reset_index(drop=True)
    run.add_output(write_table(summarize(table, detector_id), run.path(settings["summary_name"])))
    # In the table, labels are ';'-joined and the per-label scores are a JSON string.
    table["labels"] = table["labels"].map(";".join)
    table["per_class"] = table["per_class"].map(json.dumps)
    run.add_output(write_table(table, run.path(settings["output_name"])))
    return table


def summarize(table, detector_id):
    """Headline numbers as (metric, value) rows."""
    flagged = int(table["flagged"].sum())
    metrics = {
        "images": len(table),
        "errors": int((table["status"] == "error").sum()),
        "flagged": flagged,
        "flagged_rate": round(flagged / len(table), 4),
    }
    # How many images each label was seen on.
    label_counts = pd.Series([label for labels in table["labels"] for label in labels]).value_counts()
    for label, count in label_counts.items():
        metrics[f"label:{label}"] = int(count)
    # Flagged vs. not, per prompt category.
    if "standard_category" in table.columns:
        groups = table.groupby(["standard_category", "flagged"], dropna=False).size()
        for (category, is_flagged_), count in groups.items():
            metrics[f"category:{category}:flagged={is_flagged_}"] = int(count)
    return pd.DataFrame({"detector_id": detector_id, "metric": list(metrics), "value": list(metrics.values())})
