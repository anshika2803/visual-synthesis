"""Stage 5 -- generate an image for every selected prompt.

Which prompts: the deduplicated table, narrowed by ``--limit`` / ``--sample`` /
``--stratify-by`` / filters. Which model: ``generate.model`` (or ``--model``).

Writes, into the run folder:

    images/<prompt_id>_<seed>.png   the images
    generations.jsonl               one record appended after every image
    generations.parquet             the same records as a table

Incremental: a prompt that already has a record in ``generations.jsonl`` is
skipped, so a killed job continues where it stopped when it is re-run with the
same ``--run-name``.
"""

import logging
import time

import pandas as pd

from visual_synthesis.backends.image import build_image_backend, save_image
from visual_synthesis.config import get_model, stage_setting
from visual_synthesis.io import (
    append_jsonl,
    batches,
    clean,
    find_table,
    read_jsonl,
    read_table,
    select_rows,
    write_table,
)

log = logging.getLogger(__name__)

NAME = "generate"
SLUG = "s05_generate"
HELP = "Generate an image for each prompt with a diffusion model."
NEEDS_GPU = True

# Copied from the prompt table onto every generation record.
PROMPT_COLUMNS = ["source_dataset", "standard_category", "risk_level", "benchmark_level", "transformation"]


def run(cfg, run, source=None):
    """``source`` is a postprocess run folder, or any table with prompt_id and prompt."""
    settings = cfg["generate"]
    model = get_model(cfg)
    seed = stage_setting(cfg, NAME, "seed")

    prompts = select_rows(
        read_table(find_table(source, cfg["dedup"]["output_name"])),
        limit=stage_setting(cfg, NAME, "limit"),
        sample=stage_setting(cfg, NAME, "sample"),
        stratify_by=stage_setting(cfg, NAME, "stratify_by"),
        seed=seed,
        filters=cfg["stages"].get(NAME, {}).get("filters"),
    )

    records_file = run.path("generations.jsonl")
    done = set() if cfg["run"]["overwrite"] else {r["prompt_id"] for r in read_jsonl(records_file)}
    # The seed depends on the row's position, so a resumed run reproduces the same images.
    pending = [
        (position, row)
        for position, row in enumerate(prompts.to_dict("records"))
        if str(row["prompt_id"]) not in done
    ]
    log.info("%d prompts selected: %d to generate, %d already done (model=%s)",
             len(prompts), len(pending), len(prompts) - len(pending), model["id"])

    summary = {
        "rows_in": len(prompts),
        "rows_out": 0,
        "rows_skipped": len(prompts) - len(pending),
        "model": model["id"],
        "failed": [],
    }
    if cfg["run"]["dry_run"]:
        return {**summary, "planned": len(pending)}

    backend = build_image_backend(model, cfg["paths"]["models"])
    started = time.perf_counter()
    generated = logged = 0
    try:
        if pending:
            backend.load()
        for batch in batches(pending, settings["batch_size"]):
            requests = [make_request(row, seed + position, settings) for position, row in batch]
            try:
                images = backend.generate(requests)
            except Exception as exc:
                if settings["stop_on_error"]:
                    raise
                log.error("generation failed for %s: %s", [r["prompt_id"] for r in requests], exc)
                summary["failed"] += [r["prompt_id"] for r in requests]
                continue

            rows = {str(row["prompt_id"]): row for _, row in batch}
            for image in images:
                record = save_record(image, rows[image["prompt_id"]], model["id"], settings, run)
                append_jsonl(records_file, record)
                generated += 1
            if generated - logged >= settings["log_every"]:
                logged = generated
                rate = generated / (time.perf_counter() - started)
                log.info("generated %d images so far (%.2f images/s)", generated, rate)
    finally:
        backend.unload()
        write_records_table(records_file, run, settings)

    elapsed = time.perf_counter() - started
    return {
        **summary,
        "rows_out": generated,
        "elapsed_s": round(elapsed, 2),
        "images_per_min": round(generated / elapsed * 60, 2) if generated else 0,
        "images_dir": str(run.path(settings["image_subdir"])),
    }


def make_request(row, seed, settings):
    """The backend request for one prompt row. None settings fall back to the model's own."""
    return {
        "prompt_id": str(row["prompt_id"]),
        "prompt": str(row["prompt"]),
        "seed": seed,
        "negative_prompt": settings["negative_prompt"],
        "steps": settings["num_inference_steps"],
        "guidance_scale": settings["guidance_scale"],
        "width": settings["width"],
        "height": settings["height"],
    }


def save_record(image, row, model_id, settings, run):
    """Save one image to disk and return its generation record."""
    extension = "jpg" if settings["save_format"] == "jpeg" else settings["save_format"]
    filename = settings["image_template"].format(prompt_id=image["prompt_id"], seed=image["seed"])
    path = run.path(settings["image_subdir"], f"{filename}.{extension}")
    save_image(image["image"], path, settings["save_format"], settings["jpeg_quality"])
    record = {
        "prompt_id": image["prompt_id"],
        "prompt": str(row["prompt"]),
        "model": model_id,
        "image_path": str(path),
        "seed": image["seed"],
        "generation_time_s": round(image["seconds"], 4),
        "width": image["width"],
        "height": image["height"],
        "num_inference_steps": image["steps"],
        "guidance_scale": image["guidance_scale"],
        "run_id": run.id,
    }
    for column in PROMPT_COLUMNS:
        record[column] = clean(row.get(column))
    return record


def write_records_table(records_file, run, settings):
    """Rebuild ``generations.parquet`` from every record written so far."""
    records = read_jsonl(records_file)
    if not records:
        return
    # A regenerated image (--overwrite) replaces its earlier record.
    table = pd.DataFrame(records).drop_duplicates(["prompt_id", "seed"], keep="last")
    run.add_output(write_table(table, run.path(settings["output_name"])))
