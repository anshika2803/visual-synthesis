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
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from threading import Lock

import pandas as pd

from visual_synthesis.backends.device import pick_device
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
from visual_synthesis.progress import Progress

log = logging.getLogger(__name__)

NAME = "generate"
SLUG = "s05_generate"
HELP = "Generate an image for each prompt with a diffusion model."
NEEDS_GPU = True

# Copied from the prompt table onto every generation record.
PROMPT_COLUMNS = ["source_dataset", "standard_category", "risk_level", "benchmark_level", "transformation"]


def uses_processes(model, workers):
    """Whether the extra model copies must each live in their own process.

    Torch on Apple Metal is not thread-safe: two Diffusers pipelines generating
    in threads of one process abort it (an MPSGraph over-release). Separate
    processes get separate Metal contexts and work, so on MPS every copy is a
    process. CUDA and CPU copies are threads: they share the loaded weights'
    code and start faster.
    """
    if workers < 2 or model["backend"] != "diffusers":
        return False
    return pick_device(model["device"], model["device_index"]) == "mps"


# The backend owned by this spawned worker process (see ``generate_in_processes``).
_process_backend = None


def _start_process_worker(model, models_root):
    global _process_backend
    _process_backend = build_image_backend(model, models_root)
    _process_backend.load()


def _generate_in_process(requests):
    return _process_backend.generate(requests)


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
    work = list(batches(pending, settings["batch_size"]))
    worker_count = min(settings["workers"], len(work))
    in_processes = uses_processes(model, worker_count)
    log.info("%d prompts selected: %d to generate, %d already done (model=%s)",
             len(prompts), len(pending), len(prompts) - len(pending), model["id"])

    summary = {
        "rows_in": len(prompts),
        "rows_out": 0,
        "rows_skipped": len(prompts) - len(pending),
        "model": model["id"],
        "workers": worker_count,
        "worker_mode": "process" if in_processes else "thread",
        "failed": [],
    }
    if cfg["run"]["dry_run"]:
        return {**summary, "planned": len(pending)}
    if in_processes and model["dtype"] != "float16":
        log.warning(
            "running %d %s copies on Metal in float32 needs about %d GB of memory and can be slower than one; "
            "set the model's dtype to float16 (--set models.%s.dtype=float16) to roughly halve that",
            worker_count, model["id"], 5 * worker_count, model["id"],
        )

    started = time.perf_counter()
    generated = 0
    records_lock = Lock()

    def requests_for(batch):
        return [make_request(row, seed + position, settings) for position, row in batch]

    def store(batch, images):
        """Save a finished batch, append its records and tick the bar; returns the images written."""
        rows = {str(row["prompt_id"]): row for _, row in batch}
        for image in images:
            record = save_record(image, rows[image["prompt_id"]], model["id"], settings, run)
            # Copies share one manifest; serialise append+flush so each
            # completed image remains a complete JSONL record on resume.
            with records_lock:
                append_jsonl(records_file, record)
        progress.advance(len(batch))
        return len(images)

    def fail(requests, exc):
        """Record a failed batch (or re-raise with stop_on_error); returns its prompt ids."""
        if settings["stop_on_error"]:
            raise exc
        ids = [r["prompt_id"] for r in requests]
        log.error("generation failed for %s: %s", ids, exc)
        progress.advance(len(requests), failed=len(requests))
        return ids

    def generate_batches(assigned_batches):
        """Run an assigned sequence of batches through one private model instance (a thread)."""
        backend = build_image_backend(model, cfg["paths"]["models"])
        completed, failures = 0, []
        try:
            backend.load()
            for batch in assigned_batches:
                requests = requests_for(batch)
                try:
                    images = backend.generate(requests)
                except Exception as exc:
                    failures.extend(fail(requests, exc))
                    continue
                completed += store(batch, images)
        finally:
            backend.unload()
        return completed, failures

    def generate_in_processes():
        """One model per worker process; batches are handed out as workers free up."""
        completed, failures = 0, []
        executor = ProcessPoolExecutor(
            worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_process_worker,
            initargs=(model, cfg["paths"]["models"]),
        )
        with executor:
            try:
                pending_futures = {}
                for batch in work:
                    requests = requests_for(batch)
                    pending_futures[executor.submit(_generate_in_process, requests)] = (batch, requests)
                for future in as_completed(pending_futures):
                    batch, requests = pending_futures[future]
                    try:
                        images = future.result()
                    except BrokenProcessPool as exc:
                        raise RuntimeError(
                            "a generation worker process died (out of memory?); use fewer --workers"
                        ) from exc
                    except Exception as exc:
                        failures.extend(fail(requests, exc))
                        continue
                    completed += store(batch, images)
            except BaseException:
                # Do not let the pool grind through the rest of the queue before the error surfaces.
                executor.shutdown(wait=True, cancel_futures=True)
                raise
        return completed, failures

    try:
        with Progress(len(pending), "generate", "img", settings["log_every"], model_id=model["id"]) as progress:
            if in_processes:
                generated, summary["failed"] = generate_in_processes()
            elif worker_count:
                assignments = [work[index::worker_count] for index in range(worker_count)]
                with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="generate") as executor:
                    futures = [executor.submit(generate_batches, assignment) for assignment in assignments]
                    for future in as_completed(futures):
                        completed, failures = future.result()
                        generated += completed
                        summary["failed"].extend(failures)
    finally:
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
