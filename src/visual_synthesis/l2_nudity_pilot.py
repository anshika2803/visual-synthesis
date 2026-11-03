"""Prepare exact source parents and attempted L2 outputs for the standard stages."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from visual_synthesis.config import ConfigError, get_model, load_config
from visual_synthesis.l2_image_comparison import DEFAULT_PROMPT, clip_snapshot
from visual_synthesis.l2_paraphrase import (
    OUTPUT_SCHEMA,
    ROOT,
    OllamaClient,
    OllamaError,
    _atomic_text,
    _csv_text,
    _digest,
    _run_lock,
    _write_json,
    load_settings,
    parse_response,
)
from visual_synthesis.schema import STANDARD_COLUMNS, content_hash

DEFAULT_OUTPUT = ROOT / "results/runs/local/l2-nudity-project-pipeline-v2"
SOURCE = ROOT / "data/processed/unified/unified_prompt_candidates.csv"
CASES = ROOT / "config/l2_nudity_cases.json"
EXTRAS = ["evaluation_seed", "source_file", "paired_parent_id", "pair_variant", "human_validated"]


def expand_pairs(parents, outcomes, seeds, source_file):
    """A refusal leaves original-only rows; it never becomes a fabricated L2."""
    rows = []
    for parent in parents:
        parent_id = parent["prompt_id"]
        outcome = outcomes[parent_id]
        variants = [("original", parent["prompt"], "L1", "original")]
        if outcome["generation_status"] == "produced":
            variants.append(("paraphrase", outcome["paraphrased_prompt"], "L2", "paraphrase"))
        for seed in seeds:
            for variant, text, level, transformation in variants:
                row = {column: parent.get(column, "") for column in STANDARD_COLUMNS}
                row.update(
                    prompt_id=f"{parent_id}_{level}_seed{seed}",
                    prompt=text,
                    benchmark_level=level,
                    transformation=transformation,
                    parent_prompt_id=parent_id if level == "L2" else "",
                    content_hash=content_hash(text),
                    run_id="l2-nudity-boundary-pilot",
                    evaluation_seed=seed,
                    source_file=str(source_file),
                    paired_parent_id=parent_id,
                    pair_variant=variant,
                    human_validated=False,
                )
                rows.append(row)
    return rows


def prepare(output, seeds, client=None):
    cases = json.loads(CASES.read_text())
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        selected = {row["prompt_id"]: row for row in csv.DictReader(handle) if row["prompt_id"] in cases["case_ids"]}
    if set(selected) != set(cases["case_ids"]):
        raise ValueError("Selected source IDs are missing from the unified dataset.")
    parents = [selected[key] for key in cases["case_ids"]]
    if any(p["source_dataset"] != "T2I-RiskyPrompt" or p["benchmark_level"] != "L1" for p in parents):
        raise ValueError("Expected original T2I-RiskyPrompt L1 source records.")
    settings = load_settings()
    client = client or OllamaClient(settings)
    instruction = DEFAULT_PROMPT.read_text(encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    with _run_lock(output):
        metadata = {
            "source_csv": str(SOURCE),
            "source_file_sha256": _digest(SOURCE.read_bytes()),
            "parents": parents,
            "case_selection": cases,
            "qwen": client.identity(),
            "qwen_options": settings["options"],
            "instruction": instruction,
            "image_seeds": seeds,
            "human_validated": False,
            "driver_sha256": _digest(Path(__file__).read_bytes()),
        }
        metadata_path = output / "preparation.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Preparation inputs changed. Use a new --output folder.")
        _write_json(metadata_path, metadata)
        outcomes = {}
        try:
            for parent in parents:
                path = output / "qwen_attempts" / f"{parent['prompt_id']}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                else:
                    response = client.request(
                        "/api/chat",
                        {
                            "model": settings["model"],
                            "stream": False,
                            "format": OUTPUT_SCHEMA,
                            "options": settings["options"],
                            "keep_alive": "5m",
                            "messages": [
                                {"role": "system", "content": instruction},
                                {
                                    "role": "user",
                                    "content": json.dumps({"description": parent["prompt"]}, ensure_ascii=False),
                                },
                            ],
                        },
                    )
                    record = {"parent_prompt_id": parent["prompt_id"], **parse_response(response), "response": response}
                    _write_json(path, record)
                outcomes[parent["prompt_id"]] = record
                print(f"{parent['prompt_id']}: {record['generation_status']}", flush=True)
        finally:
            client.request("/api/chat", {"model": settings["model"], "messages": [], "stream": False, "keep_alive": 0})
        rows = expand_pairs(parents, outcomes, seeds, SOURCE)
        table_path = output / "paired_prompts.csv"
        _atomic_text(table_path, _csv_text(rows, STANDARD_COLUMNS + EXTRAS))
        base = load_config(use_env=False)
        if not (Path(base["paths"]["models"]) / "sd-turbo").exists():
            raise ValueError("The SD-Turbo checkpoint is not cached locally.")
        clip_path = clip_snapshot()
        overlay = {
            "paths": {"runs": str(output / "project_runs"), "logs": str(output / "project_logs")},
            "models": {"sd-turbo": {"device": "mps", "dtype": "float16"}},
            "generate": {"model": "sd-turbo", "workers": 1, "batch_size": 1},
            "nsfw": {"detector": "nudenet", "workers": 1},
            "stages": {"generate": {"input": str(table_path)}},
            "metric_models": {"clip-vit-base-patch32": {"repo_id": str(clip_path), "device": "cpu"}},
            "metrics": {"fid": {"enabled": False}},
        }
        config_path = output / "pipeline_config.json"
        _write_json(config_path, overlay)
        # Validate using the same loader as the core CLI before any images are generated.
        cfg = load_config([str(ROOT / "config/pipeline.json"), str(config_path)], use_env=False)
        summary = {
            "selected_parents": len(parents),
            "prepared_image_rows": len(rows),
            "produced_l2": sum(r["generation_status"] == "produced" for r in outcomes.values()),
            "qwen_outcomes": {key: row["generation_status"] for key, row in outcomes.items()},
            "image_model": get_model(cfg)["repo_id"],
            "human_validated": 0,
            "config_path": str(config_path),
            "paired_prompts_path": str(table_path),
        }
        _write_json(output / "preparation_summary.json", summary)
        print(json.dumps(summary, indent=2))
        print(
            "Next: visual-synthesis run all --from generate --to metrics "
            f"--config config/pipeline.json --config {config_path} --run-name nudity-boundary"
        )
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Image seeds must be distinct.")
    try:
        prepare(args.output.resolve(), args.seeds)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ConfigError, OllamaError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
