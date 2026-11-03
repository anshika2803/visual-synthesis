"""Paired local generators with offline, generator-independent diagnostic screening."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from visual_synthesis import l2_pair_review as detail
from visual_synthesis import l2_verification as nli
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

DEFAULT_OUTPUT = ROOT / "results/runs/local/l2-model-comparison-v1"
SOURCE = ROOT / "data/processed/unified/unified_prompt_candidates.csv"
INSTRUCTION = ROOT / "config/prompts/l2_rephrase_v2.txt"


def parents():
    with (ROOT / "results/runs/local/l2-pilot/parents.csv").open() as f:
        selected = [r for r in csv.DictReader(f) if r["sampling_category"] != "sexual"]
    ids = json.loads((ROOT / "config/l2_nudity_cases.json").read_text())["case_ids"]
    with SOURCE.open() as f:
        source = {r["prompt_id"]: r for r in csv.DictReader(f)}
    for key in ids:
        r = source[key]
        selected.append(
            dict(
                parent_prompt_id=key,
                original_prompt=r["prompt"],
                sampling_category="sexual",
                original_category=r["original_category"],
            )
        )
    selected.extend(
        [
            dict(
                parent_prompt_id="CONTROL_CATS",
                original_prompt="Two cats are sleeping on a blue sofa.",
                sampling_category="benign_control",
                original_category="benign",
            ),
            dict(
                parent_prompt_id="CONTROL_SPATIAL",
                original_prompt="A red bicycle is parked behind a brick wall.",
                sampling_category="benign_control",
                original_category="benign",
            ),
        ]
    )
    if len(selected) != 30 or len({r["parent_prompt_id"] for r in selected}) != 30:
        raise ValueError("Expected 28 distinct inspected source parents and two controls.")
    # Freeze the relevant fields only; category annotations remain source provenance.
    return [
        {k: r[k] for k in ("parent_prompt_id", "original_prompt", "sampling_category", "original_category")}
        for r in selected
    ]


def screen(original, created, verifier):
    prediction = nli.classify_pair(original, created, verifier)
    spatial = detail.spatial_differences(original, created)
    if spatial:
        prediction.update(label="not_L2", reason="explicit_spatial_relation_changed", spatial_differences=spatial)
    return prediction


def summarize(rows):
    groups = defaultdict(Counter)
    for row in rows:
        for category in ("ALL", row["category"]):
            counts = groups[(row["generator"], category)]
            counts["attempted"] += 1
            counts[row["generation_status"]] += 1
            if row["generation_status"] == "produced":
                counts["screen_" + row["diagnostic_label"]] += 1
    return [{"generator": m, "category": c, **counts} for (m, c), counts in sorted(groups.items())]


def run(output, models, phase):
    output.mkdir(parents=True, exist_ok=True)
    with _run_lock(output):
        settings = load_settings()
        manifest_path = output / "experiment.json"
        prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if phase != "all" and not prior:
            raise ValueError("Run the generation phase first using --phase all.")
        identities = []
        if phase == "all":
            for model in models:
                identities.append(OllamaClient({**settings, "model": model}).identity())
        else:
            identities = prior["models"]
        manifest = dict(
            parents=parents(),
            models=identities,
            instruction=INSTRUCTION.read_text(),
            schema=OUTPUT_SCHEMA,
            options=settings["options"],
            source_sha256=_digest(SOURCE.read_bytes()),
            driver_sha256=_digest(Path(__file__).read_bytes()),
            parser_sha256=_digest((ROOT / "src/visual_synthesis/l2_paraphrase.py").read_bytes()),
            human_validated=False,
            scope="Development sample: reused parents, unequal category sizes; not held out.",
        )
        if prior is not None and manifest != prior:
            raise ValueError("Frozen inputs, model identities or code changed; use a new --output.")
        _write_json(manifest_path, manifest)
        if phase == "all":
            for index, identity in enumerate(identities):
                client = OllamaClient({**settings, "model": identity["model"]})
                try:
                    for parent in manifest["parents"]:
                        path = output / f"attempts/model_{index}" / (parent["parent_prompt_id"] + ".json")
                        if path.exists():
                            continue
                        start = time.monotonic()
                        record = dict(parent_prompt_id=parent["parent_prompt_id"], generator=identity["model"])
                        try:
                            response = client.request(
                                "/api/chat",
                                dict(
                                    model=identity["model"],
                                    stream=False,
                                    format=OUTPUT_SCHEMA,
                                    options=settings["options"],
                                    keep_alive="5m",
                                    messages=[
                                        dict(role="system", content=manifest["instruction"]),
                                        dict(
                                            role="user", content=json.dumps({"description": parent["original_prompt"]})
                                        ),
                                    ],
                                ),
                            )
                            record.update(parse_response(response), raw_response=response)
                        except OllamaError as error:
                            record.update(generation_status="technical_error", paraphrased_prompt="", detail=str(error))
                        record["elapsed_seconds"] = time.monotonic() - start
                        _write_json(path, record)
                        print(
                            f"{identity['model']} {parent['parent_prompt_id']}: {record['generation_status']}",
                            flush=True,
                        )
                finally:
                    client.request("/api/chat", dict(model=identity["model"], messages=[], stream=False, keep_alive=0))
        verifier = nli.DebertaNLI()
        controls = []
        for key, original, created, expected in detail.CONTROLS:
            prediction = screen(original, created, verifier)
            controls.append(
                dict(
                    control_id=key,
                    expected=expected,
                    predicted=prediction["label"],
                    matched=prediction["label"] == expected,
                    evidence=prediction,
                )
            )
        _write_json(output / "screening_controls.json", dict(verifier=verifier.identity, records=controls))
        rows = []
        for index, identity in enumerate(identities):
            for parent in manifest["parents"]:
                path = output / f"attempts/model_{index}" / (parent["parent_prompt_id"] + ".json")
                record = json.loads(path.read_text())
                original, created = parent["original_prompt"], record.get("paraphrased_prompt", "")
                prediction = screen(original, created, verifier) if record["generation_status"] == "produced" else {}
                row = dict(
                    generator=identity["model"],
                    parent_prompt_id=parent["parent_prompt_id"],
                    category=parent["sampling_category"],
                    original_prompt=original,
                    created_prompt=created,
                    generation_status=record["generation_status"],
                    diagnostic_label=prediction.get("label", ""),
                    reason=prediction.get("reason", record.get("detail", "")),
                    lexical_similarity=SequenceMatcher(None, original.lower(), created.lower()).ratio()
                    if created
                    else "",
                    elapsed_seconds=record["elapsed_seconds"],
                    human_validated=False,
                )
                rows.append(row)
                _write_json(output / f"screening/model_{index}" / path.name, prediction)
        _atomic_text(output / "comparison.csv", _csv_text(rows, list(rows[0])))
        summary = dict(
            groups=summarize(rows),
            controls_matched=sum(c["matched"] for c in controls),
            controls_total=len(controls),
            human_validated=0,
            verifier=verifier.identity,
            screen_code_sha256=_digest(Path(nli.__file__).read_bytes() + Path(detail.__file__).read_bytes()),
            limitations=[
                "NLI plus literal spatial checks are diagnostic, not validated L2 acceptance.",
                "Lexical similarity measures wording overlap, not paraphrase quality.",
                "Development sample reuses prior parents; no category routing rule is established.",
            ],
        )
        _write_json(output / "summary.json", summary)
        cards = []
        for parent in manifest["parents"]:
            matches = [r for r in rows if r["parent_prompt_id"] == parent["parent_prompt_id"]]
            cells = "".join(
                "<td><b>"
                + html.escape(r["generator"])
                + "</b><p>"
                + html.escape(r["generation_status"] + " / " + r["diagnostic_label"])
                + "</p><p>"
                + html.escape(r["created_prompt"])
                + "</p><small>"
                + html.escape(r["reason"])
                + "</small></td>"
                for r in matches
            )
            cards.append(
                "<section><h2>"
                + html.escape(parent["parent_prompt_id"] + " — " + parent["sampling_category"])
                + "</h2><p>"
                + html.escape(parent["original_prompt"])
                + "</p><table><tr>"
                + cells
                + "</tr></table></section>"
            )
        _atomic_text(
            output / "report.html",
            "<!doctype html><meta charset='utf-8'><title>L2 generator comparison</title>"
            "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}"
            "td{vertical-align:top;width:50%;padding:16px;border:1px solid #ddd}table{width:100%;table-layout:fixed}"
            "section{margin:35px 0}small{color:#555}</style><h1>L2 text-generator comparison</h1>"
            "<p>30 inputs per model: 28 inspected source prompts and 2 benign controls. Unequal category sizes. "
            "Automated diagnostics only; no human-validated labels or final benchmark additions.</p>" + "".join(cards),
        )
        print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs=2, default=["qwen3:4b-instruct-2507-q4_K_M", "gemma2:2b"])
    parser.add_argument("--phase", choices=["all", "screen"], default="all")
    args = parser.parse_args(argv)
    if len(set(args.models)) != 2:
        parser.error("Choose two distinct models.")
    try:
        run(args.output.resolve(), args.models, args.phase)
        return 0
    except (ValueError, OSError, RuntimeError, OllamaError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
