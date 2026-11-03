"""An actual local generation-to-classification check using a separate NLI model.

NLI supplies provisional semantic screening, not a trained L0--L6 classifier.
It cannot establish benchmark validity. Synthetic controls expose some failures.
The existing generator and its frozen experiments are not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from visual_synthesis.l2_paraphrase import (
    ROOT,
    OllamaClient,
    OllamaError,
    _atomic_text,
    _csv_text,
    _digest,
    _now,
    _run_lock,
    _smoke_parents,
    _write_json,
    freeze_run,
    generate,
    load_run,
    load_settings,
)
from visual_synthesis.schema import normalize_prompt

MODEL_ID = "cross-encoder/nli-deberta-v3-base"
MODEL_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
SNAPSHOT = ROOT / "models/l2-nli/models--cross-encoder--nli-deberta-v3-base/snapshots" / MODEL_REVISION
LABELS = ("contradiction", "entailment", "neutral")  # Documented by the model author.
THRESHOLD = 0.8  # Exploratory, uncalibrated; frozen before the first control run.
MAX_TOKENS = 512
OUTPUT_SUBDIR = "classification/deberta-v3-base-v2"
CSV_COLUMNS = [
    "original_prompt",
    "created_prompt",
    "is_l2",
    "classification_reason",
    "parent_prompt_id",
    "forward_entailment",
    "reverse_entailment",
    "forward_contradiction",
    "reverse_contradiction",
    "forward_neutral",
    "reverse_neutral",
    "lexical_similarity",
    "classifier_controls_passed",
    "classification_origin",
    "human_validated",
]

# Author-defined benign controls, not human-validated or held-out research data.
# Expected labels are never provided to the inference model.
CONTROLS = [
    (
        "positive_synonyms",
        "The red bicycle is beside the brick wall.",
        "The red bike is next to the wall made of bricks.",
        "L2",
    ),
    (
        "positive_quantity_preserved",
        "Two cats are sleeping on a blue sofa.",
        "A pair of cats sleep on a couch that is blue.",
        "L2",
    ),
    (
        "positive_restructured",
        "A bowl of apples sits on a wooden kitchen table.",
        "On a wooden table in the kitchen sits a bowl containing apples.",
        "L2",
    ),
    ("positive_small_rewrite", "A portrait of an older man.", "A portrait depicting an older man.", "L2"),
    (
        "negative_unchanged",
        "A yellow bus is parked outside a school.",
        "A yellow bus is parked outside a school.",
        "not_L2",
    ),
    ("negative_quantity", "Two cats are sleeping on a blue sofa.", "Three cats are sleeping on a blue sofa.", "not_L2"),
    ("negative_location", "The box is outside the store.", "The box is inside the store.", "not_L2"),
    ("negative_negation", "The lamp is not switched on.", "The lamp is switched on.", "not_L2"),
    ("negative_roles", "The dog is chasing the cat.", "The cat is chasing the dog.", "not_L2"),
    (
        "negative_added_detail",
        "A woman is standing beside a bicycle.",
        "A woman is standing beside a bicycle while holding a red umbrella.",
        "not_L2",
    ),
    ("negative_omitted_detail", "A child in a yellow coat holds a blue balloon.", "A child holds a balloon.", "not_L2"),
    ("negative_unrelated", "A red bike is beside a brick wall.", "A boat floats on a lake.", "not_L2"),
]


REGRESSION_CASES = json.loads((ROOT / "config/l2_verification_regression.json").read_text(encoding="utf-8"))["cases"]
CONTROLS += [
    (row["control_id"], row["original_prompt"], row["created_prompt"], row["expected_label"])
    for row in REGRESSION_CASES
]


class DebertaNLI:
    """CPU-only, offline loading of the already-downloaded pinned snapshot."""

    def __init__(self, snapshot=SNAPSHOT):
        snapshot = Path(snapshot)
        if not (snapshot / "model.safetensors").exists():
            raise ValueError(f"Classifier weights not cached: {snapshot}. See docs/l2_generation_classification.md.")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
            )
            .to("cpu")
            .eval()
        )
        if self.model.config.num_labels != len(LABELS):
            raise ValueError("Unexpected NLI label count.")
        self.identity = {"model": MODEL_ID, "revision": MODEL_REVISION, "device": "cpu", "label_order": list(LABELS)}

    def predict(self, premise, hypothesis):
        encoded = self.tokenizer(premise, hypothesis, truncation=False, return_tensors="pt")
        tokens = encoded["input_ids"].shape[-1]
        if tokens > MAX_TOKENS:
            return {"error": "input_too_long", "tokens": tokens}
        with self.torch.inference_mode():
            scores = self.model(**encoded).logits.softmax(dim=-1)[0].tolist()
        return {"tokens": tokens, "scores": dict(zip(LABELS, scores, strict=True))}


def classify_pair(original, created, verifier, threshold=THRESHOLD):
    result = {"label": "uncertain", "reason": "", "forward": {}, "reverse": {}, "human_validated": False}
    if not original.strip() or not created.strip():
        result["reason"] = "missing_original_or_created_prompt"
        return result
    if normalize_prompt(original) == normalize_prompt(created):
        result.update(label="not_L2", reason="unchanged_copy")
        return result
    forward = verifier.predict(original, created)
    reverse = verifier.predict(created, original)
    result.update(forward=forward, reverse=reverse)
    if forward.get("error") or reverse.get("error"):
        result["reason"] = "classifier_input_not_evaluated"
        return result
    a, b = forward["scores"], reverse["scores"]
    if min(a["entailment"], b["entailment"]) >= threshold:
        result.update(label="L2", reason="mutual_entailment_above_exploratory_threshold")
    elif max(a["contradiction"], b["contradiction"], a["neutral"], b["neutral"]) >= threshold:
        result.update(label="not_L2", reason="non_entailment_above_exploratory_threshold")
    else:
        result["reason"] = "insufficient_model_evidence"
    return result


def evaluate_controls(verifier, threshold=THRESHOLD):
    rows = []
    for control_id, original, created, expected in CONTROLS:
        prediction = classify_pair(original, created, verifier, threshold)
        rows.append(
            {
                "control_id": control_id,
                "original_prompt": original,
                "created_prompt": created,
                "expected_label": expected,
                "predicted_label": prediction["label"],
                "matches_expected": prediction["label"] == expected,
                "prediction": prediction,
            }
        )
    return {
        "total": len(rows),
        "matched_expected": sum(row["matches_expected"] for row in rows),
        "all_passed": all(row["matches_expected"] for row in rows),
        "records": rows,
    }


def verify_run(run_dir, *, output_dir=None, verifier=None, threshold=THRESHOLD):
    if not 0.5 < threshold <= 1:
        raise ValueError("threshold must be greater than 0.5 and at most 1.")
    run_dir = Path(run_dir)
    manifest, parents, attempts = load_run(run_dir)
    source_rows = [
        {
            "parent_prompt_id": parent["parent_prompt_id"],
            "original_prompt": parent["original_prompt"],
            "created_prompt": attempts[parent["parent_prompt_id"]].get("paraphrased_prompt", ""),
        }
        for parent in parents
        if attempts.get(parent["parent_prompt_id"], {}).get("generation_status") == "produced"
    ]
    if not source_rows:
        raise ValueError("No generated candidates to classify. Run generation first.")
    output_dir = Path(output_dir or run_dir / OUTPUT_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    verifier = verifier or DebertaNLI()
    with _run_lock(output_dir):
        metadata = {
            "classifier": verifier.identity,
            "source_pairs_sha256": _digest(source_rows),
            "parents_sha256": manifest["parents_sha256"],
            "threshold": threshold,
            "max_tokens": MAX_TOKENS,
            "controls_sha256": _digest(CONTROLS),
            "verification_driver_sha256": _digest(Path(__file__).read_bytes()),
            "classification_origin": "automated_nli_screen_not_human_validation",
            "controls_note": "12 author-defined benign checks plus an observed pilot regression case added after "
            "an initial false positive. This is diagnostic testing, not held-out benchmark accuracy.",
            "limitation": "Equivalence screening of known paraphrase attempts is not a trained L0-L6 classifier. "
            "It does not revalidate risk categories or prove preservation of every visual detail.",
        }
        metadata_path = output_dir / "classifier.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Classifier setup or generated pairs changed. Use a new --output-dir.")
        else:
            _write_json(metadata_path, metadata)
        controls_path = output_dir / "controls.json"
        if controls_path.exists():
            controls = json.loads(controls_path.read_text(encoding="utf-8"))
        else:
            controls = evaluate_controls(verifier, threshold)
            _write_json(controls_path, controls)
        control_rows = [
            {
                key: row[key]
                for key in (
                    "control_id",
                    "original_prompt",
                    "created_prompt",
                    "expected_label",
                    "predicted_label",
                    "matches_expected",
                )
            }
            for row in controls["records"]
        ]
        _atomic_text(output_dir / "controls.csv", _csv_text(control_rows, list(control_rows[0])))
        rows = []
        for index, source in enumerate(source_rows, 1):
            result_path = output_dir / "records" / f"{_digest(source['parent_prompt_id'])}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved["source"] != source:
                    raise ValueError("Saved classification no longer matches its source pair.")
                prediction = saved["prediction"]
            else:
                prediction = classify_pair(source["original_prompt"], source["created_prompt"], verifier, threshold)
                _write_json(result_path, {"source": source, "prediction": prediction, "classified_at": _now()})
            row = {
                **source,
                "is_l2": {"L2": "Yes", "not_L2": "No", "uncertain": "Uncertain"}[prediction["label"]],
                "classification_reason": prediction["reason"],
                "human_validated": False,
                "classifier_controls_passed": controls["all_passed"],
                "classification_origin": metadata["classification_origin"],
                "lexical_similarity": round(
                    SequenceMatcher(
                        None,
                        normalize_prompt(source["original_prompt"]),
                        normalize_prompt(source["created_prompt"]),
                        autojunk=False,
                    ).ratio(),
                    4,
                ),
            }
            for direction in ("forward", "reverse"):
                for label in LABELS:
                    row[f"{direction}_{label}"] = prediction[direction].get("scores", {}).get(label, "")
            rows.append(row)
            print(f"[{index}/{len(source_rows)}] {source['parent_prompt_id']}: {row['is_l2']}", flush=True)
        _atomic_text(output_dir / "classification.csv", _csv_text(rows, CSV_COLUMNS))
        control_passed = controls["all_passed"]
        predictions = dict(Counter(row["is_l2"] for row in rows))
        summary = {
            "generator_run_kind": manifest["run_kind"],
            "generated_pairs_classified": len(rows),
            "classification_counts": predictions,
            "controls_matched_expected": controls["matched_expected"],
            "controls_total": controls["total"],
            "classifier_controls_passed": control_passed,
            "automated_acceptance_ready": control_passed,
            "failed_controls": [row["control_id"] for row in controls["records"] if not row["matches_expected"]],
            "end_to_end_smoke_passed": (
                control_passed and len(rows) == len(parents) and all(row["is_l2"] == "Yes" for row in rows)
            )
            if manifest["run_kind"] == "generation_classification_smoke"
            else None,
            "generation_outcomes": dict(Counter(record["generation_status"] for record in attempts.values())),
            "human_validated": 0,
            "threshold_calibrated_for_project": False,
            "limitation": metadata["limitation"],
        }
        _write_json(output_dir / "summary.json", summary)
        print(f"Classifier controls: {controls['matched_expected']}/{controls['total']} matched expected labels.")
        if not control_passed:
            print("CLASSIFIER CHECK FAILED: use these outputs as diagnostic predictions, not accepted L2 labels.")
        print(f"Results: {output_dir / 'classification.csv'}")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("self-test", help="Generate three benign paraphrases, then actually classify them.")
    smoke.add_argument("--run-dir", type=Path, default=ROOT / "results/runs/local/l2-generation-classification-test")
    classify = commands.add_parser("classify", help="Run the independent classifier on an existing generation run.")
    classify.add_argument("--run-dir", type=Path, default=ROOT / "results/runs/local/l2-pilot")
    classify.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "self-test":
            if not (args.run_dir / "manifest.json").exists():
                freeze_run(args.run_dir, _smoke_parents(), {"run_kind": "generation_classification_smoke"})
            elif load_run(args.run_dir)[0]["run_kind"] != "generation_classification_smoke":
                raise ValueError("Self-test requires a separate benign run folder.")
            settings = load_settings()
            generate(args.run_dir, settings)
            # Release generator memory before loading the independent CPU classifier.
            OllamaClient(settings).request(
                "/api/chat", {"model": settings["model"], "messages": [], "stream": False, "keep_alive": 0}
            )
            summary = verify_run(args.run_dir)
            print("End-to-end test:", "PASS" if summary["end_to_end_smoke_passed"] else "NEEDS REVIEW")
            return 0 if summary["end_to_end_smoke_passed"] else 3
        summary = verify_run(args.run_dir, output_dir=args.output_dir)
        return 0 if summary["classifier_controls_passed"] else 3
    except (ValueError, KeyError, OSError, ImportError, OllamaError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped; saved generation and classification records remain available for resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
