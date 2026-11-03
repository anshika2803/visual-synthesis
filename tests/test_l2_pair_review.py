import csv
import json

import pytest

from visual_synthesis.l2_pair_review import (
    CONTROLS,
    OUTPUT_SUBDIR,
    combine,
    parse_judgment,
    review_run,
    spatial_differences,
)
from visual_synthesis.l2_paraphrase import _smoke_parents, freeze_run, generate, load_settings


def judgment(assessment="equivalent", differences=None):
    return {
        "assessment": assessment,
        "reason": "Scene comparison.",
        "differences": differences or [],
        "rewording": "light",
    }


def response(data):
    return {"done": True, "done_reason": "stop", "message": {"content": json.dumps(data)}}


def changed_location():
    return {
        "aspect": "location",
        "original_quote": "behind",
        "created_quote": "inside",
        "reason": "The position changes from behind to inside.",
    }


class ToyNLI:
    identity = {"model": "fake-independent-nli", "revision": "fixed"}

    def __init__(self):
        self.lookup = {}
        for key, original, created, expected in CONTROLS:
            label = "entailment" if expected == "L2" or key == "negative_pilot_location_and_participants" else "neutral"
            self.lookup[(original, created)] = self.lookup[(created, original)] = label

    def predict(self, original, created):
        label = self.lookup.get((original, created), "neutral" if "resting" in original + created else "entailment")
        return {
            "tokens": 20,
            "scores": {name: 0.98 if name == label else 0.01 for name in ("entailment", "neutral", "contradiction")},
        }


class ToyJudge:
    identity = {"model": "fake-judge", "shares_generator_checkpoint": True}

    def compare(self, original, created):
        if "behind" in original and "inside" in created:
            return judgment("changed", [changed_location()])
        return judgment()


def test_spatial_review_can_reject_a_false_positive_from_nli():
    result = combine("A box behind a store.", "A box inside a store.", ToyNLI(), ToyJudge())
    assert result["nli"]["label"] == "L2"
    assert result["label"] == "not_L2"
    assert result["human_validated"] is False


def test_detail_judgment_can_reject_a_non_spatial_change():
    class AttributeJudge(ToyJudge):
        def compare(self, original, created):
            return judgment(
                "changed",
                [
                    {
                        "aspect": "colour",
                        "original_quote": "red",
                        "created_quote": "blue",
                        "reason": "The coat colour changes.",
                    }
                ],
            )

    result = combine("A woman wears a red coat.", "A woman wears a blue coat.", ToyNLI(), AttributeJudge())
    assert result["nli"]["label"] == "L2"
    assert result["label"] == "not_L2"
    assert result["detail"]["assessment"] == "changed"


@pytest.mark.parametrize(
    ("original", "created", "changed"),
    [
        ("A box behind a store.", "A box inside a store.", True),
        ("A dog sleeps beneath a wooden bench.", "A dog sleeps above a wooden bench.", True),
        ("A box is outside the warehouse.", "A box is inside the warehouse.", True),
        ("A dog sleeps beneath a wooden bench.", "A dog sleeps under a wooden bench.", False),
        ("A bike stands beside a wall.", "A bike stands next to a wall.", False),
        ("A box is behind a store.", "A box is inside a truck.", False),  # Different anchors: defer to models.
    ],
)
def test_spatial_check_handles_new_anchors_and_synonyms(original, created, changed):
    assert bool(spatial_differences(original, created)) == changed


@pytest.mark.parametrize(
    "data",
    [
        judgment("changed"),
        judgment("changed", [{**changed_location(), "original_quote": "invented evidence"}]),
        judgment("equivalent", [changed_location()]),
        {"assessment": [], "reason": "bad", "differences": [], "rewording": "light"},
        {"assessment": "equivalent"},
    ],
)
def test_invalid_or_invented_model_evidence_is_uncertain(data):
    result = parse_judgment(response(data), "A box behind a store.", "A box inside a store.")
    assert result["assessment"] == "uncertain"


def test_literal_evidence_is_preserved_and_truncation_is_not_a_rejection():
    data = judgment("changed", [changed_location()])
    assert parse_judgment(response(data), "A box behind a store.", "A box inside a store.") == data
    assert (
        parse_judgment({**response(data), "done_reason": "length"}, "original", "created")["assessment"] == "uncertain"
    )


class Generator:
    def identity(self):
        return {"model": "fake-generator", "model_digest": "fixed", "ollama_version": "test"}

    def request(self, endpoint, payload):
        original = json.loads(payload["messages"][1]["content"])["description"]
        text = original.replace("sleeping", "resting") if "cats" in original else "A rewritten view: " + original
        return response({"status": "produced", "paraphrase": text, "reason": ""})


def test_correct_rejection_does_not_fail_pipeline_and_resume_preserves_labels(tmp_path):
    run_dir = tmp_path / "run"
    freeze_run(run_dir, _smoke_parents(), {"run_kind": "generation_classification_smoke"})
    generate(run_dir, load_settings(), client=Generator())
    summary = review_run(run_dir, verifier=ToyNLI(), judge=ToyJudge())
    assert summary["pipeline_test_passed"] is True
    assert summary["all_generated_pairs_accepted"] is False
    assert summary["classification_counts"] == {"Yes": 2, "No": 1}
    assert summary["human_validated"] == 0
    assert summary["automatic_benchmark_validation_established"] is False
    output = run_dir / OUTPUT_SUBDIR / "classification.csv"
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0])[:3] == ["original_prompt", "created_prompt", "is_l2"]
    assert all(row["human_validated"] == "False" for row in rows)

    class NoInferenceNLI(ToyNLI):
        def predict(self, *_):
            pytest.fail("Resume should reuse saved evidence.")

    class NoInferenceJudge(ToyJudge):
        def compare(self, *_):
            pytest.fail("Resume should reuse saved judgments.")

    assert review_run(run_dir, verifier=NoInferenceNLI(), judge=NoInferenceJudge()) == summary
