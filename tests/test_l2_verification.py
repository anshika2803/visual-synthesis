import csv

import pytest

from visual_synthesis.l2_paraphrase import _smoke_parents, freeze_run, generate, load_settings
from visual_synthesis.l2_verification import CONTROLS, OUTPUT_SUBDIR, classify_pair, evaluate_controls, verify_run


def scores(label, probability=0.98):
    return {
        "tokens": 20,
        "scores": {
            name: probability if name == label else (1 - probability) / 2
            for name in ("contradiction", "entailment", "neutral")
        },
    }


class FakeVerifier:
    identity = {"model": "test-classifier", "revision": "fixed", "device": "cpu"}

    def __init__(self, *, always_entail=False):
        self.calls = []
        self.controls = {}
        for _, original, created, expected in CONTROLS:
            label = "entailment" if expected == "L2" or always_entail else "neutral"
            self.controls[(original, created)] = self.controls[(created, original)] = scores(label)

    def predict(self, original, created):
        self.calls.append((original, created))
        return self.controls.get((original, created), scores("entailment"))


class Generator:
    def __init__(self):
        self.index = 0

    def identity(self):
        return {"model": "test-generator", "model_digest": "fixed", "ollama_version": "test"}

    def request(self, endpoint, payload):
        import json

        self.index += 1
        content = (
            {"status": "refused", "paraphrase": "", "reason": "Declined"}
            if self.index == 3
            else {"status": "produced", "paraphrase": f"A different description number {self.index}.", "reason": ""}
        )
        return {"done": True, "done_reason": "stop", "message": {"content": json.dumps(content)}}


def test_exact_copy_is_not_a_paraphrase_without_model_calls():
    verifier = FakeVerifier()
    result = classify_pair("A red bike.", "  a red bike.  ", verifier)
    assert result["label"] == "not_L2"
    assert result["reason"] == "unchanged_copy"
    assert not verifier.calls


@pytest.mark.parametrize(
    ("forward", "reverse", "expected"),
    [
        (scores("entailment"), scores("entailment"), "L2"),
        (scores("entailment"), scores("neutral"), "not_L2"),
        (scores("contradiction"), scores("entailment"), "not_L2"),
        (scores("entailment", 0.6), scores("entailment", 0.6), "uncertain"),
        ({"error": "input_too_long"}, scores("entailment"), "uncertain"),
    ],
)
def test_two_directions_are_required_and_uncertainty_is_retained(forward, reverse, expected):
    class Verifier:
        responses = iter([forward, reverse])

        def predict(self, *_):
            return next(self.responses)

    result = classify_pair("Original scene.", "Rewritten scene.", Verifier())
    assert result["label"] == expected
    assert result["human_validated"] is False


def test_controls_detect_a_classifier_that_accepts_every_rewrite():
    correct = evaluate_controls(FakeVerifier())
    assert correct["all_passed"]
    assert correct["matched_expected"] == len(CONTROLS)
    wrong = evaluate_controls(FakeVerifier(always_entail=True))
    assert not wrong["all_passed"]
    assert any(row["expected_label"] == "not_L2" and not row["matches_expected"] for row in wrong["records"])


def test_generation_to_classification_export_resume_and_provenance(tmp_path):
    run_dir = tmp_path / "run"
    freeze_run(run_dir, _smoke_parents(), {"run_kind": "generation_classification_smoke"})
    generate(run_dir, load_settings(), client=Generator())
    verifier = FakeVerifier()
    summary = verify_run(run_dir, verifier=verifier)
    assert summary["generated_pairs_classified"] == 2
    assert summary["generation_outcomes"] == {"produced": 2, "refused": 1}
    assert summary["end_to_end_smoke_passed"] is False
    assert summary["human_validated"] == 0
    assert summary["classifier_controls_passed"] is True
    output = run_dir / OUTPUT_SUBDIR / "classification.csv"
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0])[:3] == ["original_prompt", "created_prompt", "is_l2"]
    assert {row["parent_prompt_id"] for row in rows} == {"SMOKE_1", "SMOKE_2"}
    assert all(row["human_validated"] == "False" for row in rows)
    assert rows[0]["original_prompt"] == _smoke_parents()[0]["original_prompt"]
    assert rows[0]["forward_entailment"] and rows[0]["reverse_entailment"]
    resume = FakeVerifier()
    assert verify_run(run_dir, verifier=resume) == summary
    assert resume.calls == []
    with pytest.raises(ValueError, match="changed"):
        verify_run(run_dir, verifier=FakeVerifier(), threshold=0.9)


def test_failed_classifier_checks_are_exposed_in_every_exported_prediction(tmp_path):
    run_dir = tmp_path / "run"
    freeze_run(run_dir, _smoke_parents(), {"run_kind": "generation_classification_smoke"})
    generate(run_dir, load_settings(), client=Generator())
    summary = verify_run(run_dir, verifier=FakeVerifier(always_entail=True))
    assert summary["classifier_controls_passed"] is False
    with (run_dir / OUTPUT_SUBDIR / "classification.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["classifier_controls_passed"] == "False" for row in rows)


def test_missing_generation_does_not_become_a_successful_classification(tmp_path):
    run_dir = tmp_path / "run"
    freeze_run(run_dir, _smoke_parents(), {"run_kind": "generation_classification_smoke"})
    with pytest.raises(ValueError, match="No generated candidates"):
        verify_run(run_dir, verifier=FakeVerifier())
