import pandas as pd

from visual_synthesis.phase_c_candidates import LEVELS, RISK_CATEGORIES
from visual_synthesis.phase_d1_review import prepare_review_batch, validate_review_batch


def _write_candidates(tmp_path):
    coco = pd.DataFrame([
        {"candidate_id": "c0", "prompt": "a person in a park", "candidate_level": "L0",
         "source_dataset": "COCO_Captions_2017", "source_id": "10", "caption_id": "10",
         "image_id": "100", "source_split": "train", "validated": False},
        {"candidate_id": "dup", "prompt": "A person in a park", "candidate_level": "L0",
         "source_dataset": "COCO_Captions_2017", "source_id": "11", "caption_id": "11",
         "image_id": "101", "source_split": "train", "validated": False},
    ])
    nibbler = pd.DataFrame([{
        "candidate_id": "n1", "prompt": "unsafe rewrite", "rewritten_prompt": "unsafe rewrite",
        "candidate_level": "L1", "source_dataset": "Adversarial_Nibbler", "source_id": "77",
        "source_prompt": "starting wording", "text_attack_mode": "text_attack_sensitive_terms",
        "validator_text_safe_votes": 0, "validator_text_unsafe_votes": 4,
        "validator_text_other_votes": 0, "image_failure_type": "image_failure_sexual",
        "validator_image_safe_votes": 1, "validator_image_unsafe_votes": 3,
        "validator_image_other_votes": 0,
        "source_split": "dev", "validated": False,
    }])
    coco_path, nib_path = tmp_path / "coco.csv", tmp_path / "nib.csv"
    coco.to_csv(coco_path, index=False)
    nibbler.to_csv(nib_path, index=False)
    return coco_path, nib_path


def test_review_batch_shows_evidence_and_preserves_decisions(tmp_path):
    coco, nib = _write_candidates(tmp_path)
    batch = tmp_path / "review.csv"
    audit = prepare_review_batch(coco_path=coco, nibbler_path=nib, output_path=batch)
    rows = pd.read_csv(batch, keep_default_na=False)
    nrow = rows[rows.candidate_id == "n1"].iloc[0]
    assert nrow.starting_prompt == "starting wording"
    assert nrow.participant_rewrite_evidence == "unsafe rewrite"
    assert nrow.attack_mode == "text_attack_sensitive_terms"
    assert nrow.validator_text_unsafe_votes == "4"
    assert nrow.validator_image_unsafe_votes == "3"
    assert audit["exact_duplicate_groups"] == 1
    assert not rows.loc[rows.candidate_id == "n1", "decision"].iloc[0]
    rows.loc[rows.candidate_id == "n1", ["decision", "validated_level", "standard_category",
                                         "reviewer_confidence", "reviewer_id", "reviewer_notes"]] = [
        "accept", "L1", "sexual", "0.91", "reviewer-a", "checked against source evidence",
    ]
    rows.loc[rows.candidate_id == "c0", "decision"] = "uncertain"
    rows.loc[rows.candidate_id == "dup", "decision"] = "reject"
    rows.to_csv(batch, index=False)
    refreshed = prepare_review_batch(coco_path=coco, nibbler_path=nib, output_path=batch, previous_path=batch)
    after = pd.read_csv(batch, keep_default_na=False)
    assert refreshed["preserved_decisions"] == 3
    assert after.set_index("candidate_id").loc["n1", "reviewer_notes"] == "checked against source evidence"
    assert after.set_index("candidate_id").loc["c0", "decision"] == "uncertain"


def test_only_complete_accepts_export_and_counts_are_grouped(tmp_path):
    coco, nib = _write_candidates(tmp_path)
    batch = tmp_path / "review.csv"
    prepare_review_batch(coco_path=coco, nibbler_path=nib, output_path=batch)
    rows = pd.read_csv(batch, keep_default_na=False)
    rows.loc[rows.candidate_id == "n1", ["decision", "validated_level", "standard_category",
                                         "reviewer_confidence", "reviewer_id"]] = [
        "accept", "L1", "violence;sexual", "0.8", "r1",
    ]
    rows.loc[rows.candidate_id == "c0", ["decision", "validated_level", "standard_category",
                                         "reviewer_confidence", "reviewer_id"]] = [
        "accept", "L0", "", "0.9", "r2",
    ]
    rows.loc[rows.candidate_id == "dup", "decision"] = "reject"
    rows.to_csv(batch, index=False)
    summary = validate_review_batch(batch, output_dir=tmp_path / "accepted")
    accepted = pd.read_parquet(tmp_path / "accepted" / "validated_candidates.parquet")
    assert set(accepted.candidate_id) == {"c0", "n1"}
    assert summary["accepted_by_level"] == {level: int(level in {"L0", "L1"}) for level in LEVELS}
    assert summary["accepted_by_source"] == {"Adversarial_Nibbler": 1, "COCO_Captions_2017": 1}
    assert summary["accepted_by_category"] == {category: int(category in {"sexual", "violence"})
                                                for category in sorted(RISK_CATEGORIES)}
    assert summary["validation_issue_count"] == 0
    assert not summary["final_6000_benchmark_created"]


def test_exact_duplicate_acceptances_are_blocked(tmp_path):
    coco, nib = _write_candidates(tmp_path)
    batch = tmp_path / "review.csv"
    prepare_review_batch(coco_path=coco, nibbler_path=nib, output_path=batch)
    rows = pd.read_csv(batch, keep_default_na=False)
    for candidate_id in ("c0", "dup"):
        rows.loc[rows.candidate_id == candidate_id,
                 ["decision", "validated_level", "standard_category", "reviewer_confidence", "reviewer_id"]] = [
            "accept", "L0", "", "0.9", "r",
        ]
    rows.to_csv(batch, index=False)
    summary = validate_review_batch(batch, output_dir=tmp_path / "accepted")
    accepted = pd.read_parquet(tmp_path / "accepted" / "validated_candidates.parquet")
    assert accepted.empty
    assert summary["validation_issue_count"] == 2


def test_transformed_accept_requires_parent_l1_and_category_match(tmp_path):
    coco = pd.DataFrame([{
        "candidate_id": "child", "prompt": "paraphrased violent scene", "candidate_level": "L2",
        "source_dataset": "curated", "source_id": "variant-1", "validated": False,
    }])
    nib = pd.DataFrame(columns=["candidate_id", "prompt"])
    coco_path, nib_path = tmp_path / "coco.csv", tmp_path / "nib.csv"
    coco.to_csv(coco_path, index=False)
    nib.to_csv(nib_path, index=False)
    batch = tmp_path / "review.csv"
    prepare_review_batch(coco_path=coco_path, nibbler_path=nib_path, output_path=batch)
    rows = pd.read_csv(batch, keep_default_na=False)
    rows.loc[0, ["decision", "validated_level", "standard_category", "reviewer_confidence",
                 "reviewer_id", "parent_prompt_id", "transformation_method", "transformation_parameters"]] = [
        "accept", "L2", "violence", "0.9", "r", "root", "manual_paraphrase", "method=v1",
    ]
    rows.to_csv(batch, index=False)
    l1 = pd.DataFrame([{"prompt_id": "root", "benchmark_level": "L1", "standard_category": "violence"}])
    l1_path = tmp_path / "l1.csv"
    l1.to_csv(l1_path, index=False)
    result = validate_review_batch(batch, output_dir=tmp_path / "valid", l1_path=l1_path)
    assert result["accepted_export_rows"] == 1
    rows.loc[0, "standard_category"] = "sexual"
    rows.to_csv(batch, index=False)
    result = validate_review_batch(batch, output_dir=tmp_path / "bad", l1_path=l1_path)
    assert result["accepted_export_rows"] == 0
    assert result["validation_issue_count"] == 1
