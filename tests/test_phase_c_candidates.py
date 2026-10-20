import json

import pandas as pd

from visual_synthesis.phase_c_candidates import (
    _classify_nibbler,
    prepare_coco_candidates,
    prepare_nibbler_candidates,
    review_template,
    validate_human_reviews,
)


def _coco(path, split, rows):
    path.write_text(json.dumps({"annotations": rows}), encoding="utf-8")


def test_coco_sample_is_reproducible_preserves_provenance_and_is_pending(tmp_path):
    train, val = tmp_path / "train.json", tmp_path / "val.json"
    _coco(train, "train", [
        {"id": i, "image_id": i, "caption": f"A person walking beside tree number {i}."}
        for i in range(1, 30)
    ])
    _coco(val, "val", [{"id": 101, "image_id": 101, "caption": "A person reads beside a window."}])
    a, audit = prepare_coco_candidates(train, val, seed=9, target=12)
    b, _ = prepare_coco_candidates(train, val, seed=9, target=12)
    assert a.candidate_id.tolist() == b.candidate_id.tolist()
    assert len(a) == 12 and audit["validated_rows"] == 0
    assert a.caption_id.nunique() == len(a) == a.image_id.nunique()
    assert a.validated.eq(False).all()
    assert a.human_review_required.all()


def test_level_triage_does_not_equate_every_nibbler_record_with_l4():
    coded = {"source_prompt": "masked phrase", "annotation": {"text_attack_mode": ["text_attack_coded_language"]}}
    assert _classify_nibbler(coded, 0, 4, 0)[0] == "L3"
    explicit = {"source_prompt": "A bloody corpse", "annotation": {"text_attack_mode": ["text_attack_none"]}}
    assert _classify_nibbler(explicit, 4, 1, 0)[0] == "L1"
    neutral = {"source_prompt": "A quiet landscape", "annotation": {"text_attack_mode": ["text_attack_none"]}}
    assert _classify_nibbler(neutral, 0, 4, 0)[0] == "unclassified"


def test_nibbler_uses_submitted_prompt_and_human_review_gates_validation(tmp_path):
    root = tmp_path / "nibbler"
    folder = root / "round1" / "submitted"
    folder.mkdir(parents=True)
    pd.DataFrame({"prompt": ["ordinary landscape"]}).to_csv(folder / "train.csv", index=False)
    pd.DataFrame({"prompt": ["ordinary landscape"]}).to_csv(folder / "dev.csv", index=False)
    payload = {
        "prompt": {"0": "ordinary landscape"},
        "submission_annotations": {"0": json.dumps({
            "rewritten_prompt": "a bloody landscape", "text_attack_mode": ["text_attack_none"],
            "image_failure_type": ["image_failure_violent"],
        })},
            "validation": {"0": [json.dumps({"text_safety_validation": ["text_unsafe"],
                                                "image_safety_validation": ["image_unsafe"]})]},
    }
    for split in ("train", "dev"):
        split_payload = payload.copy()
        split_payload["submission_annotations"] = dict(payload["submission_annotations"])
        if split == "dev":
            split_payload["submission_annotations"]["0"] = json.dumps({
                "rewritten_prompt": "a bloody scene", "text_attack_mode": ["text_attack_none"],
                "image_failure_type": ["image_failure_violent"],
            })
        (folder / f"{split}.json").write_text(json.dumps(split_payload), encoding="utf-8")
    # Round 4 is present but empty to exercise the production input contract.
    for split in ("train", "dev"):
        folder4 = root / "round4" / "submitted"
        folder4.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prompt": []}).to_csv(folder4 / f"{split}.csv", index=False)
        (folder4 / f"{split}.json").write_text(json.dumps({"prompt": {}, "submission_annotations": {}, "validation": {}}))
    candidates, audit = prepare_nibbler_candidates(root)
    assert candidates.iloc[0].prompt == "ordinary landscape"
    assert candidates.prompt.nunique() == len(candidates) == 1
    assert set(candidates.prompt) == {"ordinary landscape"}
    assert candidates.iloc[0].participant_rewrite_evidence in {'["a bloody landscape", "a bloody scene"]', '["a bloody scene", "a bloody landscape"]'}
    assert candidates.iloc[0].candidate_level == "unclassified"
    assert candidates.iloc[0].validator_image_unsafe_votes == 2
    assert audit["unmatched_csv_rows"] == 0
    assert audit["unique_submitted_prompt_candidates"] == 1
    reviews = review_template(candidates)
    reviews.loc[0, ["review_decision", "validated_level", "standard_category", "reviewer_confidence", "reviewer_id"]] = [
        "validated", "L1", "violence", "0.9", "reviewer-a",
    ]
    accepted, issues = validate_human_reviews(candidates, reviews)
    assert len(accepted) == 1 and accepted.iloc[0].validated
    assert issues == []
    reviews.loc[0, "reviewer_confidence"] = "1.1"
    accepted, issues = validate_human_reviews(candidates, reviews)
    assert accepted.empty and issues


def test_transformed_review_requires_parent_and_category_preservation():
    candidate = pd.DataFrame([{
        "candidate_id": "child", "prompt": "a variant", "candidate_level": "L2",
        "validated": False, "standard_category": "", "parent_prompt_id": "",
    }])
    reviews = review_template(candidate)
    reviews.loc[0, ["review_decision", "validated_level", "standard_category", "reviewer_confidence",
                    "reviewer_id", "parent_prompt_id", "transformation_method", "transformation_parameters"]] = [
        "validated", "L2", "violence", "0.9", "reviewer-a", "root", "human_paraphrase", "style=neutral",
    ]
    accepted, issues = validate_human_reviews(candidate, reviews, parent_categories={"root": "violence"})
    assert len(accepted) == 1 and not issues
    reviews.loc[0, "standard_category"] = "sexual"
    accepted, issues = validate_human_reviews(candidate, reviews, parent_categories={"root": "violence"})
    assert accepted.empty and "differs from parent" in issues[0]["issue"]
