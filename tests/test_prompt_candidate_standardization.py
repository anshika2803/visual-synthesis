import pandas as pd

from visual_synthesis.prompt_candidate_standardization import (
    CANDIDATE_COLUMNS, build_unified_candidates, find_near_duplicates,
    parse_source_metadata,
)
from visual_synthesis.schema import STANDARD_COLUMNS


def test_manifest_maps_sources_without_validating_pending_candidates(tmp_path):
    l1 = tmp_path / "l1.csv"
    ring = tmp_path / "ring.csv"
    coco = tmp_path / "coco.csv"
    nib = tmp_path / "nib.csv"
    review = tmp_path / "review.csv"
    validated = tmp_path / "validated.csv"
    pd.DataFrame([{
        "prompt_id": "I2P_1", "source_dataset": "I2P", "source_id": "1",
        "prompt": "A risky source prompt", "original_category": "violence",
        "standard_category": "violence", "risk_level": "high", "benchmark_level": "L1",
        "transformation": "original", "split": "unassigned", "is_duplicate": False,
        "parent_prompt_id": "I2P_1", "source_metadata": "src=x",
    }]).to_csv(l1, index=False)
    pd.DataFrame([{
        "prompt_id": "rab_1", "prompt": "An adversarial source prompt", "source_dataset": "ring-a-bell",
        "attack_method": "Ring-A-Bell", "attack_config": "K_16",
        "case_number": 4, "evaluation_seed": 23, "source_file": "case.csv",
        "original_category": "nudity", "standard_category": "sexual", "risk_level": "high",
        "benchmark_level": "L6", "transformation": "adversarial",
    }]).to_csv(ring, index=False)
    pd.DataFrame([{
        "candidate_id": "COCO_L0_a", "prompt": "A risky source prompt", "candidate_level": "L0",
        "caption_id": "99", "image_id": "199", "source_split": "val", "source_dataset": "COCO_Captions_2017",
        "source_id": "99", "sample_seed": "42", "sample_order": "1", "triage_confidence": "0.4",
        "review_flags": "screened_only", "validation_status": "pending_human_review",
    }]).to_csv(coco, index=False)
    pd.DataFrame([{
        "candidate_id": "NIB_hash", "prompt": "submitted actual prompt", "submitted_prompt": "submitted actual prompt",
        "candidate_level": "unclassified", "source_dataset": "Adversarial_Nibbler", "source_id": "nib-src",
        "participant_rewrite_evidence": '["description of harm"]', "rewritten_prompt": "description of harm",
        "text_attack_mode": "text_attack_coded_language", "image_failure_type": "image_failure_violent",
        "validator_text_safe_votes": "2", "validator_text_unsafe_votes": "1",
        "validator_image_unsafe_votes": "3", "suggested_standard_category": "",
        "source_provenance": '[]', "triage_confidence": "0.3", "review_flags": "unclassified",
    }]).to_csv(nib, index=False)
    pd.DataFrame([
        {"candidate_id": "STD_I2P_1", "prompt": "A risky source prompt", "proposed_level": "L1",
         "source_dataset": "I2P", "decision": "", "validated_level": "", "standard_category": "",
         "reviewer_confidence": "", "reviewer_id": "", "reviewer_notes": "", "candidate_available": "True"},
        {"candidate_id": "STD_rab_1", "prompt": "An adversarial source prompt", "proposed_level": "L6",
         "source_dataset": "ring-a-bell", "decision": "", "validated_level": "", "standard_category": "",
         "reviewer_confidence": "", "reviewer_id": "", "reviewer_notes": "", "candidate_available": "True"},
        {"candidate_id": "COCO_L0_a", "prompt": "A risky source prompt", "proposed_level": "L0",
         "source_dataset": "COCO_Captions_2017", "decision": "", "validated_level": "", "standard_category": "",
         "reviewer_confidence": "", "reviewer_id": "", "reviewer_notes": "", "candidate_available": "True"},
        {"candidate_id": "NIB_hash", "prompt": "submitted actual prompt", "proposed_level": "unclassified",
         "source_dataset": "Adversarial_Nibbler", "decision": "", "validated_level": "", "standard_category": "",
         "reviewer_confidence": "", "reviewer_id": "", "reviewer_notes": "", "candidate_available": "True"},
    ]).to_csv(review, index=False)
    pd.DataFrame(columns=["candidate_id", "validated", "validated_level", "standard_category"]).to_csv(validated, index=False)

    output, reports = tmp_path / "unified.parquet", tmp_path / "reports"
    audit = build_unified_candidates(l1_path=l1, coco_path=coco, nibbler_path=nib, ring_path=ring,
                                     review_path=review, validated_path=validated, output_path=output,
                                     reports_dir=reports, near_threshold=0.9)
    result = pd.read_parquet(output).fillna("").astype(str)
    assert CANDIDATE_COLUMNS[:len(STANDARD_COLUMNS)] == STANDARD_COLUMNS
    assert set(STANDARD_COLUMNS).issubset(result.columns)
    assert len(result) == 4
    assert result.prompt_id.nunique() == 4
    assert result.loc[result.prompt_id.eq("I2P_1"), "proposed_standard_category"].iloc[0] == "violence"
    coco_row = result[result.prompt_id.eq("COCO_L0_a")].iloc[0]
    assert coco_row.is_duplicate == "True"
    assert coco_row.duplicate_of_prompt_id == "I2P_1"
    assert coco_row.split == "val"
    assert coco_row.proposed_standard_category == ""
    coco_metadata = parse_source_metadata(coco_row.source_metadata)
    assert coco_metadata["caption_id"] == "99"
    assert coco_metadata["image_id"] == "199"
    nib_row = result[result.prompt_id.eq("NIB_hash")].iloc[0]
    nib_metadata = parse_source_metadata(nib_row.source_metadata)
    assert nib_row.prompt == "submitted actual prompt"
    assert nib_row.benchmark_level == ""
    assert nib_row.proposed_benchmark_level == "unclassified"
    assert nib_row.human_validated == "False"
    assert nib_metadata["participant_rewrite_evidence"] == '["description of harm"]'
    ring_row = result[result.prompt_id.eq("rab_1")].iloc[0]
    assert ring_row.benchmark_level == "L6"
    assert ring_row.proposed_standard_category == "sexual"
    assert ring_row.source_id == "rab_1"
    ring_metadata = parse_source_metadata(ring_row.source_metadata)
    assert ring_metadata["ring_a_bell_attack_config"] == "K_16"
    assert audit["human_validated_rows"] == 0
    assert audit["exact_duplicate_rows"] == 1
    assert audit["review_merge_audit"]["unmatched_review_rows_retained"] == 0
    assert audit["canonical_schema_validation"]["ok"] is True
    assert set(audit["input_checksums"]) >= {"standardized_l1", "coco_candidates", "corrected_nibbler_candidates",
                                              "ring_a_bell_standardized", "phase_d2_review_master"}
    assert len(audit["output_integrity"]["sha256"]) == 64
    metadata_issues = pd.read_parquet(reports / "metadata_issues.parquet")
    assert ((metadata_issues.prompt_id == "NIB_hash") & (metadata_issues.field == "risk_level")).any()
    assert (reports / "distribution_by_review_status.parquet").exists()
    assert (reports / "metadata_issues.parquet").exists()

    reviewed = pd.read_csv(review, keep_default_na=False, dtype=str)
    reviewed.loc[reviewed.candidate_id.eq("NIB_hash"), ["decision", "validated_level", "standard_category",
                                                          "reviewer_confidence", "reviewer_id"]] = [
        "accept", "L1", "violence", "0.95", "reviewer-test"]
    reviewed.to_csv(review, index=False)
    pd.DataFrame([{"candidate_id": "NIB_hash", "validated": True, "validated_level": "L1",
                   "standard_category": "violence"}]).to_csv(validated, index=False)
    approved = build_unified_candidates(l1_path=l1, coco_path=coco, nibbler_path=nib, ring_path=ring,
                                        review_path=review, validated_path=validated,
                                        output_path=tmp_path / "approved.parquet", reports_dir=tmp_path / "approved_reports")
    approved_rows = pd.read_parquet(tmp_path / "approved.parquet").fillna("").astype(str)
    approved_nib = approved_rows[approved_rows.prompt_id.eq("NIB_hash")].iloc[0]
    assert approved["human_validated_rows"] == 1
    assert approved_nib.human_validated == "True"
    assert approved_nib.validated_level == "L1"
    assert approved_nib.validated_standard_category == "violence"
    assert approved_nib.proposed_benchmark_level == "unclassified"


def test_near_duplicate_prefix_index_returns_exact_jaccard_pairs():
    frame = pd.DataFrame([
        {"prompt_id": "a", "prompt": "one two three four five six seven eight nine ten"},
        {"prompt_id": "b", "prompt": "one two three four five six seven eight nine eleven"},
        {"prompt_id": "c", "prompt": "completely different words live here today"},
    ])
    pairs, audit = find_near_duplicates(frame, threshold=0.8)
    assert len(pairs) == 1
    assert set(pairs.iloc[0][["prompt_id_a", "prompt_id_b"]]) == {"a", "b"}
    assert pairs.iloc[0].token_jaccard == 0.818182
    assert audit["verified_near_duplicate_prompt_pairs"] == 1
