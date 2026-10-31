import pandas as pd

from visual_synthesis.phase_d2_review import (
    extend_master, make_pilot_plan, merge_batch, progress_report, select_batch,
)


def _master(path):
    rows = []
    for i in range(120):
        rows.append({"candidate_id": f"C{i:03d}", "prompt": f"prompt {i}",
                     "proposed_level": "L1" if i < 60 else "L4",
                     "source_dataset": "Nibbler" if i % 2 else "COCO",
                     "suggested_standard_category": "violence" if i % 3 else "sexual",
                     "decision": "", "validated_level": "", "standard_category": "",
                     "reviewer_confidence": "", "reviewer_id": "", "reviewer_notes": "",
                     "candidate_available": "True", "participant_rewrite_evidence": "evidence"})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_select_batches_are_bounded_filtered_and_stable(tmp_path):
    master, out = tmp_path / "master.csv", tmp_path / "batch.csv"
    _master(master)
    result = select_batch(master, out, batch_size=50, level="L1", source="Nibbler", category="violence")
    batch = pd.read_csv(out, keep_default_na=False)
    assert 0 < len(batch) <= 50
    assert set(batch.proposed_level) == {"L1"}
    assert set(batch.source_dataset) == {"Nibbler"}
    assert "participant_rewrite_evidence" in batch
    assert result["batch_rows"] == len(batch)
    try:
        select_batch(master, out, batch_size=101)
    except ValueError:
        pass
    else:
        assert False, "batch size >100 must fail"


def test_merge_preserves_evidence_and_existing_values(tmp_path):
    master, batch = tmp_path / "master.csv", tmp_path / "batch.csv"
    _master(master)
    source = pd.read_csv(master, keep_default_na=False, dtype=str)
    source.loc[0, ["decision", "validated_level", "standard_category", "reviewer_confidence", "reviewer_id"]] = [
        "accept", "L1", "violence", "0.9", "reviewer-1"]
    source.loc[1, "decision"] = "reject"
    source.to_csv(batch, index=False)
    result = merge_batch(master, batch)
    merged = pd.read_csv(master, keep_default_na=False, dtype=str)
    assert result["master_rows"] == 120
    assert merged.loc[0, "decision"] == "accept"
    assert merged.loc[0, "participant_rewrite_evidence"] == "evidence"
    assert merged.loc[1, "decision"] == "reject"
    revised = pd.read_csv(master, keep_default_na=False, dtype=str)
    revised.loc[0, "decision"] = "reject"
    revised.to_csv(batch, index=False)
    merge_batch(master, batch)
    merged = pd.read_csv(master, keep_default_na=False, dtype=str)
    assert merged.loc[0, "decision"] == "reject"
    assert "decision changed from 'accept' to 'reject'" in merged.loc[0, "review_evidence"]


def test_dashboard_and_pilot_plan_keep_candidates_unvalidated(tmp_path):
    master, l1, plan = tmp_path / "master.csv", tmp_path / "l1.csv", tmp_path / "pilot.csv"
    _master(master)
    df = pd.read_csv(master, keep_default_na=False, dtype=str)
    df.loc[0, "decision"] = "accept"
    df.loc[1, "decision"] = "reject"
    df.loc[2, "decision"] = "uncertain"
    df.to_csv(master, index=False)
    report = progress_report(master)
    assert report["by_proposed_level"]["L1"]["accepted"] == 1
    assert report["by_proposed_level"]["L1"]["rejected"] == 1
    assert report["by_proposed_level"]["L1"]["uncertain"] == 1
    assert report["candidate_supply_gap_to_1000"]["L2"] == 1000

    pd.DataFrame([{"prompt_id": f"P{i}", "prompt": f"parent {i}",
                   "standard_category": ["violence", "sexual", "disturbing"][i % 3],
                   "benchmark_level": "L1"} for i in range(30)]).to_csv(l1, index=False)
    plan_result = make_pilot_plan(l1, plan, per_level=20, seed=4)
    pilot = pd.read_csv(plan, keep_default_na=False)
    assert plan_result["planned_rows"] == 80
    assert pilot.groupby("proposed_level").size().to_dict() == {"L2": 20, "L3": 20, "L4": 20, "L6": 20}
    assert pilot.candidate_prompt.eq("").all()
    assert pilot.validated.eq(False).all()
    assert pilot.parent_prompt_id.ne("").all()


def test_extend_master_adds_l1_l6_without_validating_and_preserves_reviews(tmp_path):
    master, source = tmp_path / "master.csv", tmp_path / "selected.csv"
    _master(master)
    existing = pd.read_csv(master, keep_default_na=False, dtype=str)
    existing.loc[0, "decision"] = "accept"
    existing.loc[0, "reviewer_id"] = "reviewer-a"
    existing["review_evidence"] = "prior reviewer evidence"
    existing["reviewed_at"] = "2026-10-18T10:00:00Z"
    existing.to_csv(master, index=False)
    pd.DataFrame([
        {"prompt_id": "P-L1", "prompt": "first source prompt", "benchmark_level": "L1",
         "source_dataset": "I2P", "standard_category": "violence", "source_id": "10"},
        {"prompt_id": "P-L6", "prompt": "second source prompt", "benchmark_level": "L6",
         "source_dataset": "ring-a-bell", "standard_category": "sexual", "source_id": "11"},
    ]).to_csv(source, index=False)
    audit = extend_master(master, source)
    combined = pd.read_csv(master, keep_default_na=False, dtype=str)
    assert audit["added_rows"] == 2
    assert combined.loc[combined.candidate_id.eq("STD_P-L1"), "decision"].iloc[0] == ""
    assert combined.loc[combined.candidate_id.eq("STD_P-L6"), "proposed_level"].iloc[0] == "L6"
    assert combined.loc[combined.candidate_id.eq("C000"), "decision"].iloc[0] == "accept"
    assert combined.loc[combined.candidate_id.eq("C000"), "reviewer_id"].iloc[0] == "reviewer-a"
    assert combined.loc[combined.candidate_id.eq("C000"), "review_evidence"].iloc[0] == "prior reviewer evidence"
    assert combined.loc[combined.candidate_id.eq("C000"), "reviewed_at"].iloc[0] == "2026-10-18T10:00:00Z"
