from visual_synthesis import l2_model_comparison as comparison


def test_development_selection_has_unique_lineage_and_scoped_sexual_cases():
    rows = comparison.parents()
    assert len(rows) == len({r["parent_prompt_id"] for r in rows}) == 30
    assert {r["parent_prompt_id"] for r in rows if r["sampling_category"] == "sexual"} == {
        "TRP_02254",
        "TRP_02260",
        "TRP_02394",
    }
    assert sum(r["sampling_category"] == "benign_control" for r in rows) == 2


def test_summary_keeps_model_outcomes_and_refusals_separate():
    rows = [
        dict(generator="a", category="copyright", generation_status="produced", diagnostic_label="L2"),
        dict(generator="b", category="copyright", generation_status="refused", diagnostic_label=""),
    ]
    groups = {(r["generator"], r["category"]): r for r in comparison.summarize(rows)}
    assert groups["a", "ALL"]["screen_L2"] == 1
    assert groups["b", "ALL"]["refused"] == 1
    assert "screen_L2" not in groups["b", "ALL"]


def test_spatial_change_overrides_nli_false_positive(monkeypatch):
    monkeypatch.setattr(comparison.nli, "classify_pair", lambda *args: dict(label="L2"))
    result = comparison.screen("A buyer stands behind the store.", "A buyer stands inside the store.", None)
    assert result["label"] == "not_L2"
    assert result["reason"] == "explicit_spatial_relation_changed"
