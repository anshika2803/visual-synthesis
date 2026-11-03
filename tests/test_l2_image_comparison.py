import pytest

from visual_synthesis.l2_image_comparison import image_requests, load_sources, summarize


def test_all_variants_use_matching_seeds_and_the_same_original_reference():
    pairs = [
        {
            "parent_prompt_id": "p1",
            "original": "Two sleeping cats.",
            "v1": "Two resting cats.",
            "v2": "A pair of cats sleep.",
        }
    ]
    requests = image_requests(pairs, [42, 43])
    assert len(requests) == 6
    assert {row["reference_prompt"] for row in requests} == {"Two sleeping cats."}
    for seed in (42, 43):
        assert {row["variant"] for row in requests if row["seed"] == seed} == {"original", "v1", "v2"}
    assert next(row for row in requests if row["variant"] == "v2")["prompt"] == "A pair of cats sleep."


def test_refused_rewrite_is_not_rendered_as_an_empty_image_request():
    requests = image_requests([{"parent_prompt_id": "p1", "original": "An apple.", "v1": "One apple.", "v2": ""}], [42])
    assert [row["variant"] for row in requests] == ["original", "v1"]


def make_rows(scores):
    return [
        {
            "parent_prompt_id": "p1",
            "seed": seed,
            "variant": variant,
            "clip_alignment_to_original": score,
            "nudity_flagged": False,
        }
        for seed, values in scores.items()
        for variant, score in zip(("original", "v1", "v2"), values, strict=True)
    ]


def test_comparison_preserves_each_seed_and_does_not_conflate_safety_with_alignment():
    rows = make_rows({42: [0.3, 0.2, 0.25], 43: [0.3, 0.4, 0.1]})
    summary = summarize(rows)
    assert summary["complete_matched_comparisons"] == 2
    assert summary["v2_higher_alignment_than_v1"] == 1
    assert summary["v2_lower_alignment_than_v1"] == 1
    assert summary["mean_v2_minus_v1"] == pytest.approx(-0.125)
    assert summary["nudity_flag_counts"] == {"original": 0, "v1": 0, "v2": 0}
    assert summary["general_visual_quality_measured"] is False
    assert summary["human_validated"] is False


def test_missing_alignment_is_unavailable_rather_than_zero():
    summary = summarize(make_rows({42: [None, None, None]}))
    assert summary["mean_v2_minus_v1"] is None
    assert summary["mean_alignment_by_variant"]["v2"] is None


def test_initial_image_test_cannot_silently_select_risky_benchmark_prompts(monkeypatch):
    monkeypatch.setattr(
        "visual_synthesis.l2_image_comparison.load_run", lambda _: ({"run_kind": "L2_candidate_pilot"}, [], {})
    )
    with pytest.raises(ValueError, match="benign"):
        load_sources("unused", 3)
