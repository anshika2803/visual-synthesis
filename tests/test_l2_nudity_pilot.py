from visual_synthesis.l2_nudity_pilot import expand_pairs
from visual_synthesis.stages.s05_generate import make_request


def test_refused_paraphrase_is_missing_and_original_remains_in_input():
    parents = [{"prompt_id": "P1", "prompt": "An original description.", "source_dataset": "source"}]
    outcomes = {"P1": {"generation_status": "refused", "paraphrased_prompt": ""}}
    rows = expand_pairs(parents, outcomes, [42, 43], "source.csv")
    assert len(rows) == 2
    assert all(row["benchmark_level"] == "L1" and row["pair_variant"] == "original" for row in rows)
    assert all(row["human_validated"] is False for row in rows)


def test_standard_project_stage_receives_explicit_matching_seeds_for_each_pair():
    parents = [{"prompt_id": "P1", "prompt": "A bicycle beside a wall.", "source_dataset": "source"}]
    outcomes = {"P1": {"generation_status": "produced", "paraphrased_prompt": "Beside a wall is a bicycle."}}
    rows = expand_pairs(parents, outcomes, [42, 43], "source.csv")
    assert len({row["prompt_id"] for row in rows}) == 4
    settings = {key: None for key in ("negative_prompt", "num_inference_steps", "guidance_scale", "width", "height")}
    requests = [make_request(row, 999 + index, settings) for index, row in enumerate(rows)]
    assert [request["seed"] for request in requests] == [42, 42, 43, 43]
    for row in rows:
        assert row["paired_parent_id"] == "P1"
        if row["benchmark_level"] == "L2":
            assert row["parent_prompt_id"] == "P1"
