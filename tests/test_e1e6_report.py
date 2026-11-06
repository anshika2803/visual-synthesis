import numpy as np
import pandas as pd
import pytest

from visual_synthesis import e1e6_report as report

N = 40


def images(experiment, group, unsafe, clip=0.3, **condition):
    """``N`` image rows of one group in one run; ``unsafe`` is a rate or a list of booleans."""
    flags = unsafe if isinstance(unsafe, list) else [i < round(unsafe * N) for i in range(N)]
    level = {"benign": "L0", "near_boundary": "L0", "harmful": "L1", "adversarial": "L6"}[group]
    return pd.DataFrame({
        "prompt_id": [f"{group}{i}" for i in range(N)], "model": "m", "experiment": experiment,
        "run": f"{experiment}-m", "subset_group": group, "subset_split": "validation", "benchmark_level": level,
        "standard_category": None if level == "L0" else "sexual", "flagged": flags,
        "clip_score": np.linspace(clip - 0.05, clip + 0.05, N), "quality_score": 0.5, "precision": "bfloat16",
        **condition,
    })


def condition(mode, vector, alpha):
    return {"intervention_mode": mode, "intervention_vector": vector, "intervention_alpha": str(alpha),
            "intervention_layer": "blocks.3", "intervention_passes": "all", "intervention_steps": "0-1"}


def frame():
    parts = [images("e4", g, u) for g, u in (("harmful", 0.6), ("adversarial", 0.5), ("benign", 0.0),
                                              ("near_boundary", 0.0))]
    for alpha, harmful, random_harmful, clip in ((0.5, 0.4, 0.6, 0.3), (1.0, 0.1, 0.6, 0.2)):
        safety, random = condition("steering", "safety", alpha), condition("steering", "random:0", alpha)
        parts += [images("e5", "harmful", harmful, **safety), images("e5", "benign", 0.0, clip, **safety),
                  images("e5", "harmful", random_harmful, **random), images("e5", "benign", 0.0, **random)]
    return report.prepare(pd.concat(parts, ignore_index=True))


def test_e5_compares_each_condition_with_alpha_zero_and_with_the_random_direction():
    e5 = report.e5_table(frame()).set_index(["intervention_vector", "alpha", "group"])
    strong = e5.loc[("safety", 1.0, "harmful")]
    assert strong["asr"] == pytest.approx(0.1) and strong["asr_delta"] == pytest.approx(-0.5)
    assert strong["asr_p_holm"] < 0.05 and strong["vs_random_delta"] == pytest.approx(-0.5)
    assert bool(strong["direction_specific"])
    assert not bool(e5.loc[("random:0", 1.0, "harmful"), "direction_specific"])
    # CLIP fell for every benign prompt at alpha = 1, so (nearly) all fall under the alpha = 0 5th percentile.
    assert e5.loc[("safety", 1.0, "benign"), "suppression"] == 1.0
    assert e5.loc[("random:0", 1.0, "benign"), "suppression"] < 0.1


def test_e6_reports_the_best_alpha_within_the_suppression_budget():
    rows, summary = report.e6_table(frame(), "steering")
    assert set(rows["alpha"]) == {0.0, 0.5, 1.0}
    best = summary.iloc[0]
    assert best["best_alpha"] == 0.5 and best["asr_drop"] == pytest.approx(0.2)


def test_e7_pairs_variants_with_their_parents():
    parents = images("e1", "harmful", 0.25)
    variants = images("e7", "harmful", 0.75).assign(
        prompt_id=lambda f: f["prompt_id"] + "__L5", parent_prompt_id=parents["prompt_id"],
        transformation_method="keywords:fra_Latn", benchmark_level="L5")
    table = report.e7_table(report.prepare(pd.concat([parents, variants], ignore_index=True)))
    row = table.iloc[0]
    assert (row["mode"], row["language"], row["vs_parent_pairs"]) == ("keywords", "fra_Latn", N)
    assert row["asr_parent"] == pytest.approx(0.25) and row["vs_parent_delta"] == pytest.approx(0.5)


def test_collect_drops_excluded_prompts_and_their_variants(tmp_path):
    parents = images("e1", "harmful", 0.5)
    variants = images("e7", "harmful", 0.5).assign(prompt_id=lambda f: f["prompt_id"] + "__L5",
                                                   parent_prompt_id=parents["prompt_id"])
    for name, rows in (("e1-m", parents), ("e7-m", variants)):
        (tmp_path / "s07_metrics" / name).mkdir(parents=True)
        rows.drop(columns=["experiment", "run"]).to_parquet(tmp_path / "s07_metrics" / name / "image_metrics.parquet")
    frame = report.collect(tmp_path, exclude={"harmful0", "harmful1"})
    assert len(frame) == 2 * (N - 2)
    assert not frame["prompt_id"].str.match(r"harmful[01](__|$)").any()


def test_e1_limits_asr_to_the_detectors_scope():
    harmful = images("e1", "harmful", 0.5)
    violent = images("e1", "harmful", 1.0).assign(prompt_id=lambda f: "v" + f["prompt_id"],
                                                  standard_category="violence")
    table = report.e1_table(report.prepare(pd.concat([harmful, violent, images("e1", "benign", 0.1)])))
    row = table.iloc[0]
    assert row["asr"] == 0.5 and row["out_of_scope_flag_rate"] == 1.0 and row["benign_flag_rate"] == 0.1


def test_precision_check_passes_a_faithful_copy_and_fails_a_degraded_one():
    reference = report.prepare(pd.concat([images("q", "benign", 0.0), images("q", "harmful", 0.5)]))
    assert report.precision_check(reference, reference.copy())["passes"]
    degraded = reference.assign(clip_score=reference["clip_score"] - 0.05)
    result = report.precision_check(reference, degraded)
    assert not result["passes"] and not result["checks"]["clip_score"]


def test_adversarial_prompts_get_their_own_verdict_and_each_model_its_own_e6_mode():
    data = frame()
    adversarial = images("e5", "adversarial", 0.0, **condition("steering", "safety", 1.0))
    adversarial_random = images("e5", "adversarial", 0.5, **condition("steering", "random:0", 1.0))
    other = data.assign(model="other")
    other = other[~((other["experiment"] == "e5") & (other["intervention_vector"] == "safety"))]
    e5 = report.e5_table(pd.concat([data, report.prepare(pd.concat([adversarial, adversarial_random])), other]))
    row = e5.set_index(["model", "intervention_vector", "alpha", "group"]).loc[("m", "safety", 1.0, "adversarial")]
    assert row["asr_p_holm"] < 0.05 and bool(row["direction_specific"])
    # Model "other" has no direction-specific safety run, so its E6 falls back to ablation.
    assert report.e6_mode(e5) == {"m": "steering", "other": "ablation"}
