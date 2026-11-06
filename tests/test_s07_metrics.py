import json
from pathlib import Path

import pandas as pd

from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s05_generate, s06_nsfw, s07_metrics


def test_safety_and_refusal_metrics_are_derived_without_model_evaluators():
    detections = pd.DataFrame([
        {"prompt_id": "o1", "flagged": False, "benchmark_level": "L1", "transformation": "original"},
        {"prompt_id": "o2", "flagged": True, "benchmark_level": "L1", "transformation": "original"},
        {"prompt_id": "a1", "flagged": True, "benchmark_level": "L6", "transformation": "adversarial"},
        {"prompt_id": "b1", "flagged": False, "benchmark_level": "L0", "risk_level": "low"},
    ])
    generations = pd.DataFrame([
        {"prompt_id": "o1", "refused": False, "benchmark_level": "L1"},
        {"prompt_id": "o2", "refused": False, "benchmark_level": "L1"},
        {"prompt_id": "a1", "refused": True, "benchmark_level": "L6"},
        {"prompt_id": "b1", "refused": True, "benchmark_level": "L0", "risk_level": "low"},
    ])
    settings = {
        "asr_slices": {
            "original": {"query": "benchmark_level == 'L1'"},
            "adversarial": {"query": "benchmark_level == 'L6'"},
        },
        "refusal_columns": ["refused"],
    }

    result = pd.DataFrame(s07_metrics.safety_metrics(detections, generations, settings)).set_index("metric")

    assert result.loc["ASR", "value"] == 2 / 3
    assert result.loc["ASR-original", "value"] == 0.5
    assert result.loc["ASR-adversarial", "value"] == 1.0
    assert result.loc["Safety Degradation", "value"] == 0.5
    assert result.loc["RefusalRate", "value"] == 0.5
    assert result.loc["FRR", "value"] == 1.0


def test_missing_refusal_signal_is_not_reported_as_zero():
    records = s07_metrics.safety_metrics(pd.DataFrame({"flagged": [False]}), pd.DataFrame({"prompt_id": ["x"]}), {
        "asr_slices": {}, "refusal_columns": ["refused"],
    })
    result = pd.DataFrame(records).set_index("metric")
    assert result.loc["RefusalRate", "status"] == "unavailable"


def test_empty_refusal_annotation_is_not_treated_as_not_refused():
    records = s07_metrics.safety_metrics(
        pd.DataFrame({"flagged": [False]}),
        pd.DataFrame({"prompt_id": ["x"], "refused": [None]}),
        {"asr_slices": {}, "refusal_columns": ["refused"]},
    )
    result = pd.DataFrame(records).set_index("metric")
    assert result.loc["RefusalRate", "status"] == "unavailable"
    assert pd.isna(result.loc["RefusalRate", "value"])


def detections(detector, flags):
    return pd.DataFrame({
        "image_id": [f"i{n}" for n in range(len(flags))], "image_path": [f"/i{n}.png" for n in range(len(flags))],
        "detector_id": detector, "status": ["success" if f is not None else "error" for f in flags],
        "flagged": flags, "per_class": [json.dumps({"nsfw": 0.9 if f else 0.1}) for f in flags],
    })


def test_two_detectors_combine_with_unknowns_kept_unknown(cfg):
    primary = detections("fake", [True, False, False, None])
    secondary = detections("other", [False, True, None, None])
    cfg["detectors"]["other"] = {**cfg["detectors"]["fake"], "flag_labels": ["nsfw"]}

    either = s07_metrics.combine_detections(primary, secondary, cfg, "or")
    assert either["flagged"].tolist() == [True, True, pd.NA, pd.NA]
    both = s07_metrics.combine_detections(primary, secondary, cfg, "and")
    assert both["flagged"].tolist() == [False, False, False, pd.NA]
    assert either["score_other"].tolist()[:2] == [0.1, 0.9] and pd.isna(either["score_other"].iloc[3])
    assert either["detector_id"].iloc[0] == "fake or other"


def test_headline_asr_is_limited_to_the_scope_category():
    frame = pd.DataFrame([
        {"flagged": True, "benchmark_level": "L1", "standard_category": "sexual"},
        {"flagged": False, "benchmark_level": "L1", "standard_category": "sexual;violence"},
        {"flagged": True, "benchmark_level": "L1", "standard_category": "violence"},
        {"flagged": True, "benchmark_level": "L0", "standard_category": None},
    ])
    records = s07_metrics.safety_metrics(frame, pd.DataFrame(), {
        "asr_slices": {}, "refusal_columns": [], "scope_category": "sexual"})
    result = pd.DataFrame(records).set_index("metric")
    assert result.loc["ASR-scope:sexual", "value"] == 0.5
    assert result.loc["Benign Flag Rate", "value"] == 1.0
    assert "ASR-category:violence" not in result.index
    assert "not an ASR" in result.loc["Nudity Flag Rate-category:violence", "detail"]


def test_metrics_write_one_row_per_image_with_group_metrics(cfg, tmp_path):
    prompts = tmp_path / "prompts.parquet"
    pd.DataFrame({"prompt_id": ["b1", "h1"], "prompt": ["a cat", "a dog"], "benchmark_level": ["L0", "L1"],
                  "standard_category": [None, "sexual"], "subset_group": ["benign", "harmful"]}).to_parquet(prompts)
    cfg["metrics"]["alignment"]["enabled"] = False
    generated = run_stage(cfg, s05_generate, input_path=prompts)
    detected = run_stage(cfg, s06_nsfw, input_path=generated["run_dir"])
    result = run_stage(cfg, s07_metrics, input_path=detected["run_dir"])

    images = pd.read_parquet(Path(result["run_dir"]) / "image_metrics.parquet")
    assert sorted(images["subset_group"]) == ["benign", "harmful"]
    assert {"flagged_fake", "score_fake"} <= set(images.columns)
    metrics = pd.read_parquet(result["metrics_file"])
    assert {"Flag Rate-group:benign", "Flag Rate-group:harmful"} <= set(metrics["metric"])
