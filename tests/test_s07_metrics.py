import pandas as pd

from visual_synthesis.stages import s07_metrics


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
