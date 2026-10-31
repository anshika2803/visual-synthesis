"""Existing run tables can be scored by stage 7 without model inference."""

import json

import pandas as pd

from visual_synthesis.config import load_config
from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s07_metrics
from visual_synthesis.stages.s02_standardize import complete_standard_table


def test_standardized_attack_input_and_metrics_stage(tmp_path):
    prompts = pd.DataFrame([
        {"prompt_id": f"rab_{config}", "prompt": "test prompt", "source_dataset": "ring-a-bell",
         "standard_category": "sexual", "risk_level": "high", "benchmark_level": "L6",
         "transformation": "adversarial", "attack_method": "Ring-A-Bell",
         "attack_config": config, "case_number": 1, "evaluation_seed": 7,
         "source_file": f"{config}.csv"}
        for config in ("K_16", "K_38", "K_77")
    ] + [{"prompt_id": "ordinary_L1", "prompt": "another test prompt", "source_dataset": "I2P",
          "standard_category": "sexual", "risk_level": "high", "benchmark_level": "L1",
          "transformation": "original", "attack_config": None, "case_number": None}])
    completed = complete_standard_table(prompts, "source-run")
    assert len(completed) == 4
    assert completed["content_hash"].notna().all()
    prompt_path = tmp_path / "prompts.csv"
    prompts.to_csv(prompt_path, index=False)

    generations = prompts[["prompt_id", "prompt", "attack_config", "case_number"]].copy()
    generations["generation_status"] = "success"
    generations["model"] = "test-model"
    generations["generation_time_s"] = 1.0
    generations["image_path"] = "missing-image.png"
    generations.loc[2, "generation_status"] = "error"
    generation_path = tmp_path / "generations.parquet"
    generations.to_parquet(generation_path)
    detections = pd.DataFrame([
        {"prompt_id": "rab_K_16", "status": "success", "flagged": True,
         "max_score": 0.9, "threshold": 0.3, "model": "test-model"},
        {"prompt_id": "rab_K_38", "status": "error", "flagged": None,
         "max_score": None, "threshold": 0.3, "model": "test-model"},
    ])
    detection_path = tmp_path / "detections.parquet"
    detections.to_parquet(detection_path)

    cfg = load_config(overrides={
        "paths.runs": str(tmp_path / "runs"), "paths.logs": str(tmp_path / "logs"),
        "metrics.generations_input": str(generation_path),
        "metrics.prompts_input": str(prompt_path), "metrics.alignment.enabled": False,
    })
    result = run_stage(cfg, s07_metrics, run_name="attack-test", input_path=detection_path)
    assert result["status"] == "success", result
    output = tmp_path / "runs" / "s07_metrics" / "attack-test"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["attack"]["attempts"] == 3
    assert metrics["attack"]["generation_failures"] == 1
    assert metrics["attack"]["detector_failures"] == 1
    assert metrics["attack"]["prompt_level_nudity_related_asr"] == 1 / 3
    assert len(pd.read_parquet(output / "attack_results.parquet")) == 3
