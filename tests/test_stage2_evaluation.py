import pandas as pd
import pytest

from visual_synthesis.stage2_evaluation import (
    calculate_metrics,
    image_filename,
    load_dataset,
    smoke_sample,
)


def stage2_rows():
    return pd.DataFrame([
        {"prompt_id": f"p-{case}-{config}", "prompt": "prompt", "source_dataset": "ring-a-bell",
         "attack_method": "Ring-A-Bell", "attack_config": config, "case_number": case,
         "evaluation_seed": case + 1, "source_file": f"{config}.csv"}
        for case in range(3) for config in ("K_16", "K_38", "K_77")
    ])


def test_dataset_schema_and_unique_ids(tmp_path):
    frame = stage2_rows()
    path = tmp_path / "valid.csv"
    frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert len(loaded) == 9
    assert loaded.evaluation_seed.tolist() == frame.evaluation_seed.tolist()
    frame.loc[1, "prompt_id"] = frame.loc[0, "prompt_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        load_dataset(path)


def test_dataset_missing_columns_and_values_are_rejected(tmp_path):
    frame = stage2_rows().drop(columns="source_file")
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)
    frame = stage2_rows()
    frame.loc[0, "evaluation_seed"] = None
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required values"):
        load_dataset(path)


def test_smoke_selects_matched_cases_across_configs():
    result = smoke_sample(stage2_rows(), cases=2)
    assert len(result) == 6
    assert result.case_number.nunique() == 2
    assert set(result.attack_config) == {"K_16", "K_38", "K_77"}


def test_stage2_filename_is_unique_and_seeded():
    row = stage2_rows().iloc[0].to_dict() | {"model_id": "sd-turbo"}
    first = image_filename(row, row["model_id"])
    assert first == image_filename(row, row["model_id"])
    assert first != image_filename(row, "sd-xl")
    assert first != image_filename(row | {"evaluation_seed": 7}, row["model_id"])


def test_prompt_configuration_and_case_asr_and_failures():
    generations = stage2_rows()
    generations["generation_status"] = "success"
    generations["generation_time_s"] = 2.0
    generations.loc[0, "generation_status"] = "error"
    detections = pd.DataFrame([
        {"prompt_id": row.prompt_id, "status": "success", "max_score": score, "image_path": f"{row.prompt_id}.png"}
        for row, score in zip(stage2_rows().itertuples(), [0.9, 0.1, 0.2, 0.8, 0.9, 0.1, 0.1, 0.2, 0.9])
        if row.prompt_id != "p-0-K_16"
    ])
    detections.loc[detections.prompt_id == "p-0-K_38", "status"] = "error"
    detections.loc[detections.prompt_id == "p-0-K_38", "max_score"] = None
    attempts, summary, by_config = calculate_metrics(generations, detections, threshold=0.5)
    assert summary["attempts"] == 9
    assert summary["generation_failures"] == 1
    assert summary["detector_failures"] == 1
    assert summary["prompt_level_nudity_related_asr"] == pytest.approx(3 / 9)
    assert summary["case_level_nudity_related_asr"] == pytest.approx(2 / 3)
    assert set(by_config.attack_config) == {"K_16", "K_38", "K_77"}
    assert pd.isna(attempts.loc[attempts.prompt_id == "p-0-K_38", "nsfw_flag"].iloc[0])


def test_stage1_generations_without_attack_metadata_remain_supported():
    generations = pd.DataFrame([{"prompt_id": "legacy", "generation_status": "success"}])
    detections = pd.DataFrame([{"prompt_id": "legacy", "status": "success", "max_score": 0.1}])
    attempts, summary, _ = calculate_metrics(generations, detections)
    assert summary["attempts"] == 1
    assert attempts.loc[0, "nsfw_flag"] == False
