import pandas as pd
import pytest

from visual_synthesis.benchmark import benchmark_worker, project_summary, summarize_model
from visual_synthesis.benchmark_sampling import load_prompts, sample_prompts

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L6")


def level_rows():
    return pd.DataFrame([
        {"prompt_id": f"{level}-{i}", "prompt": "a landscape", "benchmark_level": level,
         "standard_category": ["violence", "sexual"][i % 2]}
        for level in LEVELS for i in range(20)
    ])


def attack_rows():
    return pd.DataFrame([
        {"prompt_id": f"case-{case}-{config}", "prompt": "a landscape", "source_dataset": "ring-a-bell",
         "benchmark_level": "L6", "source_metadata":
         f"ring_a_bell_attack_method=Ring-A-Bell;ring_a_bell_attack_config={config};"
         f"ring_a_bell_case_number={case};ring_a_bell_evaluation_seed={case};"
         f"ring_a_bell_source_file={config}.csv"}
        for case in range(4) for config in ("K_16", "K_38", "K_77")
    ])


def test_balanced_sample_covers_levels_and_categories_and_is_reproducible():
    first, info = sample_prompts(level_rows(), 60, seed=42)
    second, _ = sample_prompts(level_rows(), 60, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first.benchmark_level.value_counts().to_dict() == dict.fromkeys(LEVELS, 10)
    assert first.groupby("benchmark_level").standard_category.nunique().eq(2).all()
    assert info["coverage_complete"]
    assert first.prompt_id.is_unique


def test_missing_level_is_reported_and_never_invented():
    source = level_rows().query("benchmark_level != 'L2'")
    sample, info = sample_prompts(source, 60)
    assert len(sample) == 60
    assert set(sample.prompt_id) <= set(source.prompt_id)
    assert info["missing_sample_levels"] == ["L2"]
    assert not info["coverage_complete"]


def test_restore_seeds_from_manifest_and_keep_matched_configurations(tmp_path):
    path = tmp_path / "candidates.csv"
    attack_rows().to_csv(path, index=False)
    frame = load_prompts(path)
    assert frame.evaluation_seed.tolist() == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    sample, info = sample_prompts(frame, 8)
    assert info["strategy"] == "matched-cases"
    assert len(sample) == 6
    assert sample.groupby("case_number").attack_config.agg(set).eq({"K_16", "K_38", "K_77"}).all()
    assert info["coverage_complete"]
    assert info["target_scope"] == "stage2_attack_prompts"


def test_tiny_matched_sample_keeps_all_three_configurations(tmp_path):
    path = tmp_path / "candidates.csv"
    attack_rows().to_csv(path, index=False)
    sample, info = sample_prompts(load_prompts(path), 2)
    assert len(sample) == 3
    assert info["actual_sample_size"] == 3


def test_lfs_pointer_and_fractional_seed_are_rejected(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        load_prompts(path)
    pd.DataFrame([{"prompt_id": "p1", "prompt": "a cat", "evaluation_seed": 1.5}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="integers"):
        load_prompts(path)


def test_repeats_keep_attack_seeds_and_model_names_in_image_paths(cfg, tmp_path):
    path = tmp_path / "attacks.csv"
    attack_rows().to_csv(path, index=False)
    sample, info = sample_prompts(load_prompts(path), 3)
    path = tmp_path / "sample.parquet"
    sample.to_parquet(path, index=False)
    result = benchmark_worker({"config": cfg, "model": "fake", "output": str(tmp_path / "pilot"),
                               "sample": str(path), "warmup": 1, "repeats": 2, "seed": 100,
                               "target_prompts": 285, "passes": 1, "evaluate": False,
                               "allow_download": False, "sampling": info})
    assert result["status"] == "success"
    for repeat in (1, 2):
        table = pd.read_parquet(tmp_path / f"pilot/runs/s05_generate/repeat-{repeat}/generations.parquet")
        assert table.seed.tolist() == sample.evaluation_seed.tolist()
        assert table.image_path.str.contains("/images/fake_").all()


def test_level_weighting_and_provisional_project_totals():
    result = {
        "status": "success", "model": "model", "images_per_prompt": 1,
        "load_seconds": 0, "warmup_seconds": 0,
        "sampling": {"strategy": "levels", "coverage_complete": True},
        "generation": {"prompt_seconds_with_io": [1] * 10 + [10], "table_write_seconds_mean": 0,
                       "mean_image_bytes": 100,
                       "mean_seconds_by_level": dict(zip(LEVELS, [1, 2, 3, 4, 5, 6], strict=True)),
                       "mean_image_bytes_by_level": dict.fromkeys(LEVELS, 100)},
        "evaluation": {"status": "success", "images": 10, "seconds": 1},
    }
    estimate = summarize_model(result, 6000, 1)
    assert estimate["mean_seconds_per_prompt_with_io"] == 3.5
    result["sampling"] = {"strategy": "levels", "coverage_complete": False, "missing_sample_levels": ["L2"]}
    result["estimate"] = summarize_model(result, 6000, 1)
    assert result["estimate"]["projection_status"] == "provisional"
    assert project_summary([result])["full_project_hours_conservative"] is None
