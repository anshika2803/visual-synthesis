import copy
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner

from visual_synthesis.backends.image import DiffusersBackend
from visual_synthesis.benchmark import benchmark_worker, merge_reports, project_summary, run_benchmark, summarize_model
from visual_synthesis.cli import cli
from visual_synthesis.config import load_config


def measured_result():
    return {
        "model": "test", "status": "success", "load_seconds": 10, "warmup_seconds": 2,
        "images_per_prompt": 2, "generation": {"prompt_seconds_with_io": [2, 4],
                                              "table_write_seconds_mean": 1, "mean_image_bytes": 1024},
        "evaluation": {"status": "success", "seconds": 20, "images": 4}, "machine": {"device": "cpu"},
    }


def test_projection_counts_loading_once_and_multiple_images_and_passes():
    estimate = summarize_model(measured_result(), target_prompts=6000, passes=3)
    assert estimate["generation_hours"] == pytest.approx(3 * (10 + 2 + 1 + 6000 * 3) / 3600)
    assert estimate["target_images"] == 36000
    assert estimate["evaluation_hours_conservative"] == 50
    assert estimate["training_status"].startswith("out_of_scope")


def test_failures_never_produce_complete_estimates():
    failed = {**measured_result(), "status": "failed"}
    assert summarize_model(failed, 6000, 1) is None
    missing_evaluator = {**measured_result(), "evaluation": {"status": "failed"}}
    estimate = summarize_model(missing_evaluator, 6000, 1)
    assert estimate["generation_hours"] > 0
    assert estimate["pipeline_hours_conservative"] is None
    assert project_summary([{**missing_evaluator, "estimate": estimate}])["status"] == "incomplete"


def test_eight_models_get_two_per_device_without_dividing_serial_time_by_four():
    results = []
    for index in range(1, 9):
        result = {**measured_result(), "model": str(index)}
        result["estimate"] = {**summarize_model(result, 6000, 1), "pipeline_hours_conservative": index}
        results.append(result)
    summary = project_summary(results)
    assert summary["pipeline_hours_conservative_measured_models"] == 36
    assert summary["four_device_wall_hours_conservative"] == 9
    assert all(len(pair["models"]) == 2 for pair in
               summary["suggested_pairs_if_four_equivalent_dedicated_devices"])
    results[0]["machine"] = {"device": "cuda", "gpu": "different"}
    assert "four_device_wall_hours_conservative" not in project_summary(results)


def test_real_worker_warmup_excluded_and_multiple_images_retained(cfg, tmp_path):
    cfg["models"]["fake"]["num_images_per_prompt"] = 2
    cfg["metrics"]["alignment"]["enabled"] = False
    sample = tmp_path / "sample.parquet"
    pd.DataFrame({"prompt_id": ["p1", "p2"], "prompt": ["a cat", "a tree"]}).to_parquet(sample)
    result = benchmark_worker({
        "config": cfg, "model": "fake", "output": str(tmp_path / "worker"), "sample": str(sample),
        "warmup": 3, "repeats": 2, "seed": 42, "target_prompts": 6000, "passes": 1,
        "evaluate": True, "allow_download": False,
    })
    assert result["status"] == "success"
    assert len(result["generation"]["prompt_seconds_with_io"]) == 4
    assert result["generation"]["images_written"] == 8
    assert result["evaluation"]["images"] == 4
    assert result["evaluation"]["status"] == "success"
    assert result["synthetic"]
    assert project_summary([result])["measured_complete_models"] == 0


def test_worker_failure_does_not_extrapolate_partial_sample(cfg, tmp_path, monkeypatch):
    class BrokenBackend:
        def load(self):
            raise RuntimeError("out of memory")

        def unload(self):
            pass

    monkeypatch.setattr("visual_synthesis.benchmark.build_image_backend", lambda *_: BrokenBackend())
    sample = tmp_path / "sample.parquet"
    pd.DataFrame({"prompt_id": ["p1"], "prompt": ["a cat"]}).to_parquet(sample)
    result = benchmark_worker({
        "config": cfg, "model": "fake", "output": str(tmp_path / "worker"), "sample": str(sample),
        "warmup": 1, "repeats": 1, "seed": 42, "target_prompts": 6000, "passes": 1,
        "evaluate": False, "allow_download": False,
    })
    assert result["status"] == "failed"
    assert result["estimate"] is None
    assert "out of memory" in result["error"]


def test_subprocess_pilot_and_merge_reject_incompatible_or_duplicate_results(cfg, tmp_path):
    cfg["metrics"]["alignment"]["enabled"] = False
    source = tmp_path / "prompts.parquet"
    pd.DataFrame({"prompt_id": ["p1", "p2"], "prompt": ["a cat", "a tree"]}).to_parquet(source)
    output = tmp_path / "pilot"
    report = run_benchmark(cfg, ["fake"], output, source, sample_size=2, warmup=1, repeats=1)
    assert report["models"][0]["evaluation"]["status"] == "success"
    assert (output / "summary.csv").exists()
    assert report["project"]["full_project_hours_conservative"] is None  # synthetic
    with pytest.raises(FileExistsError):
        run_benchmark(cfg, ["fake"], output, source)
    with pytest.raises(ValueError, match="duplicate"):
        merge_reports([output / "report.json"] * 2, tmp_path / "merged")
    altered = copy.deepcopy(report)
    altered["target_prompts"] = 7000
    (tmp_path / "altered.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="target_prompts"):
        merge_reports([output / "report.json", tmp_path / "altered.json"], tmp_path / "merged")
    second = copy.deepcopy(report)
    second["models"][0]["model"] = "another-fake"
    (tmp_path / "second.json").write_text(json.dumps(second))
    result = CliRunner().invoke(cli, ["benchmark-summary", str(output / "report.json"),
                                      str(tmp_path / "second.json"), "-o", str(tmp_path / "merged")])
    assert result.exit_code == 0, result.output
    merged = json.loads((tmp_path / "merged" / "report.json").read_text())
    assert len(merged["models"]) == 2
    assert merged["project"]["full_project_hours_conservative"] is None


def test_cli_returns_failure_for_failed_pilot(tmp_path):
    runner = CliRunner()
    cfg = {"models": {"broken": {"backend": "unsupported"}}, "generate": {"model": "broken"},
           "nsfw": {"detector": "none"}}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg))
    source = tmp_path / "prompts.csv"
    pd.DataFrame({"prompt_id": ["p1"], "prompt": ["a cat"]}).to_csv(source, index=False)
    result = runner.invoke(cli, ["benchmark", "-c", str(config_file), "--model", "broken", "-i", str(source),
                                 "-o", str(tmp_path / "pilot"), "--sample-size", "1", "--generation-only"])
    assert result.exit_code == 1
    assert "some pilots/evaluators failed" in result.output


def test_team_profile_has_eight_download_targets_and_model_specific_guidance():
    cfg = load_config(["config/pipeline.json", "config/benchmark.models.json"], use_env=False)
    assert len(cfg["download"]["models"]) == 8
    assert cfg["models"]["qwen-image"]["generation_kwargs"] == {"true_cfg_scale": 4.0}
    assert cfg["models"]["z-image-turbo"]["num_inference_steps"] == 9


def test_pipeline_specific_guidance_is_passed_without_overriding_standard_fields(cfg, monkeypatch):
    from PIL import Image

    calls = []
    generator = SimpleNamespace(manual_seed=lambda seed: seed)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Generator=lambda device: generator, inference_mode=nullcontext
    ))
    model = {**cfg["models"]["fake"], "generation_kwargs": {"true_cfg_scale": 4.0}, "negative_prompt": " "}
    backend = DiffusersBackend(model)

    def pipeline(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    backend.pipe = pipeline
    backend.generate([{"prompt_id": "p1", "prompt": "a cat", "seed": 42}])
    assert calls[0]["true_cfg_scale"] == 4.0
    assert calls[0]["negative_prompt"] == " "
    model["generation_kwargs"] = {"num_inference_steps": 1}
    with pytest.raises(ValueError, match="cannot replace"):
        backend.generate([{"prompt_id": "p1", "prompt": "a cat", "seed": 42}])
