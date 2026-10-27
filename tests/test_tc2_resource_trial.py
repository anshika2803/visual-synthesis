import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from visual_synthesis.config import ROOT

spec = importlib.util.spec_from_file_location("tc2_trial_support", ROOT / "scripts/tc2_trial_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def snapshot(available=None, files=None, filesystem=100 * support.GIB):
    value = {"path": "/models", "filesystem_free_bytes": filesystem,
             "quota": {"available_bytes": available, "available_files": files},
             "usage": {"allocated_bytes": 0}}
    return {"paths": {name: value.copy() for name in ("models", "trial_outputs")}}


def test_numeric_quota_keeps_soft_limit_and_file_limit():
    rows = support.parse_quota("Filesystem blocks quota limit grace files quota limit grace\n"
                               "server:/home 2000* 3000 4000 4days 100 200 300\n")
    assert rows[0]["available_bytes"] == 1000 * 1024
    assert rows[0]["available_files"] == 100
    assert support.parse_quota("server:/home 2G 3G 4G") == []  # Never guess quota -s units.


def test_unknown_quota_blocks_download_despite_large_filesystem():
    result = support.download_capacity(snapshot(), 8 * support.GIB, 30)
    assert not result["allowed"]
    assert "quota is unknown" in result["reason"]


def test_space_guard_counts_headroom_and_cannot_bypass_known_quota():
    assert not support.download_capacity(snapshot(6 * support.GIB), 8 * support.GIB, 30, 100)["allowed"]
    assert not support.download_capacity(snapshot(), 8 * support.GIB, 30, 9.2)["allowed"]
    assert support.download_capacity(snapshot(), 8 * support.GIB, 30, 12)["allowed"]
    assert not support.download_capacity(snapshot(100 * support.GIB, files=10), 100, 20)["allowed"]
    assert not support.download_capacity(snapshot(), 100, 20, float("nan"))["allowed"]


def test_file_plan_avoids_duplicate_weight_formats_and_rejects_unknown_sizes():
    files = [SimpleNamespace(rfilename=name, size=size) for name, size in (
        ("model_index.json", 100), ("unet/config.json", 100),
        ("unet/model.fp16.safetensors", 500), ("unet/pytorch_model.bin", 1000),
        ("unused.ckpt", 100000),
    )]
    selected = support.select_files(files, ["*.json", "unet/**"], "fp16")
    assert sum(selected.values()) == 700
    assert "unet/pytorch_model.bin" not in selected
    files[0].size = None
    with pytest.raises(ValueError, match="sizes are unavailable"):
        support.select_files(files, ["*.json", "unet/**"], "fp16")


def test_cached_model_needs_no_download_or_known_quota(cfg, tmp_path, monkeypatch):
    cfg["models"]["sd-xl"] = {**cfg["models"]["fake"], "backend": "diffusers", "repo_id": "test/sdxl"}
    root = Path(cfg["paths"]["models"]) / "sd-xl"
    (root / "unet").mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"unet": ["diffusers", "UNet2DConditionModel"]}))
    (root / "unet/model.safetensors").touch()
    monkeypatch.setattr(support, "load_config", lambda: cfg)
    monkeypatch.setattr(support, "storage_snapshot", lambda **kwargs: snapshot())
    out = tmp_path / "trial"
    out.mkdir()
    result = support.prepare("sd-xl", out)
    assert result["status"] == "cached"
    assert not result["downloads_allowed"]
    assert not (out / "download-selection.json").exists()


def test_missing_model_is_recorded_without_any_network_download(cfg, tmp_path, monkeypatch):
    cfg["models"]["sd-xl"] = {**cfg["models"]["fake"], "backend": "diffusers", "repo_id": "test/sdxl"}
    monkeypatch.setattr(support, "load_config", lambda: cfg)
    monkeypatch.setattr(support, "storage_snapshot", lambda **kwargs: snapshot())
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(try_to_load_from_cache=lambda *a, **kw: None))
    out = tmp_path / "trial"
    out.mkdir()
    result = support.prepare("sd-xl", out)
    assert result["status"] == "blocked"
    assert "downloads are disabled" in result["error"]
    assert json.loads((out / "preparation.json").read_text())["status"] == "blocked"


def test_finish_preserves_measured_resources_and_reports_disk_change(tmp_path, monkeypatch):
    (tmp_path / "benchmark").mkdir()
    support.save({"model": "sd-xl", "status": "cached"}, tmp_path / "preparation.json")
    support.save(snapshot(10 * support.GIB), tmp_path / "storage-before.json")
    support.save({"models": [{"status": "success", "load_seconds": 4.5, "warmup_seconds": 1,
                              "generation": {"prompt_seconds_inference_only": [2, 3]},
                              "resources": {"cuda_allocated_peak_bytes": 500,
                                            "rss_process_lifetime_peak_bytes": 1000}}]},
                 tmp_path / "benchmark/report.json")
    after = snapshot(9 * support.GIB)
    after["paths"]["models"]["usage"] = {"allocated_bytes": 200}
    monkeypatch.setattr(support, "storage_snapshot", lambda **kwargs: after)
    monkeypatch.setattr(support, "collect_accounting", lambda job_id: {"rows": [{"State": "RUNNING"}]})
    summary = support.finish(tmp_path, 0, "1234")
    assert summary["model_load_seconds"] == 4.5
    assert summary["inference_seconds_per_prompt"] == [2, 3]
    assert summary["resources"]["cuda_allocated_peak_bytes"] == 500
    assert summary["storage_allocated_byte_changes"]["models"] == 200


def test_storage_uses_dotenv_cache_and_freezes_it_across_environment_changes(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    (tmp_path / ".env").write_text("HF_HOME=data/cache/huggingface\n")
    cache = tmp_path / "data/cache/huggingface/hub"
    cache.mkdir(parents=True)
    (cache / "clip.bin").write_bytes(b"cached evaluator")
    monkeypatch.setattr(support, "command", lambda args: {"returncode": 1, "stdout": "", "stderr": ""})
    before = support.storage_snapshot(repo=tmp_path)
    assert before["paths"]["hf_hub"]["path"] == str(cache)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "other-cache"))
    after = support.storage_snapshot(repo=tmp_path, frozen_paths=before["paths"])
    changes, warnings = support.storage_changes(before, after)
    assert after["paths"]["hf_hub"]["path"] == str(cache)
    assert changes["hf_hub"] == 0
    assert not warnings


def test_different_cache_directories_are_not_reported_as_growth():
    before = snapshot()
    after = snapshot()
    after["paths"]["models"]["path"] = "/different-cache"
    after["paths"]["models"]["usage"] = {"allocated_bytes": 16 * support.GIB}
    changes, warnings = support.storage_changes(before, after)
    assert changes["models"] is None
    assert "different directories" in warnings["models"]["reason"]


def test_historical_storage_repair_keeps_originals_and_model_measurements(tmp_path, monkeypatch):
    before = snapshot()
    after = snapshot()
    after["paths"]["models"]["path"] = "/another-cache"
    after["paths"]["models"]["usage"] = {"allocated_bytes": 16 * support.GIB}
    support.save(before, tmp_path / "storage-before.json")
    support.save(after, tmp_path / "storage-after.json")
    summary = {"model": "sd-xl", "model_load_seconds": 95.23,
               "resources": {"cuda_allocated_peak_bytes": 11258828800},
               "storage_allocated_byte_changes": {"models": 16 * support.GIB}}
    support.save(summary, tmp_path / "trial_summary.json")
    original = (tmp_path / "trial_summary.json").read_text()
    monkeypatch.setattr(support, "storage_snapshot", lambda **kwargs: before)
    support.repair_storage(tmp_path)
    repaired = json.loads((tmp_path / "trial_summary.json").read_text())
    assert repaired["storage_allocated_byte_changes"]["models"] is None
    assert repaired["storage_recheck"]["allocated_byte_changes"]["models"] == 0
    assert repaired["model_load_seconds"] == summary["model_load_seconds"]
    assert repaired["resources"] == summary["resources"]
    assert (tmp_path / "trial_summary.before-storage-repair.json").read_text() == original
    assert json.loads((tmp_path / "storage-after.json").read_text()) == after


def test_collect_recovers_record_for_job_killed_before_exit_trap(tmp_path, monkeypatch):
    support.save({"model": "glm-image"}, tmp_path / "preparation.json")
    monkeypatch.setattr(support, "collect_accounting", lambda job_id: {"rows": [{"State": "OUT_OF_MEMORY"}]})
    monkeypatch.setattr(sys, "argv", ["support", "collect", "--output", str(tmp_path), "--job-id", "1234"])
    support.main()
    summary = json.loads((tmp_path / "trial_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["model"] == "glm-image"
    assert summary["resource_limit_states"] == ["OUT_OF_MEMORY"]


@pytest.mark.parametrize("quota,expected", [(None, "blocked"), (20 * support.GIB, "download_permitted")])
def test_new_model_only_gets_a_download_plan_after_quota_guard(cfg, tmp_path, monkeypatch, quota, expected):
    cfg["models"]["sd-xl"] = {**cfg["models"]["fake"], "backend": "diffusers", "repo_id": "test/sdxl"}
    monkeypatch.setattr(support, "load_config", lambda: cfg)
    monkeypatch.setattr(support, "storage_snapshot", lambda **kwargs: snapshot(quota))
    metadata = SimpleNamespace(siblings=[SimpleNamespace(rfilename="model_index.json", size=100),
                                        SimpleNamespace(rfilename="unet/model.safetensors", size=8 * support.GIB)])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        try_to_load_from_cache=lambda *a, **kw: None,
        HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: metadata)
    ))
    out = tmp_path / "trial"
    out.mkdir()
    result = support.prepare("sd-xl", out, allow_download=True)
    assert result["status"] == expected
    assert (out / "download-selection.json").exists() == (expected == "download_permitted")


def test_slurm_directives_match_verified_allocation_and_head_node_is_refused():
    script = ROOT / "scripts/tc2_resource_trial.sh"
    directives = dict(line.removeprefix("#SBATCH --").split("=", 1)
                      for line in script.read_text().splitlines() if line.startswith("#SBATCH --"))
    for key, value in {"account": "stud", "partition": "MGPU-TC2", "qos": "normal", "gres": "gpu:1",
                       "cpus-per-task": "10", "mem": "30G", "time": "06:00:00"}.items():
        assert directives[key] == value
    env = {key: value for key, value in os.environ.items() if key != "SLURM_JOB_ID"}
    result = subprocess.run(["bash", str(script), "sd-xl"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "head node" in result.stderr


@pytest.mark.parametrize("prepare_code,expected_exit,compute_calls", [(0, 7, 1), (20, 20, 0)])
def test_job_activates_visual_safety_and_keeps_preparation_failures(tmp_path, prepare_code,
                                                                 expected_exit, compute_calls):
    tools = tmp_path / "bin"
    tools.mkdir()
    base = tmp_path / "conda"
    (base / "etc/profile.d").mkdir(parents=True)
    (base / "etc/profile.d/conda.sh").write_text("")
    log = tmp_path / "calls.jsonl"
    dispatcher = f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if name == "git":
    print("swara/experiments" if sys.argv[1] == "branch" else "test-commit")
elif name == "conda" and sys.argv[1] == "info":
    print(os.environ["CONDA_BASE"])
elif name == "python" and sys.argv[1] == "-c":
    print("cached")
elif name == "python" and sys.argv[2] == "prepare":
    out = Path(sys.argv[sys.argv.index("--output") + 1])
    out.joinpath("preparation.json").write_text('{"status":"cached"}')
    sys.exit(int(os.environ["PREPARE_CODE"]))
elif name == "srun":
    sys.exit(7)
'''
    for name in ("python", "conda", "git", "srun"):
        path = tools / name
        path.write_text(dispatcher)
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "CALLS": str(log),
           "CONDA_BASE": str(base), "SLURM_JOB_ID": "1234", "VS_TRIAL_REPO": str(tmp_path),
           "PREPARE_CODE": str(prepare_code)}
    result = subprocess.run(["bash", str(ROOT / "scripts/tc2_resource_trial.sh"), "sd-xl"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == expected_exit, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert ["conda", "activate", "visual-safety"] in calls
    assert len([call for call in calls if call[0] == "srun"]) == compute_calls
    assert not any("prefetch" in call for call in calls)
    finish = next(call for call in calls if call[0] == "python" and "finish" in call)
    assert finish[finish.index("--exit-code") + 1] == str(expected_exit)
