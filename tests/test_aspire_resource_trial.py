import errno
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from visual_synthesis.config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("aspire_trial_support", ROOT / "scripts/aspire_trial_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def snapshot(root, used=0, free=500 * support.GIB, inodes=10000):
    return {"paths": {name: {"path": str(root / name), "filesystem_free_bytes": free,
                             "filesystem_free_inodes": inodes,
                             "usage": {"allocated_bytes": used, "stat_errors": []}}
                      for name in ("workspace", "models", "hf_home", "trial_outputs")}}


def test_guard_deducts_retained_weights_and_reserves_download_headroom(tmp_path):
    snap = snapshot(tmp_path, used=50 * support.GIB)
    assert support.download_guard(snap, 30 * support.GIB, 30, 100)["allowed"]
    assert not support.download_guard(snap, 50 * support.GIB, 30, 100)["allowed"]
    assert not support.download_guard(snapshot(tmp_path, free=support.GIB), 1, 30, 100)["allowed"]
    assert not support.download_guard(snapshot(tmp_path, inodes=10), 1, 30, 100)["allowed"]
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            support.download_guard(snap, 100, 30, invalid)


def test_storage_rejects_home_but_resolves_scratch_symlink(tmp_path, monkeypatch):
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    home.mkdir()
    scratch.mkdir()
    (home / "scratch").symlink_to(scratch, target_is_directory=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    for path in (home, home / "weights", Path("/")):
        with pytest.raises(ValueError):
            support.workspace_path(path)
    assert support.workspace_path(home / "scratch/trials") == scratch / "trials"


def test_metadata_plan_pins_configured_files_without_downloading(cfg, monkeypatch):
    cfg["models"]["sd-1.5"] = {**cfg["models"]["fake"], "repo_id": "test/sd15", "variant": "fp16",
                               "allow_patterns": ["*.json", "unet/**"]}
    metadata = SimpleNamespace(sha="a" * 40, siblings=[
        SimpleNamespace(rfilename="model_index.json", size=100),
        SimpleNamespace(rfilename="unet/model.fp16.safetensors", size=200),
        SimpleNamespace(rfilename="unet/pytorch_model.bin", size=400),
        SimpleNamespace(rfilename="unet/model.safetensors", size=400),
    ])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: metadata)))
    result = support.model_plan("sd-1.5", cfg)
    assert result["revision"] == "a" * 40
    assert result["selected_bytes"] == 300
    assert set(result["files"]) == {"model_index.json", "unet/model.fp16.safetensors"}


def test_accounting_does_not_store_environment_secrets(monkeypatch):
    job = {"job_state": "F", "Exit_status": 0, "resources_used": {"mem": "1200kb"},
           "Variable_List": "HF_TOKEN=secret", "Resource_List": {"ngpus": 1}}
    monkeypatch.setattr(support, "command", lambda args: {
        "returncode": 0, "stdout": json.dumps({"Jobs": {"42.pbs101": job}}), "stderr": ""})
    result = support.accounting("42.pbs101")
    assert result["jobs"]["42.pbs101"]["resources_used"]["mem"] == "1200kb"
    assert "secret" not in json.dumps(result)
    with pytest.raises(ValueError):
        support.accounting("-other")


def test_text_accounting_fallback_keeps_limits_and_excludes_environment(monkeypatch):
    monkeypatch.setattr(support, "command", lambda args: {
        "returncode": 0, "stdout": "    job_state = F\n    Exit_status = 271\n"
        "    resources_used.mem = 112000000kb\n    Variable_List = HF_TOKEN=secret\n", "stderr": ""})
    result = support.accounting("42.pbs101")
    assert result["jobs"]["42.pbs101"]["Exit_status"] == "271"
    assert "secret" not in json.dumps(result)


def test_collect_recovers_running_model_after_pbs_kill(tmp_path):
    support.save({"status": "running", "order": ["qwen-image", "glm-image"], "models": [],
                  "not_run": ["qwen-image", "glm-image"]}, tmp_path / "run_summary.json")
    support.save({"model": "qwen-image", "status": "running", "stage": "benchmark"},
                 tmp_path / "qwen-image/trial_summary.json")
    report = {"jobs": {"42.pbs": {"job_state": "F", "Exit_status": 271,
                                  "comment": "memory limit exceeded"}}}
    support.collect(tmp_path, report)
    result = json.loads((tmp_path / "run_summary.json").read_text())
    assert result["status"] == "failed"
    assert result["models"][0]["status"] == "interrupted"
    assert "measurement" not in result["models"][0]
    assert result["not_run"] == ["glm-image"]


def test_run_and_shell_wrapper_refuse_login_node(monkeypatch, tmp_path):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    assert support.main(["run", "--storage", str(tmp_path)]) == 2
    result = subprocess.run(["bash", str(ROOT / "scripts/aspire_resource_trial.sh")],
                            capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                               if k != "PBS_JOBID"})
    assert result.returncode == 2
    assert "qsub" in result.stderr


@pytest.mark.parametrize("failure_kind,expected", [("gpu_out_of_memory", 2), ("storage_exhausted", 1)])
def test_serial_heavy_first_run_continues_after_model_failure_but_stops_on_full_storage(
    tmp_path, monkeypatch, failure_kind, expected
):
    monkeypatch.setenv("PBS_JOBID", "42.pbs")
    monkeypatch.setattr(support, "accounting", lambda job: {"jobs": {}})
    monkeypatch.setattr(support, "command", lambda args: {"stdout": ""})
    seen = []

    def trial(model, *args):
        seen.append(model)
        return {"model": model, "status": "failed", "failure_kind": failure_kind}

    monkeypatch.setattr(support, "trial", trial)
    args = SimpleNamespace(models=list(support.ORDER[:2]), sample_size=2)
    support.run_locked(args, tmp_path)
    assert seen == list(support.ORDER[:expected])


def test_subprocess_timeout_terminates_stage_and_records_failure(tmp_path):
    log = tmp_path / "timeout.log"
    code = support.run_logged([sys.executable, "-c", "import time; time.sleep(10)"], log, 0.1)
    assert code == 124
    assert "exceeded" in log.read_text()


def test_trial_uses_existing_benchmark_and_preserves_partial_worker_peaks(cfg, tmp_path, monkeypatch):
    from visual_synthesis import benchmark_cluster

    monkeypatch.setenv("PBS_JOBID", "42.pbs")
    monkeypatch.setattr(support, "configs", lambda: cfg)
    monkeypatch.setattr(support, "storage_snapshot", lambda root, scan: snapshot(root))
    monkeypatch.setattr(benchmark_cluster, "_check_model_files", lambda *a: None)
    cfg["models"]["qwen-image"] = cfg["models"]["fake"]
    out = tmp_path / "qwen-image"
    seen = []

    def logged(argv, log, timeout, offline=True):
        seen.append((argv, offline))
        if "benchmark_resources.py" in argv[1]:
            support.save({"status": "failed", "error": "CUDA out of memory",
                          "resources": {"cuda_allocated_peak_bytes": 123}},
                         out / "benchmark/qwen-image/report.json")
        return 0

    monkeypatch.setattr(support, "run_logged", logged)
    args = SimpleNamespace(input=tmp_path / "prompts.csv", sample_size=2, allow_download=False,
                           budget_gib=100, cpu_offload=True, timeout=1800)
    result = support.trial("qwen-image", out, tmp_path, args)
    assert result["failure_kind"] == "gpu_out_of_memory"
    assert result["measurement"]["resources"]["cuda_allocated_peak_bytes"] == 123
    assert "--cpu-offload" in seen[-1][0]
    assert all(offline for _, offline in seen)


def test_collect_stdout_only_does_not_write(monkeypatch, capsys):
    monkeypatch.setattr(support, "accounting", lambda job: {"jobs": {job: {"job_state": "F"}}})
    monkeypatch.setattr(support, "save", lambda *args: pytest.fail("stdout-only wrote a file"))
    assert support.main(["collect", "--job-id", "42.pbs", "--stdout-only"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"]["42.pbs"]["job_state"] == "F"


@pytest.mark.parametrize("budget,should_download", [(1, False), (100, True)])
def test_uncached_trial_pins_only_one_model_and_checks_capacity_before_download(
    cfg, tmp_path, monkeypatch, budget, should_download
):
    from visual_synthesis import benchmark_cluster

    monkeypatch.setenv("PBS_JOBID", "42.pbs")
    monkeypatch.setattr(support, "configs", lambda: cfg)
    monkeypatch.setattr(support, "storage_snapshot", lambda root, scan: snapshot(root))
    cfg["models"]["qwen-image"] = cfg["models"]["fake"]

    def missing(*args):
        raise ValueError("not cached")

    monkeypatch.setattr(benchmark_cluster, "_check_model_files", missing)
    selection = {"model": "qwen-image", "revision": "b" * 40,
                 "selected_bytes": 50 * support.GIB, "file_count": 2,
                 "files": {"model_index.json": 100, "transformer/model.safetensors": 50 * support.GIB - 100}}
    monkeypatch.setattr(support, "model_plan", lambda *args: selection)
    commands = []
    out = tmp_path / "qwen-image"

    def logged(argv, log, timeout, offline=True):
        commands.append((argv, offline))
        if "prefetch" in argv:
            overlay = json.loads((out / "download-selection.json").read_text())
            assert list(overlay["models"]) == ["qwen-image"]
            assert overlay["models"]["qwen-image"]["revision"] == "b" * 40
            assert not offline
        if "benchmark_resources.py" in argv[1]:
            support.save({"status": "success", "resources": {}}, out / "benchmark/qwen-image/report.json")
        return 0

    monkeypatch.setattr(support, "run_logged", logged)
    args = SimpleNamespace(input=tmp_path / "prompts.csv", sample_size=2, allow_download=True,
                           budget_gib=budget, cpu_offload=False, timeout=1800, download_timeout=5400)
    result = support.trial("qwen-image", out, tmp_path, args)
    assert any("prefetch" in argv for argv, _ in commands) == should_download
    assert result["status"] == ("success" if should_download else "blocked")
    if should_download:
        assert commands[-1][1]  # The benchmark cannot download anything implicitly.


def test_storage_configuration_overrides_stale_tc2_cache_locations(tmp_path, monkeypatch):
    keys = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "VS_MODELS_DIR", "XDG_CACHE_HOME",
            "TORCH_HOME", "VS_RUNS_DIR", "VS_LOGS_DIR", "HF_HUB_DISABLE_XET", "HF_TOKEN_PATH")
    for key in keys:
        monkeypatch.setenv(key, "/old/tc2/cache")
    support.configure_storage(tmp_path)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "VS_MODELS_DIR"):
        assert Path(os.environ[key]).is_relative_to(tmp_path)


def test_moving_caches_preserves_login_path_without_copying_token(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    (old / "token").write_text("unit-test-placeholder")
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "VS_MODELS_DIR", "XDG_CACHE_HOME",
                "TORCH_HOME", "VS_RUNS_DIR", "VS_LOGS_DIR", "HF_HUB_DISABLE_XET", "HF_TOKEN_PATH"):
        monkeypatch.setenv(key, str(old))
    monkeypatch.delenv("HF_TOKEN_PATH")
    support.configure_storage(new)
    assert os.environ["HF_TOKEN_PATH"] == str(old / "token")
    assert not (new / "token").exists()


def test_collect_prints_retrieved_accounting_when_storage_is_full(tmp_path, monkeypatch, capsys):
    original = '{"status":"running","models":[],"order":[]}'
    (tmp_path / "run_summary.json").write_text(original)
    report = {"jobs": {"42.pbs": {"job_state": "F", "Exit_status": 0}}}
    monkeypatch.setattr(support, "accounting", lambda job: report)

    def no_space(*args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(support, "save", no_space)
    assert support.main(["collect", "--job-id", "42.pbs", "--output", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == report
    assert "Storage exhausted" in output.err
    assert (tmp_path / "run_summary.json").read_text() == original
