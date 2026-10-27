import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from visual_synthesis.benchmark_cluster import _check_model_files, check_environment, prefetch, prepare_inputs
from visual_synthesis.config import ROOT


def test_prepare_inputs_preserves_source_text_and_restores_attack_fields(tmp_path):
    rows = [{"prompt_id": "l1", "prompt": "a green tree", "source_dataset": "I2P", "benchmark_level": "L1"}]
    for config in ("K_16", "K_38", "K_77"):
        rows.append({"prompt_id": config, "prompt": "a blue lake", "source_dataset": "ring-a-bell",
                     "benchmark_level": "L6", "source_metadata":
                     f"ring_a_bell_attack_method=Ring-A-Bell;ring_a_bell_attack_config={config};"
                     f"ring_a_bell_case_number=7;ring_a_bell_evaluation_seed=0;ring_a_bell_source_file={config}.csv"})
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    report = prepare_inputs(path, tmp_path / "inputs")
    restored = pd.read_csv(report["outputs"]["ring_a_bell"])
    assert restored.prompt.tolist() == ["a blue lake"] * 3
    assert restored.evaluation_seed.tolist() == [0, 0, 0]
    assert set(restored.attack_config) == {"K_16", "K_38", "K_77"}
    assert pd.read_csv(report["outputs"]["l1"]).prompt.tolist() == ["a green tree"]
    assert report["sampling"]["missing_sample_levels"] == ["L0", "L2", "L3", "L4"]


def test_preflight_reports_unavailable_cuda_without_loading_a_model(cfg, tmp_path, monkeypatch):
    cfg["metrics"]["alignment"]["enabled"] = False
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.6.0", cuda=SimpleNamespace(is_available=lambda: False)
    ))
    source = tmp_path / "source.csv"
    pd.DataFrame([{"prompt_id": "p1", "prompt": "a landscape"}]).to_csv(source, index=False)
    report = check_environment(cfg, ["fake"], source, device="cuda")
    assert report["status"] == "error"
    errors = [c for c in report["checks"] if c["status"] == "error"]
    assert len(errors) == 1
    assert "allocated GPU" in errors[0]["detail"]


def test_preflight_detects_lfs_input_before_running_a_pilot(cfg, tmp_path, monkeypatch):
    cfg["metrics"]["alignment"]["enabled"] = False
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.6.0"))
    path = tmp_path / "pointer.csv"
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    report = check_environment(cfg, ["fake"], path, device="cpu")
    assert report["status"] == "error"
    assert "Git LFS pointer" in report["checks"][0]["detail"]


def test_partial_and_sharded_model_downloads_are_detected(cfg, tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Transformer2DModel"]}))
    model = {**cfg["models"]["fake"], "variant": None}
    with pytest.raises(ValueError, match="missing component"):
        _check_model_files(model, root)
    component = root / "transformer"
    component.mkdir()
    (component / "model-00001.safetensors").touch()
    (component / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001.safetensors", "b": "model-00002.safetensors"
    }}))
    with pytest.raises(ValueError, match="incomplete weight shards"):
        _check_model_files(model, root)
    (component / "model-00002.safetensors").touch()
    _check_model_files(model, root)


def test_prefetch_fetches_only_selected_models_and_alignment(cfg, tmp_path, monkeypatch):
    cfg["models"]["selected"] = {**cfg["models"]["fake"], "backend": "diffusers", "repo_id": "test/selected"}
    cfg["models"]["other"] = {**cfg["models"]["selected"], "repo_id": "test/other"}
    calls = []

    def download(repo_id, **kwargs):
        calls.append(repo_id)
        folder = Path(kwargs.get("local_dir", tmp_path / "hf-cache"))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model_index.json").write_text("{}")
        return str(folder)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    paths = prefetch(cfg, ["selected"])
    assert calls == ["test/selected", cfg["metric_models"][cfg["metrics"]["alignment"]["model"]]["repo_id"]]
    assert "other" not in paths


@pytest.mark.parametrize("preflight_exit,expected_calls", [(0, 2), (4, 1)])
def test_gpu_wrapper_checks_before_run_and_propagates_failures(tmp_path, preflight_exit, expected_calls):
    interpreter = tmp_path / "fake-python"
    interpreter.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
with open(os.environ["CALLS"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1].endswith("benchmark_cluster.py"):
    if int(os.environ["CHECK_EXIT"]):
        sys.exit(int(os.environ["CHECK_EXIT"]))
    Path(sys.argv[sys.argv.index("--output") + 1]).write_text('{"status":"ok"}')
    sys.exit(0)
Path(sys.argv[sys.argv.index("--output") + 1]).mkdir(parents=True)
sys.exit(3)
''')
    interpreter.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    output = tmp_path / "output with spaces"
    env = {**os.environ, "VS_BENCH_PYTHON": str(interpreter), "VS_BENCH_MODE": "fit",
           "VS_BENCH_OUTPUT": str(output), "CHECK_EXIT": str(preflight_exit), "CALLS": str(calls)}
    result = subprocess.run(["bash", str(ROOT / "scripts/benchmark_tc2.sh"), "sd-xl", "sd-1.5"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == (preflight_exit or 3), result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == expected_calls
    assert "--require-cache" in commands[0]
    if not preflight_exit:
        assert "--generation-only" in commands[1]
        assert (output / "preflight.json").is_file()
