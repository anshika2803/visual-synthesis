"""Pairing, real quantization API compatibility, isolated caches and PBS routing."""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from visual_synthesis.io import file_sha256  # noqa: E402
from visual_synthesis.qwen_quantization import (  # noqa: E402
    checkpoint_snapshot,
    compare_reports,
    generation_worker,
    paired_outputs,
    prepare_pairs,
    quantization_config,
)

ROOT = Path(__file__).resolve().parents[1]


def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "qwen_quantization_driver_test", ROOT / "scripts/aspire_qwen_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_table(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame({"prompt_id": [f"p{i}" for i in range(8)], "prompt": [f"prompt {i}" for i in range(8)],
                  "benchmark_level": ["L0", "L1"] * 4, "evaluation_seed": [7, None, 11, 13, 17, 19, 23, 29],
                  "case_number": [None] * 8}).to_csv(source, index=False)
    return source


def reports():
    common = {"status": "success", "images": 2, "sample_sha256": "prompts", "settings_sha256": "settings",
              "checkpoint_fingerprint": "checkpoint", "offload_policy": "model_cpu_offload"}
    return {"bf16": {**common, "mean_inference_seconds": 60, "mean_inference_and_png_seconds": 62,
                     "nf4_linear_count": 0},
            "nf4": {**common, "mean_inference_seconds": 30, "mean_inference_and_png_seconds": 31,
                    "nf4_linear_count": 4}}


def test_pairs_preserve_seeds_and_freeze_identical_repeatable_rows(tmp_path):
    source = source_table(tmp_path)
    first = prepare_pairs(source, tmp_path / "a", count=8)
    second = prepare_pairs(source, tmp_path / "b", count=8)
    assert first["sample_sha256"] == second["sample_sha256"]
    frame = pd.read_csv(tmp_path / "a/prompts.csv").set_index("prompt_id")
    assert frame.loc["p0", "evaluation_seed"] == 7
    assert frame.loc["p1", "evaluation_seed"] >= 0
    assert first["sampling"]["missing_sample_levels"]


def test_checkpoint_snapshot_detects_change_and_ignores_download_locks(tmp_path):
    (tmp_path / "model_index.json").write_text('{}')
    (tmp_path / "weights.safetensors").write_bytes(b"fake")
    first = checkpoint_snapshot(tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "lock").touch()
    assert checkpoint_snapshot(tmp_path)["fingerprint"] == first["fingerprint"]
    (tmp_path / "weights.safetensors").write_bytes(b"different")
    assert checkpoint_snapshot(tmp_path)["fingerprint"] != first["fingerprint"]


def test_real_pinned_diffusers_nf4_config_uses_transformer_only(monkeypatch):
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "0.48.2" if name == "bitsandbytes" else original(name))
    config = quantization_config()
    assert set(config.quant_mapping) == {"transformer"}
    settings = config.quant_mapping["transformer"]
    assert settings.load_in_4bit and settings.bnb_4bit_quant_type == "nf4"
    assert settings.bnb_4bit_compute_dtype == torch.bfloat16 and settings.bnb_4bit_use_double_quant


@pytest.mark.parametrize("key", ["sample_sha256", "settings_sha256", "checkpoint_fingerprint", "offload_policy"])
def test_comparison_rejects_protocol_mismatch(key):
    values = reports()
    values["nf4"][key] = "different"
    with pytest.raises(ValueError, match=key):
        compare_reports(values, 2)


def test_comparison_never_turns_failure_or_partial_run_into_speedup():
    values = reports()
    result = compare_reports(values, 2)
    assert result["inference_speedup_bf16_over_nf4"] == 2
    values["nf4"]["status"] = "failed"
    assert "inference_speedup_bf16_over_nf4" not in compare_reports(values, 2)
    values["nf4"]["status"] = "success"
    values["nf4"]["images"] = 1
    assert compare_reports(values, 2)["status"] == "incomplete"


def test_cache_isolation_does_not_reuse_shared_auth_or_mutable_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    module = driver(monkeypatch)
    for key in ("HF_HOME", "HF_TOKEN_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.setenv(key, "/shared/legacy")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    module.isolate_caches(tmp_path)
    assert os.environ["HF_HOME"].startswith(str(tmp_path))
    assert os.environ["HF_TOKEN_PATH"].startswith(str(tmp_path))
    assert os.environ["HF_TOKEN"] == ""
    assert os.environ["TRITON_CACHE_DIR"].startswith(str(tmp_path))
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_worker_generation_matches_prompts_seeds_and_saves_partial_progress(tmp_path, monkeypatch):
    from PIL import Image

    import visual_synthesis.benchmark as benchmark
    import visual_synthesis.qwen_quantization as module

    source = source_table(tmp_path)
    manifest = prepare_pairs(source, tmp_path, count=2)
    seen = {}

    class FakeLinear:
        weight = SimpleNamespace(quant_type="nf4")

    class FakeResources:
        peaks = {"rss_process_lifetime_peak_bytes": 100, "cuda_allocated_peak_bytes": 200,
                 "cuda_reserved_peak_bytes": 300}

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def sync(self):
            pass

    class FakeBackend:
        def __init__(self, arm):
            self.arm = arm
            self.pipe = SimpleNamespace(transformer=SimpleNamespace(
                modules=lambda: [FakeLinear()] if arm == "nf4" else []))
            seen[arm] = []

        def generate(self, requests):
            seen[self.arm].append(requests[0].copy())
            return [{"image": Image.new("RGB", (8, 8), (requests[0]["seed"] % 255, 0, 0))}]

        def unload(self):
            pass

    monkeypatch.setenv("PBS_JOBID", "123.pbs101")
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(nn=SimpleNamespace(Linear4bit=FakeLinear)))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.48.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 200)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 300)
    monkeypatch.setattr(benchmark, "Resources", FakeResources)
    monkeypatch.setattr(benchmark, "machine_info", lambda *_: {"packages": {}})
    monkeypatch.setattr(module, "build_backend", lambda spec, arm: FakeBackend(arm))
    values = {}
    for arm in ("bf16", "nf4"):
        spec = {"arm": arm, "output": str(tmp_path / arm), "model": {}, "sample": str(tmp_path / "prompts.csv"),
                "sample_sha256": manifest["sample_sha256"], "settings_sha256": "same",
                "checkpoint_fingerprint": "same"}
        values[arm] = generation_worker(spec)
        assert values[arm]["status"] == "success", values[arm]
        assert json.loads((tmp_path / arm / "report.json").read_text())["images"] == 2
    assert seen["bf16"] == seen["nf4"] and len(seen["bf16"]) == 3  # warmup plus two measured requests
    assert compare_reports(values, 2)["status"] == "complete"
    pairs = paired_outputs(tmp_path)
    assert len(pairs) == 2 and all(row["identical_pixels"] for row in pairs)
    assert (tmp_path / "comparison.html").is_file()
    assert file_sha256(tmp_path / "prompts.csv") == manifest["sample_sha256"]


@pytest.mark.parametrize("seconds", ["6300", "7201"])
def test_submission_uses_short_normal_queue_without_inheriting_long_job_dependency(tmp_path, seconds):
    git = tmp_path / "git"
    git.write_text('#!/bin/bash\nprintf "swara/experiments\\n"\n')
    git.chmod(0o755)
    qsub = tmp_path / "qsub"
    qsub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
                    'printf "%s\\n" "$VS_QWEN_ORDER" >> "$CAPTURE_ARGS"\nprintf "124.pbs101\\n"\n')
    qsub.chmod(0o755)
    model = tmp_path / "runtime/models/qwen-image"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}")
    capture = tmp_path / "args"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "VS_ASPIRE_PROJECT": "test-project",
           "VS_ASPIRE_BUDGET_GIB": "500", "VS_ASPIRE_STORAGE": str(tmp_path / "runtime"),
           "VS_ASPIRE_PYTHON": sys.executable, "VS_ASPIRE_GIT": str(git), "VS_QWEN_WALL_SECONDS": seconds,
           "VS_PILOT_AFTER": "26101874.pbs101", "VS_ASPIRE_QUEUE": "ai", "VS_QWEN_ORDER": "nf4-first",
           "HF_TOKEN": "fake-secret", "CAPTURE_ARGS": str(capture)}
    p = subprocess.run(["bash", str(ROOT / "scripts/submit_aspire_qwen_quantization.sh")],
                       env=env, capture_output=True, text=True)
    if seconds == "7201":
        assert p.returncode != 0 and not capture.exists()
        return
    assert p.returncode == 0, p.stderr
    args = capture.read_text()
    assert "-q\nnormal\n" in args and "walltime=01:45:00" in args
    assert "depend=" not in args and "HF_TOKEN" not in args and "fake-secret" not in args
    assert "nf4-first" in args and "qwen-quant-" in args
    assert p.stdout.strip() == "124.pbs101"


@pytest.mark.parametrize("action", ["run", "check", "evaluate"])
def test_compute_commands_require_pbs_before_loading_any_model(tmp_path, action):
    env = {key: value for key, value in os.environ.items() if key != "PBS_JOBID"}
    p = subprocess.run([sys.executable, str(ROOT / "scripts/aspire_qwen_quantization.py"), action],
                       env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "requires PBS" in p.stderr


@pytest.mark.parametrize("order", ["bf16-first", "nf4-first"])
def test_driver_prepares_private_dependency_and_freezes_worker_specs(tmp_path, monkeypatch, order):
    import visual_synthesis.qwen_quantization as workers

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("PBS_JOBID", "124.pbs101")
    module = driver(monkeypatch)
    root = tmp_path / "runtime"
    model = root / "models/qwen-image"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}")
    out = root / "outputs/qwen-quant-test"
    snapshot = {"paths": {"workspace": {"usage": {"allocated_bytes": 0},
                                       "filesystem_free_bytes": 10 * 2**30, "filesystem_free_inodes": 10000}}}
    monkeypatch.setattr(module, "storage_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(module, "accounting", lambda _: {"status": "success"})
    monkeypatch.setattr(module, "git_command", lambda *_: {"stdout": "test-commit\n"})
    monkeypatch.setattr(workers, "paired_outputs", lambda _: [None, None])
    seen = []

    def fake_run(command, log, timeout):
        Path(log).touch()
        assert timeout > 0
        if "pip" in command:
            assert "--no-deps" in command and "--only-binary=:all:" in command
            target = Path(command[command.index("--target") + 1])
            assert target == out / "dependencies"
            target.mkdir()
        elif "worker" in command:
            spec = json.loads(Path(command[-1]).read_text())
            seen.append(spec)
            arm_out = Path(spec["output"])
            arm_out.mkdir()
            report = reports()[spec["arm"]]
            report.update({key: spec[key] for key in ("sample_sha256", "settings_sha256", "checkpoint_fingerprint")})
            (arm_out / "report.json").write_text(json.dumps(report))
        else:
            assert "check" in command
        return 0

    monkeypatch.setattr(module, "run_logged", fake_run)
    args = SimpleNamespace(storage=str(root), output=out, source=None, input=source_table(tmp_path),
                           sample_size=2, seed=42, wall_seconds=6300, budget_gib=500, order=order)
    result = module.run(args)
    assert result["status"] == "complete", result
    assert [spec["arm"] for spec in seen] == (["nf4", "bf16"] if order == "nf4-first" else ["bf16", "nf4"])
    for key in ("sample", "sample_sha256", "settings_sha256", "checkpoint_fingerprint", "model", "source"):
        assert seen[0][key] == seen[1][key]
    assert not (root / ".aspire-trial.lock").exists()
    assert not (model / "dependencies").exists()
    assert str(out / "dependencies") in os.environ["PYTHONPATH"]
