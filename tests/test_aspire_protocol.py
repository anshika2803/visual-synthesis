"""Scheduling, native-precision isolation and fail-closed validation without a GPU."""

import builtins
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from visual_synthesis.config import get_model, load_config

_spec = importlib.util.spec_from_file_location(
    "aspire_protocol_runner", Path(__file__).resolve().parents[1] / "scripts/aspire_protocol.py"
)
ap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ap)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # configure() intentionally sets inherited variables for real PBS children.
    # Keep each local test's environment changes out of the rest of the suite.
    monkeypatch.setattr(ap.os, "environ", ap.os.environ.copy())


@pytest.fixture
def paths(tmp_path, monkeypatch):
    for name, value in {
        "VS_ASPIRE_PROJECT": "test-project",
        "VS_ASPIRE_BUDGET_GIB": "500",
        "VS_ASPIRE_PYTHON": sys.executable,
        "VS_ASPIRE_GIT": "/usr/bin/git",
    }.items():
        monkeypatch.setenv(name, value)
    p = ap.layout(tmp_path, "qwen-image")
    p["logs"].mkdir(parents=True)
    ap.configure(p)
    return p


def test_native_overrides_preserve_scientific_settings():
    base = load_config([ap.ROOT / "config" / n for n in ap.CONFIG_NAMES[:-1]], use_env=False)
    aspire = load_config([ap.ROOT / "config" / n for n in ap.CONFIG_NAMES], use_env=False)
    for name in ap.FULL_MODELS | ap.REDUCED_MODELS:
        before, after = base["models"][name], aspire["models"][name]
        for key in (
            "repo_id",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt",
            "generation_kwargs",
            "hook",
            "dtype",
        ):
            assert before[key] == after[key], (name, key)
        assert after["quantization"] is None
        assert after["enable_cpu_offload"] and after["device_map"] is None


@pytest.mark.parametrize("model", sorted(ap.FULL_MODELS | ap.REDUCED_MODELS))
def test_model_workspace_and_tiers(tmp_path, model):
    p = ap.layout(tmp_path, model)
    reduced = model in ap.REDUCED_MODELS
    assert p["tier"] == ("reduced" if reduced else "full")
    assert ap.TABLES[p["core_table"]][0] == (438 if reduced else 874)
    assert ap.TABLES[p["e46_table"]][0] == (198 if reduced else 392)
    assert p["e4"].name == p["report"].name == model
    assert p["work"] != ap.layout(tmp_path, "new-model")["work"]


def test_frozen_tier_cannot_be_changed(tmp_path):
    with pytest.raises(ValueError, match="frozen protocol"):
        ap.layout(tmp_path, "qwen-image", "full")
    assert ap.layout(tmp_path, "future-model", "reduced")["tier"] == "reduced"
    with pytest.raises(ValueError, match="filesystem-safe"):
        ap.layout(tmp_path, "../another-run")


def test_submission_exports_selection_without_credentials(paths, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-must-not-appear")
    command, env = ap.submission(paths, "all", 21600, after="123.pbs101", index=2, max_jobs=8)
    assert env["VS_PROTOCOL_MODEL"] == "qwen-image"
    assert env["VS_PROTOCOL_TIER"] == "reduced"
    assert command[command.index("-W") + 1] == "depend=afterok:123.pbs101"
    assert command[command.index("-l") + 1] == "walltime=06:00:00"
    assert command[-1].endswith("/scripts/aspire_protocol.pbs")
    assert "HF_TOKEN" not in env and "secret-must-not-appear" not in str(command)
    with pytest.raises(ValueError, match="dependency"):
        ap.submission(paths, "all", 21600, after="123; whoami")
    with pytest.raises(ValueError, match="seconds/job"):
        ap.submission(paths, "all", 300)


def test_stale_environment_is_replaced(paths):
    cfg = load_config(ap.configs(paths))
    assert cfg["generate"]["model"] == paths["model"]
    assert cfg["paths"]["runs"] == str(paths["runs"])
    assert cfg["run"]["seed"] == 42


@pytest.mark.parametrize("component", ["unet", "transformer"])
def test_cache_uses_model_index_not_qwen_components(tmp_path, component):
    ap.save(
        tmp_path / "model_index.json",
        {
            component: ["diffusers", "Model"],
            "tokenizer": ["transformers", "Tokenizer"],
            "scheduler": ["diffusers", "Scheduler"],
            "safety_checker": ["diffusers", "SafetyChecker"],
        },
    )
    for folder in (component, "tokenizer", "scheduler"):
        (tmp_path / folder).mkdir()
        ap.save(tmp_path / folder / "config.json", {})
    weight = tmp_path / component / "diffusion_pytorch_model.fp16.safetensors"
    weight.write_bytes(b"native weights")
    assert ap.snapshot_complete(tmp_path, generator=True)
    weight.unlink()
    (weight.parent / "model-00001-of-00002.safetensors").write_bytes(b"partial")
    assert not ap.snapshot_complete(tmp_path, generator=True)
    ap.save(
        weight.parent / "model.safetensors.index.json",
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "missing.safetensors"}},
    )
    assert not ap.snapshot_complete(tmp_path, generator=True)


def test_download_selection_avoids_duplicate_weights():
    names = [
        "model_index.json",
        "transformer/model.safetensors",
        "transformer/pytorch_model.bin",
        "model.safetensors",
        "onnx/model.onnx",
        "tokenizer/vocab.json",
    ]
    files = ap.select_snapshot_files([SimpleNamespace(rfilename=n, size=1) for n in names], ["*"])
    assert set(files) == {"model_index.json", "transformer/model.safetensors", "tokenizer/vocab.json"}


@pytest.mark.parametrize("has_safetensors", [False, True])
def test_prepare_downloads_clip_weights_and_stays_online(paths, monkeypatch, has_safetensors):
    generator = paths["models"] / "qwen-image"
    ap.save(generator / "model_index.json", {"transformer": ["diffusers", "Model"]})
    ap.save(generator / "transformer/config.json", {})
    (generator / "transformer/model.safetensors").write_bytes(b"weights")
    detector = paths["models"] / "clip-h14"
    for name in ("config.json", "preprocessor_config.json"):
        ap.save(detector / name, {})
    for name in ("model.safetensors", "h14_nsfw.pth"):
        (detector / name).write_bytes(b"weights")
    packaged = paths["work"] / "nudenet"
    packaged.mkdir()
    (packaged / "model.onnx").write_bytes(b"weights")
    monkeypatch.setitem(sys.modules, "nudenet", SimpleNamespace(__file__=str(packaged / "__init__.py")))
    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        get_device_name=lambda: "test GPU",
        get_device_properties=lambda _: SimpleNamespace(total_memory=40 * ap.GIB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="test")))
    monkeypatch.setattr(ap, "disk_guard", lambda *args: {})
    monkeypatch.setattr(ap.subprocess, "check_output", lambda *args, **kwargs: "test driver\n")
    names = [
        "config.json", "preprocessor_config.json", "vocab.json", "merges.txt",
        "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "pytorch_model.bin",
    ]
    if has_safetensors:
        names.append("model.safetensors")
    downloaded = []

    class Hub:
        def model_info(self, repo, files_metadata):
            assert ap.os.environ["HF_HUB_OFFLINE"] == "0"
            assert ap.os.environ["TRANSFORMERS_OFFLINE"] == "0"
            return SimpleNamespace(sha="test-revision", siblings=[SimpleNamespace(rfilename=n, size=1) for n in names])

    def download(repo, revision, local_dir, allow_patterns):
        assert revision == "test-revision"
        downloaded.append((repo, set(allow_patterns)))
        for name in allow_patterns:
            if name.endswith(".json"):
                ap.save(local_dir / name, {})
            else:
                (local_dir / name).write_bytes(b"weights or tokenizer")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Hub, snapshot_download=download))
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            assert ap.os.environ["HF_HUB_OFFLINE"] == "0"
            assert ap.os.environ["TRANSFORMERS_OFFLINE"] == "0"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    ap.prepare_resources(paths, 500, allow_download=True)
    assert [repo for repo, _ in downloaded] == ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"]
    expected_weight = "model.safetensors" if has_safetensors else "pytorch_model.bin"
    for _, selected in downloaded:
        assert expected_weight in selected
        assert len(selected & {"model.safetensors", "pytorch_model.bin"}) == 1
    assert all(p["reused"] is False for p in ap.read(paths["work"] / "resources.json")["plans"][-2:])
    # Production is a separate child configured offline, including after a
    # readiness job downloaded multiple previously missing evaluators.
    ap.configure(paths)
    assert ap.os.environ["HF_HUB_OFFLINE"] == ap.os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_step_resumes_and_revalidates(paths):
    runner = ap.Runner(paths, 1000)
    output = paths["work"] / "result.txt"
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(output)!r}).write_text('done')"]
    runner.step("small-task", command, lambda: output.read_text() == "done")
    resumed = ap.Runner(paths, 1000)
    resumed.step("small-task", ["command-that-must-not-run"], lambda: output.read_text())
    output.unlink()
    with pytest.raises(FileNotFoundError):
        resumed.step("small-task", command, lambda: output.read_text())


def test_failed_child_is_never_marked_complete(paths):
    runner = ap.Runner(paths, 1000)
    with pytest.raises(RuntimeError, match="exited 7"):
        runner.step("bad-task", [sys.executable, "-c", "raise SystemExit(7)"], lambda: None)
    assert "bad-task" not in runner.state["done"]


def test_timeout_terminates_child_and_preserves_checkpoint(paths, monkeypatch):
    class Child:
        pid = 999999
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("generation", timeout)
            return 0

    killed = []
    monkeypatch.setattr(ap.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(ap.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    runner = ap.Runner(paths, 1000)
    with pytest.raises(ap.Checkpoint):
        runner.step("generation", ["fake"], lambda: None)
    assert killed == [(999999, signal.SIGTERM)]
    assert not runner.state["done"]
    runner.deadline = time.monotonic() + 10
    with pytest.raises(ap.Checkpoint):
        runner.step("next", ["fake"], lambda: None)


@pytest.mark.parametrize("model", ["qwen-image", "sd-1.5"])
def test_experiment_names_counts_and_e6_selection(tmp_path, monkeypatch, model):
    p = ap.layout(tmp_path, model)
    p["logs"].mkdir(parents=True)
    ap.configure(p)
    runner = ap.Runner(p, 1000)
    generations, commands = [], []
    monkeypatch.setattr(runner, "generation", lambda *a, **k: generations.append((a, k)))
    monkeypatch.setattr(runner, "step", lambda *a: commands.append(a))
    monkeypatch.setattr(ap, "check_analysis", lambda p: None)
    monkeypatch.setattr(ap, "read_mode", lambda p: "steering")
    for phase in ("e1", "e7", "e4", "e5"):
        runner.phase(phase)
    runner.state["done"].append("e5-mode")
    runner.phase("e6")
    assert len(generations) == 15
    assert generations[0][0] == (f"e1-{model}", p["core_table"])
    assert generations[1][0] == (f"e7-{model}", "e7_v1")
    assert generations[2] == ((f"e4-{model}-capture", p["e46_table"]), {"capture": True})
    assert all(a[0].startswith(f"e6-{model}-steering-") for a, _ in generations[-4:])
    assert {k["condition"][2] for _, k in generations[-4:]} == {0.25, 0.75}


def test_validation_filter_is_passed_without_shell_eval(paths, monkeypatch):
    runner = ap.Runner(paths, 1000)
    steps = []
    monkeypatch.setattr(runner, "step", lambda *args: steps.append(args))
    monkeypatch.setattr(runner, "evaluate", lambda *args: None)
    monkeypatch.setattr(ap, "expected_table", lambda *a, **k: pd.DataFrame())
    runner.generation("e5-qwen-image-steering-safety-a0.5", paths["e46_table"], condition=("steering", "safety", 0.5))
    command = steps[0][1]
    assert 'stages.generate.filters={"subset_split":"validation"}' in command
    assert "generate.intervention.vector=safety" in command
    assert command[command.index("--mode") + 1] == "incremental"


def test_missing_verdict_and_changed_seed_are_rejected(paths):
    expected = pd.DataFrame([{"prompt_id": "a", "prompt": "a scene"}])
    target = paths["work"] / "detections"
    target.mkdir()
    (target / "detections.jsonl").write_text(
        json.dumps({"prompt_id": "a", "status": "success", "flagged": None}) + "\n"
    )
    with pytest.raises(ValueError, match="unknown verdict"):
        ap.check_detections(target, expected)
    model = get_model(load_config([ap.ROOT / "config" / n for n in ap.CONFIG_NAMES], use_env=False), "qwen-image")
    image = target / "a.png"
    image.write_bytes(b"image")
    row = {
        "prompt_id": "a",
        "prompt": "a scene",
        "model": "qwen-image",
        "generation_status": "success",
        "seed": 43,
        "precision": "bfloat16",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "image_path": str(image),
    }
    (target / "generations.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="seed or precision"):
        ap.check_generations(target, expected, model)


def test_continuation_preserves_model_and_is_bounded(paths, monkeypatch):
    for name, value in {
        "VS_ASPIRE_STORAGE": str(paths["storage"]),
        "VS_PROTOCOL_MODEL": paths["model"],
        "VS_PROTOCOL_TIER": paths["tier"],
        "VS_PROTOCOL_PHASE": "all",
        "VS_PROTOCOL_WALL_SECONDS": "21600",
        "VS_PROTOCOL_JOB_INDEX": "2",
        "VS_PROTOCOL_MAX_JOBS": "3",
        "PBS_JOBID": "123.pbs101",
    }.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(ap, "run", lambda *args: 75)
    monkeypatch.setattr(ap, "submit", lambda *args: calls.append(args) or "124.pbs101")
    assert ap.main(["run"]) == 0
    assert calls[0][0]["model"] == "qwen-image"
    assert calls[0][3:] == ("123.pbs101", 3, 3)
    monkeypatch.setenv("VS_PROTOCOL_JOB_INDEX", "3")
    assert ap.main(["run"]) == 1
    assert len(calls) == 1


def test_failure_does_not_submit_continuation(paths, monkeypatch):
    for name, value in {
        "VS_ASPIRE_STORAGE": str(paths["storage"]),
        "VS_PROTOCOL_MODEL": paths["model"],
        "VS_PROTOCOL_TIER": paths["tier"],
        "VS_PROTOCOL_PHASE": "all",
        "VS_PROTOCOL_WALL_SECONDS": "21600",
    }.items():
        monkeypatch.setenv(name, value)

    def fail(*args):
        raise RuntimeError("CUDA OOM")

    monkeypatch.setattr(ap, "run", fail)
    monkeypatch.setattr(ap, "submit", lambda *args: pytest.fail("must not resubmit a real failure"))
    assert ap.main(["run"]) == 1


def test_production_requires_readiness_before_qsub(paths, monkeypatch):
    monkeypatch.setattr(ap, "verify_code", lambda: None)
    monkeypatch.setattr(ap, "verify_inputs", lambda paths: None)
    monkeypatch.setattr(ap.subprocess, "check_output", lambda *a, **k: pytest.fail("must not submit"))
    with pytest.raises(ValueError, match="readiness job"):
        ap.submit(paths, "all", 21600)


@pytest.mark.parametrize("model", ["qwen-image", "sd-1.5"])
def test_real_pipeline_cli_with_fake_backends(tmp_path, monkeypatch, model):
    """Exercise generated CLI arguments, context joins and strict checks end to end."""
    from click.testing import CliRunner

    from visual_synthesis.cli import cli
    from visual_synthesis.stages import s07_metrics

    p = ap.layout(tmp_path, model)
    p["logs"].mkdir(parents=True)
    ap.configure(p)
    toy = p["work"] / "test-backends.json"
    ap.save(
        toy,
        {
            "models": {model: {"backend": "fake", "width": 64, "height": 64, "num_inference_steps": 2}},
            "detectors": {name: {"backend": "fake"} for name in ("nudenet", "clip-h14")},
        },
    )
    base = ap.configs(p)
    monkeypatch.setattr(ap, "configs", lambda paths: [*base, toy])

    def scores(frame, model, batch_size):
        rows = s07_metrics.existing_images(frame).assign(score=0.8)
        return rows, {"status": "ok"}

    monkeypatch.setattr(s07_metrics, "measure_alignment", scores)
    monkeypatch.setattr(s07_metrics, "measure_quality", scores)
    monkeypatch.setattr(s07_metrics, "plot_overview", lambda *args: [])
    runner = ap.Runner(p, 1000)
    calls = []

    def execute(name, command, check):
        result = CliRunner().invoke(cli, command[3:])
        assert result.exit_code == 0, (result.output, result.exception)
        check()
        calls.append(name)

    monkeypatch.setattr(runner, "step", execute)
    runner.generation(f"pilot-fit-{model}", p["core_table"], limit=2)
    assert len(calls) == 4  # generate, each detector, metrics
    output = pd.read_parquet(p["runs"] / "s07_metrics" / f"pilot-fit-{model}" / "image_metrics.parquet")
    assert len(output) == 2 and output.model.eq(model).all()


def tc2_package_mock(monkeypatch, changes=None):
    versions = ap.reference_versions()
    changes = changes or {}

    def version(name):
        if name in changes:
            if changes[name] is None:
                raise ap.importlib.metadata.PackageNotFoundError(name)
            return changes[name]
        return versions[name.lower().replace("_", "-")]

    monkeypatch.setattr(ap.importlib.metadata, "version", version)


def test_setup_reports_protocol_match_and_expected_execution_differences(paths, monkeypatch):
    tc2_package_mock(monkeypatch)
    report = ap.setup_report(paths, quiet=True)
    assert report["protocol_ok"] and report["tc2_required_versions_match"]
    assert all(v["match"] for v in report["inputs"].values())
    assert report["execution_differences"]["quantization"] == {"tc2": "fp8", "here": None}
    assert report["execution_differences"]["enable_cpu_offload"] == {"tc2": False, "here": True}
    assert report["detectors"]["clip-h14"]["threshold"] == 0.55
    assert report["checkpoint_revision_comparison"].startswith("unverified")
    assert not report["gpu_fit_checked"]


def test_setup_distinguishes_version_difference_from_missing_dependency(paths, monkeypatch):
    tc2_package_mock(monkeypatch, {"diffusers": "0.41.0"})
    report = ap.setup_report(paths, quiet=True)
    assert report["protocol_ok"] and not report["tc2_required_versions_match"]
    assert report["packages"]["diffusers"]["here"] == "0.41.0"
    tc2_package_mock(monkeypatch, {"scikit-learn": None})
    report = ap.setup_report(paths, quiet=True)
    assert not report["protocol_ok"]
    assert "Missing runtime dependency: scikit-learn" in report["issues"]


def test_setup_compares_with_immutable_tc2_settings(paths, monkeypatch):
    tc2_package_mock(monkeypatch)
    overlay = paths["work"] / "bad-settings.json"
    ap.save(overlay, {"models": {"qwen-image": {"num_inference_steps": 20}}})
    original = ap.configs(paths)
    monkeypatch.setattr(ap, "configs", lambda p: [*original, overlay])
    report = ap.setup_report(paths, quiet=True)
    assert not report["protocol_ok"]
    assert report["scientific_settings"]["num_inference_steps"] == {"tc2": 50, "here": 20, "match": False}


def test_strict_setup_cli_rejects_package_differences(paths, monkeypatch):
    tc2_package_mock(monkeypatch, {"torch": "2.6.0+cu118"})
    monkeypatch.setenv("VS_ASPIRE_STORAGE", str(paths["storage"]))
    assert ap.main(["check", "--model", "qwen-image"]) == 0
    assert ap.main(["check", "--model", "qwen-image", "--strict"]) == 2
