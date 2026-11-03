"""Resource-pilot invariants that can be checked without model weights or a GPU."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from visual_synthesis.pilot_interventions import ActivationHooks, HookPoint, fit_direction, fit_probe  # noqa: E402
from visual_synthesis.project_pilot import (  # noqa: E402
    CONDITIONS,
    EVALUATORS,
    clip_iqa_scores,
    prepare_samples,
    project_costs,
)

ROOT = Path(__file__).resolve().parents[1]


def pilot_driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("pilot_prepare_test", ROOT / "scripts/aspire_project_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_flat_clip_evaluators_reuses_generator_and_caches(cfg, tmp_path, monkeypatch):
    """CLIP repos have config.json and root weights, not Diffusers model_index.json."""
    from types import SimpleNamespace

    script = pilot_driver(monkeypatch)
    models = Path(cfg["paths"]["models"])
    generator = models / "fake"
    (generator / "unet").mkdir(parents=True)
    (generator / "model_index.json").write_text(json.dumps({"unet": ["diffusers", "UNet2DConditionModel"]}))
    (generator / "unet/model.safetensors").write_bytes(b"cached generator")
    detector = models / "clip-h14"
    detector.mkdir()
    (detector / "h14_nsfw.pth").write_bytes(b"cached detector head")

    clip_repo = "openai/clip-vit-base-patch32"
    h14_repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    payloads = {
        clip_repo: {
            "config.json": b"{}",
            "preprocessor_config.json": b"{}",
            "vocab.json": b"{}",
            "merges.txt": b"merges",
            "pytorch_model.bin": b"clip weights",
        },
        h14_repo: {
            "config.json": b"{}",
            "preprocessor_config.json": b"{}",
            "model.safetensors": b"h14 weights",
            "pytorch_model.bin": b"duplicate binary",
        },
    }
    downloads = []

    def model_info(repo, **kwargs):
        assert kwargs["files_metadata"]
        return SimpleNamespace(
            sha="a" * 40,
            siblings=[SimpleNamespace(rfilename=name, size=len(data)) for name, data in payloads[repo].items()],
        )

    def download(repo, *, revision, local_dir, allow_patterns):
        assert revision == "a" * 40
        assert "model_index.json" not in allow_patterns
        downloads.append((repo, set(allow_patterns)))
        local_dir.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            (local_dir / name).write_bytes(payloads[repo][name])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=model_info), snapshot_download=download),
    )
    monkeypatch.setattr(
        script,
        "storage_snapshot",
        lambda *args, **kw: {
            "paths": {
                "workspace": {
                    "usage": {"allocated_bytes": 0, "stat_errors": []},
                    "filesystem_free_bytes": 100 * script.GIB,
                    "filesystem_free_inodes": 10000,
                }
            }
        },
    )
    out = tmp_path / "outputs/run"
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match="insufficient"):
        script.prepare_weights("fake", cfg, tmp_path, out, 0.5, True)
    assert not downloads  # Metadata is not permission to bypass the capacity guard.
    source = script.prepare_weights("fake", cfg, tmp_path, out, 100, True)
    assert source == str(models / "openai--clip-vit-base-patch32")
    assert [repo for repo, _ in downloads] == [clip_repo, h14_repo]
    assert downloads[1][1] == {"config.json", "preprocessor_config.json", "model.safetensors"}
    assert (generator / "unet/model.safetensors").read_bytes() == b"cached generator"
    downloads.clear()
    script.prepare_weights("fake", cfg, tmp_path, out, 100, False)
    assert not downloads


@pytest.mark.parametrize("problem", ["missing-config", "missing-weights", "unknown-size", "unsafe-path"])
def test_evaluator_selection_keeps_completeness_size_and_path_guards(monkeypatch, problem):
    from types import SimpleNamespace

    script = pilot_driver(monkeypatch)
    files = {"config.json": 2, "preprocessor_config.json": 2, "pytorch_model.bin": 100}
    if problem == "missing-config":
        del files["config.json"]
    elif problem == "missing-weights":
        del files["pytorch_model.bin"]
    elif problem == "unknown-size":
        files["pytorch_model.bin"] = None
    else:
        files["../escape.json"] = 20
    siblings = [SimpleNamespace(rfilename=name, size=size) for name, size in files.items()]
    with pytest.raises(ValueError):
        script.select_evaluator_files(
            siblings, ["*.json", "pytorch_model.bin"], ["config.json", "preprocessor_config.json"]
        )


def test_generator_still_requires_diffusers_index(cfg, monkeypatch):
    from types import SimpleNamespace

    script = pilot_driver(monkeypatch)
    cfg["models"]["fake"]["repo_id"] = "example/generator"
    flat = SimpleNamespace(
        sha="b" * 40,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10),
            SimpleNamespace(rfilename="model.safetensors", size=100),
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *args, **kwargs: flat)),
    )
    with pytest.raises(ValueError, match="Diffusers weight selection"):
        script.model_plan("fake", cfg)


@pytest.mark.parametrize("spatial", [False, True])
def test_centered_projection_removes_only_direction_component(spatial):
    module = torch.nn.Identity()
    x = torch.tensor([[[7.0, 3.0, 4.0], [9.0, 5.0, 6.0]]])
    if spatial:
        x = x.transpose(1, 2).unsqueeze(-1)
    point = HookPoint("block", module, None, "none")
    with ActivationHooks([point], "ablation", torch.tensor([1.0, 0.0, 0.0]), torch.tensor([2.0, 0.0, 0.0])):
        y = module(x)
    normalized = y.squeeze(-1).transpose(1, 2) if spatial else y
    assert torch.equal(normalized[..., 0], torch.full((1, 2), 2.0))
    assert torch.equal(normalized[..., 1:], torch.tensor([[[3.0, 4.0], [5.0, 6.0]]]))
    assert not module._forward_hooks


def test_alpha_zero_is_exact_identity_and_hooks_remove_on_exception():
    module = torch.nn.Identity()
    x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="stop"):
        with ActivationHooks([HookPoint("b", module, None, "none")], "ablation", alpha=0):
            assert module(x) is x
            raise RuntimeError("stop")
    assert not module._forward_hooks


@pytest.mark.parametrize("layout", ["sequential", "batch"])
def test_capture_pools_only_conditional_pass(layout):
    module = torch.nn.Identity()
    conditional = torch.full((1, 3, 4), 2.0)
    uncond = torch.full((1, 3, 4), 99.0)
    with ActivationHooks([HookPoint("b", module, None, layout)]) as hook:
        if layout == "sequential":
            module(conditional)
            module(uncond)
        else:
            module(torch.cat([uncond, conditional]))
        assert torch.equal(hook.features()["b"], torch.full((4,), 2.0))


def test_joint_sequence_keeps_text_tokens_and_other_tuple_stream():
    class Block(torch.nn.Module):
        def forward(self, x):
            return torch.full((1, 2, 3), 77.0), x

    module = Block()
    image = torch.tensor([[[5.0, 3.0, 4.0], [7.0, 3.0, 4.0], [91.0, 92.0, 93.0]]])
    point = HookPoint("joint", module, 1, "none", image_tokens=2)
    with ActivationHooks([point], "ablation", torch.tensor([1.0, 0.0, 0.0]), torch.zeros(3)):
        text, output = module(image)
    assert torch.equal(text, torch.full((1, 2, 3), 77.0))
    assert torch.equal(output[:, 2:], image[:, 2:])
    assert torch.equal(output[:, :2, 0], torch.zeros(1, 2))


def test_direction_and_cpu_probe_learn_from_discovery():
    x = np.array([[-3.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    center, d, scale = fit_direction(x, y)
    assert torch.allclose(center, torch.tensor([-2.5, 0.0]))
    assert torch.equal(d, torch.tensor([1.0, 0.0])) and scale == 5
    probe = fit_probe(x, y)
    predictions = ((torch.from_numpy(x) - probe["mean"]) / probe["std"]) @ probe["weight"] + probe["bias"]
    assert predictions[:2].lt(0).all() and predictions[2:].gt(0).all()
    with pytest.raises(ValueError, match="zero/nonfinite"):
        fit_direction(np.zeros((4, 2)), y)


def test_quality_score_favors_positive_anchor_and_is_scale_invariant():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scores = clip_iqa_scores(torch.tensor([[10.0, 0.0], [0.0, 10.0]]), anchors)
    assert scores[0] > 0.99 and scores[1] < 0.01


def test_sample_splits_and_seeds_are_shared_and_disjoint(tmp_path):
    rows = [
        {
            "prompt_id": str(i),
            "prompt": f"resource sample {i}",
            "benchmark_level": "L0" if i < 30 else "L1",
            "standard_category": "benign" if i < 30 else "sexual",
        }
        for i in range(60)
    ]
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    manifests = []
    for name in ("one", "two"):
        manifests.append(
            prepare_samples(source, tmp_path / name, baseline=12, discovery_per_class=2, validation_per_class=2)
        )
    assert manifests[0]["tables_sha256"] == manifests[1]["tables_sha256"]
    baseline = pd.read_csv(tmp_path / "one/baseline.csv")
    capture = pd.read_csv(tmp_path / "one/capture.csv")
    validation = pd.read_csv(tmp_path / "one/validation.csv")
    discovery = capture[capture.pilot_split.eq("discovery")]
    assert set(discovery.prompt_id).isdisjoint(baseline.prompt_id)
    merged = baseline.merge(validation, on="prompt_id")
    assert len(merged) == 4
    assert (merged.evaluation_seed_x == merged.evaluation_seed_y).all()
    assert len(baseline) == 12 and len(capture) == 8


def complete_reports():
    return {
        name: {"status": "success", "mean_seconds_per_image": 1.0, "fixed_seconds": 60.0, "cpu_learning_seconds": 2.0}
        for name in ["baseline", "capture", "learn", *CONDITIONS, *EVALUATORS]
    }


def test_project_counts_and_unknown_failures_are_not_zero_cost():
    reports = complete_reports()
    result = project_costs(reports)
    assert result["target_images"] == 10100
    assert result["generation"]["E5-safety-ablation"]["images"] == 600
    assert result["generation"]["E6-random-ablation"]["images"] == 600
    assert result["generation_hours"] == pytest.approx((10100 + 14 * 60) / 3600)
    assert result["evaluation_hours"] > 0
    reports["capture"] = {"status": "failed", "error": "CUDA out of memory"}
    result = project_costs(reports)
    assert result["status"] == "incomplete" and result["missing_successful_phases"] == ["capture"]
    assert "generation_plus_evaluation_hours" not in result


def test_projection_splits_long_stages_for_six_hour_jobs():
    reports = complete_reports()
    reports["baseline"]["mean_seconds_per_image"] = 90.0
    result = project_costs(reports)
    assert result["generation"]["baseline"]["jobs_estimated"] > 25


def test_generation_capture_learning_and_identity_workflow_without_weights(tmp_path, monkeypatch):
    """Exercise on-disk handoffs and hooks with a deterministic tiny CPU backend."""
    from types import SimpleNamespace

    from PIL import Image

    import visual_synthesis.backends.image as image_module
    import visual_synthesis.config as config_module
    from visual_synthesis.project_pilot import generation, learn

    class TupleStream(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(1, 1, 4), x

    class TinyBackend:
        def load(self):
            self.pipe = SimpleNamespace(
                _all_hooks=[object()],
                transformer=SimpleNamespace(transformer_blocks=torch.nn.ModuleList([TupleStream() for _ in range(5)])),
            )

        def generate(self, requests):
            outputs = []
            for row in requests:
                x = torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(row["seed"]))
                for _ in range(2):
                    for block in self.pipe.transformer.transformer_blocks:
                        _, x = block(x)
                        block(torch.zeros_like(x))  # Qwen's sequential negative pass.
                color = int(abs(float(x.sum())) * 10) % 255
                outputs.append({"image": Image.new("RGB", (4, 4), color=(color, color, color))})
            return outputs

        def unload(self):
            self.pipe = None

    class Probe:
        peaks = {}

        def sync(self):
            pass

    model = {"id": "qwen-image", "family": "qwen_image"}
    monkeypatch.setattr(config_module, "get_model", lambda *args: model.copy())
    monkeypatch.setattr(image_module, "build_image_backend", lambda *args: TinyBackend())
    rows = [
        {
            "prompt_id": str(i),
            "prompt": f"sample {i}",
            "evaluation_seed": i + 42,
            "pilot_split": "discovery" if i < 4 else "validation",
            "pilot_label": i % 2,
            "case_number": i if i < 4 else None,
        }
        for i in range(8)
    ]
    source = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    base = {"model": "qwen-image", "config": {"paths": {"models": str(tmp_path)}}, "sample": str(source)}
    for phase in ("baseline", "capture"):
        result = {}
        generation({**base, "phase": phase, "output": str(tmp_path / phase)}, result, Probe())
        assert result["images"] == 8
        if phase == "capture":
            assert result["activation_bytes"] > 0 and len(result["last_hook_calls"]) == 5
    learn_out = tmp_path / "learn"
    learn_out.mkdir()
    learned = {}
    learn({"capture": str(tmp_path / "capture"), "output": str(learn_out)}, learned, Probe())
    assert learned["discovery_images"] == 4 and learned["validation_images"] == 4
    for phase in ("alpha-zero", "safety-ablation"):
        result = {}
        generation(
            {
                **base,
                "phase": phase,
                "output": str(tmp_path / phase),
                "direction": str(learn_out / "direction.npz"),
                "baseline": str(tmp_path / "baseline"),
            },
            result,
            Probe(),
        )
        assert result["images"] == 8 and all(result["last_hook_calls"].values())
        if phase == "alpha-zero":
            assert result["alpha_zero_equivalent"] is True


def test_aggregate_keeps_latest_failure_visible(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("aspire_project_pilot_test", ROOT / "scripts/aspire_project_pilot.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for i, status in enumerate(("success", "failed")):
        path = tmp_path / "runs" / f"run-{i}" / "pilot_summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "model": "qwen-image",
                    "status": status,
                    "output": str(path.parent),
                    "projection": {"status": "incomplete"},
                    "time_utc": f"2026-11-0{i + 1}T00:00:00+00:00",
                }
            )
        )
        # Collecting old accounting can update mtime; select by run creation time.
        os.utime(path, (200 - i, 200 - i))
    result = script.aggregate(tmp_path, tmp_path / "aggregate.json")
    assert result["models"]["qwen-image"]["status"] == "failed"
    assert result["status"] == "incomplete"
    assert "generation_plus_evaluation_hours" not in result


@pytest.mark.parametrize("queue", ["normal", "ai", "g1"])
def test_submission_routes_pbs_output_and_exports_token_path_not_value(tmp_path, queue):
    git = tmp_path / "git"
    git.write_text('#!/bin/bash\nprintf "swara/experiments\\n"\n')
    git.chmod(0o755)
    qsub = tmp_path / "qsub"
    qsub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nprintf "12345.pbs101\\n"\n')
    qsub.chmod(0o755)
    capture = tmp_path / "args.txt"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "VS_ASPIRE_PROJECT": "test-project",
        "VS_ASPIRE_BUDGET_GIB": "500",
        "VS_ASPIRE_STORAGE": str(tmp_path / "workspace"),
        "VS_ASPIRE_PYTHON": sys.executable,
        "VS_ASPIRE_GIT": str(git),
        "VS_ASPIRE_QUEUE": queue,
        "CAPTURE_ARGS": str(capture),
        "HF_TOKEN_PATH": str(tmp_path / "token"),
        "VS_PILOT_AFTER": "12344.pbs101",
    }
    process = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_aspire_project_pilot.sh"), "qwen-image"],
        env=env,
        capture_output=True,
        text=True,
    )
    if queue == "g1":
        assert process.returncode != 0 and "normal or ai" in process.stderr
        assert not capture.exists()
        return
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "12345.pbs101"
    args = capture.read_text()
    assert f"-q\n{queue}\n" in args
    assert "depend=afterany:12344.pbs101" in args and "HF_TOKEN_PATH" in args
    assert str(tmp_path / "workspace/logs") in args and ".pbs.log" in args
    assert "qwen-image" in process.stderr


@pytest.mark.parametrize("credentials", ["legacy", "standard", "missing", "empty"])
def test_submission_normalizes_legacy_token_filename_without_exposing_it(tmp_path, credentials):
    git = tmp_path / "git"
    git.write_text('#!/bin/bash\nprintf "swara/experiments\\n"\n')
    git.chmod(0o755)
    qsub = tmp_path / "qsub"
    qsub.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$HF_TOKEN_PATH" > "$CAPTURE_TOKEN_PATH"\nprintf "12345.pbs101\\n"\n'
    )
    qsub.chmod(0o755)
    legacy = tmp_path / "hf_fake-legacy-filename"
    if credentials != "missing":
        legacy.write_text("" if credentials == "empty" else "fake-legacy-credential")
    canonical = tmp_path / "runtime/hf-home/token"
    if credentials == "standard":
        canonical.parent.mkdir(parents=True)
        canonical.write_text("fake-existing-credential")
    capture = tmp_path / "submitted-token-path"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "VS_ASPIRE_PROJECT": "test-project",
        "VS_ASPIRE_BUDGET_GIB": "500",
        "VS_ASPIRE_STORAGE": str(tmp_path / "runtime"),
        "VS_ASPIRE_PYTHON": sys.executable,
        "VS_ASPIRE_GIT": str(git),
        "HF_TOKEN_PATH": str(legacy),
        "CAPTURE_TOKEN_PATH": str(capture),
        "VS_PILOT_AFTER": "",
    }
    process = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_aspire_project_pilot.sh"), "qwen-image"],
        env=env,
        capture_output=True,
        text=True,
    )
    combined = process.stdout + process.stderr
    for sensitive in (legacy.name, "fake-legacy-credential", "fake-existing-credential"):
        assert sensitive not in combined
    if credentials in ("missing", "empty"):
        assert process.returncode != 0
        assert "hf auth login" in process.stderr
        assert not capture.exists() and not canonical.exists()
    else:
        assert process.returncode == 0, process.stderr
        assert capture.read_text().strip() == str(canonical)
        assert canonical.read_text() == (
            "fake-existing-credential" if credentials == "standard" else "fake-legacy-credential"
        )
        assert legacy.read_text() == "fake-legacy-credential"
        if credentials == "legacy":
            assert canonical.stat().st_mode & 0o777 == 0o600
        assert not list(canonical.parent.glob(".token-*"))


def test_aggregate_reports_extended_core_and_mixed_scopes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("pilot_scope_test", ROOT / "scripts/aspire_project_pilot.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for model in script.ORDER:
        path = tmp_path / "outputs" / model / "pilot_summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "status": "success",
                    "output": str(path.parent),
                    "projection": project_costs(complete_reports()),
                    "core_projection": project_costs(complete_reports(), baseline=1200),
                    "reduced_core_projection": project_costs(
                        complete_reports(), baseline=600, capture=250, validation=150
                    ),
                }
            )
        )
    result = script.aggregate(tmp_path, tmp_path / "result.json")
    assert result["status"] == "conditional"
    assert result["scenarios"]["extended_6000"]["images"] == 80800
    assert result["scenarios"]["full_core"]["images"] == 42400
    assert result["scenarios"]["mixed_core"]["images"] == 31800
    assert result["scenarios"]["mixed_core"]["evaluation_hours"] > 0


@pytest.mark.parametrize("selection", ["all", "remaining"])
def test_group_submission_chains_models_and_uses_one_runtime_tree(tmp_path, selection):
    git = tmp_path / "git"
    git.write_text('#!/bin/bash\nprintf "swara/experiments\\n"\n')
    git.chmod(0o755)
    qsub = tmp_path / "qsub"
    qsub.write_text("""#!/bin/bash
n=0
if [ -f "$JOB_COUNTER" ]; then read -r n < "$JOB_COUNTER"; fi
n=$((n+1))
printf '%s\\n' "$n" > "$JOB_COUNTER"
printf '%s\\n' "$@" >> "$CAPTURE_ARGS"
printf '%s\\n' "$HF_TOKEN_PATH" >> "$CAPTURE_ARGS"
printf '%s.pbs101\\n' "$((12345+n))"
""")
    qsub.chmod(0o755)
    token = tmp_path / "hf_fake-filename"
    token.write_text("not-a-real-token")
    capture = tmp_path / "args.txt"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "VS_ASPIRE_PROJECT": "test-project",
        "VS_ASPIRE_BUDGET_GIB": "500",
        "VS_ASPIRE_STORAGE": str(tmp_path / "runtime"),
        "VS_ASPIRE_PYTHON": sys.executable,
        "VS_ASPIRE_GIT": str(git),
        "CAPTURE_ARGS": str(capture),
        "JOB_COUNTER": str(tmp_path / "counter"),
        "HF_TOKEN_PATH": str(token),
        "VS_PILOT_AFTER": "12000.pbs101" if selection == "remaining" else "",
    }
    p = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_aspire_project_pilot.sh"), selection],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    expected = ["glm-image", "flux-dev", "sd-3.5-large", "z-image-turbo", "pixart-sigma", "sd-xl", "sd-1.5"]
    if selection == "all":
        expected.insert(0, "qwen-image")
    assert [line.split()[0] for line in lines] == expected
    count = len(expected)
    args = capture.read_text()
    assert args.count("depend=afterany:") == 7
    if selection == "remaining":
        assert "depend=afterany:12000.pbs101" in args
    for i in range(1, count):
        assert f"depend=afterany:{12345 + i}.pbs101" in args
    assert len(list((tmp_path / "runtime/outputs").iterdir())) == count
    assert args.count(".pbs.log") == count
    arguments = args.splitlines()
    destinations = [Path(arguments[i + 1]) for i, arg in enumerate(arguments) if arg == "-o"]
    assert len({p.name for p in destinations}) == count
    assert {p.parent for p in destinations} == {tmp_path / "runtime/logs"}
    assert "not-a-real-token" not in args
    assert token.name not in args + p.stdout + p.stderr
    assert args.count(str(tmp_path / "runtime/hf-home/token")) == count
    assert (tmp_path / "runtime/hf-home/token").read_text() == token.read_text()
