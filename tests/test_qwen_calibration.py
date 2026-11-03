"""Preserve paired evidence and expose destructive edits before larger GPU studies."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("calibration_test", ROOT / "scripts/aspire_qwen_calibration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pilot(tmp_path):
    """A completed miniature pilot with independently recorded image/input hashes."""
    from visual_synthesis.io import file_sha256
    from visual_synthesis.project_pilot import save_json

    source = tmp_path / "pilot"
    (source / "inputs").mkdir(parents=True)
    (source / "baseline/images").mkdir(parents=True)
    (source / "learn").mkdir()
    (source / "capture").mkdir()
    rows, images = [], []
    for i in range(4):
        row = {
            "prompt_id": f"p{i}",
            "prompt": f"scene {i}",
            "evaluation_seed": 10 + i,
            "pilot_label": "0" if i < 2 else "1",
        }
        rows.append(row)
        path = source / "baseline/images" / f"{i}.png"
        path.write_bytes(f"image {i}".encode())
        images.append({"condition": "baseline", **row, "image_path": str(path), "sha256": file_sha256(path)})
    pd.DataFrame(rows).to_csv(source / "inputs/validation.csv", index=False)
    (source / "baseline/images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in images))
    np.savez(
        source / "learn/direction.npz", direction=np.array([1.0, 0.0]), center=np.zeros(2), scale=5.0, layer_index=0
    )
    save_json(
        {
            "layers": ["transformer.transformer_blocks.0"],
            "rows": [{"prompt_id": "training-only", "split": "discovery", "label": 0}],
        },
        source / "capture/activation_metadata.json",
    )
    save_json(
        {
            "status": "success",
            "model": "qwen-image",
            "pbs_job_id": "123.pbs",
            "input": {"tables_sha256": {"validation": file_sha256(source / "inputs/validation.csv")}},
            "phases": {
                "baseline": {
                    "model_settings": {"dtype": "bfloat16", "enable_cpu_offload": True},
                    "mean_seconds_per_image": 62.0,
                },
                "learn": {"selected_layer": "transformer.transformer_blocks.0"},
            },
        },
        source / "pilot_summary.json",
    )
    return source


def test_plan_reuses_seeds_baselines_and_matched_control_strengths(driver, pilot):
    before = {str(p): p.read_bytes() for p in pilot.rglob("*") if p.is_file()}
    plan = driver.make_plan(pilot, [f"p{i}" for i in range(4)])
    assert plan["new_images"] == 34 and plan["reused_baseline_images"] == 4
    assert plan["identity_ids"] == ["p0", "p2"]
    assert [int(p["evaluation_seed"]) for p in plan["prompts"]] == [10, 11, 12, 13]
    for alpha in driver.DEFAULT_ALPHAS:
        arms = [a for a in plan["arms"] if a["alpha"] == alpha]
        assert {a["vector"] for a in arms} == {"safety", "random"}
    assert before == {str(p): p.read_bytes() for p in pilot.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["image", "input", "leak", "direction", "ids"])
def test_plan_rejects_unreliable_source_pairs(driver, pilot, problem):
    ids = [f"p{i}" for i in range(4)]
    if problem == "image":
        (pilot / "baseline/images/0.png").write_bytes(b"changed image")
    elif problem == "input":
        with (pilot / "inputs/validation.csv").open("a") as stream:
            stream.write("changed input")
    elif problem == "leak":
        path = pilot / "capture/activation_metadata.json"
        value = json.loads(path.read_text())
        value["rows"][0]["prompt_id"] = "p0"
        path.write_text(json.dumps(value))
    elif problem == "direction":
        np.savez(
            pilot / "learn/direction.npz", direction=np.array([2.0, 0.0]), center=np.zeros(2), scale=5.0, layer_index=0
        )
    else:
        ids[-1] = ids[0]
    with pytest.raises(ValueError):
        driver.make_plan(pilot, ids)


@pytest.mark.parametrize("alphas", [[0], [1], [-0.1], [float("nan")], [float("inf")], [0.1, 0.1]])
def test_bad_strengths_fail_before_generation(driver, alphas):
    with pytest.raises(ValueError):
        driver.strength_grid(alphas)


def test_diagnostics_observe_both_cfg_passes_without_changing_rng_or_output(driver):
    torch = pytest.importorskip("torch")
    from visual_synthesis.pilot_interventions import ActivationHooks, HookPoint

    module = torch.nn.Identity()
    point = HookPoint("block", module, None, "sequential")
    d = torch.tensor([1.0, 0.0])
    x = torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)
    seen = []
    state = torch.get_rng_state().clone()
    handle = module.register_forward_hook(driver.diagnostic_hook(point, d, 0.1, 10.0, seen))
    try:
        with ActivationHooks([point], "steering", d, torch.zeros(2), 10.0, 0.1):
            y = module(x)
            module(x)
    finally:
        handle.remove()
    assert torch.equal(x, torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16))
    assert torch.equal(y, torch.tensor([[[2.0, 4.0]]], dtype=torch.bfloat16))
    assert torch.equal(state, torch.get_rng_state())
    assert [s["cfg_pass"] for s in seen] == ["conditional", "unconditional"]
    assert seen[0]["hidden_rms"] == pytest.approx(5 / 2**0.5)
    assert seen[0]["requested_shift_to_hidden_rms"] == pytest.approx(0.2)
    assert not module._forward_hooks


@pytest.mark.parametrize("break_identity", [False, True])
def test_generation_reuses_one_load_and_stops_on_baseline_mismatch(driver, pilot, monkeypatch, break_identity):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from PIL import Image

    torch = pytest.importorskip("torch")
    import visual_synthesis.backends.image as image_module
    import visual_synthesis.benchmark as benchmark
    import visual_synthesis.config as config_module
    import visual_synthesis.pilot_interventions as intervention_module
    from visual_synthesis.io import file_sha256

    module = torch.nn.Identity()
    point = intervention_module.HookPoint("transformer.transformer_blocks.0", module, None, "sequential")

    class Backend:
        loads = 0
        unloaded = False

        def load(self):
            self.loads += 1
            self.pipe = SimpleNamespace(_all_hooks=[object()])

        def generate(self, requests):
            x = torch.tensor([[[float(requests[0]["seed"]), 20.0]]])
            value = module(x)
            module(x)  # Separate unconditional pass, as in the Qwen adapter.
            color = max(0, min(255, round(float(value[0, 0, 0]) * 10)))
            return [{"image": Image.new("RGB", (8, 8), (color, 40, 80))}]

        def unload(self):
            self.unloaded = True

    backend = Backend()
    records = driver.jsonl(pilot / "baseline/images.jsonl")
    for row in records:
        image_module.save_image(backend.generate([{"seed": row["evaluation_seed"]}])[0]["image"], row["image_path"])
        row["sha256"] = file_sha256(row["image_path"])
    (pilot / "baseline/images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    plan = driver.make_plan(pilot, [f"p{i}" for i in range(4)], [0.01])
    plan["model_settings"]["device"] = "cuda"
    if break_identity:
        plan["baseline_images"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(config_module, "get_model", lambda *a: dict(plan["model_settings"]))
    monkeypatch.setattr(image_module, "build_image_backend", lambda *a: backend)
    monkeypatch.setattr(intervention_module, "hook_points", lambda *a: [point])
    monkeypatch.setattr(benchmark, "machine_info", lambda *a: {"test": True})
    monkeypatch.setattr(benchmark, "Resources", lambda *a: nullcontext(SimpleNamespace(sync=lambda: None, peaks={})))
    out = pilot.parent / "calibration"
    out.mkdir()
    (out / "all-images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in plan["baseline_images"]))
    if break_identity:
        with pytest.raises(ValueError, match="alpha-zero differs"):
            driver.generate(plan, {"paths": {"models": "/unused"}}, out, float("inf"))
        assert len(driver.jsonl(out / "all-images.jsonl")) == 4
        assert not (out / "safety-steering-a0.01").exists()
    else:
        result = driver.generate(plan, {"paths": {"models": "/unused"}}, out, float("inf"))
        assert result["identity_passed"] and result["images"] == 10
        assert len(driver.jsonl(out / "all-images.jsonl")) == 14  # Four reused baselines + ten new.
        assert len(driver.jsonl(out / "hook_diagnostics.jsonl")) == 20
    assert backend.loads == 1 and backend.unloaded
    assert not module._forward_hooks


def test_summary_keeps_flag_drop_and_quality_collapse_visible(driver, tmp_path):
    from visual_synthesis.io import file_sha256

    out = tmp_path
    prompts = [{"prompt_id": "benign", "pilot_label": "0"}, {"prompt_id": "risk", "pilot_label": "1"}]
    arms = [{"condition": "safety-steering-a0.1", "alpha": 0.1, "vector": "safety"}]
    images, pixels = [], []
    for condition in ("baseline", arms[0]["condition"]):
        for row in prompts:
            path = out / f"{condition}-{row['prompt_id']}.png"
            path.write_bytes(b"audited image")
            images.append(
                {
                    "condition": condition,
                    "prompt_id": row["prompt_id"],
                    "image_path": str(path),
                    "sha256": file_sha256(path),
                }
            )
            if condition != "baseline":
                pixels.append({"condition": condition, "prompt_id": row["prompt_id"], "near_blank_heuristic": True})
    (out / "all-images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in images))
    (out / "generation_report.json").write_text(json.dumps({"pixel_statistics": pixels}))
    for metric in ("nudenet", "clip-h14", "alignment", "clip-iqa"):
        (out / metric).mkdir()
        rows = []
        for image in images:
            base = image["condition"] == "baseline"
            payload = (
                {"flagged": base and image["prompt_id"] == "risk"}
                if metric in ("nudenet", "clip-h14")
                else {"score": 0.9 if base else 0.1}
            )
            rows.append({"condition": image["condition"], "prompt_id": image["prompt_id"], **payload})
        (out / metric / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = driver.summarize({"prompts": prompts, "arms": arms}, out)
    risk = next(r for r in result if r["condition"] != "baseline" and r["group"] == "sexual-risk")
    assert risk["union_flagged"] == 0 and risk["near_blank_count"] == 1
    assert risk["clip_iqa_mean"] == 0.1
    (out / "clip-iqa/scores.jsonl").write_text("")
    with pytest.raises(ValueError, match="missing/duplicate"):
        driver.summarize({"prompts": prompts, "arms": arms}, out)
