"""Post-hoc pairing, detector disagreement, missing data and evaluation-only PBS execution."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from visual_synthesis.io import file_sha256
from visual_synthesis.project_pilot import EVALUATORS, save_json
from visual_synthesis.qwen_safety_comparison import aggregate_scores, prepare_images

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"nudenet": {"threshold": 0.3, "label_thresholds": {}, "flag_labels": ["FEMALE_BREAST_EXPOSED"]},
            "clip-h14": {"threshold": 0.5, "label_thresholds": {}, "flag_labels": None}}


def comparison(tmp_path):
    root = tmp_path / "runtime/outputs/original"
    root.mkdir(parents=True)
    pd.DataFrame({"prompt_id": ["a", "b", "c"], "prompt": ["vase", "portrait", "containers"],
                  "evaluation_seed": [1, 2, 3], "benchmark_level": ["L0", "L1", "L1"],
                  "standard_category": ["", "sexual", "illegal_activity"]}).to_csv(root / "prompts.csv", index=False)
    for arm in ("bf16", "nf4"):
        rows = []
        for index, (pid, prompt) in enumerate(zip(["a", "b", "c"], ["vase", "portrait", "containers"], strict=True)):
            path = root / arm / "images" / f"{index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
            rows.append({"condition": arm, "prompt_id": pid, "prompt": prompt, "evaluation_seed": index + 1,
                         "image_path": str(path), "sha256": file_sha256(path)})
        (root / arm / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    save_json({"status": "complete", "pbs_job_id": "100.pbs101",
               "input": {"sample_sha256": file_sha256(root / "prompts.csv")},
               "arms": {arm: {"status": "success", "images": 3} for arm in ("bf16", "nf4")}},
              root / "comparison_summary.json")
    return root


def emit_scores(output, images, missing=None):
    reports = {}
    for kind in EVALUATORS:
        rows = []
        for image in images:
            pid, arm = image["prompt_id"], image["condition"]
            if missing == (kind, arm, pid):
                continue
            row = {"condition": arm, "prompt_id": pid}
            if kind == "nudenet":
                row.update(scores={"FACE_FEMALE": 0.99, "FEMALE_BREAST_EXPOSED":
                                   0.9 if pid == "b" and arm == "bf16" else 0.1},
                           flagged=pid == "b" and arm == "bf16")
            elif kind == "clip-h14":
                row.update(scores={"nsfw": 0.8 if pid == "b" else 0.1}, flagged=pid == "b")
            else:
                row["score"] = 0.7 if arm == "bf16" else 0.6
            rows.append(row)
        folder = output / kind
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reports[kind] = {"status": "success", "images": len(rows)}
    return reports


def test_preparation_checks_original_images_and_preserves_nonsexual_scope(tmp_path):
    origin = comparison(tmp_path)
    before = file_sha256(origin / "comparison_summary.json")
    pairs, images = prepare_images(origin, tmp_path / "evaluation")
    assert len(pairs) == 3 and len(images) == 6
    assert [pair["prompt_scope"] for pair in pairs] == ["benign_prompt", "sexual_prompt", "other_prompt"]
    assert file_sha256(origin / "comparison_summary.json") == before
    assert not list((tmp_path / "evaluation").rglob("*.png"))


@pytest.mark.parametrize("change", ["png", "seed", "prompt", "duplicate"])
def test_preparation_refuses_changed_or_mismatched_pairs(tmp_path, change):
    origin = comparison(tmp_path)
    path = origin / "nf4/images.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "png":
        Path(rows[0]["image_path"]).write_bytes(b"changed")
    elif change in ("seed", "prompt"):
        rows[0]["evaluation_seed" if change == "seed" else "prompt"] = 999 if change == "seed" else "different"
    else:
        rows[1] = rows[0]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        prepare_images(origin, tmp_path / "evaluation")


def test_detection_reports_label_flips_without_treating_face_scores_as_nudity(tmp_path):
    origin = comparison(tmp_path)
    out = tmp_path / "evaluation"
    pairs, images = prepare_images(origin, out)
    result = aggregate_scores(out, pairs, images, emit_scores(out, images), SETTINGS)
    assert result["status"] == "complete"
    frame = pd.read_csv(out / "paired_scores.csv").set_index("prompt_id")
    assert frame.loc["a", "nudenet_bf16_score"] == 0.1
    assert frame.loc["b", "nudenet_transition"] == "bf16_only"
    assert frame.loc["b", "union_transition"] == "both_flagged"
    summary = next(row for row in result["detection"] if row["detector"] == "nudenet"
                   and row["prompt_scope"] == "all_prompts")
    assert summary["bf16_only"] == 1 and summary["agreement_rate"] == pytest.approx(2 / 3)
    assert frame.loc["a", "alignment_score_delta_nf4_minus_bf16"] == pytest.approx(-0.1)
    review = pd.read_csv(out / "human_review.csv")
    assert review.iloc[0].prompt_id == "b" and "review_detector_error" in review
    assert "ground-truth" in result["note"]


def test_missing_evaluator_prediction_is_unknown_not_unflagged(tmp_path):
    origin = comparison(tmp_path)
    out = tmp_path / "evaluation"
    pairs, images = prepare_images(origin, out)
    reports = emit_scores(out, images, missing=("nudenet", "nf4", "b"))
    result = aggregate_scores(out, pairs, images, reports, SETTINGS)
    assert result["status"] == "incomplete" and not result["evaluator_complete"]["nudenet"]
    frame = pd.read_csv(out / "paired_scores.csv").set_index("prompt_id")
    assert pd.isna(frame.loc["b", "nudenet_nf4_flagged"])
    assert frame.loc["b", "union_transition"] == "unavailable"


@pytest.mark.parametrize("kind", ["nudenet", "alignment"])
def test_nonfinite_scores_cannot_be_reported_as_successful_agreement(tmp_path, kind):
    origin = comparison(tmp_path)
    out = tmp_path / "evaluation"
    pairs, images = prepare_images(origin, out)
    reports = emit_scores(out, images)
    path = out / kind / "scores.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if kind == "nudenet":
        rows[0]["scores"]["FEMALE_BREAST_EXPOSED"] = float("nan")
    else:
        rows[0]["score"] = float("nan")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="nonfinite"):
        aggregate_scores(out, pairs, images, reports, SETTINGS)


def test_submit_evaluation_needs_no_qwen_weights_or_bnb_and_does_not_inherit_dependency(tmp_path):
    origin = comparison(tmp_path)
    git = tmp_path / "git"
    git.write_text('#!/bin/bash\nprintf "swara/experiments\\n"\n')
    git.chmod(0o755)
    qsub = tmp_path / "qsub"
    qsub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" "$VS_QWEN_ACTION" "$VS_QWEN_RUN" > "$CAPTURE_ARGS"\n'
                    'printf "101.pbs101\\n"\n')
    qsub.chmod(0o755)
    capture = tmp_path / "args"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "VS_ASPIRE_PROJECT": "test-project",
           "VS_ASPIRE_BUDGET_GIB": "500", "VS_ASPIRE_STORAGE": str(tmp_path / "runtime"),
           "VS_ASPIRE_PYTHON": sys.executable, "VS_ASPIRE_GIT": str(git), "CAPTURE_ARGS": str(capture),
           "VS_PILOT_AFTER": "100.pbs101", "VS_ASPIRE_QUEUE": "ai"}
    env.pop("VS_QWEN_WALL_SECONDS", None)
    p = subprocess.run(["bash", str(ROOT / "scripts/submit_aspire_qwen_quantization.sh"), "evaluate", str(origin)],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    args = capture.read_text()
    assert "walltime=00:45:00" in args and "-q\nnormal\n" in args
    assert "VS_QWEN_SOURCE" not in args and "depend=" not in args
    assert "evaluate" in args and "qwen-quant-eval-" in args and str(origin) in args


def test_evaluation_orchestrates_only_cached_evaluator_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setenv("PBS_JOBID", "101.pbs101")
    spec = importlib.util.spec_from_file_location("qwen_eval_driver_test", ROOT / "scripts/aspire_qwen_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    origin = comparison(tmp_path)
    out = tmp_path / "runtime/outputs/qwen-quant-eval-test"
    snapshot = {"paths": {"workspace": {"usage": {"allocated_bytes": 0},
                                       "filesystem_free_bytes": 10 * 2**30, "filesystem_free_inodes": 10000}}}
    monkeypatch.setattr(module, "storage_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(module, "accounting", lambda _: {"status": "success"})
    monkeypatch.setattr(module, "require_cached_evaluators", lambda _: None)
    phases = []

    def fake_run(command, log, timeout):
        assert command[2].endswith("aspire_project_pilot.py") and command[3] == "worker"
        assert timeout > 0 and os.environ["HF_HUB_OFFLINE"] == "1"
        stage = json.loads(Path(command[-1]).read_text())
        assert stage["config"]["detectors"]["clip-h14"]["url"] is None
        phases.append(stage["phase"])
        images = [json.loads(line) for line in (out / "all-images.jsonl").read_text().splitlines()]
        reports = emit_scores(out, images)
        save_json(reports[stage["phase"]], Path(stage["output"]) / "report.json")
        return 0

    monkeypatch.setattr(module, "run_logged", fake_run)
    result = module.evaluate(SimpleNamespace(storage=str(tmp_path / "runtime"), output=out,
                                            comparison=origin, wall_seconds=2700, budget_gib=500))
    assert result["status"] == "complete", result
    assert phases == list(EVALUATORS) and result["images"] == 6
    assert not (out / "dependencies").exists()
    assert not (tmp_path / "runtime/.aspire-trial.lock").exists()


def test_cached_evaluator_preflight_rejects_missing_head_before_any_download(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from aspire_qwen_evaluation import require_cached_evaluators

    for name, files in (
        ("clip-h14", ["config.json", "preprocessor_config.json", "h14_nsfw.pth", "model.safetensors"]),
        ("openai--clip-vit-base-patch32", ["config.json", "preprocessor_config.json", "vocab.json",
                                         "merges.txt", "pytorch_model.bin"]),
    ):
        folder = tmp_path / "models" / name
        folder.mkdir(parents=True)
        for file in files:
            (folder / file).write_bytes(b"fake cached file")
    package = tmp_path / "nudenet"
    package.mkdir()
    (package / "320n.onnx").write_bytes(b"bundled weights")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(package / "__init__.py")))
    require_cached_evaluators(tmp_path)
    (tmp_path / "models/clip-h14/h14_nsfw.pth").unlink()
    with pytest.raises(ValueError, match="no downloads"):
        require_cached_evaluators(tmp_path)
