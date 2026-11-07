"""Exercise the PBS launcher with recorded CLI calls, without a GPU or downloads."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visual_synthesis.config import get_model, load_config

ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "scripts/aspire_protocol.pbs"

# Run the actual job-config builder; record external pipeline calls. Other
# scientific components have their own tests against real/fake model backends.
DRIVER = """
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
body = sys.stdin.read() if args and args[0] == '-' else ''
with open(os.environ['CALLS'], 'a') as out:
    out.write(json.dumps({'args': args, 'body': body, 'offline': os.environ.get('HF_HUB_OFFLINE')}) + '\\n')
if body and 'base = load_config' in body:
    raise SystemExit(subprocess.run([sys.executable, *args], input=body, text=True).returncode)
if body and 'modes = e6_mode' in body:
    print('steering')
if args[:3] == ['-u', '-m', 'visual_synthesis'] and args[3] == os.environ.get('FAIL_ACTION'):
    raise SystemExit(7)
"""


@pytest.fixture
def launch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (repo / "config").symlink_to(ROOT / "config", target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(f"#!{sys.executable}\n" + DRIVER)
    python.chmod(0o755)
    flock = bin_dir / "flock"
    flock.write_text("#!/bin/sh\nexit 0\n")
    flock.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PBS_JOBID": "123.pbs101",
        "PBS_O_WORKDIR": str(repo),
        "PBS_O_PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "TMPDIR": str(tmp_path),
        "VS_ASPIRE_PYTHON": str(python),
        "VS_PROTOCOL_MODEL": "qwen-image",
        "VS_PROTOCOL_PHASE": "all",
        "VS_PROTOCOL_TIER": "reduced",
        "VS_ASPIRE_STORAGE": str(repo / "runtime"),
        "VS_ASPIRE_RUNS": str(repo / "data/runs"),
        "CALLS": str(calls),
    }

    def run(**changes):
        calls.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(PBS)], env={**env, **changes}, capture_output=True, text=True)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, recorded, repo

    return run


def cli_calls(calls, action):
    return [c["args"] for c in calls if c["args"][:3] == ["-u", "-m", "visual_synthesis"] and c["args"][3] == action]


def name(args):
    return args[args.index("--run-name") + 1]


def test_all_uses_existing_scripts_and_frozen_conditions(launch):
    result, calls, repo = launch()
    assert result.returncode == 0, result.stdout + result.stderr
    generated = cli_calls(calls, "generate")
    assert len(generated) == 16  # two-image pilot plus 15 production runs
    assert [name(c) for c in generated[:4]] == [
        "pilot-fit-qwen-image",
        "e1-qwen-image",
        "e7-qwen-image",
        "e4-qwen-image-capture",
    ]
    e5 = [c for c in generated if name(c).startswith("e5-")]
    e6 = [c for c in generated if name(c).startswith("e6-")]
    assert len(e5) == 8 and len(e6) == 4
    assert all('stages.generate.filters={"subset_split":"validation"}' in c for c in e5 + e6)
    assert all("generate.intervention.mode=steering" in c for c in e6)
    assert all(c[c.index("--mode") + 1] == "incremental" for c in generated)
    assert len(cli_calls(calls, "nsfw")) == 32
    assert len(cli_calls(calls, "metrics")) == 16
    assert any("scripts/validate_hooks.py" in c["args"] for c in calls)
    assert any("scripts/e4_analyze.py" in c["args"] for c in calls)
    assert len([c for c in calls if "scripts/e1e6_report.py" in c["args"]]) == 2
    assert not (repo / "runtime/protocol-v1").exists()
    assert not list(repo.rglob("ready.json")) and not list(repo.rglob("resources.json"))
    # Live configs are temporary; an audit copy stays with the job logs.
    config_call = next(c for c in calls if "base = load_config" in c["body"])
    assert not Path(config_call["args"][2]).exists()
    assert (repo / "logs/aspire/123.pbs101.config.json").is_file()
    assert "COMPLETE qwen-image phase=all" in result.stdout


@pytest.mark.parametrize("model,tier", [("qwen-image", "reduced"), ("sd-1.5", "full")])
def test_config_preserves_model_science_and_reuses_cache(launch, model, tier):
    result, calls, repo = launch(VS_PROTOCOL_MODEL=model, VS_PROTOCOL_TIER=tier, VS_PROTOCOL_PHASE="ready")
    assert result.returncode == 0, result.stdout + result.stderr
    overlay = repo / "logs/aspire/123.pbs101.config.json"
    files = [ROOT / "config" / n for n in ("pipeline.json", "pipeline.hpc.json", "e1e6.json")]
    before = get_model(load_config(files, use_env=False), model)
    after = get_model(load_config([*files, overlay], use_env=False), model)
    for key in (
        "repo_id",
        "dtype",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "negative_prompt",
        "generation_kwargs",
        "hook",
    ):
        assert after[key] == before[key]
    assert after["quantization"] is None and after["device_map"] is None and after["enable_cpu_offload"]
    saved = json.loads(overlay.read_text())
    assert saved["paths"]["models"] == str(repo / "runtime/models")
    assert saved["paths"]["runs"] == str(repo / "data/runs")
    generated = cli_calls(calls, "generate")
    assert len(generated) == 1 and generated[0][generated[0].index("--limit") + 1] == "2"
    assert all(c["offline"] == "1" for c in calls if "scripts/validate_hooks.py" in c["args"])
    assert all(c["offline"] == "0" for c in calls if "snapshot_download" in c["body"])


def test_failed_stage_stops_production_and_cleans_temp_config(launch):
    result, calls, _ = launch(FAIL_ACTION="nsfw")
    assert result.returncode == 7
    assert "FAILED: step=pilot-fit-qwen-image-nudenet exit=7" in result.stdout
    assert len(cli_calls(calls, "generate")) == 1
    config_call = next(c for c in calls if "base = load_config" in c["body"])
    assert not Path(config_call["args"][2]).exists()


def test_unknown_model_fails_before_pipeline_commands(launch):
    result, calls, _ = launch(VS_PROTOCOL_MODEL="not-a-configured-model")
    assert result.returncode != 0
    assert not cli_calls(calls, "generate")


def test_frozen_tier_cannot_be_changed(launch):
    result, calls, _ = launch(VS_PROTOCOL_TIER="full")
    assert result.returncode == 2 and not calls
    assert "Frozen tier for qwen-image is reduced" in result.stdout


def python_blocks():
    body = PBS.read_text().splitlines()
    blocks, capture = [], None
    for line in body:
        if "<<'PY'" in line:
            capture = []
        elif line == "PY" and capture is not None:
            blocks.append("\n".join(capture))
            capture = None
        elif capture is not None:
            capture.append(line)
    return blocks


def test_every_inline_python_block_compiles():
    for index, body in enumerate(python_blocks()):
        compile(body, f"aspire_protocol.pbs block {index}", "exec")
    subprocess.run(["bash", "-n", str(PBS)], check=True)


@pytest.mark.parametrize(("quality", "error"), [(np.nan, "missing scores"), (np.inf, "non-finite scores"), (0.8, None)])
def test_metric_validation_checks_quality(tmp_path, quality, error):
    """The PBS check must fail even when the metrics CLI returns success."""
    body = next(b for b in python_blocks() if "def records(path)" in b)
    expected = pd.read_parquet(ROOT / "data/processed/e1e6/core_v1_reduced.parquet").head(2)
    model_id, run = "qwen-image", "pilot-fit-qwen-image"
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "generate": {"model": model_id},
                "paths": {"runs": str(tmp_path)},
                "models": {model_id: {"quantization": None}},
            }
        )
    )
    generation = tmp_path / "s05_generate" / run
    generation.mkdir(parents=True)
    records = []
    for prompt in expected.itertuples():
        image = generation / f"{prompt.prompt_id}.png"
        image.write_bytes(b"image")
        records.append(
            {
                "prompt_id": prompt.prompt_id,
                "generation_status": "success",
                "image_path": str(image),
                "seed": 42,
                "model": model_id,
                "precision": "bfloat16",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
            }
        )
    (generation / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for detector in ("nudenet", "laion"):
        target = tmp_path / "s06_nsfw" / f"{run}-{detector}"
        target.mkdir(parents=True)
        (target / "detections.jsonl").write_text(
            "".join(
                json.dumps({"prompt_id": r["prompt_id"], "status": "success", "flagged": False}) + "\n" for r in records
            )
        )
    target = tmp_path / "s07_metrics" / run
    target.mkdir(parents=True)
    pd.DataFrame(
        {
            "prompt_id": expected.prompt_id,
            "flagged": False,
            "flagged_nudenet": False,
            "flagged_clip-h14": False,
            "clip_score": 0.8,
            "quality_score": quality,
            "seed": "42",
            "model": model_id,
            "precision": "bfloat16",
        }
    ).to_parquet(target / "image_metrics.parquet")
    result = subprocess.run(
        [sys.executable, "-", run, "core_v1_reduced", "2", "0", "0", str(cfg_file)],
        input=body,
        text=True,
        cwd=ROOT,
        capture_output=True,
    )
    if error:
        assert result.returncode != 0 and error in result.stderr
    else:
        assert result.returncode == 0, result.stderr
