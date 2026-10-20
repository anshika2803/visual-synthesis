import json

import pandas as pd
from click.testing import CliRunner

from visual_synthesis.cli import cli
from visual_synthesis.config import ROOT


def test_stage2_smoke_cli_runs_with_fake_backends(tmp_path):
    rows = []
    for config in ("K_16", "K_38", "K_77"):
        rows.append({
            "prompt_id": f"rab_{config.lower()}_0000", "prompt": "a simple landscape",
            "source_dataset": "ring-a-bell", "attack_method": "Ring-A-Bell",
            "attack_config": config, "case_number": 0, "evaluation_seed": 21,
            "source_file": f"{config}.csv",
        })
    dataset = tmp_path / "stage2.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    config = tmp_path / "stage2-test.json"
    config.write_text(json.dumps({
        "models": {"stage2-fake": {"backend": "fake", "width": 8, "height": 8}},
        "detectors": {"stage2-fake": {"backend": "fake", "threshold": 0.5}},
        "generate": {"model": "stage2-fake", "num_inference_steps": 1, "width": 8, "height": 8},
        "nsfw": {"detector": "stage2-fake"},
        "stage2": {"output_dir": str(tmp_path / "results"), "smoke_cases": 1},
    }))

    result = CliRunner().invoke(cli, [
        "stage2", "--smoke", "--cases", "1", "--input", str(dataset), "--run-name", "fake-smoke",
        "-c", str(ROOT / "config" / "pipeline.json"), "-c", str(config),
    ])

    assert result.exit_code == 0, result.output
    out = tmp_path / "results" / "stage2-fake" / "fake-smoke"
    attempts = pd.read_csv(out / "attack_results.csv")
    assert len(attempts) == 3
    assert set(attempts.attack_config) == {"K_16", "K_38", "K_77"}
    assert attempts.evaluation_seed.tolist() == [21, 21, 21]
    assert (out / "generation_results.csv").exists()
    assert (out / "nsfw_results.csv").exists()
