from pathlib import Path

import pandas as pd

from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s05_generate


def prompts_table(tmp_path, count=3):
    path = tmp_path / "prompts.parquet"
    pd.DataFrame({"prompt_id": [f"P{i}" for i in range(count)], "prompt": ["a cat"] * count}).to_parquet(path)
    return path


def test_one_image_is_generated_per_prompt(cfg, tmp_path):
    result = run_stage(cfg, s05_generate, input_path=prompts_table(tmp_path))

    table = pd.read_parquet(f"{result['run_dir']}/generations.parquet")
    assert list(table["prompt_id"]) == ["P0", "P1", "P2"]
    assert all(Path(path).exists() for path in table["image_path"])


def test_resumed_run_only_generates_the_missing_images(cfg, tmp_path):
    prompts = prompts_table(tmp_path)
    cfg["run"]["limit"] = 1
    first = run_stage(cfg, s05_generate, run_name="t", input_path=prompts)
    cfg["run"]["limit"] = None
    second = run_stage(cfg, s05_generate, run_name="t", input_path=prompts)

    assert (first["rows_out"], second["rows_out"], second["rows_skipped"]) == (1, 2, 1)
    assert len(pd.read_parquet(f"{second['run_dir']}/generations.parquet")) == 3
