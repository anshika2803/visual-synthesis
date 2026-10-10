import pandas as pd
import pytest

from visual_synthesis.config import ConfigError, load_config
from visual_synthesis.pipeline import STAGES, run_chain


def test_whole_pipeline_runs_end_to_end(cfg):
    results = run_chain(cfg, STAGES, run_name="e2e")

    assert [r["status"] for r in results] == ["success"] * 6
    detections = pd.read_parquet(f"{results[-1]['run_dir']}/detections.parquet")
    assert sorted(detections["prompt_id"]) == ["TOY_000", "TOY_001", "TOY_002"]
    assert set(detections["standard_category"]) == {"sexual", "violence", "sexual;violence"}


def test_config_typo_is_an_error(tmp_path):
    config_file = tmp_path / "bad.json"
    config_file.write_text('{"generate": {"modle": "sd-turbo"}}')
    with pytest.raises(ConfigError, match="generate.modle"):
        load_config([config_file])
