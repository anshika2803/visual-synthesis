import pandas as pd

from visual_synthesis.pipeline import run_stage
from visual_synthesis.schema import STANDARD_COLUMNS
from visual_synthesis.stages import s02_standardize


def test_dataset_is_converted_to_the_standard_schema(cfg):
    result = run_stage(cfg, s02_standardize, run_name="t")

    table = pd.read_parquet(f"{result['run_dir']}/toy.parquet")
    assert list(table.columns) == STANDARD_COLUMNS
    assert list(table["prompt_id"]) == ["TOY_000", "TOY_001", "TOY_002", "TOY_003"]
    assert table.loc[2, "standard_category"] == "sexual;violence"
    assert table.loc[2, "risk_level"] == "high"


def test_rerun_skips_a_dataset_that_is_already_converted(cfg):
    run_stage(cfg, s02_standardize, run_name="t")
    again = run_stage(cfg, s02_standardize, run_name="t")
    assert again["datasets"]["toy"]["status"] == "skipped"
