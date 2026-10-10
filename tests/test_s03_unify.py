import pandas as pd

from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s02_standardize, s03_unify


def test_standardized_tables_are_merged(cfg):
    run_stage(cfg, s02_standardize)
    result = run_stage(cfg, s03_unify, run_name="t")

    table = pd.read_parquet(f"{result['run_dir']}/unified.parquet")
    assert len(table) == 4
    assert result["per_dataset"] == {"Toy": 4}


def test_rerun_skips_when_the_inputs_did_not_change(cfg):
    run_stage(cfg, s02_standardize)
    run_stage(cfg, s03_unify, run_name="t")
    again = run_stage(cfg, s03_unify, run_name="t")
    assert again["skipped"] is True
