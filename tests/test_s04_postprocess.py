import pandas as pd

from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s02_standardize, s03_unify, s04_postprocess


def test_duplicates_are_removed_and_written_separately(cfg):
    run_stage(cfg, s02_standardize)
    run_stage(cfg, s03_unify)
    result = run_stage(cfg, s04_postprocess)

    unique = pd.read_parquet(f"{result['run_dir']}/unified_deduplicated.parquet")
    duplicates = pd.read_parquet(f"{result['run_dir']}/duplicates.parquet")
    assert list(unique["prompt_id"]) == ["TOY_000", "TOY_001", "TOY_002"]
    assert list(duplicates["prompt_id"]) == ["TOY_003"]
    assert duplicates.loc[0, "duplicate_of_prompt_id"] == "TOY_000"
