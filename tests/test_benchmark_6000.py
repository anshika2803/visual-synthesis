import pandas as pd

from visual_synthesis.benchmark_6000 import _canonicalize, _stratified_sample, prepare


def test_stratified_sampling_is_reproducible_and_exact():
    frame = pd.DataFrame({"prompt_id": range(30), "standard_category": ["a"] * 20 + ["b"] * 10})
    a = _stratified_sample(frame, 12, 42)
    b = _stratified_sample(frame, 12, 42)
    assert list(a.prompt_id) == list(b.prompt_id)
    assert len(a) == 12
    assert a.standard_category.value_counts().to_dict() == {"a": 8, "b": 4}


def test_parent_and_child_share_split():
    source = pd.DataFrame([{
        "prompt_id": "parent", "prompt": "harmful example", "source_dataset": "test",
        "standard_category": "violence", "risk_level": "high", "parent_prompt_id": "",
    }, {
        "prompt_id": "child", "prompt": "a transformed example", "source_dataset": "test",
        "standard_category": "violence", "risk_level": "high", "parent_prompt_id": "parent",
    }])
    parent = _canonicalize(source.iloc[[0]], "L1", approved_only=False, source_name="test", seed=42)
    child = _canonicalize(source.iloc[[1]], "L2", approved_only=False, source_name="test", seed=42)
    assert parent.iloc[0].split == child.iloc[0].split


def test_partial_inputs_report_shortfall_without_final_file(tmp_path):
    l1 = pd.DataFrame([{
        "prompt_id": f"l1_{i}", "source_dataset": "L1source", "source_id": i,
        "prompt": f"unique harmful prompt {i}", "standard_category": "violence",
        "risk_level": "high", "benchmark_level": "L1", "transformation": "original",
        "parent_prompt_id": f"l1_{i}",
    } for i in range(1002)])
    ring = pd.DataFrame([{
        "prompt_id": f"RAB_{i}", "source_dataset": "Ring-A-Bell", "prompt": f"attack prompt {i}",
        "attack_method": "method", "attack_config": "K_16", "case_number": i,
        "evaluation_seed": i, "benchmark_level": "L6", "transformation": "adversarial",
    } for i in range(3)])
    l1_path, ring_path = tmp_path / "l1.csv", tmp_path / "ring.csv"
    l1.to_csv(l1_path, index=False)
    ring.to_csv(ring_path, index=False)
    audit = prepare(l1_path=l1_path, ring_path=ring_path, output_dir=tmp_path / "out", seed=42)
    assert audit["counts_per_level"]["L1"] == 1000
    assert audit["counts_per_level"]["L6"] == 3
    assert audit["shortfalls"]["L6"] == 997
    assert not audit["all_levels_reach_1000"]
    assert (tmp_path / "out" / "benchmark_6000_candidates.parquet").exists()
    assert not (tmp_path / "out" / "unified_benchmark_6000.parquet").exists()
    saved = pd.read_parquet(tmp_path / "out" / "benchmark_6000_candidates.parquet")
    assert set(ring.prompt_id).issubset(set(saved.prompt_id))
    assert {"attack_method", "attack_config", "case_number", "evaluation_seed"}.issubset(saved.columns)
    assert saved.prompt_id.nunique() == len(saved)
    assert saved.prompt.str.strip().ne("").all()
    assert set(saved.transformation) <= {"original", "paraphrase", "obfuscation", "indirect", "adversarial"}


def test_unapproved_transform_is_review_only_and_not_counted(tmp_path):
    l1 = pd.DataFrame([{
        "prompt_id": "parent", "source_dataset": "L1source", "source_id": "parent",
        "prompt": "a reviewed source prompt", "standard_category": "violence",
        "risk_level": "high", "benchmark_level": "L1", "transformation": "original",
        "parent_prompt_id": "parent",
    }])
    ring = pd.DataFrame([{
        "prompt_id": "rab_1", "source_dataset": "Ring-A-Bell", "prompt": "attack prompt",
        "attack_config": "K_16", "benchmark_level": "L6", "transformation": "adversarial",
    }])
    l2 = pd.DataFrame([{
        "prompt_id": "child", "source_dataset": "curated", "prompt": "meaning-preserving variant",
        "standard_category": "violence", "risk_level": "high", "parent_prompt_id": "parent",
        "review_status": "pending",
    }])
    paths = [tmp_path / name for name in ("l1.csv", "ring.csv", "l2.csv")]
    for frame, path in zip((l1, ring, l2), paths):
        frame.to_csv(path, index=False)
    audit = prepare(l1_path=paths[0], ring_path=paths[1], extra_inputs=[("L2", paths[2])],
                    output_dir=tmp_path / "out", seed=42)
    assert audit["counts_per_level"]["L2"] == 0
    review = pd.read_parquet(tmp_path / "out" / "benchmark_6000_candidates_review.parquet")
    assert review.loc[0, "prompt_id"] == "child"
    assert review.loc[0, "review_status"] == "pending"
    assert audit["parent_references_valid"]
