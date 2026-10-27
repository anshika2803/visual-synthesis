"""Sampling and provenance shared by the local and cluster resource pilots."""

import numpy as np
import pandas as pd

from visual_synthesis.benchmark_6000 import LEVELS
from visual_synthesis.io import read_table, select_rows
from visual_synthesis.prompt_candidate_standardization import parse_source_metadata
from visual_synthesis.stage2_evaluation import ATTACK_CONFIGS, REQUIRED_COLUMNS

SAMPLING = ("auto", "levels", "matched-cases", "category", "random")
ATTACK_FIELDS = ("attack_method", "attack_config", "case_number", "evaluation_seed", "source_file")


def load_prompts(path):
    """Read real prompt text and restore attack metadata retained by Phase E1."""
    with open(path, "rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer; run git lfs pull for this input first")
    frame = read_table(path).reset_index(drop=True)
    required = {"prompt_id", "prompt"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("input must be a nonempty prompt table with prompt_id and prompt")
    if frame[list(required)].isna().any().any() or frame["prompt_id"].astype(str).duplicated().any():
        raise ValueError("prompt IDs must be unique and prompt/ID values must be non-null")
    if frame.prompt.astype(str).str.strip().eq("").any():
        raise ValueError("prompt text must not be blank")
    if "source_metadata" in frame:
        metadata = frame.source_metadata.fillna("").map(parse_source_metadata)
        for field in ATTACK_FIELDS:
            restored = metadata.map(lambda row, field=field: row.get(f"ring_a_bell_{field}"))
            if restored.notna().any():
                existing = frame[field] if field in frame else pd.Series(None, index=frame.index, dtype=object)
                frame[field] = existing.where(existing.notna(), restored)
    for field in ("evaluation_seed", "case_number"):
        if field in frame:
            numeric = pd.to_numeric(frame[field], errors="raise")
            if (numeric.dropna() % 1 != 0).any() or (numeric.dropna() < 0).any():
                raise ValueError(f"{field} values must be nonnegative integers")
            frame[field] = numeric.astype("Int64")
    if "source_dataset" in frame:
        ring = frame[frame.source_dataset.astype(str).str.lower().eq("ring-a-bell")]
        if not ring.empty and any(field not in ring or ring[field].isna().any() for field in ATTACK_FIELDS):
            raise ValueError("Ring-A-Bell rows need complete attack metadata, including their original evaluation_seed")
    return frame


def _category_sample(frame, count, seed):
    """Cover categories where the budget permits, then fill with random rows."""
    count = min(count, len(frame))
    if count == len(frame) or "standard_category" not in frame:
        return frame.sample(n=count, random_state=seed)
    groups = list(frame.groupby("standard_category", dropna=False, sort=False))
    rng = np.random.default_rng(seed)
    weights = np.array([len(group) for _, group in groups], dtype=float)
    chosen = rng.choice(len(groups), min(count, len(groups)), replace=False, p=weights / weights.sum())
    anchors = pd.concat([groups[i][1].sample(n=1, random_state=seed) for i in chosen])
    remainder = frame.drop(index=anchors.index).sample(n=count - len(anchors), random_state=seed)
    return pd.concat([anchors, remainder])


def sample_prompts(frame, sample_size=60, seed=42, sampling="auto"):
    """Keep levels balanced, or all three attack configurations per matched case.

    Missing levels remain explicit. No new prompts are generated to fill gaps.
    """
    if sampling not in SAMPLING or sample_size < 1:
        raise ValueError(f"sampling must be one of {SAMPLING} and sample_size must be positive")
    levels = frame.get("benchmark_level", pd.Series("", index=frame.index)).fillna("").astype(str)
    stage2 = all(field in frame and frame[field].notna().all() for field in REQUIRED_COLUMNS)
    if sampling == "auto":
        sampling = "matched-cases" if stage2 else "levels" if levels.isin(LEVELS).any() else "category"
    if sampling == "matched-cases":
        if not stage2 or not set(frame.attack_config.astype(str)).issubset(ATTACK_CONFIGS):
            raise ValueError("matched-cases sampling needs complete Stage 2 metadata and known attack configurations")
        grouped = frame.groupby("case_number").attack_config.agg(list)
        matched = [case for case, configs in grouped.items()
                   if len(configs) == len(ATTACK_CONFIGS) and set(configs) == set(ATTACK_CONFIGS)]
        if not matched:
            raise ValueError("no complete matched cases contain exactly one row per attack configuration")
        cases = pd.Series(matched).sample(n=min(len(matched), max(1, sample_size // 3)), random_state=seed)
        sample = frame[frame.case_number.isin(cases)].sort_values(["case_number", "attack_config"])
    elif sampling == "levels":
        available = [level for level in LEVELS if (levels == level).any()]
        if not available:
            raise ValueError("levels sampling needs a benchmark_level column with recognized levels")
        # Water-fill quotas without replacing prompts when a level has few rows.
        quota = dict.fromkeys(available, 0)
        capacity = {level: int((levels == level).sum()) for level in available}
        for _ in range(min(sample_size, int(levels.isin(LEVELS).sum()))):
            eligible = [level for level in available if quota[level] < capacity[level]]
            level = min(eligible, key=lambda item: quota[item])
            quota[level] += 1
        sample = pd.concat([_category_sample(frame[levels == level], count, seed)
                            for level, count in quota.items() if count])
    else:
        sample = (select_rows(frame, sample_size, "random", seed=seed) if sampling == "random"
                  else _category_sample(frame, sample_size, seed))
    sample = sample.sort_index().reset_index(drop=True)
    sample_levels = sample.get("benchmark_level", pd.Series("", index=sample.index)).fillna("").astype(str)
    missing = [] if sampling == "matched-cases" else [level for level in LEVELS if level not in set(sample_levels)]
    info = {
        "strategy": sampling, "requested_sample_size": sample_size, "actual_sample_size": len(sample),
        "source_level_counts": {str(k): int(v) for k, v in levels.value_counts().items()},
        "sample_level_counts": {str(k): int(v) for k, v in sample_levels.value_counts().items()},
        "target_scope": "stage2_attack_prompts" if sampling == "matched-cases" else "balanced_six_levels",
        "missing_sample_levels": missing, "coverage_complete": not missing,
        "seed_policy": "preserve evaluation_seed when present; deterministic per-repeat fallback otherwise",
        "excluded_unknown_level_rows": int((~levels.isin(LEVELS)).sum()) if sampling == "levels" else 0,
    }
    if "human_validated" in frame:
        info["human_validated_source_rows"] = int(frame.human_validated.astype(str).str.lower().eq("true").sum())
        info["candidate_input"] = info["human_validated_source_rows"] < len(frame)
    return sample, info
