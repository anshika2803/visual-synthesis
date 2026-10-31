"""Prepare and audit a balanced 6,000-prompt benchmark without inventing examples.

Records from the local, already-standardized L1 and Ring-A-Bell inputs are
preselected as candidates; this is not human validation. L0/L2/L3/L4 and
supplemental L6 rows require an explicit ``review_status=approved`` in their
supplied table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

from visual_synthesis.config import load_config
from visual_synthesis.io import read_table, write_tabular
from visual_synthesis.schema import (
    BENCHMARK_LEVELS, STANDARD_COLUMNS, TRANSFORMATIONS,
    content_hash, normalize_prompt, split_categories,
)

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L6")
EXPECTED_TRANSFORM = {"L0": "original", "L1": "original", "L2": "paraphrase",
                      "L3": "obfuscation", "L4": "indirect", "L6": "adversarial"}
TARGET = 1000


def _empty(value):
    return pd.isna(value) or str(value).strip() == ""


def _approved(row, level):
    if str(row.get("review_status", "")).strip().casefold() not in {"approved", "validated"}:
        return False
    if str(row.get("transformation", "")).strip() != EXPECTED_TRANSFORM[level]:
        return False
    if level in {"L2", "L3", "L4"}:
        required = ("parent_prompt_id", "transformation_method", "transformation_parameters")
        if any(_empty(row.get(column)) for column in required):
            return False
    return True


def _stable_split(group_id, seed):
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return "test" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def _stratified_sample(frame, n, seed):
    """Proportional deterministic sampling by the source's category combination."""
    if len(frame) <= n:
        return frame.copy()
    keys = frame["standard_category"].fillna("").astype(str)
    groups = [(key, group) for key, group in frame.groupby(keys, sort=True, dropna=False)]
    quotas = {key: math.floor(len(group) * n / len(frame)) for key, group in groups}
    remainder = n - sum(quotas.values())
    ranked = sorted(groups, key=lambda pair: (-(len(pair[1]) * n / len(frame) - quotas[pair[0]]), pair[0]))
    for key, _ in ranked[:remainder]:
        quotas[key] += 1
    pieces = [group.sample(n=quotas[key], random_state=seed) for key, group in groups if quotas[key]]
    return pd.concat(pieces).sort_index()


def _canonicalize(frame, level, *, approved_only, source_name, seed):
    frame = frame.copy()
    if approved_only:
        frame = frame[frame.apply(lambda row: _approved(row, level), axis=1)].copy()
    if "prompt" not in frame:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    frame = frame[frame["prompt"].fillna("").astype(str).str.strip().ne("")].copy()
    frame["benchmark_level"] = level
    frame["transformation"] = EXPECTED_TRANSFORM[level]
    if "source_dataset" not in frame:
        frame["source_dataset"] = source_name
    if "source_id" not in frame:
        frame["source_id"] = frame.get("prompt_id", pd.Series(frame.index, index=frame.index)).astype(str)
    if "prompt_id" not in frame:
        frame["prompt_id"] = [f"{source_name}_{level}_{i:06d}" for i in range(len(frame))]
    if "original_category" not in frame:
        frame["original_category"] = ""
    if "standard_category" not in frame:
        frame["standard_category"] = ""
    if "risk_level" not in frame:
        frame["risk_level"] = "low" if level == "L0" else ""
    if "parent_prompt_id" not in frame:
        frame["parent_prompt_id"] = ""
    if "source_metadata" not in frame:
        frame["source_metadata"] = ""
    frame["content_hash"] = frame["prompt"].map(content_hash)
    frame["is_duplicate"] = False
    frame["duplicate_of_prompt_id"] = ""
    frame["duplicate_group_id"] = ""
    # Preserve source split when present only as a requested grouping hint; all
    # train/test assignments below derive from the parent group deterministically.
    roots = frame["parent_prompt_id"].fillna("").astype(str).str.strip()
    roots = roots.where(roots.ne(""), frame["prompt_id"].astype(str))
    frame["parent_prompt_id"] = frame["parent_prompt_id"].fillna("")
    frame["split"] = roots.map(lambda root: _stable_split(root, seed))
    frame["run_id"] = f"benchmark6000-seed{seed}"
    return frame


def _near_duplicate_pairs(frame, threshold=0.9):
    """Return token-set near duplicates; intended as an audit, not auto-removal."""
    prompts = frame[["prompt_id", "prompt"]].drop_duplicates("prompt_id")
    token_sets = []
    for row in prompts.itertuples(index=False):
        tokens = set(re.findall(r"\w+", normalize_prompt(row.prompt)))
        token_sets.append((str(row.prompt_id), tokens))
    pairs = []
    for (id_a, a), (id_b, b) in combinations(token_sets, 2):
        if not a or not b:
            continue
        score = len(a & b) / len(a | b)
        if score >= threshold and a != b:
            pairs.append({"prompt_id_a": id_a, "prompt_id_b": id_b, "token_jaccard": round(score, 4)})
    return pairs


def _metadata_issues(frame):
    issues = []
    for row in frame.to_dict("records"):
        pid = str(row.get("prompt_id", ""))
        level = str(row.get("benchmark_level", ""))
        if level not in LEVELS:
            issues.append({"prompt_id": pid, "issue": "invalid benchmark_level"})
        if str(row.get("transformation", "")) != EXPECTED_TRANSFORM.get(level):
            issues.append({"prompt_id": pid, "issue": "level/transformation mismatch"})
        if _empty(row.get("prompt")):
            issues.append({"prompt_id": pid, "issue": "missing prompt"})
        if str(row.get("risk_level", "")) not in {"low", "medium", "high"}:
            issues.append({"prompt_id": pid, "issue": "missing or invalid risk_level"})
        if level != "L0" and _empty(row.get("standard_category")):
            issues.append({"prompt_id": pid, "issue": "missing category; requires review"})
        cats = split_categories(row.get("standard_category", ""))
        unknown = sorted(set(cats) - {"sexual", "violence", "disturbing", "illegal_activity",
                                      "political_sensitivity", "copyright"})
        if unknown:
            issues.append({"prompt_id": pid, "issue": f"unknown categories: {unknown}"})
        if level in {"L2", "L3", "L4"}:
            for column in ("parent_prompt_id", "transformation_method", "transformation_parameters"):
                if _empty(row.get(column)):
                    issues.append({"prompt_id": pid, "issue": f"transformed row missing {column}"})
    ids = set(frame["prompt_id"].astype(str)) if "prompt_id" in frame else set()
    for row in frame.to_dict("records"):
        parent = str(row.get("parent_prompt_id", "")).strip()
        if parent and parent not in ids:
            issues.append({"prompt_id": str(row.get("prompt_id", "")), "issue": f"missing parent {parent}"})
    return issues


def prepare(*, l1_path, ring_path, extra_inputs=(), output_dir, seed=42, dry_run=False):
    output_dir = Path(output_dir)
    l1 = read_table(l1_path).fillna("")
    l1 = l1[l1["benchmark_level"].astype(str).eq("L1")].copy()
    l1 = _canonicalize(l1, "L1", approved_only=False, source_name="unified_benchmark_l1", seed=seed)
    # The unified L1 file is the deduplicated source of record. Remove exact
    # normalized duplicates before sampling; source files are not concatenated.
    l1 = l1.drop_duplicates("prompt", keep="first", ignore_index=True)
    l1 = _stratified_sample(l1, TARGET, seed)

    ring = read_table(ring_path).fillna("")
    ring = _canonicalize(ring, "L6", approved_only=False, source_name="Ring-A-Bell", seed=seed)
    frames = [l1, ring]
    input_counts = {"L1": len(l1), "L6": len(ring)}
    review = []
    for level, path in extra_inputs:
        candidate = read_table(path).fillna("")
        approved_rows = candidate[candidate.apply(lambda row: _approved(row, level), axis=1)]
        approved = _canonicalize(approved_rows, level, approved_only=False,
                                 source_name=Path(path).stem, seed=seed)
        pending = _canonicalize(candidate, level, approved_only=False,
                                source_name=Path(path).stem, seed=seed)
        approved_ids = set(approved["prompt_id"].astype(str))
        for row in pending[~pending["prompt_id"].astype(str).isin(approved_ids)].to_dict("records"):
            was_approved = str(row.get("review_status", "")).strip().casefold() in {"approved", "validated"}
            row["review_status"] = "pending"
            if was_approved:
                row["review_reason"] = "required transformation label, parent, method, or parameters missing/invalid"
            else:
                row["review_reason"] = "explicit review_status=approved or validated required"
            review.append(row)
        input_counts[level] = len(approved)
        frames.append(approved)

    candidates = pd.concat(frames, ignore_index=True, sort=False)
    # Keep exact duplicate diagnostics and exclude duplicate text from the
    # candidate set, preserving the first source row deterministically.
    candidates["_norm"] = candidates["prompt"].map(normalize_prompt)
    exact_mask = candidates["_norm"].duplicated(keep="first")
    exact_duplicates = candidates.loc[exact_mask, ["prompt_id", "prompt", "benchmark_level"]].copy()
    candidates = candidates.loc[~exact_mask].drop(columns="_norm").reset_index(drop=True)
    missing = {level: max(0, TARGET - int((candidates["benchmark_level"] == level).sum())) for level in LEVELS}
    excess = {level: max(0, int((candidates["benchmark_level"] == level).sum()) - TARGET) for level in LEVELS}
    # Limit an overfull level deterministically. L1 was stratified above; other
    # levels use the same stratified sampler after review approval.
    selected = []
    for level in LEVELS:
        subset = candidates[candidates["benchmark_level"] == level]
        if len(subset) > TARGET:
            subset = _stratified_sample(subset, TARGET, seed)
        selected.append(subset)
    candidates = pd.concat(selected, ignore_index=True, sort=False)
    # Recompute per-level gaps after exact duplicate removal/sampling.
    level_counts = {level: int((candidates["benchmark_level"] == level).sum()) for level in LEVELS}
    missing = {level: max(0, TARGET - level_counts[level]) for level in LEVELS}
    issues = _metadata_issues(candidates)
    ids = set(candidates["prompt_id"].astype(str))
    duplicate_ids = candidates[candidates["prompt_id"].astype(str).duplicated(keep=False)]
    if len(duplicate_ids):
        issues.append({"issue": f"{len(duplicate_ids)} rows have duplicate prompt_id values"})
    valid_for_final = all(count == TARGET for count in level_counts.values()) and not issues
    near = _near_duplicate_pairs(candidates)
    # Include each source column plus standard columns, preserving Ring-A-Bell
    # attack configuration, case number, seed, and source file.
    for col in STANDARD_COLUMNS:
        if col not in candidates:
            candidates[col] = ""
    candidates["source_id"] = candidates["source_id"].astype(str)
    ordered = STANDARD_COLUMNS + [c for c in candidates.columns if c not in STANDARD_COLUMNS and not c.startswith("_")]
    candidates = candidates[ordered]
    distribution = (candidates.groupby(["benchmark_level", "source_dataset", "standard_category", "risk_level"],
                                       dropna=False).size().rename("count").reset_index())
    audit = {
        "target_per_level": TARGET, "seed": seed, "levels": list(LEVELS),
        "counts_per_level": level_counts,
        "counts_per_source_dataset": {str(k): int(v) for k, v in candidates["source_dataset"].value_counts(dropna=False).items()},
        "risk_category_counts": dict(sorted(Counter(
            category for value in candidates["standard_category"] for category in split_categories(value)
        ).items())),
        "risk_level_counts": {str(k): int(v) for k, v in candidates["risk_level"].value_counts(dropna=False).items()},
        "input_eligible_counts": input_counts,
        "shortfalls": missing, "excess_before_cap": excess,
        "exact_duplicate_rows_excluded": len(exact_duplicates),
        "near_duplicate_pairs_reported": len(near), "near_duplicate_threshold": 0.9,
        "missing_or_invalid_metadata": issues,
        "parent_references_valid": not any("parent" in x.get("issue", "") for x in issues),
        "all_levels_reach_1000": valid_for_final,
        "complete_benchmark_written": valid_for_final,
        "limitations": ([
            "This run creates a candidate manifest only; no unified_benchmark_6000.parquet is written."
        ] if not valid_for_final else []) + [
            "Acquired COCO/Nibbler candidates remain unreviewed and are not counted toward the benchmark.",
            "The preselected L1 and Ring-A-Bell rows are source candidates, not human-validated by this command.",
            "L2/L3/L4 and supplemental L6 require human-approved rows with provenance and parent IDs.",
            "Ring-A-Bell prompt-data licensing needs resolution before redistribution.",
            "Near-duplicate detection uses token-set Jaccard and reports pairs for human review.",
        ],
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_tabular(candidates, output_dir / "benchmark_6000_candidates.parquet")
        write_tabular(distribution, output_dir / "benchmark_6000_candidates_distribution.parquet")
        review_columns = STANDARD_COLUMNS + ["review_status", "review_reason"]
        review_frame = pd.DataFrame(review)
        for col in review_columns:
            if col not in review_frame:
                review_frame[col] = ""
        write_tabular(review_frame, output_dir / "benchmark_6000_candidates_review.parquet")
        write_tabular(exact_duplicates, output_dir / "benchmark_6000_exact_duplicates_excluded.parquet")
        write_tabular(pd.DataFrame(near), output_dir / "benchmark_6000_near_duplicates.parquet")
        (output_dir / "benchmark_6000_candidates_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if valid_for_final:
            write_tabular(candidates, output_dir / "unified_benchmark_6000.parquet")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.json", help="pipeline config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="audit inputs without writing output")
    parser.add_argument("--validate-only", action="store_true", help="validate and report without writing output")
    parser.add_argument("--output-dir", default="data/processed/unified")
    parser.add_argument("--l1", default=None)
    parser.add_argument("--ring-a-bell", default=None)
    parser.add_argument("--input", action="append", default=[], metavar="LEVEL=TABLE",
                        help="reviewed source input for L0/L2/L3/L4/L6; rows need review_status=approved")
    args = parser.parse_args(argv)
    cfg = load_config([args.config])
    extras = []
    for spec in args.input:
        level, sep, path = spec.partition("=")
        if not sep or level not in {"L0", "L2", "L3", "L4", "L6"}:
            parser.error(f"invalid --input {spec!r}; expected L0/L2/L3/L4/L6=TABLE")
        extras.append((level, path))
    audit = prepare(l1_path=args.l1 or cfg["benchmark"]["l1_input"],
                    ring_path=args.ring_a_bell or cfg["datasets"]["ring-a-bell"]["input_table"], extra_inputs=extras,
                    output_dir=args.output_dir, seed=args.seed, dry_run=args.dry_run or args.validate_only)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["all_levels_reach_1000"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
