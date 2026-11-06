"""The frozen prompt tables of E1-E7 (sections 3.1 and 4 of docs/e1_e6_research_plan.md).

    core_v1   E1-E3 (and E7's parents): up to 200 prompts per level L0-L4, L6
    e46_v1    E4-E6: benign + harmful (discovery and validation), near_boundary, adversarial (validation)
    quant_v1  the precision check: 50 benign + 50 harmful prompts from e46_v1's validation split

Candidates come from the Phase D2 review master; standardized rows (I2P,
T2I-RiskyPrompt, Ring-A-Bell) keep their own labels and attack metadata. Only
reviewer-accepted rows are used, unless ``--provisional``: then the proposed
levels are used and the files are named ``*_provisional`` -- for pipeline checks,
never for reported numbers.

Prompts that mention or imply a minor are removed from every group first.
Every row gets evaluation_seed = 42, subset_group and subset_split; next to each
table an audit JSON records its counts, prompt lengths and SHA-256.

    python scripts/make_e1e6_subsets.py --provisional
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.config import DEFAULTS
from visual_synthesis.io import file_sha256, read_table, stratified_sample, write_json, write_table
from visual_synthesis.schema import risk_level, split_categories

SEED = 42
MASTER = Path("data/processed/benchmark_prep/phase_d2/review_master.parquet")
STANDARDIZED = Path("data/processed/unified/benchmark_6000_candidates.parquet")
OUTPUT = Path("data/processed/e1e6")

LEVEL_TRANSFORMATION = {"L0": "original", "L1": "original", "L2": "paraphrase", "L3": "obfuscation",
                        "L4": "indirect", "L5": "multilingual", "L6": "adversarial"}
CORE_GROUPS = {"L0": "benign", "L1": "harmful", "L2": "harmful", "L3": "harmful", "L4": "harmful", "L6": "adversarial"}

# Deliberately broad: a false exclusion costs one prompt, a miss is unacceptable.
MINOR = re.compile(
    r"\b(?:child|children|kid|kids|kiddie|boys?|girls?|teens?|teenage|teenagers?|minors?|underage|under-age|young|"
    r"youth|juvenile|schoolgirls?|schoolboys?|schoolchild|students?|pupils?|bab(?:y|ies)|toddlers?|infants?|"
    r"lolita|loli|shota|preteens?|pre-teens?|daughters?|sons?|nieces?|nephews?|little|tween)\b"
    r"|\b(?:1[0-7]|[1-9])\s*-?\s*(?:yo|y/o|years?\s*-?\s*old)\b",
    re.IGNORECASE,
)
PERSON = re.compile(
    r"\b(?:man|men|woman|women|person|people|lady|ladies|guy|guys|gentleman|couple|players?|skiers?|surfers?|"
    r"skateboarders?|riders?|chef|workers?|tourists?|crowd|bride|groom|adults?)\b", re.IGNORECASE)
# Benign captions that share surface features with sexual prompts (E6's near-boundary group).
NEAR_BOUNDARY = re.compile(
    r"\b(?:swimsuits?|swimwear|bikinis?|beach|bath|bathtub|bathing|shower|beds?|bedroom|statues?|sculptures?|"
    r"paintings?|ballet|dancers?|dancing|embrac\w*|hug\w*|kiss\w*|pool|sunbath\w*|shirtless|towels?|massage|yoga)\b",
    re.IGNORECASE)

COLUMNS = [
    "prompt_id", "prompt", "source_dataset", "standard_category", "risk_level", "benchmark_level", "transformation",
    "parent_prompt_id", "transformation_method", "attack_method", "attack_config", "case_number",
    "evaluation_seed", "subset_group", "subset_split",
]


def blank(series):
    return series.astype("string").str.strip().replace("", pd.NA)


def load_candidates(master_path=MASTER, standardized_path=STANDARDIZED, provisional=False):
    """Every candidate in the standard columns. Reviewed mode keeps accepted rows at their validated level."""
    master = read_table(master_path)
    if not provisional:
        master = master[master["decision"].astype(str).str.strip().eq("accept")]
    master = master.reset_index(drop=True)
    standardized = read_table(standardized_path).drop_duplicates("prompt_id").set_index("prompt_id")
    frame = pd.DataFrame({
        "prompt_id": master["candidate_id"].astype(str).str.removeprefix("STD_"),
        "prompt": master["prompt"].astype(str),
        "source_dataset": master["source_dataset"].astype(str),
        "benchmark_level": blank(master["proposed_level" if provisional else "validated_level"]),
        "parent_prompt_id": blank(master["parent_prompt_id"]),
        "transformation_method": blank(master["transformation_method"]),
    })
    # A reviewer's category wins; then the source dataset's own label; then the triage suggestion.
    frame["standard_category"] = (blank(master["standard_category"])
                                  .fillna(blank(frame["prompt_id"].map(standardized["standard_category"])))
                                  .fillna(blank(master["suggested_standard_category"])))
    for column in ("attack_method", "attack_config", "case_number"):
        frame[column] = frame["prompt_id"].map(standardized[column]) if column in standardized else None
    taxonomy = DEFAULTS["taxonomy"]
    frame["risk_level"] = frame["standard_category"].fillna("").map(lambda v: risk_level(split_categories(v), taxonomy))
    frame["transformation"] = frame["benchmark_level"].map(LEVEL_TRANSFORMATION)
    return frame[frame["benchmark_level"].isin(list(LEVEL_TRANSFORMATION))].reset_index(drop=True)


def safe(frame):
    """Drop every prompt that mentions or implies a minor."""
    return frame[~frame["prompt"].str.contains(MINOR)]


def has_category(frame, category):
    return frame["standard_category"].fillna("").map(lambda v: category in split_categories(v))


def take(frame, n, seed=SEED):
    return frame.sample(n=min(n, len(frame)), random_state=seed)


def length_matched(pool, targets, seed=SEED):
    """For each target in turn, the unused pool row with the closest word count."""
    if len(pool) < len(targets):
        raise ValueError(f"only {len(pool)} candidates to match {len(targets)} prompts")
    pool = pool.sample(frac=1, random_state=seed)  # ties break randomly, but reproducibly
    lengths = pool["prompt"].str.split().str.len().to_numpy(float)
    used = np.zeros(len(pool), bool)
    picked = []
    for length in targets["prompt"].str.split().str.len():
        index = int(np.argmin(np.where(used, np.inf, np.abs(lengths - length))))
        used[index] = True
        picked.append(index)
    return pool.iloc[picked]


def core_table(candidates, per_level=200, seed=SEED):
    """L1 first takes the parents of the L2-L4 candidates, and L2-L4 their linked children
    first, so E2's paired comparisons exist; every remaining slot is sampled."""
    candidates = safe(candidates)
    children = candidates[candidates["benchmark_level"].isin(["L2", "L3", "L4"])]
    linked_to = set(children["parent_prompt_id"].dropna())
    parts, l1_ids = [], set()
    for level, group in CORE_GROUPS.items():
        rows = candidates[candidates["benchmark_level"] == level]
        if level == "L1":
            first = take(rows[rows["prompt_id"].isin(linked_to)], per_level, seed)
            rows = pd.concat([first, fill(rows.drop(first.index), per_level - len(first), seed, stratify=True)])
            l1_ids = set(rows["prompt_id"])
        elif level in ("L2", "L3", "L4"):
            first = take(rows[rows["parent_prompt_id"].isin(l1_ids)], per_level, seed)
            rows = pd.concat([first, fill(rows.drop(first.index), per_level - len(first), seed)])
        else:
            rows = take(rows, per_level, seed)
        parts.append(rows.assign(subset_group=group, subset_split="core"))
    return pd.concat(parts, ignore_index=True)


def fill(rows, n, seed=SEED, stratify=False):
    """``n`` sampled rows; stratified by first category (keeps L1's category mix) when asked."""
    if n <= 0:
        return rows.iloc[:0]
    if stratify and len(rows) > n:
        rows = rows.assign(_stratum=rows["standard_category"].fillna("none").str.split(";").str[0])
        return stratified_sample(rows, n, "_stratum", seed).drop(columns="_stratum")
    return take(rows, n, seed)


def e46_table(candidates, per_split=100, near=50, adversarial=50, seed=SEED):
    """Harmful = L1 sexual; benign = person captions length-matched to them; both split 50/50."""
    harmful = take(safe(candidates[(candidates["benchmark_level"] == "L1") & has_category(candidates, "sexual")]),
                   2 * per_split, seed)
    harmful = harmful.assign(subset_group="harmful",
                             subset_split=["discovery"] * min(per_split, len(harmful))
                             + ["validation"] * max(0, len(harmful) - per_split))
    coco = safe(candidates[candidates["benchmark_level"] == "L0"])
    surface = coco["prompt"].str.contains(NEAR_BOUNDARY)
    pool = coco[~surface & coco["prompt"].str.contains(PERSON)]
    benign = []
    for split in ("discovery", "validation"):
        chosen = length_matched(pool, harmful[harmful["subset_split"] == split], seed)
        pool = pool.drop(chosen.index)
        benign.append(chosen.assign(subset_group="benign", subset_split=split))
    parts = [harmful, *benign,
             take(coco[surface], near, seed).assign(subset_group="near_boundary", subset_split="validation"),
             take(safe(candidates[candidates["benchmark_level"] == "L6"]), adversarial, seed)
             .assign(subset_group="adversarial", subset_split="validation")]
    return pd.concat(parts, ignore_index=True)


def quant_table(e46, n=50, seed=SEED):
    validation = e46[e46["subset_split"] == "validation"]
    parts = [take(validation[validation["subset_group"] == group], n, seed) for group in ("benign", "harmful")]
    return pd.concat(parts, ignore_index=True).assign(subset_split="quant")


def write(frame, path, sources):
    """The table (standard columns, seed 42) and its audit JSON; returns the audit."""
    frame = frame.assign(evaluation_seed=SEED).reindex(columns=COLUMNS).reset_index(drop=True)
    if frame["prompt_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate prompt_id values")
    write_table(frame, path)
    words = frame["prompt"].str.split().str.len()
    audit = {
        "table": str(path), "rows": len(frame), "sha256": file_sha256(path), "evaluation_seed": SEED,
        "provisional": "_provisional" in path.stem,
        "counts": {f"{g}/{s}": int(n) for (g, s), n in frame.groupby(["subset_group", "subset_split"]).size().items()},
        "levels": frame["benchmark_level"].value_counts().to_dict(),
        "categories": frame["standard_category"].fillna("none").value_counts().to_dict(),
        "mean_words": {group: round(float(words[frame["subset_group"] == group].mean()), 2)
                       for group in frame["subset_group"].unique()},
        "sources": sources,
    }
    write_json(audit, path.with_suffix(".json"))
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--standardized", type=Path, default=STANDARDIZED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--provisional", action="store_true", help="use proposed levels; for pipeline checks only")
    parser.add_argument("--per-level", type=int, default=200)
    parser.add_argument("--per-split", type=int, default=100, help="E4-E6 harmful (and benign) prompts per split")
    parser.add_argument("--near", type=int, default=50)
    parser.add_argument("--adversarial", type=int, default=50)
    args = parser.parse_args(argv)

    candidates = load_candidates(args.master, args.standardized, args.provisional)
    if candidates.empty:
        raise SystemExit("no reviewed candidates yet: finish the Phase D2 review, or pass --provisional")
    sources = {str(path): file_sha256(path) for path in (args.master, args.standardized)}
    suffix = "_provisional" if args.provisional else ""
    core = core_table(candidates, args.per_level)
    e46 = e46_table(candidates, args.per_split, args.near, args.adversarial)
    audits = [write(frame, args.output_dir / f"{name}_v1{suffix}.parquet", sources)
              for name, frame in (("core", core), ("e46", e46), ("quant", quant_table(e46)))]
    print(json.dumps({"excluded_minor": int(candidates["prompt"].str.contains(MINOR).sum()),
                      "tables": [{k: a[k] for k in ("table", "rows", "counts", "mean_words")} for a in audits]},
                     indent=2))
    return 0
