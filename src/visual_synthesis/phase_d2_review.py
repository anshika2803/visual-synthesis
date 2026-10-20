"""Manage Phase D1 candidate review batches and prepare a non-generated pilot plan.

All commands are local and deterministic. Candidate preparation does not validate
examples; only explicit reviewer decisions can enter the accepted export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from visual_synthesis.phase_c_candidates import LEVELS, RISK_CATEGORIES
from visual_synthesis.phase_d1_review import (
    DECISIONS, REVIEW_FIELDS, _duplicate_diagnostics, _preserve_decisions,
)
from visual_synthesis.schema import normalize_prompt, split_categories

REVIEW_METADATA_FIELDS = tuple(REVIEW_FIELDS) + ("reviewed_at",)


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    if "candidate_id" not in frame:
        raise ValueError(f"{path} has no candidate_id")
    if frame.candidate_id.astype(str).duplicated().any():
        raise ValueError(f"{path} has duplicate candidate_id rows")
    return frame


def _category_values(row: dict) -> set[str]:
    values: set[str] = set()
    for field in ("standard_category", "suggested_standard_category", "category_assignment"):
        values.update(split_categories(str(row.get(field, ""))))
    return values


def select_batch(master_path: Path, output_path: Path, *, batch_size: int = 75,
                 level: str | None = None, source: str | None = None,
                 category: str | None = None, offset: int = 0,
                 include_uncertain: bool = False) -> dict:
    if not 50 <= batch_size <= 100:
        raise ValueError("batch_size must be between 50 and 100")
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    frame = _read(master_path)
    available = frame.get("candidate_available", pd.Series(["true"] * len(frame))).str.casefold().ne("false")
    decisions = frame.get("decision", pd.Series([""] * len(frame))).str.casefold()
    eligible = available & (decisions.eq("") | (include_uncertain & decisions.eq("uncertain")))
    if level:
        eligible &= frame.proposed_level.astype(str).str.casefold().eq(level.casefold())
    if source:
        eligible &= frame.source_dataset.astype(str).str.casefold().eq(source.casefold())
    selected = frame.loc[eligible].copy()
    if category:
        wanted = category.casefold()
        mask = selected.apply(lambda row: wanted in {v.casefold() for v in _category_values(row.to_dict())}, axis=1)
        selected = selected.loc[mask]
    selected = selected.sort_values("candidate_id", kind="mergesort").iloc[offset:offset + batch_size]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    return {"batch_rows": len(selected), "batch_size": batch_size, "filters": {
        "proposed_level": level or "all", "source": source or "all", "category": category or "all",
        "offset": offset, "include_uncertain": include_uncertain}, "output": str(output_path)}


def merge_batch(master_path: Path, batch_path: Path, *, output_path: Path | None = None) -> dict:
    master, batch = _read(master_path), _read(batch_path)
    master_ids = set(master.candidate_id.astype(str))
    unknown = sorted(set(batch.candidate_id.astype(str)) - master_ids)
    if unknown:
        raise ValueError(f"batch contains candidate IDs absent from master: {unknown[:5]}")
    target = output_path or master_path
    by_id = master.set_index(master.candidate_id.astype(str), drop=False)
    changed = 0
    for incoming in batch.to_dict("records"):
        cid = str(incoming["candidate_id"])
        idx = by_id.index.get_loc(cid)
        for field in REVIEW_METADATA_FIELDS:
            value = str(incoming.get(field, "")).strip()
            if value:
                if field == "decision" and value.casefold() not in DECISIONS:
                    raise ValueError(f"invalid decision for {cid}: {value}")
                if field == "validated_level" and value not in LEVELS:
                    raise ValueError(f"invalid validated_level for {cid}: {value}")
                if field == "standard_category":
                    cats = set(split_categories(value))
                    if cats - RISK_CATEGORIES:
                        raise ValueError(f"invalid standard_category for {cid}: {value}")
                prior = str(master.at[idx, field]) if field in master else ""
                if prior and prior != value:
                    # Explicit edits are allowed; they remain auditable in evidence.
                    old_evidence = str(master.at[idx, "review_evidence"]) if "review_evidence" in master else ""
                    note = f"{field} changed from {prior!r} to {value!r}"
                    master.at[idx, "review_evidence"] = "; ".join(filter(None, [old_evidence, note]))
                if field not in master:
                    master[field] = ""
                master.at[idx, field] = value
                changed += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(target, index=False)
    return {"master_rows": len(master), "batch_rows": len(batch), "review_fields_merged": changed,
            "output": str(target)}


def extend_master(master_path: Path, standardized_candidates_path: Path,
                  output_path: Path | None = None, *, near_threshold: float = 0.9) -> dict:
    """Append preselected L1/L6 source candidates without treating them as reviewed."""
    master = _read(master_path)
    source = pd.read_csv(standardized_candidates_path, keep_default_na=False, dtype=str)
    required = {"prompt_id", "prompt", "benchmark_level", "source_dataset", "standard_category"}
    if required - set(source):
        raise ValueError(f"standardized candidate source missing columns: {sorted(required - set(source))}")
    source = source[source.benchmark_level.isin(["L1", "L6"])].copy()
    source_ids = set(master.candidate_id.astype(str))
    rows = []
    for row in source.to_dict("records"):
        cid = "STD_" + str(row["prompt_id"])
        if cid in source_ids:
            continue
        source_id = str(row.get("source_id", row["prompt_id"]))
        metadata = {key: row.get(key, "") for key in (
            "original_category", "risk_level", "transformation", "source_metadata",
            "attack_method", "attack_config", "case_number", "evaluation_seed", "source_file",
        )}
        rows.append({
            "candidate_id": cid, "prompt": row["prompt"], "proposed_level": row["benchmark_level"],
            "triage_confidence": "", "source_dataset": row["source_dataset"], "source_id": source_id,
            "caption_id": "", "image_id": "", "source_split": str(row.get("split", "")),
            "starting_prompt": row["prompt"], "participant_rewrite_evidence": "",
            "attack_mode": str(row.get("attack_method", "")), "image_failure_type": "",
            "validator_text_safe_votes": "", "validator_text_unsafe_votes": "",
            "validator_text_other_votes": "", "validator_image_safe_votes": "",
            "validator_image_unsafe_votes": "", "validator_image_other_votes": "",
            "suggested_standard_category": row["standard_category"],
            "source_provenance": json.dumps(metadata, ensure_ascii=False),
            "evidence_summary": "preselected standardized source candidate; human review required",
        })
    addition = pd.DataFrame(rows)
    combined = pd.concat([master, addition], ignore_index=True, sort=False)
    combined, near_pairs = _duplicate_diagnostics(combined, near_threshold)
    previous_path = master_path
    combined, preservation = _preserve_decisions(combined, previous_path)
    target = output_path or master_path
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False)
    pd.DataFrame(near_pairs, columns=["candidate_id_a", "candidate_id_b", "token_jaccard", "threshold"]).to_csv(
        target.with_name("duplicate_pairs.csv"), index=False)
    by_level = combined.loc[combined.candidate_id.str.startswith("STD_")].proposed_level.value_counts().to_dict()
    return {"added_rows": len(addition), "added_by_level": {str(k): int(v) for k, v in by_level.items()},
            "master_rows": len(combined), "near_duplicate_pairs": len(near_pairs),
            "preserved_decisions": preservation["preserved_decisions"],
            "output": str(target)}


def progress_report(master_path: Path, validated_path: Path | None = None) -> dict:
    frame = _read(master_path)
    decisions = frame.get("decision", pd.Series([""] * len(frame))).str.strip().str.casefold()
    available = frame.get("candidate_available", pd.Series(["true"] * len(frame))).str.casefold().ne("false")
    levels = list(LEVELS) + ["unclassified"]
    by_level = {}
    for level in levels:
        subset = frame.loc[available & frame.proposed_level.astype(str).eq(level)]
        d = decisions.loc[subset.index]
        by_level[level] = {"candidates": len(subset), "reviewed": int(d.isin(DECISIONS).sum()),
                           "accepted": int(d.eq("accept").sum()), "rejected": int(d.eq("reject").sum()),
                           "uncertain": int(d.eq("uncertain").sum()),
                           "remaining": int((~d.isin(DECISIONS)).sum())}
    accepted = frame.loc[available & decisions.eq("accept")]
    accepted_by_source = accepted.source_dataset.value_counts().sort_index().to_dict() if len(accepted) else {}
    accepted_by_category = Counter(cat for value in accepted.get("standard_category", [])
                                   for cat in split_categories(value))
    source_pool_by_category = Counter(cat for _, row in frame.loc[available].iterrows()
                                      for cat in _category_values(row.to_dict()))
    validated_counts = Counter()
    if validated_path and validated_path.exists():
        validated = _read(validated_path)
        level_field = "validated_level" if "validated_level" in validated else "benchmark_level"
        if level_field in validated:
            validated_counts.update(validated[level_field].astype(str))
    return {"review_master": str(master_path), "total_available_candidates": int(available.sum()),
            "by_proposed_level": by_level,
            "candidate_supply_gap_to_1000": {level: max(0, 1000 - by_level[level]["candidates"]) for level in LEVELS},
            "accepted_by_source": {str(k): int(v) for k, v in accepted_by_source.items()},
            "accepted_by_category": {cat: int(accepted_by_category.get(cat, 0)) for cat in sorted(RISK_CATEGORIES)},
            "candidate_pool_by_category": {cat: int(source_pool_by_category.get(cat, 0)) for cat in sorted(RISK_CATEGORIES)},
            "human_validated_manifest": str(validated_path) if validated_path else "not supplied",
            "human_validated_by_level": {level: int(validated_counts.get(level, 0)) for level in LEVELS},
            "shortfall_to_1000_human_validated_by_level": {level: max(0, 1000 - int(validated_counts.get(level, 0))) for level in LEVELS},
            "note": "Accepted decisions are review progress; only rows in the Phase D1 validated export count toward validated totals."}


def make_pilot_plan(l1_path: Path, output_path: Path, *, per_level: int = 25, seed: int = 42) -> dict:
    if not 20 <= per_level <= 30:
        raise ValueError("per_level must be between 20 and 30")
    l1 = pd.read_csv(l1_path, keep_default_na=False, dtype=str)
    required = {"prompt_id", "prompt", "standard_category"}
    if required - set(l1):
        raise ValueError(f"L1 source missing columns: {sorted(required - set(l1))}")
    if "benchmark_level" in l1:
        l1 = l1[l1.benchmark_level.eq("L1")]
    l1 = l1[l1.standard_category.map(lambda v: bool(set(split_categories(v)) & RISK_CATEGORIES))]
    l1 = l1.drop_duplicates("prompt_id")
    # Select deterministic parent set with risk-category coverage; all transformed
    # variants remain blank until an approved human or local process creates them.
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in l1.to_dict("records"):
        cats = sorted(set(split_categories(row["standard_category"])) & RISK_CATEGORIES)
        if not cats:
            continue
        primary = cats[0]
        pools[primary].append(row)
    for category in pools:
        pools[category].sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['prompt_id']}".encode("utf-8")).hexdigest())
    categories = sorted(pools)
    chosen = []
    while len(chosen) < per_level:
        advanced = False
        for category in categories:
            if pools[category] and len(chosen) < per_level:
                chosen.append((category, pools[category].pop(0)))
                advanced = True
        if not advanced:
            break
    if len(chosen) < per_level:
        raise ValueError(f"only {len(chosen)} eligible L1 parents available; requested {per_level}")
    plans = {
        "L2": ("human-authored meaning-preserving paraphrase", "Meaning and risk remain; wording/syntax changes materially; no new risk added."),
        "L3": ("seeded local character/word obfuscation; edit log required", "Obfuscation is visible and recoverable; record exact edits; preserve semantic risk."),
        "L4": ("human-authored indirect/contextual framing", "Harmful intent is indirect/contextual; no explicit L1 wording; category preserved."),
        "L6": ("human-authored adversarial rewrite informed by Ring-A-Bell method", "Explicitly targets model behavior or safety boundary; do not infer attack success from wording."),
    }
    rows = []
    for level, (method, criteria) in plans.items():
        for index, (category, parent) in enumerate(chosen, 1):
            rows.append({"pilot_candidate_id": f"PILOT_{level}_{index:03d}", "candidate_prompt": "",
                         "proposed_level": level, "risk_category": parent["standard_category"],
                         "primary_category_stratum": category, "parent_prompt_id": parent["prompt_id"],
                         "parent_prompt": parent["prompt"], "source_dataset": parent.get("source_dataset", "standardized_L1"),
                         "source_id": parent.get("source_id", ""), "transformation_method_planned": method,
                         "transformation_parameters": "", "provenance": f"Phase D2 pilot plan; seed={seed}; no transformation executed",
                         "quality_review_criteria": criteria, "decision": "", "reviewer_id": "",
                         "validated": False, "status": "planned_not_generated"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {"per_level": per_level, "planned_rows": len(rows), "parent_count": len(chosen),
            "parents_by_primary_category": dict(Counter(category for category, _ in chosen)),
            "generated_prompt_rows": 0, "validated_rows": 0, "output": str(output_path)}


def render_dashboard(report: dict, output: Path | None = None, fmt: str = "markdown") -> str:
    if fmt == "json":
        content = json.dumps(report, indent=2) + "\n"
    elif fmt == "markdown":
        lines = ["# Phase D2 review progress", "", "| Proposed level | Candidates | Reviewed | Accepted | Rejected | Uncertain | Remaining |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for level, row in report["by_proposed_level"].items():
            lines.append(f"| {level} | {row['candidates']} | {row['reviewed']} | {row['accepted']} | {row['rejected']} | {row['uncertain']} | {row['remaining']} |")
        lines += ["", "## Accepted by source", ""]
        lines += [f"- {source}: {count}" for source, count in report["accepted_by_source"].items()] or ["- None"]
        lines += ["", "## Accepted by risk category", ""]
        lines += [f"- {category}: {count}" for category, count in report["accepted_by_category"].items()]
        lines += ["", "## Candidate supply gap to 1,000", "", "These are proposed-level candidate gaps before human validation; unclassified rows are excluded.", ""]
        lines += [f"- {level}: {count}" for level, count in report["candidate_supply_gap_to_1000"].items()]
        lines += ["", "## Candidate pool by risk category", ""]
        lines += [f"- {category}: {count}" for category, count in report["candidate_pool_by_category"].items()]
        content = "\n".join(lines) + "\n"
    else:
        raise ValueError("format must be markdown or json")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    return content


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    batch = sub.add_parser("select-batch")
    batch.add_argument("--master", type=Path, required=True); batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--batch-size", type=int, default=75); batch.add_argument("--level")
    batch.add_argument("--source"); batch.add_argument("--category"); batch.add_argument("--offset", type=int, default=0)
    batch.add_argument("--include-uncertain", action="store_true")
    merge = sub.add_parser("merge-batch")
    merge.add_argument("--master", type=Path, required=True); merge.add_argument("--batch", type=Path, required=True)
    merge.add_argument("--output", type=Path)
    extend = sub.add_parser("extend-master", help="append preselected standardized L1/L6 candidates")
    extend.add_argument("--master", type=Path, required=True); extend.add_argument("--standardized-candidates", type=Path, required=True)
    extend.add_argument("--output", type=Path); extend.add_argument("--near-threshold", type=float, default=0.9)
    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("--master", type=Path, required=True); dashboard.add_argument("--output", type=Path)
    dashboard.add_argument("--validated", type=Path, help="Phase D1 validated_candidates.csv export")
    dashboard.add_argument("--format", choices=("markdown", "json"), default="markdown")
    pilot = sub.add_parser("pilot-plan")
    pilot.add_argument("--l1", type=Path, required=True); pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--per-level", type=int, default=25); pilot.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "select-batch":
        result = select_batch(args.master, args.output, batch_size=args.batch_size, level=args.level,
                              source=args.source, category=args.category, offset=args.offset,
                              include_uncertain=args.include_uncertain)
    elif args.command == "merge-batch":
        result = merge_batch(args.master, args.batch, output_path=args.output)
    elif args.command == "extend-master":
        result = extend_master(args.master, args.standardized_candidates, output_path=args.output,
                               near_threshold=args.near_threshold)
    elif args.command == "dashboard":
        report = progress_report(args.master, args.validated)
        print(render_dashboard(report, args.output, args.format), end="")
        return 0
    else:
        result = make_pilot_plan(args.l1, args.output, per_level=args.per_level, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
