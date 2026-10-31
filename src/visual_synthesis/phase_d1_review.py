"""Human-in-the-loop review batches for Phase C benchmark candidates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from visual_synthesis.io import read_table
from visual_synthesis.phase_c_candidates import LEVELS, RISK_CATEGORIES
from visual_synthesis.schema import normalize_prompt, split_categories

DECISIONS = ("accept", "reject", "uncertain")
REVIEW_FIELDS = (
    "decision", "validated_level", "standard_category", "reviewer_confidence",
    "reviewer_id", "reviewer_notes", "parent_prompt_id", "transformation_method",
    "transformation_parameters",
)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_prompt(text)))


def _frame_from_candidates(coco: pd.DataFrame, nibbler: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in coco.to_dict("records"):
        rows.append({
            "candidate_id": str(row["candidate_id"]), "prompt": str(row.get("prompt", "")),
            "proposed_level": str(row.get("candidate_level", "L0")),
            "triage_confidence": row.get("triage_confidence", ""),
            "source_dataset": str(row.get("source_dataset", "COCO_Captions_2017")),
            "source_id": str(row.get("source_id", row.get("caption_id", ""))),
            "caption_id": str(row.get("caption_id", "")), "image_id": str(row.get("image_id", "")),
            "source_split": str(row.get("source_split", "")),
            "starting_prompt": "", "participant_rewrite_evidence": "",
            "attack_mode": "", "image_failure_type": "",
            "validator_text_safe_votes": "", "validator_text_unsafe_votes": "",
            "validator_text_other_votes": "", "validator_image_safe_votes": "",
            "validator_image_unsafe_votes": "", "validator_image_other_votes": "",
            "suggested_standard_category": "",
            "source_provenance": "",
            "evidence_summary": str(row.get("review_flags", "")),
        })
    for row in nibbler.to_dict("records"):
        rows.append({
            "candidate_id": str(row["candidate_id"]), "prompt": str(row.get("submitted_prompt") or row.get("prompt", "")),
            "proposed_level": str(row.get("candidate_level", "unclassified")),
            "triage_confidence": row.get("triage_confidence", ""),
            "source_dataset": str(row.get("source_dataset", "Adversarial_Nibbler")),
            "source_id": str(row.get("source_id", "")), "caption_id": "", "image_id": "",
            "source_split": str(row.get("source_split", "")),
            "starting_prompt": str(row.get("source_prompt", row.get("submitted_prompt", ""))),
            "participant_rewrite_evidence": str(row.get("participant_rewrite_evidence", row.get("rewritten_prompt", ""))),
            "attack_mode": str(row.get("text_attack_mode", "")),
            "image_failure_type": str(row.get("image_failure_type", "")),
            "validator_text_safe_votes": row.get("validator_text_safe_votes", ""),
            "validator_text_unsafe_votes": row.get("validator_text_unsafe_votes", ""),
            "validator_text_other_votes": row.get("validator_text_other_votes", ""),
            "validator_image_safe_votes": row.get("validator_image_safe_votes", ""),
            "validator_image_unsafe_votes": row.get("validator_image_unsafe_votes", ""),
            "validator_image_other_votes": row.get("validator_image_other_votes", ""),
            "suggested_standard_category": str(row.get("suggested_standard_category", "")),
            "source_provenance": str(row.get("source_provenance", "")),
            "evidence_summary": str(row.get("review_flags", "")),
        })
    return pd.DataFrame(rows)


def _duplicate_diagnostics(frame: pd.DataFrame, near_threshold: float) -> tuple[pd.DataFrame, list[dict]]:
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for row in frame.to_dict("records"):
        norm = normalize_prompt(row.get("prompt", ""))
        if norm:
            exact_groups[norm].append(str(row["candidate_id"]))
    exact_of = {}
    exact_gid = {}
    for norm, ids in exact_groups.items():
        if len(ids) > 1:
            ordered = sorted(ids)
            group_id = "EXACT_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
            for candidate_id in ordered:
                exact_of[candidate_id] = [other for other in ordered if other != candidate_id]
                exact_gid[candidate_id] = group_id

    token_sets = {str(row.candidate_id): _tokens(str(row.prompt)) for row in frame.itertuples(index=False)}
    normalized = {str(row.candidate_id): normalize_prompt(str(row.prompt)) for row in frame.itertuples(index=False)}
    ids = sorted(token_sets)
    near_pairs = []
    near_by_id: dict[str, list[dict]] = defaultdict(list)
    for id_a, id_b in itertools.combinations(ids, 2):
        a, b = token_sets[id_a], token_sets[id_b]
        if not a or not b or normalized[id_a] == normalized[id_b]:
            continue
        score = len(a & b) / len(a | b)
        if score >= near_threshold:
            pair = {"candidate_id_a": id_a, "candidate_id_b": id_b,
                    "token_jaccard": round(score, 4), "threshold": near_threshold}
            near_pairs.append(pair)
            near_by_id[id_a].append({"candidate_id": id_b, "token_jaccard": pair["token_jaccard"]})
            near_by_id[id_b].append({"candidate_id": id_a, "token_jaccard": pair["token_jaccard"]})
    out = frame.copy()
    out["exact_duplicate_group"] = out.candidate_id.map(exact_gid).fillna("")
    out["exact_duplicate_candidate_ids"] = out.candidate_id.map(
        lambda cid: ";".join(exact_of.get(str(cid), [])))
    out["near_duplicate_matches"] = out.candidate_id.map(
        lambda cid: json.dumps(near_by_id.get(str(cid), []), separators=(",", ":")))
    return out, near_pairs


def _preserve_decisions(fresh: pd.DataFrame, previous_path: Path | None) -> tuple[pd.DataFrame, dict]:
    for column in REVIEW_FIELDS:
        fresh[column] = ""
    if previous_path is None or not previous_path.exists():
        fresh["candidate_available"] = True
        return fresh, {"preserved_decisions": 0, "stale_review_rows_retained": 0, "new_candidate_rows": len(fresh)}
    previous = pd.read_csv(previous_path, keep_default_na=False, dtype=str)
    # Phase D2 may keep review timestamps and an edit-history field alongside
    # the standard D1 review columns. Carry those through candidate refreshes.
    auxiliary_review_fields = [field for field in ("reviewed_at", "review_evidence") if field in previous]
    for column in auxiliary_review_fields:
        if column not in fresh:
            fresh[column] = ""
    if "candidate_id" not in previous:
        raise ValueError("previous review batch has no candidate_id column")
    if previous.candidate_id.astype(str).duplicated().any():
        raise ValueError("previous review batch contains duplicate candidate_id rows")
    prev = previous.set_index(previous.candidate_id.astype(str), drop=False)
    fresh_ids = set(fresh.candidate_id.astype(str))
    preserved = 0
    for index, row in fresh.iterrows():
        cid = str(row["candidate_id"])
        if cid not in prev.index:
            continue
        prior = prev.loc[cid]
        if isinstance(prior, pd.DataFrame):
            raise ValueError(f"previous review batch has ambiguous candidate_id: {cid}")
        for field in (*REVIEW_FIELDS, *auxiliary_review_fields):
            if field in prior:
                fresh.at[index, field] = prior[field]
        if any(str(prior.get(field, "")).strip() for field in (*REVIEW_FIELDS, *auxiliary_review_fields)):
            preserved += 1
    stale = previous[~previous.candidate_id.astype(str).isin(fresh_ids)].copy()
    if len(stale):
        for column in fresh.columns:
            if column not in stale:
                stale[column] = ""
        stale["candidate_available"] = False
        fresh["candidate_available"] = True
        fresh = pd.concat([fresh, stale[fresh.columns]], ignore_index=True)
    else:
        fresh["candidate_available"] = True
    return fresh, {
        "preserved_decisions": preserved, "stale_review_rows_retained": len(stale),
        "new_candidate_rows": len(fresh) - len(stale) - preserved,
    }


def prepare_review_batch(*, coco_path: Path, nibbler_path: Path, output_path: Path,
                         previous_path: Path | None = None, near_threshold: float = 0.9) -> dict:
    if not 0 <= near_threshold <= 1:
        raise ValueError("near_threshold must be between 0 and 1")
    coco = pd.read_csv(coco_path, keep_default_na=False, dtype=str)
    nibbler = pd.read_csv(nibbler_path, keep_default_na=False, dtype=str)
    candidates = _frame_from_candidates(coco, nibbler)
    if candidates.candidate_id.astype(str).duplicated().any():
        raise ValueError("candidate manifests have duplicate candidate_id values")
    candidates, duplicate_pairs = _duplicate_diagnostics(candidates, near_threshold)
    review, preservation = _preserve_decisions(candidates, previous_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_path, index=False)
    duplicate_path = output_path.with_name("duplicate_pairs.csv")
    pd.DataFrame(duplicate_pairs, columns=["candidate_id_a", "candidate_id_b", "token_jaccard", "threshold"]).to_csv(
        duplicate_path, index=False)
    exact_group_count = int(review.exact_duplicate_group.astype(bool).sum())
    audit = {
        "candidate_rows": len(candidates), "review_batch_rows": len(review),
        "exact_duplicate_candidate_rows": exact_group_count,
        "exact_duplicate_groups": int(review.loc[review.exact_duplicate_group.ne(""), "exact_duplicate_group"].nunique()),
        "near_duplicate_pairs": len(duplicate_pairs), "near_duplicate_threshold": near_threshold,
        "decision_counts": {decision: int(review.decision.astype(str).str.casefold().eq(decision).sum())
                            for decision in DECISIONS},
        **preservation,
        "review_batch": str(output_path), "duplicate_pairs": str(duplicate_path),
    }
    output_path.with_name("review_batch_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def _parse_categories(value: str) -> set[str]:
    return set(split_categories(value))


def validate_review_batch(review_path: Path, *, output_dir: Path,
                          l1_path: Path | None = None) -> dict:
    review = pd.read_csv(review_path, keep_default_na=False, dtype=str)
    required = {"candidate_id", "prompt", "source_dataset", *REVIEW_FIELDS}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"review batch is missing columns: {missing}")
    duplicate_review = set(review.loc[review.candidate_id.duplicated(keep=False), "candidate_id"].astype(str))
    available = review[review.get("candidate_available", pd.Series([True] * len(review))).astype(str).str.casefold().ne("false")]
    accepted = []
    issues = []
    accepted_by_id = {}
    parent_categories = {}
    if l1_path:
        l1 = read_table(l1_path)
        if {"prompt_id", "standard_category"}.issubset(l1.columns):
            for row in l1.to_dict("records"):
                if row.get("benchmark_level", "L1") == "L1":
                    parent_categories[str(row["prompt_id"])] = str(row["standard_category"])

    for row in available.to_dict("records"):
        cid = str(row["candidate_id"])
        decision = str(row.get("decision", "")).strip().casefold()
        if decision not in DECISIONS:
            if decision:
                issues.append({"candidate_id": cid, "severity": "error", "issue": f"invalid decision: {decision}"})
            continue
        if decision != "accept":
            continue
        if cid in duplicate_review:
            issues.append({"candidate_id": cid, "severity": "error", "issue": "duplicate review rows"})
            continue
        level = str(row.get("validated_level", "")).strip()
        if level not in LEVELS:
            issues.append({"candidate_id": cid, "severity": "error", "issue": "accepted row has missing/invalid level"})
            continue
        categories = _parse_categories(str(row.get("standard_category", "")))
        if level == "L0":
            if categories:
                issues.append({"candidate_id": cid, "severity": "error", "issue": "L0 must not carry a risk category"})
                continue
        elif not categories or categories - RISK_CATEGORIES:
            issues.append({"candidate_id": cid, "severity": "error", "issue": "missing or unknown standard risk category"})
            continue
        try:
            confidence = float(row.get("reviewer_confidence", ""))
        except (ValueError, TypeError):
            confidence = -1
        if not 0 <= confidence <= 1 or not str(row.get("reviewer_id", "")).strip():
            issues.append({"candidate_id": cid, "severity": "error", "issue": "reviewer id/confidence missing or invalid"})
            continue
        if not normalize_prompt(row.get("prompt", "")):
            issues.append({"candidate_id": cid, "severity": "error", "issue": "empty prompt"})
            continue
        parent = str(row.get("parent_prompt_id", "")).strip()
        if level in {"L2", "L3", "L4"}:
            if not parent:
                issues.append({"candidate_id": cid, "severity": "error", "issue": "transformed prompt requires parent_prompt_id"})
                continue
            if not row.get("transformation_method", "").strip() or not row.get("transformation_parameters", "").strip():
                issues.append({"candidate_id": cid, "severity": "error", "issue": "transformed prompt requires method and parameters"})
                continue
            if parent in accepted_by_id:
                parent_level = accepted_by_id[parent]["validated_level"]
                parent_cats = _parse_categories(accepted_by_id[parent]["standard_category"])
            elif parent in parent_categories:
                parent_level = "L1"
                parent_cats = _parse_categories(parent_categories[parent])
            else:
                # Parent candidate may appear later in the CSV; resolve in a second pass.
                parent_level, parent_cats = "", set()
            if parent_level and parent_level != "L1":
                issues.append({"candidate_id": cid, "severity": "error", "issue": "transformation parent must be L1"})
                continue
            if parent_level and not parent_cats:
                issues.append({"candidate_id": cid, "severity": "error", "issue": "parent L1 has no risk category"})
                continue
            if parent_cats and categories != parent_cats:
                issues.append({"candidate_id": cid, "severity": "error", "issue": "risk category differs from parent"})
                continue
            if not parent_level:
                row["_pending_parent_check"] = True
        row["validated_level"] = level
        row["standard_category"] = ";".join(sorted(categories))
        row["reviewer_confidence"] = confidence
        accepted.append(row)
        accepted_by_id[cid] = row

    # Resolve transformed rows whose accepted L1 parent appears later in the batch.
    kept = []
    accepted_ids = {str(row["candidate_id"]) for row in accepted}
    for row in accepted:
        if not row.pop("_pending_parent_check", False):
            kept.append(row)
            continue
        parent = str(row.get("parent_prompt_id", "")).strip()
        p = accepted_by_id.get(parent)
        if p and p.get("validated_level") == "L1":
            if _parse_categories(p.get("standard_category", "")) == _parse_categories(row.get("standard_category", "")):
                kept.append(row)
            else:
                issues.append({"candidate_id": row["candidate_id"], "severity": "error", "issue": "risk category differs from parent"})
        else:
            issues.append({"candidate_id": row["candidate_id"], "severity": "error", "issue": "parent must identify an accepted L1 prompt or standardized L1 source row"})
    accepted = kept

    # Exact normalized duplicates cannot both be accepted in the validated manifest.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in accepted:
        groups[normalize_prompt(row["prompt"])].append(row)
    duplicate_ids = {str(row["candidate_id"]) for group in groups.values() if len(group) > 1 for row in group}
    for cid in sorted(duplicate_ids):
        issues.append({"candidate_id": cid, "severity": "error", "issue": "accepted exact duplicate; resolve duplicate group before export"})
    accepted = [row for row in accepted if str(row["candidate_id"]) not in duplicate_ids]

    for row in accepted:
        row.update({"candidate_level": row["validated_level"], "benchmark_level": row["validated_level"],
                    "category_assignment": "human_reviewed", "review_decision": "accept",
                    "validated": True, "validation_status": "human_validated", "human_review_required": False})
    accepted_columns = list(review.columns) + ["validated", "validation_status", "human_review_required"]
    accepted_frame = pd.DataFrame(accepted, columns=list(dict.fromkeys(accepted_columns)))
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_frame.to_csv(output_dir / "validated_candidates.csv", index=False)
    issues_frame = pd.DataFrame(issues, columns=["candidate_id", "severity", "issue"])
    issues_frame.to_csv(output_dir / "review_validation_issues.csv", index=False)
    counts_level = accepted_frame.groupby("validated_level").size().to_dict() if len(accepted_frame) else {}
    counts_source = accepted_frame.groupby("source_dataset").size().to_dict() if len(accepted_frame) else {}
    category_counts = Counter(category for value in accepted_frame.get("standard_category", [])
                              for category in split_categories(value))
    source_names = sorted(set(available.source_dataset.astype(str)))
    action_counts = {decision: int(review.decision.astype(str).str.casefold().eq(decision).sum())
                     for decision in DECISIONS}
    summary = {
        "review_rows": len(review), "decision_counts": action_counts,
        "accepted_export_rows": len(accepted_frame),
        "accepted_by_level": {level: int(counts_level.get(level, 0)) for level in LEVELS},
        "accepted_by_source": {source: int(counts_source.get(source, 0)) for source in source_names},
        "accepted_by_category": {category: int(category_counts.get(category, 0))
                                 for category in sorted(RISK_CATEGORIES)},
        "shortfall_to_1000_by_level": {level: max(0, 1000 - int(counts_level.get(level, 0)))
                                       for level in LEVELS},
        "validation_issue_count": len(issues),
        "near_duplicate_pairs_are_review_warnings_not_automatic_rejections": True,
        "final_6000_benchmark_created": False,
        "outputs": {"accepted": str(output_dir / "validated_candidates.csv"),
                    "issues": str(output_dir / "review_validation_issues.csv")},
    }
    (output_dir / "accepted_counts.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="create/refresh a review batch while preserving human decisions")
    prep.add_argument("--coco", type=Path, required=True)
    prep.add_argument("--nibbler", type=Path, required=True)
    prep.add_argument("--review-csv", type=Path, required=True)
    prep.add_argument("--previous", type=Path, help="previous edited batch to preserve; defaults to --review-csv if it exists")
    prep.add_argument("--near-threshold", type=float, default=0.9)
    validate = sub.add_parser("validate", help="export only complete human accept decisions")
    validate.add_argument("--review-csv", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--l1", type=Path, help="standardized L1 source used to verify transformed parents")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        previous = args.previous if args.previous else args.review_csv if args.review_csv.exists() else None
        result = prepare_review_batch(coco_path=args.coco, nibbler_path=args.nibbler,
                                      output_path=args.review_csv, previous_path=previous,
                                      near_threshold=args.near_threshold)
    else:
        result = validate_review_batch(args.review_csv, output_dir=args.output_dir, l1_path=args.l1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
