"""Build a schema-aligned, review-aware prompt-candidate manifest for Phase E1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from visual_synthesis.config import DEFAULTS
from visual_synthesis.io import write_tabular
from visual_synthesis.schema import (
    BENCHMARK_LEVELS, RISK_LEVELS, STANDARD_COLUMNS, content_hash, normalize_prompt,
    risk_level as risk_for_categories, split_categories, validate_table,
)

RUN_ID = "phase-e1-unified-prompt-candidates-v1"
LEVEL_TO_TRANSFORMATION = {
    "L0": "original", "L1": "original", "L2": "paraphrase",
    "L3": "obfuscation", "L4": "indirect", "L5": "multilingual",
    "L6": "adversarial",
}
EXTRA_COLUMNS = [
    "source_candidate_id", "proposed_benchmark_level", "proposed_standard_category",
    "proposed_transformation", "reviewed_level", "reviewed_standard_category",
    "validated_level", "validated_standard_category", "review_decision", "review_status",
    "reviewer_confidence", "reviewer_id", "reviewer_notes", "triage_confidence",
    "human_validated", "validation_status",
]
CANDIDATE_COLUMNS = STANDARD_COLUMNS + EXTRA_COLUMNS
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).fillna("").astype(str)
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def _as_bool(value) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _source_metadata(existing: str = "", **extra) -> str:
    """Preserve canonical semicolon key/value metadata with safely escaped values."""
    pairs = []
    if existing:
        for item in str(existing).split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                if key.strip():
                    pairs.append((key.strip(), value))
            elif item:
                pairs.append(("source_metadata_original_unparsed", item))
    present = {key for key, _ in pairs}
    for key, value in extra.items():
        if value is None or value == "":
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key in present:
            pairs = [(old_key, old_value) for old_key, old_value in pairs if old_key != key]
        pairs.append((key, str(value)))
        present.add(key)
    return ";".join(f"{key}={quote(value, safe=':/_-. ') }" for key, value in pairs)


def parse_source_metadata(value: str) -> dict[str, str]:
    """Decode source_metadata values for downstream audit or contributor tooling."""
    from urllib.parse import unquote

    result = {}
    for item in str(value).split(";"):
        if "=" in item:
            key, val = item.split("=", 1)
            result[key] = unquote(val)
    return result


def _base_record(row: dict, *, origin_candidate_id: str = "", proposed_level: str | None = None,
                 proposed_category: str | None = None, source_metadata: str | None = None,
                 triage_confidence: str = "") -> dict:
    prompt_id = str(row.get("prompt_id", "")).strip()
    prompt = str(row.get("prompt", "")).strip()
    categories = split_categories(row.get("standard_category", ""))
    level = proposed_level if proposed_level is not None else str(row.get("benchmark_level", "")).strip()
    known_level = level if level in BENCHMARK_LEVELS else ""
    source_risk = str(row.get("risk_level", "")).strip()
    current_risk = source_risk if source_risk in RISK_LEVELS else ""
    if known_level == "L0" and not categories:
        current_risk = "low"
    elif categories:
        current_risk = risk_for_categories(categories, DEFAULTS["taxonomy"])
    transform = str(row.get("transformation", "")).strip()
    if known_level:
        transform = LEVEL_TO_TRANSFORMATION.get(known_level, transform)
    data = {
        "prompt_id": prompt_id,
        "source_dataset": str(row.get("source_dataset", "")).strip(),
        "source_id": str(row.get("source_id", "")).strip(),
        "prompt": prompt,
        "original_category": str(row.get("original_category", "")).strip(),
        "standard_category": ";".join(categories),
        "risk_level": current_risk,
        "benchmark_level": known_level,
        "transformation": transform,
        "split": str(row.get("split", "")).strip(),
        "is_duplicate": False,
        "duplicate_of_prompt_id": "",
        "duplicate_group_id": "",
        "parent_prompt_id": str(row.get("parent_prompt_id", "")).strip(),
        "source_metadata": source_metadata if source_metadata is not None else str(row.get("source_metadata", "")),
        "content_hash": content_hash(prompt),
        "run_id": str(row.get("run_id", "")).strip() or RUN_ID,
        "source_candidate_id": origin_candidate_id,
        "proposed_benchmark_level": level,
        "proposed_standard_category": ";".join(split_categories(
            (row.get("proposed_standard_category", "") or row.get("standard_category", ""))
            if proposed_category is None else proposed_category)),
        "proposed_transformation": LEVEL_TO_TRANSFORMATION.get(level, ""),
        "reviewed_level": "", "reviewed_standard_category": "",
        "validated_level": "", "validated_standard_category": "",
        "review_decision": "", "review_status": "not_reviewed",
        "reviewer_confidence": "", "reviewer_id": "", "reviewer_notes": "",
        "triage_confidence": str(triage_confidence), "human_validated": False,
        "validation_status": "not_human_validated",
    }
    return data


def _canonical_source_rows(l1_path: Path, ring_path: Path) -> list[dict]:
    rows = []
    for path, expected in ((l1_path, {"L1"}), (ring_path, {"L6"})):
        frame = _read_csv(path)
        missing = {"prompt_id", "prompt", "source_dataset", "benchmark_level"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing canonical columns: {sorted(missing)}")
        if "source_id" not in frame:
            if expected == {"L6"}:
                # The Ring-A-Bell standardized table uses prompt_id as its
                # stable source-row ID and has no separate source_id column.
                frame["source_id"] = frame["prompt_id"]
            else:
                raise ValueError(f"{path} is missing canonical source_id")
        frame = frame[frame.benchmark_level.isin(expected)]
        for row in frame.to_dict("records"):
            extras = {}
            for key in ("attack_method", "attack_config", "case_number", "evaluation_seed", "source_file"):
                if row.get(key, "") != "":
                    extras[f"ring_a_bell_{key}"] = row[key]
            metadata = _source_metadata(row.get("source_metadata", ""), **extras)
            rows.append(_base_record(row, source_metadata=metadata))
    return rows


def _candidate_rows(coco_path: Path, nibbler_path: Path) -> list[dict]:
    rows = []
    coco = _read_csv(coco_path)
    for row in coco.to_dict("records"):
        cid = str(row["candidate_id"])
        prompt_id = cid
        meta = _source_metadata(
            caption_id=row.get("caption_id"), image_id=row.get("image_id"),
            candidate_id=cid, source_split=row.get("source_split"),
            sampling_seed=row.get("sample_seed"), sample_order=row.get("sample_order"),
            triage_confidence=row.get("triage_confidence"), review_flags=row.get("review_flags"),
            validation_status=row.get("validation_status"),
        )
        source = {
            "prompt_id": prompt_id, "source_dataset": row.get("source_dataset", "COCO_Captions_2017"),
            "source_id": row.get("caption_id", row.get("source_id", "")), "prompt": row.get("prompt", ""),
            "original_category": "", "standard_category": "", "risk_level": "low",
            "benchmark_level": row.get("candidate_level", "L0"), "transformation": "original",
            "split": row.get("source_split", ""), "parent_prompt_id": "", "run_id": RUN_ID,
        }
        rows.append(_base_record(source, origin_candidate_id=cid,
                                 proposed_level=row.get("candidate_level", "L0"),
                                 proposed_category="", source_metadata=meta,
                                 triage_confidence=row.get("triage_confidence", "")))

    nibbler = _read_csv(nibbler_path)
    for row in nibbler.to_dict("records"):
        cid = str(row["candidate_id"])
        level = str(row.get("candidate_level", "unclassified"))
        failure = str(row.get("image_failure_type", ""))
        meta = _source_metadata(
            candidate_id=cid, submitted_prompt=row.get("submitted_prompt", row.get("prompt", "")),
            participant_rewrite_evidence=row.get("participant_rewrite_evidence", ""),
            participant_rewritten_prompt=row.get("rewritten_prompt", ""),
            attack_modes=row.get("text_attack_mode", ""), image_failure_type=failure,
            validator_text_safe_votes=row.get("validator_text_safe_votes"),
            validator_text_unsafe_votes=row.get("validator_text_unsafe_votes"),
            validator_text_other_votes=row.get("validator_text_other_votes"),
            validator_image_safe_votes=row.get("validator_image_safe_votes"),
            validator_image_unsafe_votes=row.get("validator_image_unsafe_votes"),
            validator_image_other_votes=row.get("validator_image_other_votes"),
            nibbler_source_provenance=row.get("source_provenance", ""),
            duplicate_source_record_count=row.get("duplicate_source_record_count"),
            source_json_key=row.get("source_json_key"), nibbler_round=row.get("nibbler_round"),
            metadata_matched=row.get("metadata_matched"), triage_confidence=row.get("triage_confidence"),
            review_flags=row.get("review_flags"), exact_overlap_with_existing_l1=row.get("exact_overlap_with_existing_l1"),
        )
        source = {
            "prompt_id": cid, "source_dataset": row.get("source_dataset", "Adversarial_Nibbler"),
            "source_id": row.get("source_id", ""), "prompt": row.get("submitted_prompt", row.get("prompt", "")),
            "original_category": failure, "standard_category": "", "risk_level": "",
            "benchmark_level": level if level in BENCHMARK_LEVELS else "",
            "transformation": LEVEL_TO_TRANSFORMATION.get(level, ""),
            "split": row.get("source_split", ""), "parent_prompt_id": "", "run_id": RUN_ID,
        }
        rows.append(_base_record(source, origin_candidate_id=cid, proposed_level=level,
                                 proposed_category=row.get("suggested_standard_category", ""),
                                 source_metadata=meta, triage_confidence=row.get("triage_confidence", "")))
    return rows


def _review_key_to_prompt_id(candidate_id: str) -> str:
    value = str(candidate_id)
    return value[4:] if value.startswith("STD_") else value


def _merge_review(rows: list[dict], review_path: Path, validated_path: Path | None) -> tuple[list[dict], dict]:
    review = _read_csv(review_path)
    known = {str(row["prompt_id"]): row for row in rows}
    review_mapped = {}
    mismatches, unmatched = [], []
    for row in review.to_dict("records"):
        cid = str(row.get("candidate_id", ""))
        pid = _review_key_to_prompt_id(cid)
        if pid not in known:
            unmatched.append(cid)
            continue
        target = known[pid]
        if normalize_prompt(row.get("prompt", "")) != normalize_prompt(target.get("prompt", "")):
            mismatches.append({"candidate_id": cid, "prompt_id": pid, "issue": "review prompt differs from source prompt"})
            unmatched.append(cid)
            continue
        review_mapped[pid] = row
        target["source_candidate_id"] = cid

    validated_by_id = {}
    if validated_path and validated_path.exists():
        validated = _read_csv(validated_path)
        if "candidate_id" in validated:
            validated_by_id = {_review_key_to_prompt_id(row["candidate_id"]): row
                               for row in validated.to_dict("records")}
    for pid, target in known.items():
        candidate_id = str(target.get("source_candidate_id", ""))
        # For source records outside the D2 review queue, their canonical ID is
        # not equivalent to a source_candidate_id; the review map is authoritative.
        rv = review_mapped.get(pid)
        if rv is None:
            target["review_status"] = "not_reviewed"
            continue
        decision = str(rv.get("decision", "")).strip().casefold()
        if str(rv.get("candidate_available", "true")).strip().casefold() == "false":
            target["review_status"] = "candidate_unavailable"
            target["review_decision"] = decision
            continue
        target["review_decision"] = decision
        target["reviewed_level"] = str(rv.get("validated_level", "")).strip()
        target["reviewed_standard_category"] = ";".join(split_categories(rv.get("standard_category", "")))
        target["reviewer_confidence"] = str(rv.get("reviewer_confidence", ""))
        target["reviewer_id"] = str(rv.get("reviewer_id", ""))
        target["reviewer_notes"] = str(rv.get("reviewer_notes", ""))
        target["parent_prompt_id"] = str(rv.get("parent_prompt_id", target.get("parent_prompt_id", ""))) or target.get("parent_prompt_id", "")
        if decision in {"reject", "uncertain"}:
            target["review_status"] = decision
        elif decision == "accept":
            accepted = validated_by_id.get(pid)
            if accepted is not None and _as_bool(accepted.get("validated", "true")):
                target["human_validated"] = True
                target["validation_status"] = "human_validated"
                target["review_status"] = "human_validated"
                target["validated_level"] = str(accepted.get("validated_level", "")).strip()
                target["validated_standard_category"] = ";".join(split_categories(accepted.get("standard_category", "")))
                target["benchmark_level"] = target["validated_level"]
                target["standard_category"] = target["validated_standard_category"]
                target["risk_level"] = risk_for_categories(split_categories(target["standard_category"]), DEFAULTS["taxonomy"])
                target["transformation"] = LEVEL_TO_TRANSFORMATION.get(target["validated_level"], target["transformation"])
            else:
                target["review_status"] = "accepted_pending_validation"
        else:
            target["review_status"] = "pending"
    # D2 review rows not represented by the source manifests are retained as
    # review-master-only candidate rows rather than silently discarded.
    for cid in unmatched:
        row = next(r for r in review.to_dict("records") if str(r.get("candidate_id", "")) == cid)
        prompt_id = "REVIEW_" + hashlib.sha256(cid.encode("utf-8")).hexdigest()[:16]
        level = str(row.get("proposed_level", "unclassified"))
        categories = str(row.get("suggested_standard_category", "") or row.get("standard_category", ""))
        metadata = _source_metadata(
            review_candidate_id=cid, d2_source_provenance=row.get("source_provenance", ""),
            participant_rewrite_evidence=row.get("participant_rewrite_evidence", ""),
            attack_mode=row.get("attack_mode", ""), image_failure_type=row.get("image_failure_type", ""),
            validator_text_safe_votes=row.get("validator_text_safe_votes"),
            validator_text_unsafe_votes=row.get("validator_text_unsafe_votes"),
            validator_image_safe_votes=row.get("validator_image_safe_votes"),
            validator_image_unsafe_votes=row.get("validator_image_unsafe_votes"),
        )
        source = {"prompt_id": prompt_id, "source_dataset": row.get("source_dataset", "review_master"),
                  "source_id": row.get("source_id", cid), "prompt": row.get("prompt", ""),
                  "original_category": row.get("image_failure_type", ""), "standard_category": "",
                  "risk_level": "", "benchmark_level": level if level in BENCHMARK_LEVELS else "",
                  "transformation": LEVEL_TO_TRANSFORMATION.get(level, ""), "split": row.get("source_split", ""),
                  "parent_prompt_id": row.get("parent_prompt_id", ""), "run_id": RUN_ID}
        target = _base_record(source, origin_candidate_id=cid, proposed_level=level,
                              proposed_category=categories, source_metadata=metadata,
                              triage_confidence=row.get("triage_confidence", ""))
        target["review_decision"] = str(row.get("decision", "")).strip().casefold()
        target["reviewed_level"] = str(row.get("validated_level", "")).strip()
        target["reviewed_standard_category"] = ";".join(split_categories(row.get("standard_category", "")))
        target["reviewer_confidence"] = str(row.get("reviewer_confidence", ""))
        target["reviewer_id"] = str(row.get("reviewer_id", ""))
        target["reviewer_notes"] = str(row.get("reviewer_notes", ""))
        target["review_status"] = target["review_decision"] or "pending"
        rows.append(target)
        known[prompt_id] = target

    return rows, {"review_master_rows": len(review), "matched_review_rows": len(review_mapped),
                  "unmatched_review_rows_retained": len(unmatched), "prompt_mismatches": mismatches,
                  "unmatched_candidate_ids": unmatched}


def mark_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["is_duplicate"] = False
    out["duplicate_of_prompt_id"] = ""
    out["duplicate_group_id"] = ""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, text in enumerate(out.prompt.astype(str)):
        normalized = normalize_prompt(text)
        if normalized:
            groups[normalized].append(idx)
    duplicate_rows = []
    for normalized, indices in groups.items():
        if len(indices) < 2:
            continue
        # Keep the first source in deterministic input order as the anchor:
        # existing standardized rows precede newly prepared candidate sources.
        ordered = sorted(indices)
        group_id = "dup_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        first_id = str(out.iloc[ordered[0]].prompt_id)
        for position, idx in enumerate(ordered):
            out.iat[idx, out.columns.get_loc("duplicate_group_id")] = group_id
            if position:
                out.iat[idx, out.columns.get_loc("is_duplicate")] = True
                out.iat[idx, out.columns.get_loc("duplicate_of_prompt_id")] = first_id
            duplicate_rows.append({"prompt_id": str(out.iloc[idx].prompt_id),
                                   "duplicate_of_prompt_id": "" if position == 0 else first_id,
                                   "duplicate_group_id": group_id,
                                   "source_dataset": str(out.iloc[idx].source_dataset),
                                   "source_id": str(out.iloc[idx].source_id),
                                   "prompt": str(out.iloc[idx].prompt)})
    return out, pd.DataFrame(duplicate_rows, columns=[
        "prompt_id", "duplicate_of_prompt_id", "duplicate_group_id", "source_dataset", "source_id", "prompt"])


def find_near_duplicates(frame: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, dict]:
    """Exact token-set Jaccard verification with a prefix-filter candidate index."""
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    prompt_ids_by_norm: dict[str, list[str]] = defaultdict(list)
    tokens_by_norm: dict[str, set[str]] = {}
    lineage = {str(row.prompt_id): {"source_dataset": str(getattr(row, "source_dataset", "")),
                                    "source_id": str(getattr(row, "source_id", ""))}
               for row in frame.itertuples(index=False)}
    for row in frame[["prompt_id", "prompt"]].to_dict("records"):
        norm = normalize_prompt(row["prompt"])
        tokens = set(TOKEN_RE.findall(norm))
        if not tokens:
            continue
        prompt_ids_by_norm[norm].append(str(row["prompt_id"]))
        tokens_by_norm[norm] = tokens
    norms = sorted(tokens_by_norm)
    frequency = Counter(token for tokens in tokens_by_norm.values() for token in tokens)
    ordered_tokens = {norm: sorted(tokens_by_norm[norm], key=lambda token: (frequency[token], token)) for norm in norms}
    postings: dict[str, list[str]] = defaultdict(list)
    pairs: set[tuple[str, str]] = set()
    candidate_comparisons = 0
    for norm in norms:
        tokens = tokens_by_norm[norm]
        prefix_length = len(tokens) - math.ceil(threshold * len(tokens)) + 1
        prefix = ordered_tokens[norm][:max(1, prefix_length)]
        for token in prefix:
            for other in postings[token]:
                other_tokens = tokens_by_norm[other]
                if min(len(tokens), len(other_tokens)) / max(len(tokens), len(other_tokens)) < threshold:
                    continue
                candidate_comparisons += 1
                pairs.add((other, norm))
        for token in prefix:
            postings[token].append(norm)
    records = []
    for norm_a, norm_b in sorted(pairs):
        a, b = tokens_by_norm[norm_a], tokens_by_norm[norm_b]
        score = len(a & b) / len(a | b)
        if score < threshold:
            continue
        for id_a in sorted(prompt_ids_by_norm[norm_a]):
            for id_b in sorted(prompt_ids_by_norm[norm_b]):
                records.append({"prompt_id_a": id_a, "prompt_id_b": id_b,
                                "source_dataset_a": lineage[id_a]["source_dataset"],
                                "source_id_a": lineage[id_a]["source_id"],
                                "source_dataset_b": lineage[id_b]["source_dataset"],
                                "source_id_b": lineage[id_b]["source_id"],
                                "token_jaccard": round(score, 6), "threshold": threshold})
    result = pd.DataFrame(records, columns=["prompt_id_a", "prompt_id_b", "source_dataset_a", "source_id_a",
                                             "source_dataset_b", "source_id_b", "token_jaccard", "threshold"])
    return result, {"unique_normalized_prompts": len(norms), "candidate_comparisons": candidate_comparisons,
                    "verified_near_duplicate_prompt_pairs": len(result), "threshold": threshold,
                    "similarity": "token-set Jaccard; exact verification after prefix and length filtering"}


def _distributions(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def counts(series, column):
        normalized = series.replace("", "(missing)")
        out = normalized.value_counts(dropna=False).rename_axis(column).reset_index(name="rows")
        return out.sort_values(column, kind="mergesort").reset_index(drop=True)

    cats = []
    for row in frame[["prompt_id", "source_dataset", "proposed_standard_category", "validated_standard_category"]].to_dict("records"):
        for label_type in ("proposed", "validated"):
            field = f"{label_type}_standard_category"
            values = split_categories(row[field]) or (["(missing)"] if label_type == "proposed" else [])
            for category in values:
                cats.append({"label_type": label_type, "category": category,
                             "source_dataset": row["source_dataset"], "prompt_id": row["prompt_id"]})
    return {
        "by_source": counts(frame.source_dataset, "source_dataset"),
        "by_benchmark_level": counts(frame.proposed_benchmark_level, "proposed_benchmark_level"),
        "by_validated_level": counts(frame.validated_level, "validated_level"),
        "by_transformation": counts(frame.transformation, "transformation"),
        "by_review_status": counts(frame.review_status, "review_status"),
        "by_risk_level": counts(frame.risk_level, "risk_level"),
        "by_category": pd.DataFrame(cats, columns=["label_type", "category", "source_dataset", "prompt_id"])
            .groupby(["label_type", "category", "source_dataset"], dropna=False).size().reset_index(name="rows")
            .sort_values(["label_type", "category", "source_dataset"], kind="mergesort").reset_index(drop=True),
    }


def _metadata_issues(frame: pd.DataFrame, taxonomy: set[str]) -> pd.DataFrame:
    issues = []
    ids = set(frame.prompt_id.astype(str))
    for row in frame.to_dict("records"):
        pid = str(row["prompt_id"])
        if not str(row.get("source_id", "")).strip():
            issues.append({"prompt_id": pid, "severity": "warning", "field": "source_id", "issue": "missing original source ID"})
        if not str(row.get("split", "")).strip():
            issues.append({"prompt_id": pid, "severity": "info", "field": "split", "issue": "source split unavailable"})
        if not str(row.get("standard_category", "")).strip() and row.get("benchmark_level") != "L0":
            issues.append({"prompt_id": pid, "severity": "info", "field": "standard_category", "issue": "category is unknown or awaiting human review"})
        risk = str(row.get("risk_level", "")).strip()
        if not risk and row.get("benchmark_level") != "L0":
            issues.append({"prompt_id": pid, "severity": "info", "field": "risk_level", "issue": "risk level unresolved because no validated category is available"})
        elif risk and risk not in RISK_LEVELS:
            issues.append({"prompt_id": pid, "severity": "warning", "field": "risk_level", "issue": f"invalid risk level: {risk}"})
        unknown = sorted(set(split_categories(row.get("proposed_standard_category", ""))) - taxonomy)
        if unknown:
            issues.append({"prompt_id": pid, "severity": "warning", "field": "proposed_standard_category", "issue": f"unknown proposed categories: {unknown}"})
        unknown_standard = sorted(set(split_categories(row.get("standard_category", ""))) - taxonomy)
        if unknown_standard:
            issues.append({"prompt_id": pid, "severity": "warning", "field": "standard_category", "issue": f"unknown categories: {unknown_standard}"})
        level = str(row.get("proposed_benchmark_level", ""))
        if level and level not in BENCHMARK_LEVELS and level != "unclassified":
            issues.append({"prompt_id": pid, "severity": "warning", "field": "proposed_benchmark_level", "issue": f"noncanonical proposed level: {level}"})
        if level == "unclassified":
            issues.append({"prompt_id": pid, "severity": "info", "field": "benchmark_level", "issue": "candidate level is unclassified; canonical benchmark_level left blank"})
        if level in {"L2", "L3", "L4", "L5"}:
            parent = str(row.get("parent_prompt_id", "")).strip()
            if not parent:
                issues.append({"prompt_id": pid, "severity": "warning", "field": "parent_prompt_id", "issue": "transformed candidate has no parent_prompt_id"})
            elif parent not in ids:
                issues.append({"prompt_id": pid, "severity": "warning", "field": "parent_prompt_id", "issue": f"parent reference not present in unified candidates: {parent}"})
    return pd.DataFrame(issues, columns=["prompt_id", "severity", "field", "issue"])


def _write_source_audits(paths: dict[str, Path]) -> dict:
    result = {}
    for name, path in paths.items():
        if not path or not path.exists():
            result[name] = {"path": str(path) if path else "", "present": False}
            continue
        digest = _file_sha256(path)
        summary = {"path": str(path), "present": True, "sha256": digest, "bytes": path.stat().st_size}
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    summary["audit_keys"] = sorted(payload.keys())
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        result[name] = summary
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_unified_candidates(*, l1_path: Path, coco_path: Path, nibbler_path: Path,
                             ring_path: Path, review_path: Path, validated_path: Path | None,
                             output_path: Path, reports_dir: Path, near_threshold: float = 0.9,
                             selected_path: Path | None = None, audit_paths: dict[str, Path] | None = None) -> dict:
    rows = _canonical_source_rows(l1_path, ring_path)
    rows.extend(_candidate_rows(coco_path, nibbler_path))
    if len({row["prompt_id"] for row in rows}) != len(rows):
        duplicates = pd.Series([row["prompt_id"] for row in rows]).loc[lambda values: values.duplicated(False)].tolist()
        raise ValueError(f"prompt_id collision across source inputs: {duplicates[:10]}")
    rows, review_audit = _merge_review(rows, review_path, validated_path)
    frame = pd.DataFrame(rows)
    # Refresh hashes and IDs after all sources and review-master-only rows are combined.
    frame["content_hash"] = frame.prompt.astype(str).map(content_hash)
    frame, exact_groups = mark_exact_duplicates(frame)
    near_pairs, near_audit = find_near_duplicates(frame, near_threshold)
    taxonomy = set(DEFAULTS["taxonomy"])
    issues = _metadata_issues(frame, taxonomy)
    frame = frame[CANDIDATE_COLUMNS]
    if frame.prompt_id.astype(str).duplicated().any():
        raise ValueError("unified candidate prompt IDs must be unique")
    if frame.prompt.astype(str).str.strip().eq("").any():
        raise ValueError("unified candidate manifest cannot contain empty prompts")
    canonical_validation = validate_table(frame, DEFAULTS["taxonomy"])
    if canonical_validation["missing_columns"] or any("unknown standard categories" in err or "empty prompt" in err for err in canonical_validation["errors"]):
        raise ValueError(f"candidate manifest failed canonical schema checks: {canonical_validation['errors']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_tabular(frame, output_path)
    write_tabular(exact_groups, reports_dir / "exact_duplicate_lineage.parquet")
    write_tabular(near_pairs, reports_dir / "near_duplicate_pairs.parquet")
    write_tabular(issues, reports_dir / "metadata_issues.parquet")
    distributions = _distributions(frame)
    for name, table in distributions.items():
        write_tabular(table, reports_dir / f"distribution_{name}.parquet")

    l1 = _read_csv(l1_path)
    ring = _read_csv(ring_path)
    selected = _read_csv(selected_path) if selected_path and selected_path.exists() else pd.DataFrame()
    review = _read_csv(review_path)
    validated = _read_csv(validated_path) if validated_path and validated_path.exists() else pd.DataFrame()
    audit = {
        "schema_columns": CANDIDATE_COLUMNS,
        "canonical_schema_columns": STANDARD_COLUMNS,
        "supplemental_review_columns": EXTRA_COLUMNS,
        "run_id": RUN_ID,
        "rows": len(frame),
        "source_counts": {str(k): int(v) for k, v in frame.source_dataset.value_counts().items()},
        "proposed_level_counts": {str(k): int(v) for k, v in frame.proposed_benchmark_level.value_counts().items()},
        "canonical_benchmark_level_counts": {str(k) if str(k) else "(missing)": int(v)
                                                for k, v in frame.benchmark_level.replace("", "(missing)").value_counts().items()},
        "validated_level_counts": {str(k): int(v) for k, v in frame.loc[frame.human_validated, "validated_level"].value_counts().items()},
        "review_status_counts": {str(k): int(v) for k, v in frame.review_status.value_counts().items()},
        "review_decision_counts": {str(k) if str(k) else "(blank)": int(v) for k, v in frame.review_decision.replace("", "(blank)").value_counts().items()},
        "human_validated_rows": int(frame.human_validated.astype(bool).sum()),
        "exact_duplicate_rows": int(frame.is_duplicate.astype(bool).sum()),
        "exact_duplicate_groups": int(exact_groups.duplicate_group_id.nunique()) if len(exact_groups) else 0,
        "cross_source_exact_duplicate_groups": int(exact_groups.groupby("duplicate_group_id").source_dataset.nunique().gt(1).sum()) if len(exact_groups) else 0,
        "near_duplicate_audit": near_audit,
        "metadata_issue_counts": {str(k): int(v) for k, v in issues.severity.value_counts().items()} if len(issues) else {},
        "canonical_schema_validation": canonical_validation,
        "review_merge_audit": review_audit,
        "input_rows": {"existing_standardized_l1": len(l1), "ring_a_bell_l6": len(ring),
                       "coco_candidates": len(_read_csv(coco_path)), "nibbler_candidates": len(_read_csv(nibbler_path)),
                       "phase_d2_review_rows": len(review), "phase_d2_validated_rows": len(validated),
                       "selected_candidate_manifest_rows": len(selected)},
        "source_audits": _write_source_audits(audit_paths or {}),
        "input_checksums": _write_source_audits({
            "standardized_l1": l1_path, "coco_candidates": coco_path,
            "corrected_nibbler_candidates": nibbler_path, "ring_a_bell_standardized": ring_path,
            "phase_d2_review_master": review_path,
            "phase_d2_validated_export": validated_path,
            "selected_candidate_manifest": selected_path,
        }),
        "output_integrity": {"path": str(output_path), "bytes": output_path.stat().st_size,
                             "sha256": _file_sha256(output_path)},
        "outputs": {"manifest": str(output_path), "reports_dir": str(reports_dir)},
        "final_6000_benchmark_created": False,
    }
    (reports_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    _write_readme_report(audit, reports_dir)
    return audit


def _write_readme_report(audit: dict, reports_dir: Path) -> None:
    lines = ["# Unified prompt-candidate standardization (Phase E1)", "",
             f"Rows: {audit['rows']:,}. Human-validated rows: {audit['human_validated_rows']:,}.", "",
             "## Proposed benchmark levels", ""]
    lines += [f"- {level}: {count:,}" for level, count in sorted(audit["proposed_level_counts"].items())]
    lines += ["", "## Review status", ""]
    lines += [f"- {status}: {count:,}" for status, count in sorted(audit["review_status_counts"].items())]
    lines += ["", "## Duplicates and metadata", "",
              f"- Exact duplicate rows retained with source lineage: {audit['exact_duplicate_rows']:,}.",
              f"- Exact duplicate groups: {audit['exact_duplicate_groups']:,}; cross-source groups: {audit['cross_source_exact_duplicate_groups']:,}.",
              f"- Near-duplicate pairs at Jaccard ≥ {audit['near_duplicate_audit']['threshold']}: {audit['near_duplicate_audit']['verified_near_duplicate_prompt_pairs']:,}.",
              f"- Metadata issues: {sum(audit['metadata_issue_counts'].values()):,}.",
              "- Proposed and validated level/category fields are separate. Only the D1 validated export sets `human_validated=true`.",
              "- This is a candidate manifest; no final 6,000-prompt benchmark is assembled.", "",
              "See adjacent `distribution_*.parquet`, `exact_duplicate_lineage.parquet`, `near_duplicate_pairs.parquet`, and `metadata_issues.parquet` for detail."]
    (reports_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1", type=Path, default=Path("data/runs/s04_postprocess/real1/unified_deduplicated.parquet"))
    parser.add_argument("--coco", type=Path, default=Path("data/processed/benchmark_prep/phase_c/coco_l0_candidates.parquet"))
    parser.add_argument("--nibbler", type=Path, default=Path("data/processed/benchmark_prep/phase_d2/nibbler_submitted_candidates_corrected.parquet"))
    parser.add_argument("--ring-a-bell", type=Path, default=Path("data/processed/standardized/ring-a-bell.parquet"))
    parser.add_argument("--review-master", type=Path, default=Path("data/processed/benchmark_prep/phase_d2/review_master.parquet"))
    parser.add_argument("--validated", type=Path, default=Path("data/processed/benchmark_prep/phase_d2/validated/validated_candidates.parquet"))
    parser.add_argument("--selected", type=Path, default=Path("data/processed/unified/benchmark_6000_candidates.parquet"))
    parser.add_argument("--phase-c-audit", type=Path, default=Path("data/processed/benchmark_prep/phase_c/phase_c_audit.json"))
    parser.add_argument("--phase-d2-audit", type=Path, default=Path("data/processed/benchmark_prep/phase_d2/phase_d2_audit.json"))
    parser.add_argument("--phase-d2-review-audit", type=Path, default=Path("data/processed/benchmark_prep/phase_d2/review_batch_audit.json"))
    parser.add_argument("--phase-d2-validated-audit", type=Path, default=Path("data/processed/benchmark_prep/phase_d2/validated/accepted_counts.json"))
    parser.add_argument("--selected-audit", type=Path, default=Path("data/processed/unified/benchmark_6000_candidates_audit.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/unified/unified_prompt_candidates.parquet"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/processed/unified/unified_prompt_candidates_reports"))
    parser.add_argument("--near-threshold", type=float, default=0.9)
    args = parser.parse_args(argv)
    result = build_unified_candidates(
        l1_path=args.l1, coco_path=args.coco, nibbler_path=args.nibbler,
        ring_path=args.ring_a_bell, review_path=args.review_master,
        validated_path=args.validated, output_path=args.output, reports_dir=args.reports_dir,
        near_threshold=args.near_threshold, selected_path=args.selected,
        audit_paths={"phase_c": args.phase_c_audit, "phase_d2": args.phase_d2_audit,
                     "phase_d2_review": args.phase_d2_review_audit,
                     "phase_d2_validated": args.phase_d2_validated_audit,
                     "selected_candidate_audit": args.selected_audit},
    )
    print(json.dumps({"rows": result["rows"], "source_counts": result["source_counts"],
                      "proposed_level_counts": result["proposed_level_counts"],
                      "human_validated_rows": result["human_validated_rows"],
                      "exact_duplicate_rows": result["exact_duplicate_rows"],
                      "near_duplicate_pairs": result["near_duplicate_audit"]["verified_near_duplicate_prompt_pairs"],
                      "metadata_issue_counts": result["metadata_issue_counts"],
                      "output": result["outputs"]["manifest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
