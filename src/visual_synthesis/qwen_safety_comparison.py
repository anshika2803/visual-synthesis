"""Post-hoc evaluator agreement on existing BF16/NF4 pairs, not safety ground truth."""

import json
import math
import re
from pathlib import Path

import pandas as pd

from visual_synthesis.benchmark_sampling import load_prompts
from visual_synthesis.io import file_sha256
from visual_synthesis.project_pilot import EVALUATORS, read_jsonl, save_json

DETECTORS = ("nudenet", "clip-h14")


def prompt_scope(row):
    if row.get("benchmark_level") == "L0":
        return "benign_prompt"
    categories = set(re.split(r"[,;|]", str(row.get("standard_category", "")).lower()))
    return "sexual_prompt" if "sexual" in {value.strip() for value in categories} else "other_prompt"


def prepare_images(comparison, output):
    """Freeze original PNG identities and labels, without copying or regenerating them."""
    comparison, output = Path(comparison).resolve(), Path(output)
    summary = json.loads((comparison / "comparison_summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("evaluation requires a completed paired comparison")
    expected_hash = summary["input"]["sample_sha256"]
    if file_sha256(comparison / "prompts.csv") != expected_hash:
        raise ValueError("original prompt table changed")
    table = load_prompts(comparison / "prompts.csv").astype(object)
    prompts = {str(row["prompt_id"]): row for row in table.where(pd.notna(table), "").to_dict("records")}
    rows, pairs = [], []
    for arm in ("bf16", "nf4"):
        report = summary["arms"][arm]
        records = read_jsonl(comparison / arm / "images.jsonl")
        if report.get("status") != "success" or len(records) != len(prompts) or report.get("images") != len(prompts):
            raise ValueError(f"incomplete {arm} image manifest")
        seen = set()
        for record in records:
            pid = str(record["prompt_id"])
            if pid not in prompts or pid in seen:
                raise ValueError("image IDs do not match the original paired prompts")
            seen.add(pid)
            prompt = prompts[pid]
            if (record["condition"] != arm or record["prompt"] != prompt["prompt"] or
                    record["evaluation_seed"] != int(prompt["evaluation_seed"])):
                raise ValueError("image prompt/seed/condition differs from the frozen comparison")
            image = Path(record["image_path"]).resolve()
            if not image.is_relative_to(comparison / arm / "images") or not image.is_file():
                raise ValueError("image is missing or outside its original arm directory")
            if file_sha256(image) != record["sha256"]:
                raise ValueError("saved PNG changed since generation")
            row = {**record, "image_path": str(image), "benchmark_level": prompt.get("benchmark_level", ""),
                   "standard_category": prompt.get("standard_category", ""), "prompt_scope": prompt_scope(prompt)}
            rows.append(row)
            if arm == "bf16":
                pairs.append({"prompt_id": pid, "evaluation_seed": record["evaluation_seed"],
                              "prompt": record["prompt"], "benchmark_level": row["benchmark_level"],
                              "standard_category": row["standard_category"], "prompt_scope": row["prompt_scope"]})
    output.mkdir(parents=True, exist_ok=True)
    (output / "all-images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    save_json({"comparison": str(comparison), "sample_sha256": expected_hash,
               "pairs": len(pairs), "images": len(rows), "original_generation_job": summary["pbs_job_id"],
               "scope": "Prompt categories are source labels, not reviewed image labels."},
              output / "input_manifest.json")
    return pairs, rows


def flag_transition(bf16, nf4):
    if bf16 is None or nf4 is None:
        return "unavailable"
    if bf16 and nf4:
        return "both_flagged"
    if bf16:
        return "bf16_only"
    return "nf4_only" if nf4 else "neither_flagged"


def aggregate_scores(output, pairs, images, reports, detector_settings):
    """Keep missing predictions unknown and raw detector scores separate from flags."""
    output = Path(output)
    expected = {(row["condition"], row["prompt_id"]) for row in images}
    scores, complete = {}, {}
    for kind in EVALUATORS:
        path = output / kind / "scores.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        index = {(row["condition"], row["prompt_id"]): row for row in rows}
        if len(index) != len(rows) or not set(index).issubset(expected):
            raise ValueError("evaluator produced duplicate or unexpected image IDs")
        for row in rows:
            values = row["scores"].values() if kind in DETECTORS else [row["score"]]
            if any(not isinstance(value, float | int) or not math.isfinite(value) for value in values):
                raise ValueError("evaluator produced nonfinite or invalid scores")
            if kind in DETECTORS and not isinstance(row.get("flagged"), bool):
                raise ValueError("evaluator flag must be a boolean")
        scores[kind] = index
        complete[kind] = reports.get(kind, {}).get("status") == "success" and set(index) == expected
    result_rows = []
    for pair in pairs:
        row = dict(pair)
        pid = pair["prompt_id"]
        for kind in EVALUATORS:
            for arm in ("bf16", "nf4"):
                payload = scores[kind].get((arm, pid))
                prefix = f"{kind}_{arm}"
                if kind in DETECTORS:
                    row[f"{prefix}_flagged"] = payload["flagged"] if payload else None
                    raw = payload["scores"] if payload else None
                    labels = detector_settings[kind]["flag_labels"]
                    if labels is None:
                        labels = list(raw or {})
                    row[f"{prefix}_score"] = max((raw.get(label, 0.0) for label in labels), default=0.0) \
                        if raw is not None else None
                    row[f"{prefix}_raw_scores"] = json.dumps(raw, sort_keys=True) if raw is not None else None
                else:
                    row[f"{prefix}_score"] = payload["score"] if payload else None
            a, b = (row[f"{kind}_{arm}_score"] for arm in ("bf16", "nf4"))
            row[f"{kind}_score_delta_nf4_minus_bf16"] = b - a if a is not None and b is not None else None
            if kind in DETECTORS:
                row[f"{kind}_transition"] = flag_transition(
                    row[f"{kind}_bf16_flagged"], row[f"{kind}_nf4_flagged"])
        for arm in ("bf16", "nf4"):
            flags = [row[f"{kind}_{arm}_flagged"] for kind in DETECTORS]
            row[f"union_{arm}_flagged"] = any(flags) if all(value is not None for value in flags) else None
        row["union_transition"] = flag_transition(row["union_bf16_flagged"], row["union_nf4_flagged"])
        result_rows.append(row)
    frame = pd.DataFrame(result_rows)
    frame.to_csv(output / "paired_scores.csv", index=False)
    review = frame.copy()
    review["detector_disagreement_or_missing"] = review[
        [f"{kind}_transition" for kind in (*DETECTORS, "union")]].isin(
        ["bf16_only", "nf4_only", "unavailable"]).any(axis=1)
    review["review_bf16_nudity"] = ""
    review["review_nf4_nudity"] = ""
    review["review_content_changed"] = ""
    review["review_detector_error"] = ""
    review["review_quality_or_alignment_loss"] = ""
    review["review_notes"] = ""
    review.sort_values("detector_disagreement_or_missing", ascending=False, kind="stable").to_csv(
        output / "human_review.csv", index=False)
    summaries = []
    groups = [("all_prompts", frame), *[(name, group) for name, group in frame.groupby("prompt_scope")]]
    for scope, group in groups:
        for kind in (*DETECTORS, "union"):
            transitions = group[f"{kind}_transition"]
            available = transitions.ne("unavailable")
            count = int(available.sum())
            counts = transitions.value_counts()
            summaries.append({"prompt_scope": scope, "detector": kind, "total_pairs": len(group),
                              "scored_pairs": count, "both_flagged": int(counts.get("both_flagged", 0)),
                              "bf16_only": int(counts.get("bf16_only", 0)),
                              "nf4_only": int(counts.get("nf4_only", 0)),
                              "neither_flagged": int(counts.get("neither_flagged", 0)),
                              "agreement_rate": float(transitions[available].isin(
                                  ["both_flagged", "neither_flagged"]).mean()) if count else None,
                              "bf16_flag_rate": float(group.loc[available, f"{kind}_bf16_flagged"].mean())
                              if count else None,
                              "nf4_flag_rate": float(group.loc[available, f"{kind}_nf4_flagged"].mean())
                              if count else None})
    pd.DataFrame(summaries).to_csv(output / "detection_summary.csv", index=False)
    quality = {kind: {arm: float(frame[f"{kind}_{arm}_score"].dropna().mean())
                     if frame[f"{kind}_{arm}_score"].notna().any() else None for arm in ("bf16", "nf4")}
               for kind in ("alignment", "clip-iqa")}
    return {"status": "complete" if all(complete.values()) else "incomplete", "evaluator_complete": complete,
            "detector_settings": detector_settings, "detection": summaries, "quality_alignment_means": quality,
            "note": "Detector agreement/flag rates are not ground-truth safety, accuracy or attack success. "
                    "Nonsexual categories need separate assessment. Review content and detector errors; "
                    "Small pilot samples cannot establish equivalence. "
                    "CLIP-IQA is the existing B/32 two-anchor variant."}
