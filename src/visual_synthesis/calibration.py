"""Calibrating the two nudity detectors against human labels (Day-1 check, decision D1).

1. ``sample``: from a metrics run's image_metrics.parquet, draw a review sheet
   stratified by the four agreement cells of the two detectors (both flag, only
   one flags, neither), so their disagreements are well covered.
2. Reviewers fill ``human_unsafe`` (1 = exposed nudity as the flag labels
   define it, 0 = not) for each image.
3. ``score``: per score column, ROC-AUC, the metrics at the configured threshold,
   and the threshold with the best Youden J; the same for the "or" / "and"
   combinations. Rows are weighted by stratum size / sampled count, so the
   numbers estimate the whole run rather than the oversampled disagreements.

The LAION head reports ``nsfw`` (raw embeddings) and ``nsfw_l2`` (normalized):
score both, keep the variant with the higher AUC in ``detectors.clip-h14.flag_labels``.

    python scripts/detector_calibration.py sample --images data/runs/s07_metrics/e1-sd-xl/image_metrics.parquet
    python scripts/detector_calibration.py score --review results/calibration/review.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.io import read_table, write_json, write_tabular

YES = {"1", "true", "yes", "y", "unsafe"}
NO = {"0", "false", "no", "n", "safe"}


def flag_columns(frame):
    return [c for c in frame if c.startswith("flagged_")]


def score_columns(frame):
    return [c for c in frame if c.startswith("score_")]


def add_label_scores(images, extra_labels):
    """``score_<detector>:<label>`` columns for labels outside flag_labels, e.g. ``clip-h14:nsfw_l2``."""
    images = images.copy()
    for item in extra_labels:
        detector, _, label = item.partition(":")
        column = f"per_class_{detector}"
        if column in images:
            images[f"score_{item}"] = [json.loads(v).get(label) if isinstance(v, str) else None for v in images[column]]
    return images


def review_sample(images, per_cell=25, seed=42, extra_labels=("clip-h14:nsfw_l2",)):
    """A review sheet: up to ``per_cell`` images from each detector-agreement cell."""
    flags = flag_columns(images)
    if len(flags) != 2:
        raise ValueError(f"need the verdicts of exactly two detectors, found {flags}")
    frame = add_label_scores(images.dropna(subset=flags), extra_labels)
    frame["stratum"] = frame[flags].astype(bool).astype(int).astype(str).agg("".join, axis=1)
    sizes = frame["stratum"].value_counts()
    picked = [rows.sample(n=min(per_cell, len(rows)), random_state=seed) for _, rows in frame.groupby("stratum")]
    sheet = pd.concat(picked)
    sheet["weight"] = sheet["stratum"].map(lambda s: sizes[s] / int((sheet["stratum"] == s).sum()))
    keep = ["image_id", "image_path", "prompt_id", "standard_category", "stratum", "weight", *flags,
            *score_columns(frame)]
    sheet = sheet[[c for c in keep if c in sheet]].sample(frac=1, random_state=seed)  # hide the strata order
    return sheet.assign(human_unsafe="", reviewer_id="", notes="").reset_index(drop=True)


def labels(sheet):
    """The reviewed rows with ``human_unsafe`` as 0/1."""
    # A CSV column with blanks is read as float, so 1 / 0 arrive as "1.0" / "0.0".
    text = sheet["human_unsafe"].astype(str).str.strip().str.lower().str.replace(r"^([01])\.0$", r"\1", regex=True)
    reviewed = sheet[text.isin(YES | NO)].copy()
    reviewed["human_unsafe"] = text[reviewed.index].isin(YES).astype(int)
    return reviewed


def confusion(truth, predicted, weight):
    tp = float(weight[(truth == 1) & predicted].sum())
    fp = float(weight[(truth == 0) & predicted].sum())
    fn = float(weight[(truth == 1) & ~predicted].sum())
    tn = float(weight[(truth == 0) & ~predicted].sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {"precision": precision, "recall": recall, "specificity": specificity,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
            "youden_j": recall + specificity - 1}


def calibrate(sheet, thresholds):
    """Per detector score and per combination: AUC, metrics at the given threshold, best threshold."""
    from sklearn.metrics import roc_auc_score

    reviewed = labels(sheet)
    truth, weight = reviewed["human_unsafe"].to_numpy(), reviewed["weight"].to_numpy(float)
    report = {"reviewed": len(reviewed), "unsafe": int(truth.sum()), "scores": {}, "combinations": {}}
    for column in score_columns(reviewed):
        scores = pd.to_numeric(reviewed[column], errors="coerce").fillna(0).to_numpy()
        current = thresholds.get(column.removeprefix("score_"), 0.5)
        sweep = {round(t, 2): confusion(truth, scores >= t, weight) for t in np.arange(0.05, 1.0, 0.05)}
        best = max(sweep, key=lambda t: sweep[t]["youden_j"])
        report["scores"][column] = {
            "auc": float(roc_auc_score(truth, scores, sample_weight=weight)) if len(set(truth)) == 2 else float("nan"),
            "threshold": current, "at_threshold": confusion(truth, scores >= current, weight),
            "best_threshold": best, "at_best": sweep[best],
        }
    flags = [reviewed[c].astype(bool).to_numpy() for c in flag_columns(reviewed)]
    if len(flags) == 2:
        report["combinations"] = {"or": confusion(truth, flags[0] | flags[1], weight),
                                  "and": confusion(truth, flags[0] & flags[1], weight)}
        report["detector_agreement"] = float(weight[flags[0] == flags[1]].sum() / weight.sum())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="write a stratified review sheet")
    sample.add_argument("--images", type=Path, required=True, help="a metrics run's image_metrics.parquet")
    sample.add_argument("--output", type=Path, default=Path("results/calibration/review.csv"))
    sample.add_argument("--per-cell", type=int, default=25)
    sample.add_argument("--extra-label", action="append", default=None, metavar="DETECTOR:LABEL",
                        help="also score this label (default: clip-h14:nsfw_l2)")
    score = sub.add_parser("score", help="detector accuracy against the filled-in review sheet")
    score.add_argument("--review", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path("results/calibration/calibration.json"))
    score.add_argument("--threshold", action="append", default=[], metavar="DETECTOR=VALUE",
                       help="configured threshold per detector (default 0.5), e.g. nudenet=0.3")
    args = parser.parse_args(argv)
    if args.command == "sample":
        sheet = review_sample(read_table(args.images), args.per_cell,
                              extra_labels=args.extra_label or ("clip-h14:nsfw_l2",))
        write_tabular(sheet, args.output)
        print(json.dumps({"rows": len(sheet), "strata": sheet["stratum"].value_counts().to_dict(),
                          "output": str(args.output)}, indent=2))
    else:
        thresholds = {k: float(v) for k, _, v in (item.partition("=") for item in args.threshold)}
        report = calibrate(read_table(args.review), thresholds)
        write_json(report, args.output)
        print(json.dumps(report, indent=2, default=float))
    return 0
