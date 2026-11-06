"""Experiment E4: does one layer's activation space separate harmful from benign prompts?

Input: an E4 capture run (``generate`` with ``generate.capture.enabled``) over
e46_v1. Each image has ``activations/<image>.npz`` with, per hooked layer, the
pooled vector averaged over steps (``<layer>/mean``) and per step (``<layer>/steps``).

Discovery split only (it alone chooses):
    per layer   5-fold CV logistic-regression ROC-AUC and accuracy, AUC of the mean-difference score
    layer       the highest CV AUC; among layers within 0.01 of it, the earliest (rule fixed in advance)
    directions  d = (mu_harmful - mu_benign) / |mu_harmful - mu_benign|, the probe's normal, 3 random unit vectors

Validation split, reported only: held-out AUCs per layer, and every group's
projection onto d in units of |mu_harmful - mu_benign| (benign 0, harmful 1).

Ablations: split-half cosine of d (is the direction stable?) and AUC per
denoising step (when does the separation appear?). A high AUC shows association
only: different prompts are expected to separate. E5 tests causation.

    python scripts/e4_analyze.py --capture-run data/runs/s05_generate/e4-sd-xl-capture --output results/e4/sd-xl
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.io import find_table, read_table, write_json

LABELS = {"benign": 0, "harmful": 1}
RANDOM_SEEDS = (0, 1, 2)
TIE = 0.01


def load(frame, layer, kind="mean"):
    """``<layer>/<kind>`` of every row's activation file, stacked (float32)."""
    return np.stack([np.load(path)[f"{layer}/{kind}"].astype(np.float32) for path in frame["activation_path"]])


def labelled(frame):
    """The benign and harmful rows, and their 0/1 labels."""
    rows = frame[frame["subset_group"].isin(LABELS)]
    return rows, rows["subset_group"].map(LABELS).to_numpy()


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def cross_validate(features, labels, folds=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_validate as sklearn_cv

    split = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = sklearn_cv(probe(), features, labels, cv=split, scoring=("roc_auc", "accuracy"))
    return {
        "cv_auc": float(scores["test_roc_auc"].mean()), "cv_auc_sd": float(scores["test_roc_auc"].std()),
        "cv_accuracy": float(scores["test_accuracy"].mean()), "cv_accuracy_sd": float(scores["test_accuracy"].std()),
    }


def auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else math.nan


def mean_difference(features, labels):
    """``(mu_benign, mu_harmful, unit direction, norm of the difference)``."""
    mu_benign, mu_harmful = features[labels == 0].mean(axis=0), features[labels == 1].mean(axis=0)
    norm = float(np.linalg.norm(mu_harmful - mu_benign))
    return mu_benign, mu_harmful, (mu_harmful - mu_benign) / norm, norm


def probe_direction(fitted):
    """The probe's decision normal in raw activation units, as a unit vector."""
    weights = fitted[-1].coef_[0] / fitted[0].scale_
    return weights / np.linalg.norm(weights)


def random_direction(dim, seed):
    vector = np.random.default_rng(seed).standard_normal(dim)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def choose_layer(table, tie=TIE):
    """The pre-registered rule: highest mean CV AUC; within ``tie`` of it, the earliest layer."""
    best = table["cv_auc"].max()
    return table.loc[table["cv_auc"] >= best - tie, "layer"].iloc[0]


def split_half_cosine(features, labels, repeats=20, seed=42):
    """Mean cosine between directions estimated on two disjoint, label-balanced halves."""
    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(repeats):
        halves = ([], [])
        for label in (0, 1):
            shuffled = rng.permutation(np.flatnonzero(labels == label))
            halves[0].extend(shuffled[: len(shuffled) // 2])
            halves[1].extend(shuffled[len(shuffled) // 2 :])
        first, second = (mean_difference(features[half], labels[half])[2] for half in halves)
        cosines.append(float(first @ second))
    return float(np.mean(cosines))


def step_aucs(discovery, labels, validation, validation_labels, layer):
    """AUC on validation of the per-step mean difference, one row per denoising step."""
    train, held = load(discovery, layer, "steps"), load(validation, layer, "steps")
    rows = []
    for step in range(train.shape[1]):
        mu_benign, _, direction, _ = mean_difference(train[:, step], labels)
        rows.append({"layer": layer, "step": step,
                     "validation_auc": auc(validation_labels, (held[:, step] - mu_benign) @ direction)})
    return rows


def analyze(generations, layers, seed=42):
    """``(layer table, arrays for directions.npz, group projections, step AUCs, statistics)``."""
    rows = generations[generations["generation_status"].eq("success") & generations["activation_path"].notna()]
    discovery, labels = labelled(rows[rows["subset_split"] == "discovery"])
    validation = rows[rows["subset_split"] == "validation"]
    held, held_labels = labelled(validation)
    if len(set(labels)) != 2:
        raise ValueError("the discovery split needs both benign and harmful prompts with activations")

    table, arrays, projections, steps = [], {}, [], []
    for layer in layers:
        features = load(discovery, layer)
        held_features = load(held, layer) if len(held) else np.zeros((0, features.shape[1]), np.float32)
        mu_benign, mu_harmful, direction, norm = mean_difference(features, labels)
        fitted = probe().fit(features, labels)
        normal = probe_direction(fitted)
        table.append({
            "layer": layer, "dim": features.shape[1], **cross_validate(features, labels, seed=seed),
            "discovery_direction_auc": auc(labels, (features - mu_benign) @ direction),
            "validation_probe_auc": (auc(held_labels, fitted.decision_function(held_features))
                                     if len(held) else math.nan),
            "validation_direction_auc": auc(held_labels, (held_features - mu_benign) @ direction),
            "norm_delta": norm, "probe_cosine": float(normal @ direction),
            "split_half_cosine": split_half_cosine(features, labels, seed=seed),
        })
        arrays.update({f"{layer}/mu_benign": mu_benign, f"{layer}/mu_harmful": mu_harmful,
                       f"{layer}/mean_diff": direction, f"{layer}/probe": normal})
        arrays.update({f"{layer}/random_{s}": random_direction(len(direction), s) for s in RANDOM_SEEDS})
        if len(validation):
            projection = (load(validation, layer) - mu_benign) @ direction / norm
            for group, values in pd.Series(projection).groupby(validation["subset_group"].to_numpy()):
                projections.append({"layer": layer, "group": group, "n": len(values),
                                    "mean": float(values.mean()), "sd": float(values.std())})
        if len(set(held_labels)) == 2:
            steps.extend(step_aucs(discovery, labels, held, held_labels, layer))

    table = pd.DataFrame(table)
    layer = choose_layer(table)
    statistics = {
        "layer": layer,
        "rule": f"highest discovery CV ROC-AUC; earliest layer within {TIE}",
        "norm_delta": float(table.set_index("layer").loc[layer, "norm_delta"]),
        "seed": seed,
        "n": {split: frame["subset_group"].value_counts().to_dict()
              for split, frame in (("discovery", discovery), ("validation", validation))},
        "prompt_ids": {"discovery": discovery["prompt_id"].astype(str).tolist(),
                       "validation": validation["prompt_id"].astype(str).tolist()},
        "layers": table.to_dict("records"),
        "versions": versions(),
    }
    return table, arrays, pd.DataFrame(projections), pd.DataFrame(steps), statistics


def versions():
    import sklearn

    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def capture_layers(generations):
    """The layers saved in the first activation file, in hook order."""
    with np.load(generations["activation_path"].dropna().iloc[0]) as first:
        return list(dict.fromkeys(name.rpartition("/")[0] for name in first.files))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-run", type=Path, required=True, help="E4 generate run folder (or its table)")
    parser.add_argument("--output", type=Path, required=True, help="e.g. results/e4/<model>")
    parser.add_argument("--layers", nargs="*", help="default: every captured layer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    generations = read_table(find_table(args.capture_run, "generations.parquet"))
    layers = args.layers or capture_layers(generations)
    table, arrays, projections, steps, statistics = analyze(generations, layers, args.seed)
    statistics["capture_run"] = str(args.capture_run)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "layer_auc.csv", index=False)
    projections.to_csv(args.output / "group_projection.csv", index=False)
    steps.to_csv(args.output / "step_auc.csv", index=False)
    np.savez(args.output / "directions.npz", **{k: np.asarray(v, np.float32) for k, v in arrays.items()})
    write_json(statistics, args.output / "statistics.json")
    columns = ["layer", "cv_auc", "cv_auc_sd", "validation_direction_auc", "split_half_cosine"]
    print(table[columns].to_string(index=False))
    print(json.dumps({"layer": statistics["layer"], "output": str(args.output)}, indent=2))
    return 0
