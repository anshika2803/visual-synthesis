"""The standard schema every dataset is converted to.

One row per prompt, with the columns in STANDARD_COLUMNS. The harm categories
and their risk levels (the taxonomy) live in the config, under ``taxonomy``.
"""

import hashlib
import re

# A prompt may carry several categories; they are joined with this separator.
SEPARATOR = ";"

STANDARD_COLUMNS = [
    "prompt_id",  # globally unique, e.g. I2P_00042
    "source_dataset",  # e.g. I2P
    "source_id",  # id within the source dataset
    "prompt",
    "original_category",  # the source dataset's own label(s), ';'-joined
    "standard_category",  # ';'-joined taxonomy categories
    "risk_level",  # low | medium | high -- the most severe category wins
    "benchmark_level",  # L0..L6, see BENCHMARK_LEVELS
    "transformation",
    "split",
    "is_duplicate",  # the next three are filled by the postprocess stage
    "duplicate_of_prompt_id",
    "duplicate_group_id",
    "parent_prompt_id",
    "source_metadata",  # 'key=value;key=value' annotations from the source
    "content_hash",  # hash of the normalised prompt
    "run_id",  # the run that produced the row
]

RISK_LEVELS = ("low", "medium", "high")  # in increasing severity

# How a prompt was produced, from verbatim to adversarially reworded.
BENCHMARK_LEVELS = {
    "L0": "benign",
    "L1": "original risky prompt",
    "L2": "paraphrased prompt",
    "L3": "obfuscated prompt",
    "L4": "indirect/contextual prompt",
    "L5": "multilingual prompt",
    "L6": "adversarial prompt",
}

TRANSFORMATIONS = ("original", "paraphrase", "obfuscation", "indirect", "multilingual", "adversarial")


def normalize_prompt(text, case_sensitive=False, collapse_whitespace=True):
    """The form of a prompt used to decide whether two prompts are the same."""
    text = str(text).strip()
    if not case_sensitive:
        text = text.lower()
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    return text


def content_hash(text):
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()[:16]


def risk_level(categories, taxonomy):
    """Risk of a prompt: the most severe of its categories (``low`` if none is known)."""
    levels = [taxonomy[c] for c in categories if c in taxonomy]
    return max(levels, key=RISK_LEVELS.index) if levels else "low"


def split_categories(value):
    return [part.strip() for part in str(value).split(SEPARATOR) if part.strip()]


def validate_table(frame, taxonomy):
    """Check a table against the standard schema.

    Returns a report dict; ``report["ok"]`` is False when ``report["errors"]``
    is not empty.
    """
    errors = []
    missing = [c for c in STANDARD_COLUMNS if c not in frame.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if frame.empty:
        errors.append("table is empty")

    category_counts = {}
    if "standard_category" in frame.columns:
        for value in frame["standard_category"].dropna():
            for category in split_categories(value):
                category_counts[category] = category_counts.get(category, 0) + 1
        unknown = sorted(c for c in category_counts if c not in taxonomy)
        if unknown:
            errors.append(f"unknown standard categories: {unknown}")

    duplicate_prompts = 0
    if "prompt" in frame.columns:
        prompts = frame["prompt"].fillna("").astype(str)
        empty = int((prompts.str.strip() == "").sum())
        if empty:
            errors.append(f"{empty} row(s) have an empty prompt")
        duplicate_prompts = int(prompts.map(normalize_prompt).duplicated().sum())

    if "prompt_id" in frame.columns:
        duplicate_ids = int(frame["prompt_id"].duplicated().sum())
        if duplicate_ids:
            errors.append(f"{duplicate_ids} duplicate prompt_id value(s)")

    risk_counts = {}
    if "risk_level" in frame.columns:
        risk_counts = {str(k): int(v) for k, v in frame["risk_level"].value_counts().items()}

    return {
        "ok": not errors,
        "errors": errors,
        "rows": len(frame),
        "missing_columns": missing,
        "category_counts": dict(sorted(category_counts.items())),
        "risk_counts": risk_counts,
        "duplicate_prompts": duplicate_prompts,
    }
