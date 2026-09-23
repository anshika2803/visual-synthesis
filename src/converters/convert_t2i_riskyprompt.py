import csv
import os

from src.standard_schema import STANDARD_FIELDS
from src.taxonomy import RISK_LEVELS

INPUT_FILE = "data/processed/t2i_riskyprompt.csv"
OUTPUT_FILE = "data/processed/standardized/t2i_riskyprompt.csv"


# --------------------------------------------------
# Mapping from original dataset categories
# to our common benchmark taxonomy
# --------------------------------------------------

CATEGORY_MAP = {
    "Pornography--Borderline_Pornography": "sexual",
    "Pornography--Explicit_Pornography": "sexual",

    "Violence--Bloody_Content": "violence",
    "Violence--Weapons_and_Conflicts": "violence",

    "Disturbing_Content--Terrifying_Content": "disturbing",
    "Disturbing_Content--Other_Disturbing_Content": "disturbing",

    "Illegal_Activities--Drug_Crimes": "illegal_activity",
    "Illegal_Activities--Illegal_Trade": "illegal_activity",
    "Illegal_Activities--Theft_and_Robbery": "illegal_activity",
    "Illegal_Activities--Other_Illegal_Content": "illegal_activity",

    "Political_Sensitivity--Political_Figures": "political_sensitivity",
    "Political_Sensitivity--Political_Metaphor": "political_sensitivity",

    "Copyright_Infringement--Cartoon_Characters": "copyright",
    "Copyright_Infringement--LOGO": "copyright",
}




os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# --------------------------------------------------
# Read source dataset
# --------------------------------------------------

with open(INPUT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


standardized_rows = []


# --------------------------------------------------
# Convert each prompt
# --------------------------------------------------

for row in rows:

    original_categories = row["category"].split(";")

    # Map every original category
    standard_categories = []

    for category in original_categories:

        if category not in CATEGORY_MAP:
            raise ValueError(
                f"Unknown category encountered: {category}"
            )

        mapped = CATEGORY_MAP[category]

        if mapped not in standard_categories:
            standard_categories.append(mapped)

    # If a prompt has multiple categories,
    # preserve all standardized categories.
    standard_category = ";".join(standard_categories)

    # Use the highest-risk category for risk_level.
    risk_levels = [
        RISK_LEVELS[category]
        for category in standard_categories
    ]

    if "high" in risk_levels:
        risk_level = "high"
    elif "medium" in risk_levels:
        risk_level = "medium"
    else:
        risk_level = "low"

    standardized_rows.append({
        "prompt_id": row["prompt_id"],
        "source_dataset": "T2I-RiskyPrompt",
        "source_id": row["prompt_id"].replace("TRP_", ""),
        "prompt": row["prompt"],
        "original_category": row["category"],
        "standard_category": standard_category,
        "risk_level": risk_level,
        "benchmark_level": "L1",
        "transformation": "original",
        "split": "unassigned",
        "is_duplicate": "false",
        "parent_prompt_id": row["prompt_id"],
        "source_metadata": "",
    })


# --------------------------------------------------
# Write standardized dataset
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=STANDARD_FIELDS,
    )

    writer.writeheader()
    writer.writerows(standardized_rows)


print(f"Converted {len(standardized_rows)} prompts.")
print(f"Saved to: {OUTPUT_FILE}")