import csv

from src.standard_schema import STANDARD_FIELDS
from src.taxonomy import RISK_LEVELS


INPUT_FILE = "data/raw/i2p/i2p_benchmark.csv"
OUTPUT_FILE = "data/processed/standardized/i2p.csv"


CATEGORY_MAP = {
    "sexual": "sexual",
    "violence": "violence",
    "shocking": "disturbing",
    "self-harm": "disturbing",
    "illegal activity": "illegal_activity",
    "harassment": "disturbing",
    "hate": "political_sensitivity",
}


RISK_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


with open(INPUT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


output_rows = []

for index, row in enumerate(rows):

    original_categories = [
        category.strip()
        for category in row["categories"].split(",")
    ]

    standard_categories = []

    for category in original_categories:
        if category not in CATEGORY_MAP:
            raise ValueError(
                f"Unknown I2P category: {category}"
            )

        mapped = CATEGORY_MAP[category]

        if mapped not in standard_categories:
            standard_categories.append(mapped)

    risk_levels = [
        RISK_LEVELS[category]
        for category in standard_categories
    ]

    risk_level = max(
        risk_levels,
        key=lambda x: RISK_ORDER[x]
    )

    prompt_id = f"I2P_{index:05d}"

    output_rows.append({
        "prompt_id": prompt_id,
        "source_dataset": "I2P",
        "source_id": str(index),
        "prompt": row["prompt"],
        "original_category": row["categories"],
        "standard_category": ";".join(standard_categories),
        "risk_level": risk_level,
        "benchmark_level": "L1",
        "transformation": "original",
        "split": "unassigned",
        "is_duplicate": "false",
        "parent_prompt_id": prompt_id,
        "source_metadata": (
            f"hard={row['hard']};"
            f"inappropriate_percentage={row['inappropriate_percentage']};"
            f"nudity_percentage={row['nudity_percentage']};"
            f"q16_percentage={row['q16_percentage']};"
            f"sd_safety_percentage={row['sd_safety_percentage']};"
            f"prompt_toxicity={row['prompt_toxicity']}"
        ),
    })


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
    writer.writerows(output_rows)


print("Converted:", len(output_rows))
print("Saved to:", OUTPUT_FILE)
