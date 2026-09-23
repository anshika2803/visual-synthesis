import csv
import re

from src.standard_schema import STANDARD_FIELDS


INPUT_FILE = "data/processed/unified/unified_candidates.csv"
OUTPUT_FILE = "data/processed/unified/unified_deduplicated.csv"


def normalize_prompt(text):
    """Normalize text only for duplicate comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


with open(INPUT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


seen = set()
duplicate_count = 0


for row in rows:

    normalized = normalize_prompt(row["prompt"])

    if normalized in seen:
        row["is_duplicate"] = "true"
        duplicate_count += 1
    else:
        row["is_duplicate"] = "false"
        seen.add(normalized)


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
    writer.writerows(rows)


print("Input records:", len(rows))
print("Exact duplicates:", duplicate_count)
print("Unique prompts:", len(rows) - duplicate_count)
print("Saved to:", OUTPUT_FILE)