import csv
import glob
import os

from src.standard_schema import STANDARD_FIELDS


INPUT_DIR = "data/processed/standardized"
OUTPUT_FILE = "data/processed/unified/unified_candidates.csv"


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# Find all standardized datasets
input_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.csv"))
)

print("Standardized datasets found:")

for file in input_files:
    print(" -", file)


if not input_files:
    raise ValueError("No standardized CSV files found.")


all_rows = []


# Read every standardized dataset
for file in input_files:

    with open(file, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Make sure the dataset follows our schema
        missing_fields = [
            field
            for field in STANDARD_FIELDS
            if field not in reader.fieldnames
        ]

        if missing_fields:
            raise ValueError(
                f"{file} is missing fields: {missing_fields}"
            )

        rows = list(reader)

        print(f"{file}: {len(rows)} records")

        all_rows.extend(rows)


# Check prompt IDs are unique
prompt_ids = [row["prompt_id"] for row in all_rows]

if len(prompt_ids) != len(set(prompt_ids)):
    raise ValueError("Duplicate prompt_id values detected.")


# Write unified candidate dataset
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
    writer.writerows(all_rows)


print()
print("Unified candidate dataset created.")
print("Total records:", len(all_rows))
print("Saved to:", OUTPUT_FILE)