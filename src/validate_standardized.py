import csv

from src.standard_schema import STANDARD_FIELDS


INPUT_FILE = "data/processed/standardized/t2i_riskyprompt.csv"

ALLOWED_CATEGORIES = {
    "sexual",
    "violence",
    "disturbing",
    "illegal_activity",
    "political_sensitivity",
    "copyright",
}


with open(INPUT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


print("Records:", len(rows))

# Check columns
missing_columns = [
    field for field in STANDARD_FIELDS
    if field not in reader.fieldnames
]

print("Missing columns:", missing_columns)

# Check prompts
missing_prompts = sum(
    1 for row in rows
    if not row["prompt"].strip()
)

print("Missing prompts:", missing_prompts)

# Check standardized categories
unknown_categories = set()

for row in rows:
    for category in row["standard_category"].split(";"):
        if category not in ALLOWED_CATEGORIES:
            unknown_categories.add(category)

print("Unknown categories:", unknown_categories)

# Final result
if (
    len(rows) == 6433
    and not missing_columns
    and missing_prompts == 0
    and not unknown_categories
):
    print("VALIDATION PASSED")
else:
    print("VALIDATION FAILED")