import csv
import json
import os


INPUT_FILE = "data/raw/t2i-riskyprompt/T2I-RiskyPrompt.json"
OUTPUT_FILE = "data/processed/t2i_riskyprompt.csv"


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# Load original dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# Convert to our standard format
rows = []

for idx, item in data.items():

    prompt = item["prompt"]
    labels = item.get("label", {})

    # Preserve all original labels
    categories = ";".join(labels.keys())

    rows.append({
        "prompt_id": f"TRP_{int(idx):05d}",
        "prompt": prompt,
        "dataset": "T2I-RiskyPrompt",
        "category": categories,
        "risk_level": "",
        "transformation": "original",
    })


# Save standardized CSV
fieldnames = [
    "prompt_id",
    "prompt",
    "dataset",
    "category",
    "risk_level",
    "transformation",
]


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)


print(f"Converted {len(rows)} prompts.")
print(f"Saved to: {OUTPUT_FILE}")
