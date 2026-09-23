import csv
import os
import time

import torch
from diffusers import StableDiffusionPipeline


# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "models/sd-turbo"
INPUT_CSV = "data/processed/baseline_test.csv"
OUTPUT_DIR = "results/images"
RESULTS_CSV = "results/tables/baseline_results.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)


# -----------------------------
# Load model ONCE
# -----------------------------

print("Loading SD-Turbo...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    variant="fp16",
    torch_dtype=torch.float16,
    local_files_only=True,
)

pipe = pipe.to("cuda")

print("MODEL LOADED ON:", torch.cuda.get_device_name(0))


# -----------------------------
# Read prompts
# -----------------------------

with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


# -----------------------------
# Generate images + save results
# -----------------------------

fieldnames = [
    "prompt_id",
    "prompt",
    "dataset",
    "category",
    "risk_level",
    "transformation",
    "image_path",
    "generation_time",
    "model",
]


with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as results_file:

    writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:

        prompt_id = row["prompt_id"]
        prompt = row["prompt"]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{prompt_id}.png"
        )

        print(f"\nGenerating {prompt_id}")
        print(f"Prompt: {prompt}")

        start_time = time.time()

        image = pipe(
            prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
        ).images[0]

        image.save(output_path)

        elapsed = time.time() - start_time

        writer.writerow({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "dataset": row["dataset"],
            "category": row["category"],
            "risk_level": row["risk_level"],
            "transformation": row["transformation"],
            "image_path": output_path,
            "generation_time": f"{elapsed:.3f}",
            "model": "SD-Turbo",
        })

        results_file.flush()

        print(f"Saved: {output_path}")
        print(f"Time: {elapsed:.2f} seconds")


print("\nALL PROMPTS COMPLETED.")
print(f"Results saved to: {RESULTS_CSV}")