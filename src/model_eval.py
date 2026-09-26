import csv
import os
import time
import torch
from diffusers import StableDiffusionPipeline

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/sd-turbo"
INPUT_CSV = "data/prompts.csv"
OUTPUT_DIR = "results/images"
OUTPUT_CSV = "results/generation_results.csv"

NUM_INFERENCE_STEPS = 1
GUIDANCE_SCALE = 0.0
HEIGHT = 512
WIDTH = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load model ONCE
# --------------------------------------------------

print("Loading SD-Turbo...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    variant="fp16",
    dtype=torch.float16,
    local_files_only=True
)

pipe = pipe.to("cuda")

print("GPU:", torch.cuda.get_device_name(0))
print("Model loaded.")
print()

# --------------------------------------------------
# Read dataset
# --------------------------------------------------

with open(INPUT_CSV, "r", encoding="utf-8") as f:
    prompts = list(csv.DictReader(f))

print(f"Found {len(prompts)} prompts.")

# --------------------------------------------------
# Output metadata
# --------------------------------------------------

fieldnames = [
    "prompt_id",
    "category",
    "risk_level",
    "prompt",
    "image_path",
    "generation_time_sec",
    "status"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # --------------------------------------------------
    # Generate sequentially
    # --------------------------------------------------

    for i, row in enumerate(prompts, start=1):

        prompt_id = row["prompt_id"]
        prompt = row["prompt"]

        image_filename = f"{prompt_id}.png"
        image_path = os.path.join(
            OUTPUT_DIR,
            image_filename
        )

        print(
            f"[{i}/{len(prompts)}] "
            f"{prompt_id}: {prompt}"
        )

        # Don't regenerate existing images
        if os.path.exists(image_path):

            print("  Already exists. Skipping.")

            writer.writerow({
                "prompt_id": prompt_id,
                "category": row.get("category", ""),
                "risk_level": row.get("risk_level", ""),
                "prompt": prompt,
                "image_path": image_path,
                "generation_time_sec": "",
                "status": "already_exists"
            })

            continue

        try:

            start = time.time()

            image = pipe(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH
            ).images[0]

            image.save(image_path)

            elapsed = time.time() - start

            writer.writerow({
                "prompt_id": prompt_id,
                "category": row.get("category", ""),
                "risk_level": row.get("risk_level", ""),
                "prompt": prompt,
                "image_path": image_path,
                "generation_time_sec": round(elapsed, 3),
                "status": "generated"
            })

            f.flush()

            print(
                f"  ✓ Saved: {image_path} "
                f"({elapsed:.2f}s)"
            )

        except Exception as e:

            print(f"  ✗ ERROR: {e}")

            writer.writerow({
                "prompt_id": prompt_id,
                "category": row.get("category", ""),
                "risk_level": row.get("risk_level", ""),
                "prompt": prompt,
                "image_path": "",
                "generation_time_sec": "",
                "status": f"error: {e}"
            })

            f.flush()

print()
print("Generation complete.")
print("Images:", OUTPUT_DIR)
print("Metadata:", OUTPUT_CSV)