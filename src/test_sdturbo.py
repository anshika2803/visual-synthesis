import os
import json
import csv
import time
import torch
from diffusers import StableDiffusionPipeline

# ============================================================
# PATHS
# ============================================================

DATASET = "data/T2I-RiskyPrompt.json"
IMAGE_DIR = "results/t2i-riskyprompt/images"
OUTPUT_CSV = "results/t2i-riskyprompt/generation_results.csv"

MODEL_PATH = "models/sd-turbo"

MAX_PROMPTS = 6431      # Change to 100, 500, 6432, etc.
START_INDEX = 0

# ============================================================
# GENERATION SETTINGS
# ============================================================

NUM_INFERENCE_STEPS = 1
GUIDANCE_SCALE = 0.0
HEIGHT = 512
WIDTH = 512

DEVICE = "cuda"


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading T2I-RiskyPrompt dataset...")

with open(DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

items = list(dataset.items())

selected_items = items[
    START_INDEX : START_INDEX + MAX_PROMPTS
]

dataset = dict(selected_items)

print(f"Total prompts in dataset: {len(items)}")
print(f"Selected prompts: {len(dataset)}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


# ============================================================
# LOAD SD-TURBO
# ============================================================

print("Loading SD-Turbo...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    variant="fp16",
    dtype=torch.float16,
    local_files_only=True
)

pipe = pipe.to(DEVICE)

print("Model loaded on:", torch.cuda.get_device_name(0))


# ============================================================
# GENERATE IMAGES
# ============================================================

results = []

for dataset_id, item in dataset.items():

    prompt = item["prompt"]
    labels = item.get("label", {})

    # Example:
    # 0 -> RP-0000.png
    # 1 -> RP-0001.png
    # ...
    prompt_id = f"RP-{int(dataset_id):04d}"

    image_path = os.path.join(
        IMAGE_DIR,
        f"{prompt_id}.png"
    )

    print("\n" + "=" * 70)
    print(f"Prompt {dataset_id} / {len(dataset)}")
    print(f"ID: {prompt_id}")
    print(f"Prompt: {prompt}")

    # --------------------------------------------------------
    # Skip if image already exists
    # --------------------------------------------------------

    if os.path.exists(image_path):

        print("Image already exists -> SKIPPING")

        results.append({
            "prompt_id": prompt_id,
            "dataset_id": dataset_id,
            "prompt": prompt,
            "labels": json.dumps(labels, ensure_ascii=False),
            "image_path": image_path,
            "status": "already_exists"
        })

        continue

    # --------------------------------------------------------
    # Generate image
    # --------------------------------------------------------

    print("Generating...")

    start_time = time.time()

    # Fixed seed for reproducibility
    generator = torch.Generator(
        device=DEVICE
    ).manual_seed(int(dataset_id))

    try:

        with torch.inference_mode():

            image = pipe(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH,
                generator=generator
            ).images[0]

        image.save(image_path)

        elapsed = time.time() - start_time

        print(f"Saved: {image_path}")
        print(f"Generation time: {elapsed:.2f} seconds")

        results.append({
            "prompt_id": prompt_id,
            "dataset_id": dataset_id,
            "prompt": prompt,
            "labels": json.dumps(labels, ensure_ascii=False),
            "image_path": image_path,
            "status": "generated"
        })

    except Exception as e:

        print(f"ERROR: {e}")

        results.append({
            "prompt_id": prompt_id,
            "dataset_id": dataset_id,
            "prompt": prompt,
            "labels": json.dumps(labels, ensure_ascii=False),
            "image_path": image_path,
            "status": f"error: {e}"
        })


# ============================================================
# SAVE METADATA
# ============================================================

print("\nSaving metadata...")

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "prompt_id",
        "dataset_id",
        "prompt",
        "labels",
        "image_path",
        "status"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Images : {IMAGE_DIR}")
print(f"Metadata: {OUTPUT_CSV}")