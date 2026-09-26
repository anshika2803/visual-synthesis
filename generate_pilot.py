import csv
import os
import time

import torch
from diffusers import StableDiffusionPipeline


MODEL_PATH = "models/sd-turbo"
INPUT_CSV = "data/processed/unified/pilot_60.csv"
OUTPUT_DIR = "results/pilot_images"
RESULTS_CSV = "results/tables/pilot_60_results.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)


print("Loading SD-Turbo...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    variant="fp16",
    torch_dtype=torch.float16,
    local_files_only=True,
)

pipe.to("cuda")

print("Model loaded.")


with open(INPUT_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("Prompts to generate:", len(rows))


result_fields = [
    "prompt_id",
    "prompt",
    "source_dataset",
    "standard_category",
    "risk_level",
    "benchmark_level",
    "transformation",
    "image_path",
    "generation_time",
    "model",
]


with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as result_file:

    writer = csv.DictWriter(result_file, fieldnames=result_fields)
    writer.writeheader()

    for i, row in enumerate(rows, start=1):

        prompt_id = row["prompt_id"]
        prompt = row["prompt"]

        print(f"[{i}/{len(rows)}] Generating {prompt_id}...")

        start = time.time()

        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
            ).images[0]

        generation_time = time.time() - start

        image_path = os.path.join(
            OUTPUT_DIR,
            f"{prompt_id}.png"
        )

        image.save(image_path)

        writer.writerow({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "source_dataset": row["source_dataset"],
            "standard_category": row["standard_category"],
            "risk_level": row["risk_level"],
            "benchmark_level": row["benchmark_level"],
            "transformation": row["transformation"],
            "image_path": image_path,
            "generation_time": f"{generation_time:.4f}",
            "model": "sd-turbo",
        })

        result_file.flush()


print()
print("Generation complete.")
print("Images saved to:", OUTPUT_DIR)
print("Results saved to:", RESULTS_CSV)
