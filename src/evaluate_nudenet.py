import os
import csv
from nudenet import NudeDetector

IMAGE_DIR = "results/t2i-riskyprompt/images"
OUTPUT_FILE = "results/t2i-riskyprompt/nudenet_results.csv"

detector = NudeDetector()

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

results = []

images = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
)

print(f"Found {len(images)} images")

for i, filename in enumerate(images, 1):

    image_path = os.path.join(IMAGE_DIR, filename)

    print(f"[{i}/{len(images)}] {filename}")

    try:
        detections = detector.detect(image_path)

        results.append({
            "image_id": os.path.splitext(filename)[0],
            "image_path": image_path,
            "nudenet_detections": str(detections),
            "status": "success"
        })

    except Exception as e:

        results.append({
            "image_id": os.path.splitext(filename)[0],
            "image_path": image_path,
            "nudenet_detections": "",
            "status": f"error: {e}"
        })


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "image_path",
            "nudenet_detections",
            "status"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nDONE")
print(f"Results saved to: {OUTPUT_FILE}")