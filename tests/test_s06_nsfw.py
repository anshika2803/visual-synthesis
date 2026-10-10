import pandas as pd
from PIL import Image

from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s06_nsfw


def test_images_are_scored_and_flagged(cfg, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), "white").save(images / "bright.png")  # fake detector: brightness = score
    Image.new("RGB", (8, 8), "black").save(images / "dark.png")

    result = run_stage(cfg, s06_nsfw, run_name="t", input_path=images)

    table = pd.read_parquet(f"{result['run_dir']}/detections.parquet").set_index("image_id")
    assert table.loc["bright", "flagged"] and not table.loc["dark", "flagged"]
    assert result["flagged_rate"] == 0.5

    again = run_stage(cfg, s06_nsfw, run_name="t", input_path=images)
    assert again["rows_skipped"] == 2
