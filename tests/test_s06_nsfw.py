from threading import Barrier, Lock
from time import sleep

import pandas as pd
from PIL import Image

from visual_synthesis.backends.detectors import FakeDetector
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


def test_configured_workers_use_separate_detector_backends(cfg, tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(6):
        Image.new("RGB", (8, 8), "white" if index % 2 else "black").save(images / f"image-{index}.png")

    cfg["nsfw"]["workers"] = 3
    state = {"active": 0, "maximum": 0}
    state_lock = Lock()
    barrier = Barrier(3)

    class SlowFakeDetector:
        def __init__(self, detector):
            self.delegate = FakeDetector(detector)

        def load(self):
            self.delegate.load()
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            barrier.wait()

        def detect(self, path):
            sleep(0.01)
            return self.delegate.detect(path)

        def unload(self):
            self.delegate.unload()
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(s06_nsfw, "build_detector", lambda detector, models_root: SlowFakeDetector(detector))
    result = run_stage(cfg, s06_nsfw, input_path=images)

    table = pd.read_parquet(f"{result['run_dir']}/detections.parquet")
    assert result["workers"] == 3
    assert state["maximum"] == 3
    assert len(table) == 6
