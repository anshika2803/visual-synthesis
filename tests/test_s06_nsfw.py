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


def test_generation_failures_without_images_do_not_break_context_join(cfg, tmp_path):
    from visual_synthesis.io import write_table

    image = tmp_path / "one.png"
    Image.new("RGB", (8, 8), "white").save(image)
    write_table(pd.DataFrame([
        {"prompt_id": "ok", "image_path": str(image), "attack_config": "K_16"},
        {"prompt_id": "failed", "image_path": None, "attack_config": "K_38"},
    ]), tmp_path / cfg["generate"]["output_name"])

    context = s06_nsfw.load_generation_context(tmp_path, cfg)
    assert context[image.stem]["prompt_id"] == "ok"
    assert len(context) == 1


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


def test_detector_that_is_not_thread_safe_gets_one_worker(cfg, tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), "white").save(images / f"image-{index}.png")
    cfg["nsfw"]["workers"] = 3
    monkeypatch.setattr(s06_nsfw, "supports_threads", lambda detector: False)

    result = run_stage(cfg, s06_nsfw, input_path=images)

    assert result["workers"] == 1
    assert len(pd.read_parquet(f"{result['run_dir']}/detections.parquet")) == 4


def test_only_the_metal_safety_checker_is_not_thread_safe(monkeypatch):
    from visual_synthesis.backends import detectors, device

    assert detectors.supports_threads({"backend": "nudenet", "device": "auto"})
    assert detectors.supports_threads({"backend": "fake", "device": "auto"})
    monkeypatch.setattr(device, "pick_device", lambda preference, index=None: "mps")
    assert not detectors.supports_threads({"backend": "safety_checker", "device": "auto"})
    monkeypatch.setattr(device, "pick_device", lambda preference, index=None: "cpu")
    assert detectors.supports_threads({"backend": "safety_checker", "device": "auto"})
