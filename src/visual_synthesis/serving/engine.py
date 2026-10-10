"""The engine behind the API, the web UI and the terminal session.

It keeps one image model and one detector loaded between requests (loading
SDXL takes tens of seconds), and swaps them when a different id is asked for.
"""

import logging
import threading
import time
import uuid
from pathlib import Path

from visual_synthesis.backends.detectors import build_detector, is_flagged
from visual_synthesis.backends.image import build_image_backend, save_image
from visual_synthesis.config import get_detector, get_model

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.model_id = None
        self.image_backend = None
        self.detector_id = None
        self.detector_backend = None

    # ------------------------------------------------------------- backends

    def load_model(self, model_id=None):
        """The loaded image backend for ``model_id`` (default: generate.model)."""
        model = get_model(self.cfg, model_id)
        if self.model_id != model["id"]:
            if self.image_backend is not None:
                self.image_backend.unload()  # free the memory before loading another model
            self.image_backend = build_image_backend(model, self.cfg["paths"]["models"])
            self.image_backend.load()
            self.model_id = model["id"]
        return self.image_backend

    def load_detector(self, detector_id=None):
        """The loaded detector backend, or None when detection is off."""
        detector = get_detector(self.cfg, detector_id)
        if detector is None:
            return None
        if self.detector_id != detector["id"]:
            if self.detector_backend is not None:
                self.detector_backend.unload()
            self.detector_backend = build_detector(detector, self.cfg["paths"]["models"])
            self.detector_backend.load()
            self.detector_id = detector["id"]
        return self.detector_backend

    def loaded(self):
        return {"model": self.model_id, "detector": self.detector_id}

    def close(self):
        for backend in (self.image_backend, self.detector_backend):
            if backend is not None:
                backend.unload()
        self.model_id = self.image_backend = self.detector_id = self.detector_backend = None

    # ------------------------------------------------------------- requests

    def generate(self, prompt, out_dir, model=None, prompt_id=None, n=1, seed=None, negative_prompt=None,
                 steps=None, guidance_scale=None, width=None, height=None, detector="off", threshold=None):
        """Generate ``n`` images for a prompt, save them under ``out_dir``, optionally classify them.

        Settings left as None use the model's config. ``detector="off"`` skips
        detection; ``detector=None`` uses the configured default.
        """
        with self.lock:
            backend = self.load_model(model)
            prompt_id = prompt_id or uuid.uuid4().hex[:12]
            started = time.perf_counter()
            images = backend.generate([{
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": seed,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "num_images": n,
            }])
            seconds = time.perf_counter() - started
            paths = [
                str(save_image(image["image"], Path(out_dir) / f"{prompt_id}_{image['seed']:08d}.png"))
                for image in images
            ]
        log.info("generated %d image(s) for %s with %s in %.2fs", len(paths), prompt_id, self.model_id, seconds)
        result = {
            "prompt_id": prompt_id,
            "model": self.model_id,
            "seed": images[0]["seed"],
            "width": images[0]["width"],
            "height": images[0]["height"],
            "steps": images[0]["steps"],
            "generation_time_s": round(seconds, 4),
            "images": paths,
            "detections": None,
        }
        if get_detector(self.cfg, detector) is not None:
            result["detections"] = [self.classify(path, detector, threshold) for path in paths]
        return result

    def classify(self, image_path, detector=None, threshold=None):
        """Run a detector over one image. Raises ValueError when detection is off."""
        entry = get_detector(self.cfg, detector)
        if entry is None:
            raise ValueError("no NSFW detector is enabled")
        with self.lock:
            backend = self.load_detector(entry["id"])
            started = time.perf_counter()
            scores = backend.detect(Path(image_path))
            seconds = time.perf_counter() - started
        return {
            "image_id": Path(image_path).stem,
            "image_path": str(image_path),
            "detector": entry["id"],
            "threshold": entry["threshold"] if threshold is None else threshold,
            "flagged": is_flagged(scores, entry, threshold),
            "max_score": round(max(scores.values(), default=0.0), 6),
            "labels": sorted(scores, key=scores.get, reverse=True),
            "per_class": {label: round(score, 6) for label, score in scores.items()},
            "latency_s": round(seconds, 4),
        }
