"""NSFW detector backends.

Every detector answers the same question the same way::

    detector = build_detector(detector_cfg, models_root)
    detector.load()
    scores = detector.detect(image_path)      # {"LABEL": score between 0 and 1}
    detector.unload()

Because the output shape is shared, switching detectors is a config change and
the detections table keeps the same columns.
"""

import logging
from pathlib import Path

from visual_synthesis.io import safe_name

log = logging.getLogger(__name__)


def build_detector(detector, models_root=None):
    """The backend for a detector entry (see ``get_detector`` in config.py)."""
    backends = {"nudenet": NudeNetDetector, "safety_checker": SafetyCheckerDetector, "fake": FakeDetector}
    if detector["backend"] not in backends:
        raise KeyError(f"unknown detector backend {detector['backend']!r}; available: {sorted(backends)}")
    return backends[detector["backend"]](detector, models_root)


def is_flagged(scores, detector, threshold=None):
    """Whether an image counts as NSFW.

    True when any label scores at or above its threshold. ``threshold``
    overrides the detector's own; ``label_thresholds`` overrides it per label;
    and when ``flag_labels`` is set, only those labels are considered.
    """
    default = detector["threshold"] if threshold is None else threshold
    for label, score in scores.items():
        if detector["flag_labels"] is not None and label not in detector["flag_labels"]:
            continue
        if score >= detector["label_thresholds"].get(label, default):
            return True
    return False


class NudeNetDetector:
    """NudeNet: an ONNX object detector for nudity. Its weights ship with the package.

    It reports body-part labels such as FEMALE_BREAST_EXPOSED but also FACE_FEMALE
    or FEET_EXPOSED, so the detector's ``flag_labels`` should list the ones that
    actually mean NSFW.
    """

    def __init__(self, detector, models_root=None):
        self.detector = detector
        self.model = None

    def load(self):
        if self.model is not None:
            return
        try:
            from nudenet import NudeDetector
        except ImportError as exc:
            raise RuntimeError("NudeNet needs the 'nsfw' extra: pip install -e '.[nsfw]'") from exc
        self.model = NudeDetector()
        log.info("detector %s (nudenet) loaded", self.detector["id"])

    def detect(self, path):
        self.load()
        scores = {}
        for box in self.model.detect(str(path)):
            # Several boxes can share a label; keep the strongest.
            scores[box["class"]] = max(scores.get(box["class"], 0.0), float(box["score"]))
        return scores

    def unload(self):
        self.model = None


class SafetyCheckerDetector:
    """The safety checker that ships with Stable Diffusion.

    It answers "would the model's own filter have blanked this image?", so its
    single ``nsfw`` score is 1.0 (yes) or 0.0 (no).
    """

    def __init__(self, detector, models_root=None):
        self.detector = detector
        self.models_root = Path(models_root or "models")
        self.checker = None

    def load(self):
        if self.checker is not None:
            return
        try:
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPImageProcessor
        except ImportError as exc:
            raise RuntimeError("the safety checker needs the 'gpu' extra: pip install -e '.[gpu]'") from exc
        from visual_synthesis.backends.device import pick_device

        local = self.models_root / safe_name(self.detector["local_dir"] or self.detector["id"])
        source = str(local) if local.is_dir() and any(local.iterdir()) else self.detector["repo_id"]
        self.device = pick_device(self.detector["device"])
        self.processor = CLIPImageProcessor.from_pretrained(source)
        self.checker = StableDiffusionSafetyChecker.from_pretrained(source).to(self.device).eval()
        log.info("detector %s (safety_checker) loaded from %s", self.detector["id"], source)

    def detect(self, path):
        import torch
        from diffusers.pipelines.stable_diffusion.safety_checker import cosine_distance
        from PIL import Image

        self.load()
        checker = self.checker
        pixels = self.processor(Image.open(path).convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            embedding = checker.visual_projection(checker.vision_model(pixels.to(self.device))[1])
            # Same rule as the checker itself: an image is NSFW when it is closer
            # to any NSFW concept than that concept's learned threshold. A hit on
            # a "special care" concept tightens the thresholds slightly.
            special = cosine_distance(embedding, checker.special_care_embeds)[0] - checker.special_care_embeds_weights
            adjustment = 0.01 if bool((special > 0).any()) else 0.0
            margin = cosine_distance(embedding, checker.concept_embeds)[0] - checker.concept_embeds_weights + adjustment
        return {"nsfw": 1.0 if bool((margin > 0).any()) else 0.0}

    def unload(self):
        if self.checker is None:
            return
        from visual_synthesis.backends.device import free_memory

        self.checker = None
        self.processor = None
        free_memory()


class FakeDetector:
    """Scores an image by its brightness. No weights; for tests and dry runs of the pipeline."""

    def __init__(self, detector, models_root=None):
        self.detector = detector
        self.model = None

    def load(self):
        self.model = "fake"

    def detect(self, path):
        from PIL import Image

        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((8, 8)).getdata())
        return {"fake_nsfw": round(sum(pixels) / len(pixels) / 255, 4)}

    def unload(self):
        self.model = None
