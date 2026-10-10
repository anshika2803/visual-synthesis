"""Image generation backends.

A backend turns requests into images::

    backend = build_image_backend(model, models_root)
    backend.load()
    images = backend.generate([{"prompt_id": "p1", "prompt": "a cat", "seed": 42}])
    backend.unload()

A request is a dict with ``prompt_id``, ``prompt`` and, optionally, ``seed``,
``negative_prompt``, ``steps``, ``guidance_scale``, ``width``, ``height`` and
``num_images``. Anything left out (or None) falls back to the model's config.

Each produced image is a dict: ``prompt_id``, ``image`` (a PIL image), ``seed``,
``width``, ``height``, ``steps``, ``guidance_scale`` and ``seconds``.
"""

import logging
import random
import time
from pathlib import Path
from threading import Lock

from visual_synthesis.io import safe_name

log = logging.getLogger(__name__)

# Diffusers uses Accelerate's process-global ``init_empty_weights`` context while
# loading with low_cpu_mem_usage=True. Concurrent from_pretrained calls can leave
# a component on the meta device (notably on MPS), so pipeline construction must
# be serialized. Inference itself remains parallel across backend instances.
_PIPELINE_LOAD_LOCK = Lock()

# models.<id>.family -> diffusers pipeline class
PIPELINES = {
    "auto": "AutoPipelineForText2Image",
    "stable_diffusion": "StableDiffusionPipeline",
    "stable_diffusion_xl": "StableDiffusionXLPipeline",
}


def build_image_backend(model, models_root=None):
    """The backend for a model entry (see ``get_model`` in config.py)."""
    if model["backend"] == "fake":
        return FakeBackend(model)
    if model["backend"] == "diffusers":
        return DiffusersBackend(model, models_root)
    raise KeyError(f"unknown image backend {model['backend']!r}; available: diffusers, fake")


def save_image(image, path, fmt="png", quality=95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        image.save(path, format="PNG")
    else:
        image.convert("RGB").save(path, format="JPEG" if fmt in ("jpeg", "jpg") else "WEBP", quality=quality)
    return path


def with_defaults(request, model):
    """A request with every missing setting filled in from the model's config."""

    def pick(key, model_key=None):
        value = request.get(key)
        return model[model_key or key] if value is None else value

    seed = request.get("seed")
    return {
        "prompt_id": request["prompt_id"],
        "prompt": request["prompt"],
        "negative_prompt": pick("negative_prompt"),
        "seed": random.randrange(2**31) if seed is None else int(seed),
        "steps": pick("steps", "num_inference_steps"),
        "guidance_scale": pick("guidance_scale"),
        "width": pick("width"),
        "height": pick("height"),
        "num_images": pick("num_images", "num_images_per_prompt"),
    }


class DiffusersBackend:
    """Any diffusers text-to-image pipeline: SD-Turbo, SDXL, ..."""

    def __init__(self, model, models_root=None):
        self.model = model
        self.models_root = Path(models_root or "models")
        self.pipe = None

    def weights_source(self):
        """The local weights folder when it was downloaded, else the HuggingFace repo id."""
        local = self.models_root / safe_name(self.model["local_dir"] or self.model["id"])
        if local.is_dir() and any(local.iterdir()):
            return str(local)
        return self.model["repo_id"]

    def load(self):
        if self.pipe is not None:
            return
        try:
            import diffusers
        except ImportError as exc:
            raise RuntimeError("image generation needs the 'gpu' extra: pip install -e '.[gpu]'") from exc
        from visual_synthesis.backends.device import pick_device, pick_dtype

        model = self.model
        device = pick_device(model["device"], model["device_index"])
        dtype = pick_dtype(model["dtype"], device)
        source = self.weights_source()
        log.info("loading model %s from %s (family=%s, dtype=%s)", model["id"], source, model["family"], dtype)

        import transformers

        # Keep job logs readable.
        diffusers.utils.logging.disable_progress_bar()
        transformers.utils.logging.disable_progress_bar()
        # diffusers renamed `torch_dtype` to `dtype`; 0.41 is known to accept the new name.
        newer = tuple(int(part) for part in diffusers.__version__.split(".")[:2]) >= (0, 41)
        options = {"dtype" if newer else "torch_dtype": dtype}
        if model["variant"]:
            options["variant"] = model["variant"]
        if model["revision"]:
            options["revision"] = model["revision"]
        if Path(source).is_dir():
            options["local_files_only"] = True
        # This is a safety benchmark: by default it records what the model
        # produces *before* any filtering.
        if not model["safety_checker"] and model["family"] == "stable_diffusion":
            options["safety_checker"] = None

        started = time.perf_counter()
        # See _PIPELINE_LOAD_LOCK above. Keep the critical section to loading;
        # completed pipelines can still generate concurrently in worker threads.
        with _PIPELINE_LOAD_LOCK:
            pipe = getattr(diffusers, PIPELINES[model["family"]]).from_pretrained(source, **options).to(device)
        if not model["safety_checker"] and getattr(pipe, "safety_checker", None) is not None:
            pipe.safety_checker = None
        if model["enable_attention_slicing"]:
            pipe.enable_attention_slicing()
        if model["enable_xformers"]:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                log.warning("xformers is unavailable, continuing without it: %s", exc)
        if model["scheduler"]:
            pipe.scheduler = getattr(diffusers, model["scheduler"]).from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        log.info("model %s ready on %s in %.1fs", model["id"], device, time.perf_counter() - started)

    def generate(self, requests):
        import torch

        self.load()
        images = []
        for request in requests:
            request = with_defaults(request, self.model)
            call = {
                "prompt": request["prompt"],
                "num_inference_steps": request["steps"],
                "guidance_scale": request["guidance_scale"],
                "width": request["width"],
                "height": request["height"],
                "num_images_per_prompt": request["num_images"],
                "generator": torch.Generator("cpu").manual_seed(request["seed"]),
            }
            if request["negative_prompt"]:
                call["negative_prompt"] = request["negative_prompt"]

            started = time.perf_counter()
            with torch.inference_mode():
                produced = self.pipe(**call).images
            seconds = time.perf_counter() - started
            for index, image in enumerate(produced):
                images.append({
                    "prompt_id": request["prompt_id"],
                    "image": image,
                    "seed": request["seed"] + index,
                    "width": image.width,
                    "height": image.height,
                    "steps": request["steps"],
                    "guidance_scale": request["guidance_scale"],
                    "seconds": seconds,
                })
        return images

    def unload(self):
        if self.pipe is None:
            return
        from visual_synthesis.backends.device import free_memory

        self.pipe = None
        free_memory()
        log.info("model %s unloaded", self.model["id"])


class FakeBackend:
    """Draws a flat colour derived from the seed. No weights, no GPU.

    Lets the whole pipeline (and the tests) run in seconds: set
    ``"backend": "fake"`` on a model entry.
    """

    def __init__(self, model):
        self.model = model
        self.pipe = None

    def load(self):
        self.pipe = "fake"

    def generate(self, requests):
        from PIL import Image

        self.load()
        images = []
        for request in requests:
            request = with_defaults(request, self.model)
            for index in range(request["num_images"]):
                seed = request["seed"] + index
                colour = (seed * 37 % 256, seed * 101 % 256, seed * 17 % 256)
                images.append({
                    "prompt_id": request["prompt_id"],
                    "image": Image.new("RGB", (request["width"], request["height"]), colour),
                    "seed": seed,
                    "width": request["width"],
                    "height": request["height"],
                    "steps": request["steps"],
                    "guidance_scale": request["guidance_scale"],
                    "seconds": 0.0,
                })
        return images

    def unload(self):
        self.pipe = None
