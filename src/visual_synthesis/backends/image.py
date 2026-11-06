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
``width``, ``height``, ``steps``, ``guidance_scale`` and ``seconds``. With
``hooks`` (experiments E4-E6, see activations.py) it also has ``activations``
(captured layers) and ``intervention`` (what the edit did).
"""

import hashlib
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
    "stable_diffusion_3": "StableDiffusion3Pipeline",
    "pixart_sigma": "PixArtSigmaPipeline",
    "flux": "FluxPipeline",
    "qwen_image": "QwenImagePipeline",
    "z_image": "ZImagePipeline",
    "glm_image": "GlmImagePipeline",
}

# Trained in bfloat16; in float16 their activations overflow into black or NaN images.
BFLOAT16_FAMILIES = {"flux", "qwen_image", "z_image", "glm_image"}

# models.<id>.quantization: how the weights of models.<id>.quantize_components are stored.
QUANTIZATION = {
    "fp8": "float8 weight-only (torchao)",
    "int8": "8-bit LLM.int8 (bitsandbytes)",
    "nf4": "4-bit NormalFloat (bitsandbytes)",
    "fp4": "4-bit float (bitsandbytes)",
}


def build_image_backend(model, models_root=None, hooks=None):
    """The backend for a model entry (see ``get_model`` in config.py).

    ``hooks``: ``{"capture_layers": [...], "intervention": {...} or None}`` (see s05_generate.hook_settings).
    """
    if model["backend"] == "fake":
        return FakeBackend(model, hooks)
    if model["backend"] == "diffusers":
        return DiffusersBackend(model, models_root, hooks)
    raise KeyError(f"unknown image backend {model['backend']!r}; available: diffusers, fake")


def precision(model):
    """How the weights are stored, e.g. ``bfloat16`` or ``bfloat16+fp8``; recorded with every image."""
    dtype = model["dtype"] or "auto"
    return f"{dtype}+{model['quantization']}" if model["quantization"] else dtype


def quantization_config(name, components, dtype):
    """A diffusers PipelineQuantizationConfig storing ``components`` in ``name`` precision."""
    import diffusers
    import transformers

    def config(library):
        if name == "fp8":
            from torchao.quantization import Float8WeightOnlyConfig

            return library.TorchAoConfig(Float8WeightOnlyConfig())
        if name == "int8":
            return library.BitsAndBytesConfig(load_in_8bit=True)
        return library.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=name, bnb_4bit_compute_dtype=dtype)

    # Text encoders are transformers models; the denoiser is a diffusers model.
    text = ("text_encoder", "vision_language_encoder")
    mapping = {part: config(transformers if part.startswith(text) else diffusers) for part in components}
    return diffusers.PipelineQuantizationConfig(quant_mapping=mapping)


def save_image(image, path, fmt="png", quality=95, compress_level=6):
    """``compress_level`` (0-9) only trades PNG file size for speed: the pixels are the same."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        image.save(path, format="PNG", compress_level=compress_level)
    else:
        image.convert("RGB").save(path, format="JPEG" if fmt in ("jpeg", "jpg") else "WEBP", quality=quality)
    return path


def batch_groups(requests, hooks=False):
    """Consecutive requests that can share one pipeline call.

    Hooks capture and edit per image, so with hooks every request is its own call;
    so is a request for several images. Others group while their settings match.
    """
    groups = []
    for request in requests:
        key = tuple(request[k] for k in ("negative_prompt", "steps", "guidance_scale", "width", "height"))
        joinable = not hooks and request["num_images"] == 1
        if groups and joinable and groups[-1][0] == key and groups[-1][1]:
            groups[-1][2].append(request)
        else:
            groups.append((key, joinable, [request]))
    return [group for _, _, group in groups]


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

    def __init__(self, model, models_root=None, hooks=None):
        self.model = model
        self.models_root = Path(models_root or "models")
        self.hook_settings = hooks
        self.hooks = None
        self.last_probe = None  # the hooks' full result for the latest image (hook_validation.py)
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
        if model["family"] in BFLOAT16_FAMILIES and str(dtype) == "torch.float16":
            log.warning("%s is a bfloat16 model; float16 often gives black or NaN images", model["id"])
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
        if model["quantization"]:
            options["quantization_config"] = quantization_config(
                model["quantization"], model["quantize_components"], dtype)
        # This is a safety benchmark: by default it records what the model
        # produces *before* any filtering.
        if not model["safety_checker"] and model["family"] == "stable_diffusion":
            options["safety_checker"] = None

        started = time.perf_counter()
        # See _PIPELINE_LOAD_LOCK above. Keep the critical section to loading;
        # completed pipelines can still generate concurrently in worker threads.
        with _PIPELINE_LOAD_LOCK:
            pipeline_class = getattr(diffusers, PIPELINES[model["family"]], None)
            if pipeline_class is None:
                raise RuntimeError(f"diffusers {diffusers.__version__} has no {PIPELINES[model['family']]}; "
                                   "upgrade diffusers and transformers for this model")
            if model["device_map"]:
                # Weights go shard by shard straight to the GPU: host RAM stays small (TC2: 30 GB per user).
                options["device_map"] = model["device_map"]
            pipe = pipeline_class.from_pretrained(source, **options)
            if model["device_map"]:
                log.info("weights loaded with device_map=%s", model["device_map"])
            elif model.get("enable_cpu_offload"):
                import torch

                pipe.enable_model_cpu_offload(device=torch.device(device))
            else:
                try:
                    pipe = pipe.to(device)
                except Exception as exc:
                    log.warning("could not move pipeline to %s directly (%s); enabling model cpu offload", device, exc)
                    import torch

                    pipe.enable_model_cpu_offload(device=torch.device(device))
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
        if self.hook_settings:
            from visual_synthesis.activations import ActivationHooks

            self.hooks = ActivationHooks(pipe, model["hook"], **self.hook_settings)
        self.pipe = pipe
        log.info("model %s ready on %s in %.1fs", model["id"], device, time.perf_counter() - started)

    def generate(self, requests):
        """One pipeline call per group of compatible requests (see ``batch_groups``)."""
        self.load()
        requests = [with_defaults(request, self.model) for request in requests]
        images = []
        for group in batch_groups(requests, hooks=self.hooks is not None):
            if len(group) == 1:
                images += self.call(group)
                continue
            try:
                images += self.call(group)
            except Exception as exc:  # e.g. a pipeline without list prompts, or out of memory
                log.warning("a batch of %d failed (%s: %s); generating them one at a time",
                            len(group), type(exc).__name__, str(exc)[:200])
                self.free_cache()
                for request in group:
                    images += self.call([request])
        return images

    def call(self, group):
        """Generate ``group`` (requests with identical settings) in one pipeline call."""
        import torch

        first = group[0]
        batched = len(group) > 1
        call = {
            "prompt": [r["prompt"] for r in group] if batched else first["prompt"],
            "num_inference_steps": first["steps"],
            "guidance_scale": first["guidance_scale"],
            "width": first["width"],
            "height": first["height"],
            "num_images_per_prompt": first["num_images"],
            # One generator per prompt, seeded as if it ran alone: the same starting noise.
            "generator": ([torch.Generator("cpu").manual_seed(r["seed"]) for r in group] if batched
                          else torch.Generator("cpu").manual_seed(first["seed"])),
        }
        if first["negative_prompt"]:
            call["negative_prompt"] = [first["negative_prompt"]] * len(group) if batched else first["negative_prompt"]
        extras = self.model["generation_kwargs"]
        if set(call) & set(extras):
            raise ValueError(f"generation_kwargs cannot replace request fields: {sorted(set(call) & set(extras))}")
        call.update(extras)
        if self.hooks:
            if first["num_images"] != 1:
                raise ValueError("activation capture and intervention need num_images_per_prompt=1")
            self.hooks.begin(first["width"], first["height"], first["steps"])

        started = time.perf_counter()
        with torch.inference_mode():
            produced = self.pipe(**call).images
        seconds = (time.perf_counter() - started) / len(group)
        probe = self.last_probe = self.hooks.end() if self.hooks else None
        images = []
        per_request = len(produced) // len(group)
        for position, request in enumerate(group):
            for index, image in enumerate(produced[position * per_request:(position + 1) * per_request]):
                images.append({
                    "prompt_id": request["prompt_id"],
                    "image": image,
                    "seed": request["seed"] + index,
                    "width": image.width,
                    "height": image.height,
                    "steps": request["steps"],
                    "guidance_scale": request["guidance_scale"],
                    "seconds": seconds,
                    "batch_size": len(group),
                    **({"activations": probe["capture"], "intervention": probe["stats"]} if probe else {}),
                })
        return images

    def free_cache(self):
        from visual_synthesis.backends.device import free_memory

        free_memory()

    def unload(self):
        if self.pipe is None:
            return
        from visual_synthesis.backends.device import free_memory

        if self.hooks:
            self.hooks.remove()
            self.hooks = None
        self.pipe = None
        free_memory()
        log.info("model %s unloaded", self.model["id"])


class FakeBackend:
    """Draws a flat colour derived from the seed. No weights, no GPU.

    Lets the whole pipeline (and the tests) run in seconds: set
    ``"backend": "fake"`` on a model entry. With hooks, its "activations" are a
    hashed bag of the prompt's words, so E4's analysis can run end to end too.
    """

    def __init__(self, model, hooks=None):
        self.model = model
        self.hooks = hooks
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
                    **(self.fake_hooks(request) if self.hooks else {}),
                })
        return images

    def fake_hooks(self, request, dim=16):
        import numpy as np

        words = np.zeros(dim, np.float32)
        for word in request["prompt"].lower().split():
            words[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1
        steps = np.stack([words * (1 + step / request["steps"]) for step in range(request["steps"])])
        capture = {layer: {"steps": steps.astype(np.float16), "mean": steps.mean(axis=0)}
                   for layer in self.hooks["capture_layers"]}
        intervention = self.hooks["intervention"]
        alpha = intervention["alpha"] if intervention else 0.0
        stats = {"steps": request["steps"], "edited_calls": request["steps"] if alpha else 0, "relative_change": alpha}
        return {"activations": capture, "intervention": stats}

    def unload(self):
        self.pipe = None
