"""Activation capture and intervention inside a diffusers pipeline (experiments E4-E6).

One mechanism serves every model; what differs per model is ``models.<id>.hook``:

    layers            module paths to hook, e.g. "unet.mid_block.attentions.0"
    output_index      the element of a tuple output that is the image stream
    cfg               where the conditional pass is inside the denoiser calls of one step:
                        batch_uncond_first  one call, batch [uncond, cond]  (SD 1.5, SDXL, SD 3.5, PixArt)
                        batch_cond_first    one call, batch [cond, uncond]  (Z-Image)
                        sequential          a cond call, then an uncond call (FLUX, Qwen-Image, GLM-Image)
                      A batch of one, or one call per step, means no CFG: every row is conditional.
    pixels_per_token  DiT streams are (B, tokens, D) with tokens = (H / p) * (W / p); None for UNet maps
    joint             image_first | image_last when the stream also holds the text tokens (Z-Image)

Every call checks the stream against the image size, so a hook pointed at the
text stream raises instead of silently editing the wrong tokens.

Capture: per layer and denoising step, the mean image-token (or spatial) vector
of the conditional pass, taken before any edit. Intervention, on one layer and
every image token (decision D9 of docs/e1_e6_research_plan.md):

    ablation   h' = h - alpha * ((h - mu_benign) . d) d
    steering   h' = h - alpha * scale * d
"""

import json
from functools import partial
from pathlib import Path

import numpy as np

CFG_LAYOUTS = ("batch_uncond_first", "batch_cond_first", "sequential")
JOINT_LAYOUTS = ("image_first", "image_last")
MODES = ("ablation", "steering")
PASSES = ("all", "cond")


def ablate(h, mu, d, alpha):
    """Move ``h``'s component along unit ``d`` towards ``mu``'s. Features are the last axis."""
    return h - alpha * ((h - mu) @ d)[..., None] * d


def steer(h, d, scale, alpha):
    """Shift ``h`` by ``-alpha * scale`` along unit ``d``."""
    return h - alpha * scale * d


def resolve(root, path):
    """``resolve(pipe, "unet.down_blocks.1")`` is ``pipe.unet.down_blocks[1]``."""
    for part in path.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    return root


def describe(value):
    """``(Tensor(2, 4096, 1536), None)``: the structure of a module output, for validation reports."""
    if isinstance(value, tuple):
        return "(" + ", ".join(describe(item) for item in value) + ")"
    if hasattr(value, "shape"):
        return f"Tensor{tuple(value.shape)}"
    return type(value).__name__


def load_intervention(settings):
    """``generate.intervention`` plus the E4 arrays it points at, in the form ActivationHooks takes."""
    e4_dir = Path(settings["e4_dir"])
    layer = settings["layer"] or json.loads((e4_dir / "statistics.json").read_text())["layer"]
    arrays = np.load(e4_dir / "directions.npz")
    vector = settings["vector"]
    key = {"safety": "mean_diff", "probe": "probe"}.get(vector, f"random_{vector.partition(':')[2]}")
    if f"{layer}/{key}" not in arrays.files:
        layers = sorted({name.rpartition("/")[0] for name in arrays.files})
        raise KeyError(f"{e4_dir / 'directions.npz'} has no {layer}/{key}; layers: {layers}")
    direction, mu = arrays[f"{layer}/{key}"], arrays[f"{layer}/mu_benign"]
    delta = arrays[f"{layer}/mu_harmful"] - mu
    # Steering moves the harmful mean's projection onto the benign mean's. A random
    # direction has no such projection, so it gets the same norm |delta| instead.
    scale = float(np.linalg.norm(delta)) if vector.startswith("random") else float(delta @ direction)
    return {
        "layer": layer, "mode": settings["mode"], "vector": vector, "alpha": float(settings["alpha"]),
        "direction": direction.astype(np.float32), "mu": mu.astype(np.float32), "scale": scale,
        "passes": settings["passes"], "steps": tuple(settings["steps"]),
    }


class ActivationHooks:
    """Forward hooks that capture and/or edit the hooked layers of a pipeline's denoiser.

        hooks = ActivationHooks(pipe, model["hook"], capture_layers, intervention)
        hooks.begin(width, height, steps)    # before each pipeline call
        pipe(...)
        result = hooks.end()                 # capture, uncond, stats, outputs
        hooks.remove()                       # the pipeline behaves as before
    """

    def __init__(self, root, hook, capture_layers=(), intervention=None, images=1):
        self.hook, self.intervention, self.images = hook, intervention, images
        self.capture_layers = list(capture_layers)
        layers = list(dict.fromkeys(self.capture_layers + ([intervention["layer"]] if intervention else [])))
        if not layers:
            raise ValueError("nothing to hook: no capture layers and no intervention")
        denoiser = resolve(root, layers[0].split(".")[0])
        self.handles = [denoiser.register_forward_pre_hook(self.on_call, with_kwargs=True)]
        self.handles += [resolve(root, layer).register_forward_hook(partial(self.on_output, layer)) for layer in layers]
        self.vectors = {}
        self.begin()

    def begin(self, width=None, height=None, steps=None):
        """Reset the per-image state; the image size fixes how many image tokens a DiT stream has."""
        pixels = self.hook["pixels_per_token"]
        self.tokens = (height // pixels) * (width // pixels) if pixels and width and height else None
        self.steps = steps
        self.step, self.call, self.last_t = -1, 0, None
        self.buffers = {layer: {"cond": [], "uncond": []} for layer in self.capture_layers}
        self.stats = {"edited": 0, "change": 0.0}
        self.outputs = {}

    def on_call(self, module, args, kwargs):
        """Track the step. Batch layouts: every denoiser call is one step (SD 1.5's PNDM
        scheduler repeats a timestep, so timesteps cannot be trusted there). Sequential:
        a new timestep starts a step, a repeated one is that step's uncond call."""
        if self.hook["cfg"] != "sequential":
            self.step, self.call = self.step + 1, 0
            return
        t = kwargs.get("timestep", kwargs.get("t", args[1] if len(args) > 1 else None))
        if t is None:
            raise RuntimeError("hook.cfg=sequential needs the denoiser's timestep argument, but the call has none")
        value = None if t is None else float(t.flatten()[0]) if hasattr(t, "flatten") else float(t)
        if value is None or value != self.last_t:
            self.step, self.call = self.step + 1, 0
        else:
            self.call += 1
        self.last_t = value

    def rows(self, batch):
        """``(cond, uncond)``: the batch rows of the current call that belong to each pass."""
        everything, nothing = slice(None), slice(0, 0)
        if self.hook["cfg"] == "sequential":
            if batch != self.images:
                raise RuntimeError(f"hook.cfg=sequential, but a call has {batch} rows: this pipeline batches CFG")
            return (everything, nothing) if self.call == 0 else (nothing, everything)
        if batch == self.images:
            return everything, nothing
        if batch != 2 * self.images:
            raise RuntimeError(f"a batch of {batch} rows is neither {self.images} (no CFG) nor {2 * self.images} (CFG)")
        first, second = slice(0, self.images), slice(self.images, batch)
        return (second, first) if self.hook["cfg"] == "batch_uncond_first" else (first, second)

    def region(self, layer, stream):
        """A view of the image part of ``stream`` with the features last: (B, tokens, D) or (B, H, W, C)."""
        if stream.dim() == 4:
            return stream.movedim(1, -1)
        if stream.dim() != 3:
            raise RuntimeError(f"{layer}: expected a (B, C, H, W) map or a (B, tokens, D) stream, "
                               f"got {describe(stream)}")
        if self.tokens is None:
            raise RuntimeError(f"{layer}: a token stream needs hook.pixels_per_token and the image size")
        found, joint = stream.shape[1], self.hook["joint"]
        if found == self.tokens:
            return stream
        if found > self.tokens and joint == "image_first":
            return stream[:, : self.tokens]
        if found > self.tokens and joint == "image_last":
            return stream[:, -self.tokens :]
        raise RuntimeError(f"{layer}: the stream has {found} tokens but the image has {self.tokens}; "
                           "check hook.output_index and hook.joint")

    def vector(self, name, device):
        if (name, device) not in self.vectors:
            import torch

            self.vectors[name, device] = torch.from_numpy(self.intervention[name]).to(device)
        return self.vectors[name, device]

    def on_output(self, layer, module, args, output):
        import torch

        self.outputs.setdefault(layer, describe(output))
        index = self.hook["output_index"]
        if isinstance(output, tuple):
            if index is None:
                raise RuntimeError(f"{layer} returns {describe(output)}; set hook.output_index")
            index %= len(output)
            stream = output[index]
        else:
            stream = output
        if not torch.is_tensor(stream):
            raise RuntimeError(f"{layer}: output element {index} is {describe(stream)}, not a tensor")
        image = self.region(layer, stream)
        cond, uncond = self.rows(stream.shape[0])

        if layer in self.buffers:
            for name, rows in (("cond", cond), ("uncond", uncond)):
                selected = image[rows]
                if len(selected):
                    self.buffers[layer][name].append(selected.float().flatten(1, -2).mean(dim=(0, 1)).cpu().numpy())

        settings = self.intervention
        if not settings or layer != settings["layer"]:
            return None
        start, end = settings["steps"]
        # end = 1 is inclusive: PNDM makes one call more than num_inference_steps.
        fraction = self.step / self.steps if self.steps else 0.0
        if not (start <= fraction and (fraction < end or end >= 1)):
            return None
        rows = slice(None) if settings["passes"] == "all" else cond
        h = image[rows].float()
        if not h.numel():
            return None
        d, mu = self.vector("direction", h.device), self.vector("mu", h.device)
        if settings["mode"] == "ablation":
            edited = ablate(h, mu, d, settings["alpha"])
        else:
            edited = steer(h, d, settings["scale"], settings["alpha"])
        stored = edited.to(stream.dtype)
        self.stats["edited"] += 1
        # Measured after rounding to the model's dtype: what the next block really receives.
        self.stats["change"] += float((stored.float() - h).norm() / h.norm().clamp_min(1e-12))
        # Edit a copy: the block's own output (and anything holding a reference to it) stays untouched.
        new = stream.clone()
        self.region(layer, new)[rows] = stored
        return output[:index] + (new,) + output[index + 1 :] if isinstance(output, tuple) else new

    def end(self):
        """What one pipeline call produced; raises when a hook never did its job."""
        missing = [layer for layer, buffer in self.buffers.items() if not buffer["cond"]]
        if missing:
            raise RuntimeError(f"no conditional pass reached {missing}; check hook.cfg")
        settings, edited = self.intervention, self.stats["edited"]
        if settings and settings["alpha"] and not edited:
            raise RuntimeError(f"the intervention never edited {settings['layer']}; check the layer and hook.cfg")
        capture = {}
        for layer, buffer in self.buffers.items():
            steps = np.stack(buffer["cond"])
            capture[layer] = {"steps": steps.astype(np.float16), "mean": steps.mean(axis=0).astype(np.float32)}
        return {
            "capture": capture,
            "uncond": {layer: np.stack(b["uncond"]) for layer, b in self.buffers.items() if b["uncond"]},
            "stats": {"steps": self.step + 1, "edited_calls": edited,
                      "relative_change": self.stats["change"] / edited if edited else 0.0},
            "outputs": dict(self.outputs),
        }

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
