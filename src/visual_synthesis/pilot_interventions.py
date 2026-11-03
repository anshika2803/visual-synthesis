"""Pooled activation hooks for resource pilots, not a validated safety intervention."""

from dataclasses import dataclass


@dataclass
class HookPoint:
    name: str
    module: object
    output_index: int | None
    cfg_layout: str
    image_tokens: int | None = None


def hook_points(pipe, model):
    """Adapters inspected against Diffusers 0.41; runtime shapes are checked too.

    GLM deliberately profiles the DiT decoder, not its autoregressive generator.
    Z-Image basic text-to-image uses [image, text] tokens; exclude the text tail.
    """
    family = model["family"]
    if family in ("stable_diffusion", "stable_diffusion_xl"):
        points = []
        for name in ("down_blocks", "mid_block", "up_blocks"):
            value = getattr(pipe.unet, name)
            blocks = list(value) if name != "mid_block" else [value]
            for i, block in enumerate(blocks):
                path = f"unet.{name}" if name == "mid_block" else f"unet.{name}.{i}"
                points.append(HookPoint(path, block, 0 if name == "down_blocks" else None, "batch"))
        indices = [1, 2, 4, 6, 7] if family == "stable_diffusion" else [1, 2, 3, 4, 5]
        return [points[i] for i in indices]
    if family == "z_image":
        # Only the configured 2-D text-to-image path; no omni/reference images.
        tokens = (model["height"] // (pipe.vae_scale_factor * 2)) * (model["width"] // (pipe.vae_scale_factor * 2))
        blocks = [
            HookPoint(f"transformer.layers.{i}", block, None, "none", tokens)
            for i, block in enumerate(pipe.transformer.layers)
        ]
    else:
        layout = "sequential" if family in ("qwen_image", "glm_image") else ("none" if family == "flux" else "batch")
        output_index = {"qwen_image": 1, "flux": 1, "glm_image": 0, "stable_diffusion_3": 1, "pixart_sigma": None}[
            family
        ]
        names = ["transformer_blocks", "single_transformer_blocks"] if family == "flux" else ["transformer_blocks"]
        blocks = [
            HookPoint(f"transformer.{name}.{i}", block, output_index, layout)
            for name in names
            for i, block in enumerate(getattr(pipe.transformer, name))
        ]
    indices = sorted({round(fraction * (len(blocks) - 1)) for fraction in (0, 0.25, 0.5, 0.75, 1)})
    if len(indices) != 5:
        raise ValueError("resource capture expects five distinct candidate blocks")
    return [blocks[i] for i in indices]


class ActivationHooks:
    """Immediately pool activations to CPU; never retain full device outputs.

    Capture uses the conditional half/pass. Interventions affect all CFG passes.
    Centered ablation computes h - alpha * ((h-mu_b).d) * d, using a small
    per-token projection and one fused addcmul instead of a full fp32 copy.
    """

    def __init__(self, points, mode="capture", direction=None, center=None, scale=1.0, alpha=1.0):
        if mode not in ("capture", "ablation", "steering"):
            raise ValueError("unknown hook mode")
        self.points, self.mode = points, mode
        self.direction, self.center, self.scale, self.alpha = direction, center, scale, alpha
        self.handles = []
        self.reset()

    def reset(self):
        self.calls = {p.name: 0 for p in self.points}
        self.counts, self.sums, self.shapes = {}, {}, {}
        self.device_vectors = {}

    def _hook(self, point):
        def apply(module, args, output):
            import torch

            del module, args
            hidden = output if point.output_index is None else output[point.output_index]
            if not torch.is_tensor(hidden) or hidden.ndim not in (3, 4):
                raise ValueError(f"{point.name}: expected BND/BCHW image stream")
            if point.image_tokens is not None:
                if hidden.ndim != 3 or hidden.shape[1] < point.image_tokens:
                    raise ValueError("Z-Image image-token prefix does not match runtime shape")
                hidden = hidden[:, : point.image_tokens]
            self.shapes[point.name] = list(hidden.shape)
            call = self.calls[point.name]
            self.calls[point.name] += 1
            if self.mode == "capture":
                if point.cfg_layout == "sequential" and call % 2:
                    return output
                if point.cfg_layout == "batch":
                    if hidden.shape[0] != 2:
                        raise ValueError("expected batch-one CFG with [unconditional, conditional] layout")
                    hidden = hidden[1:]
                elif hidden.shape[0] != 1:
                    raise ValueError("pilot expects exactly one image per prompt")
                axes = (0, 2, 3) if hidden.ndim == 4 else (0, 1)
                pooled = hidden.detach().mean(dim=axes).float().cpu()
                if not torch.isfinite(pooled).all():
                    raise ValueError("nonfinite pooled activations")
                self.sums[point.name] = self.sums.get(point.name, 0) + pooled
                self.counts[point.name] = self.counts.get(point.name, 0) + 1
                return output
            if self.alpha == 0:
                return output  # Exact identity, no rounding or allocation.
            channels_last = hidden.movedim(1, -1) if hidden.ndim == 4 else hidden
            key = (str(hidden.device), hidden.dtype)
            if key not in self.device_vectors:
                d = self.direction.to(device=hidden.device, dtype=hidden.dtype)
                center = self.center.to(device=hidden.device, dtype=hidden.dtype)
                if d.ndim != 1 or len(d) != channels_last.shape[-1]:
                    raise ValueError("direction does not match selected image channels")
                self.device_vectors[key] = (d, (center * d).sum())
            d, offset = self.device_vectors[key]
            projection = (
                (channels_last @ d - offset).unsqueeze(-1)
                if self.mode == "ablation"
                else (torch.as_tensor(self.scale, device=hidden.device, dtype=hidden.dtype))
            )
            changed = torch.addcmul(channels_last, projection, d, value=-float(self.alpha))
            changed = changed.movedim(-1, 1) if hidden.ndim == 4 else changed
            if point.image_tokens is not None:
                # Preserve joint text tokens; only replace the image prefix.
                original = output if point.output_index is None else output[point.output_index]
                changed = torch.cat((changed, original[:, point.image_tokens :]), dim=1)
            if point.output_index is None:
                return changed
            parts = list(output)
            parts[point.output_index] = changed
            return tuple(parts) if isinstance(output, tuple) else parts

        return apply

    def features(self):
        if any(not self.calls[p.name] for p in self.points):
            raise ValueError("a selected block was never called")
        for point in self.points:
            if point.cfg_layout == "sequential" and self.calls[point.name] % 2:
                raise ValueError("unexpected sequential CFG call count")
        return {p.name: self.sums[p.name] / self.counts[p.name] for p in self.points}

    def __enter__(self):
        for point in self.points:
            self.handles.append(point.module.register_forward_hook(self._hook(point)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.device_vectors.clear()


def fit_direction(features, labels):
    import torch

    x = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    y = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("discovery needs benign and sexual-harmful examples")
    center = x[y == 0].mean(dim=0)
    delta = x[y == 1].mean(dim=0) - center
    scale = delta.norm()
    if not torch.isfinite(scale) or scale < 1e-8:
        raise ValueError("discovery direction is zero/nonfinite")
    return center, delta / scale, float(scale)


def fit_probe(features, labels, iterations=50):
    """A small CPU logistic probe; its observed fit time is not a scaling law."""
    import torch

    x = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    y = torch.as_tensor(labels, dtype=torch.float32, device="cpu")
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5)
    x = (x - mean) / std
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.02)
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(x @ weight + bias, y) + 0.01 * weight.square().mean()
        )
        loss.backward()
        optimizer.step()
    return {"weight": weight.detach(), "bias": bias.detach(), "mean": mean, "std": std}
