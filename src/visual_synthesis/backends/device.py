"""Choosing the compute device and weight dtype.

The same config has to work on a macOS laptop (Metal, no half precision), a
CUDA node, and a CPU-only machine.
"""

import logging

log = logging.getLogger(__name__)


def pick_device(preference="auto", index=None):
    """Return a torch device string: ``cuda[:n]``, ``mps`` or ``cpu``.

    ``auto`` tries CUDA, then Metal, then CPU; a specific preference raises if
    that device is not available.
    """
    import torch

    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if preference == "cuda" and not cuda:
        raise RuntimeError("device 'cuda' was requested but CUDA is not available")
    if preference == "mps" and not mps:
        raise RuntimeError("device 'mps' was requested but Metal is not available")

    if preference == "cuda" or (preference == "auto" and cuda):
        device = f"cuda:{index or 0}"
        free, total = torch.cuda.mem_get_info(index or 0)
        log.info("device %s (%s): %.1f of %.1f GB free",
                 device, torch.cuda.get_device_name(index or 0), free / 2**30, total / 2**30)
        return device
    device = "mps" if preference == "mps" or (preference == "auto" and mps) else "cpu"
    log.info("device %s", device)
    return device


def pick_dtype(requested, device):
    """The torch dtype to load weights in.

    Half precision only pays off on CUDA: Metal does not support it reliably
    and on CPU it is slower, so both fall back to float32.
    """
    import torch

    if not device.startswith("cuda"):
        if requested not in (None, "float32"):
            log.warning("dtype %s is not usable on %s; using float32", requested, device)
        return torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(requested, torch.float16)


def free_memory():
    """Release cached GPU memory after a model is unloaded."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
