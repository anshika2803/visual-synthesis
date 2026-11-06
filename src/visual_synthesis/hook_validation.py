"""The gate before E4: does a model's hook read and edit the right tensor?

Runs the real pipeline for a few steps on two benign prompts and checks, for
``models.<id>.hook`` (validate SDXL first, then each model in turn):

    capture        every layer returns the expected structure and image stream (shapes are
                   checked on every call), and the conditional pass is reached on every step
    cfg layout     at step 0 the unconditional activations are identical for both prompts (same
                   noise, same empty prompt) while the conditional ones differ
    alpha = 0      the image equals the unhooked baseline
    edit           steering and ablation at alpha = 1 change the activations, and steering the image
    removal        after the hooks are removed the image equals the baseline again

A working capture does not imply a working intervention, so both are checked.
The report goes to results/hook_validation/<model>.json; the exit code is 1 on any failure.

    sbatch scripts/slurm_run.sh scripts/validate_hooks.py -c config/pipeline.json -c config/e1e6.json --model sd-xl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from visual_synthesis.activations import ActivationHooks
from visual_synthesis.backends.image import DiffusersBackend
from visual_synthesis.config import get_model, load_config
from visual_synthesis.io import write_json

PROMPTS = ("a red bicycle leaning against a brick wall", "a bowl of fruit on a wooden kitchen table")


def pixels(backend, prompt):
    image = backend.generate([{"prompt_id": "check", "prompt": prompt, "seed": 42}])[0]["image"]
    return np.asarray(image.convert("RGB"), np.float32)


def hooked(backend, hook, capture=(), intervention=None):
    """Replace the backend's hooks; returns the new ActivationHooks."""
    if backend.hooks:
        backend.hooks.remove()
    backend.hooks = ActivationHooks(backend.pipe, hook, capture, intervention) if capture or intervention else None
    return backend.hooks


def validate(cfg, model_id, steps=2, layer=None):
    model = {**get_model(cfg, model_id), "num_inference_steps": steps}
    hook = model["hook"]
    if not hook["layers"]:
        raise ValueError(f"models.{model_id}.hook.layers is empty")
    backend = DiffusersBackend(model, cfg["paths"]["models"])
    backend.load()
    checks, report = [], {"model": model_id, "steps": steps, "hook": hook, "layers": {}}

    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    baseline = pixels(backend, PROMPTS[0])
    noise = float(np.abs(pixels(backend, PROMPTS[0]) - baseline).mean())
    check("deterministic", noise == 0, f"mean |pixel diff| of two identical runs = {noise:.4f}")
    changed = max(1.0, 10 * noise)  # a real edit must exceed run-to-run noise by far

    probes = []
    for prompt in PROMPTS:
        hooked(backend, hook, hook["layers"])
        pixels(backend, prompt)
        probes.append(backend.last_probe)
    for name in hook["layers"]:
        first, second = probes[0], probes[1]
        cond = [p["capture"][name]["steps"].astype(np.float32) for p in (first, second)]
        info = {"output": first["outputs"][name], "dim": int(cond[0].shape[1]), "cond_steps": len(cond[0]),
                "uncond_steps": len(first["uncond"].get(name, []))}
        report["layers"][name] = info
        check(f"{name}: conditional pass on every step", len(cond[0]) == first["stats"]["steps"],
              f"{len(cond[0])} of {first['stats']['steps']} steps")
        check(f"{name}: prompts differ in the conditional pass", not np.allclose(cond[0], cond[1]))
        if name in first["uncond"]:
            # Only at step 0: afterwards the two prompts' latents have diverged.
            same = np.allclose(first["uncond"][name][0], second["uncond"][name][0], rtol=1e-3, atol=1e-3)
            check(f"{name}: unconditional pass at step 0 is prompt-independent", same, "else hook.cfg is wrong")

    layer = layer or hook["layers"][len(hook["layers"]) // 2]
    mean = probes[0]["capture"][layer]["mean"]
    direction = np.random.default_rng(0).standard_normal(len(mean)).astype(np.float32)
    direction /= np.linalg.norm(direction)
    edit = {"layer": layer, "mode": "steering", "vector": "random:0", "direction": direction, "mu": mean,
            "scale": float(np.linalg.norm(mean)), "passes": "all", "steps": (0.0, 1.0)}
    for mode, alpha in (("steering", 0.0), ("steering", 1.0), ("ablation", 1.0)):
        hooked(backend, hook, intervention={**edit, "mode": mode, "alpha": alpha})
        difference = float(np.abs(pixels(backend, PROMPTS[0]) - baseline).mean())
        stats = backend.last_probe["stats"]
        label = f"{layer}: {mode} alpha={alpha:g}"
        if alpha == 0:
            check(f"{label} leaves the image unchanged", difference <= noise, f"mean |pixel diff| = {difference:.4f}")
        else:
            check(f"{label} edits the activations", stats["edited_calls"] > 0 and stats["relative_change"] > 0,
                  f"{stats['edited_calls']} calls, relative change {stats['relative_change']:.4f}")
            if mode == "steering":
                check(f"{label} changes the image", difference > changed, f"mean |pixel diff| = {difference:.4f}")
    hooked(backend, hook)
    difference = float(np.abs(pixels(backend, PROMPTS[0]) - baseline).mean())
    check("removing the hooks restores the baseline", difference <= noise, f"mean |pixel diff| = {difference:.4f}")
    backend.unload()
    report.update(checks=checks, passed=all(c["passed"] for c in checks), intervention_layer=layer)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", action="append", default=[])
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--layer", help="the layer for the intervention checks (default: the middle one)")
    parser.add_argument("--output", type=Path, default=Path("results/hook_validation"))
    args = parser.parse_args(argv)
    report = validate(load_config(args.config or None), args.model, args.steps, args.layer)
    write_json(report, args.output / f"{args.model}.json")
    for item in report["checks"]:
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['check']}  {item['detail']}")
    print(json.dumps({name: info["output"] for name, info in report["layers"].items()}, indent=2))
    return 0 if report["passed"] else 1
