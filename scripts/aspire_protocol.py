#!/usr/bin/env python3
"""Reusable ASPIRE execution adapter for main's E1–E7 model protocol.

The login-node submit path imports no ML libraries. Heavy preparation and all
pipeline commands run inside PBS. A successful readiness job is required before
production. Production checkpoints between commands and can submit a bounded
afterok continuation before its walltime expires. Errors never auto-continue.
"""

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE_COMMIT = "16671d1da7b927e01c72176ddaebac4bef41e956"
GIB = 1024**3
PHASES = ("check", "ready", "all", "e1", "e7", "e4", "e5", "e6", "report")
CONFIG_NAMES = ("pipeline.json", "pipeline.hpc.json", "e1e6.json", "e1e6.aspire.json")
REDUCED_MODELS = {"sd-3.5-large", "flux-dev", "qwen-image", "glm-image"}
FULL_MODELS = {"sd-1.5", "sd-xl", "pixart-sigma", "z-image-turbo"}
TABLES = {
    "core_v1": (874, "bf1457da64d4b498bced4c3be3dbf614b88ddd919d7af40cb1e20ad39d83f839"),
    "e46_v1": (392, "1f5438fc278e1b618b320e66bda16fd0cd3a080d51ecab376dbe8bf6e7c19a33"),
    "core_v1_reduced": (438, "558dcce8c2585ea0f1e2329681adac1a89b64c7693dcdeb40009c82199c3fe4d"),
    "e46_v1_reduced": (198, "82fec053db40d63d4e4d2afc0dbf9047fa76acdcd204051b0999641cbc8ac370"),
    "e7_v1": (282, "d0f0ce7b2c033eb27a0aa0224a7f571e468f00ae31d091717002a6be21ccd912"),
}
EXCLUSIONS_HASH = "2d24b18e78f16dab1f711cc2716415b17c3b173a4633da6b25c524ea22d682f1"
PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "nudenet",
    "onnxruntime",
    "Pillow",
    "huggingface-hub",
    "torchmetrics",
    "safetensors",
    "tokenizers",
    "scipy",
    "torchvision",
    "sentencepiece",
    "protobuf",
)
TC2_REQUIRED = ("torch", "diffusers", "transformers", "nudenet", "torchmetrics")
JOB_ID = re.compile(r"\d+(?:\.[A-Za-z0-9_.-]+)?")


class Checkpoint(Exception):
    """The command used this job's time budget; its incremental outputs are retained."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def storage_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("VS_ASPIRE_STORAGE must be absolute")
    path = path.resolve()
    if path == Path(path.anchor) or path.is_relative_to(Path.home().resolve()):
        raise ValueError("Use resolved scratch/project storage outside the quota-limited home")
    return path


def layout(storage, model, tier="auto"):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", model):
        raise ValueError("Model ID must be a filesystem-safe configured name")
    frozen_tier = "reduced" if model in REDUCED_MODELS else "full"
    if tier == "auto":
        tier = frozen_tier
    if tier not in ("full", "reduced"):
        raise ValueError("Tier must be auto, full or reduced")
    if model in FULL_MODELS | REDUCED_MODELS and tier != frozen_tier:
        raise ValueError(f"The frozen protocol assigns {model} to the {frozen_tier} tier")
    suffix = "_reduced" if tier == "reduced" else ""
    work = storage / "protocol-v1" / f"{model}-{tier}"
    return {
        "model": model,
        "tier": tier,
        "core_table": "core_v1" + suffix,
        "e46_table": "e46_v1" + suffix,
        "storage": storage,
        "work": work,
        "models": storage / "models",
        "runs": work / "runs",
        "logs": work / "logs",
        "config": work / "resolved-paths.json",
        "e4": work / "results/e4" / model,
        "report": work / "results/e1e6" / model,
    }


def configs(paths):
    return [*(ROOT / "config" / name for name in CONFIG_NAMES), paths["config"]]


def config_args(paths):
    return [part for path in configs(paths) for part in ("-c", str(path))]


def verify_model_name(paths):
    # Keep submission/dry-run independent of pandas/torch on the login node.
    models = {}
    for name in CONFIG_NAMES:
        models.update(read(ROOT / "config" / name).get("models", {}))
    if paths["model"] not in models:
        raise ValueError(f"Unknown model {paths['model']}; configure it in config/e1e6.aspire.json first")


def verify_inputs(paths):
    """Check frozen bytes before importing pandas or scheduling a GPU job."""
    for name, (_, digest) in TABLES.items():
        if name not in (paths["core_table"], paths["e46_table"], "e7_v1"):
            continue
        path = ROOT / "data/processed/e1e6" / f"{name}.parquet"
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Frozen input mismatch: {path}. Run git lfs pull for data/processed/e1e6/*")
    if sha(ROOT / "results/review/excluded_prompts.csv") != EXCLUSIONS_HASH:
        raise ValueError("The protocol exclusion list changed")


def verify_code():
    git = os.environ.get("VS_ASPIRE_GIT", "git")
    subprocess.run([git, "-C", str(ROOT), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], check=True)
    # Snapshot actual source/config at readiness. A later main update or config
    # edit cannot silently alter an in-progress model's experiment.


def selected_model(paths):
    from visual_synthesis.config import get_model, load_config

    return get_model(load_config(configs(paths), use_env=False), paths["model"])


def identity(paths):
    files = [
        *ROOT.glob("src/visual_synthesis/**/*.py"),
        *(ROOT / "config" / n for n in CONFIG_NAMES),
        ROOT / "scripts/aspire_protocol.py",
        ROOT / "scripts/aspire_protocol.pbs",
        ROOT / "scripts/validate_hooks.py",
        ROOT / "scripts/e4_analyze.py",
        ROOT / "scripts/e1e6_report.py",
        paths["config"],
    ]
    try:
        packages = {p: importlib.metadata.version(p) for p in PACKAGES}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Missing protocol dependency: {exc.name}; install it in the selected environment") from exc
    return {
        "files": {str(p): sha(p) for p in sorted(files)},
        "packages": packages,
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "tables": {n: digest for n, (_, digest) in TABLES.items()},
        "model": paths["model"],
        "tier": paths["tier"],
        "execution": "ASPIRE / configured native precision and offload; no automatic quantization fallback",
    }


def reference_versions():
    """Normalize pip distribution names, preserving exact versions/CUDA build suffixes."""
    versions = {}
    git = os.environ.get("VS_ASPIRE_GIT", "git")
    text = subprocess.check_output(
        [git, "-C", str(ROOT), "show", f"{BASE_COMMIT}:results/environment/tc2-visual-safety-pip-freeze.txt"], text=True
    )
    for line in text.splitlines():
        if "==" in line and not line.startswith(("#", "-")):
            name, version = line.split("==", 1)
            versions[re.sub(r"[-_.]+", "-", name).lower()] = version
    return versions


def frozen_reference_config():
    from visual_synthesis.config import load_config

    git = os.environ.get("VS_ASPIRE_GIT", "git")
    with tempfile.TemporaryDirectory(prefix="aspire-tc2-reference-") as directory:
        files = []
        for name in CONFIG_NAMES[:-1]:
            data = subprocess.check_output([git, "-C", str(ROOT), "show", f"{BASE_COMMIT}:config/{name}"])
            path = Path(directory) / name
            path.write_bytes(data)
            files.append(path)
        return load_config(files, use_env=False)


def setup_report(paths, quiet=False):
    """Compare the account with the actual TC2 setup file without loading model weights."""
    versions = reference_versions()
    packages, issues = {}, []
    for name in PACKAGES:
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
            issues.append(f"Missing runtime dependency: {name}")
        expected = versions.get(re.sub(r"[-_.]+", "-", name).lower())
        packages[name] = {"tc2": expected, "here": actual, "match": actual == expected and actual is not None}
    inputs = {}
    for name in (paths["core_table"], paths["e46_table"], "e7_v1"):
        file = ROOT / "data/processed/e1e6" / f"{name}.parquet"
        digest = sha(file) if file.is_file() else None
        match = digest == TABLES[name][1]
        inputs[name] = {
            "expected_rows": TABLES[name][0],
            "expected_sha256": TABLES[name][1],
            "actual_sha256": digest,
            "match": match,
        }
        if not match:
            issues.append(f"Frozen input mismatch: {name}; run git lfs pull")
    exclusion = ROOT / "results/review/excluded_prompts.csv"
    if not exclusion.is_file() or sha(exclusion) != EXCLUSIONS_HASH:
        issues.append("Protocol exclusion list differs")

    science, execution, detector_settings, checkpoint = {}, {}, {}, "unverified"
    try:
        from visual_synthesis.config import get_model, load_config

        reference = frozen_reference_config()
        actual_cfg = load_config(configs(paths), use_env=False)
        model = get_model(actual_cfg, paths["model"])
        if paths["model"] in reference["models"]:
            baseline = get_model(reference, paths["model"])
            for key in (
                "repo_id",
                "revision",
                "family",
                "variant",
                "dtype",
                "width",
                "height",
                "num_inference_steps",
                "guidance_scale",
                "negative_prompt",
                "generation_kwargs",
                "hook",
                "safety_checker",
            ):
                match = model[key] == baseline[key]
                science[key] = {"tc2": baseline[key], "here": model[key], "match": match}
                if not match:
                    issues.append(f"Scientific model setting differs: {key}")
            execution = {
                key: {"tc2": baseline[key], "here": model[key]}
                for key in ("device_map", "quantization", "enable_cpu_offload")
                if baseline[key] != model[key]
            }
            checkpoint = (
                "pinned and matches"
                if model["revision"] and model["revision"] == baseline["revision"]
                else "unverified: TC2 config does not pin an immutable model revision"
            )
        else:
            checkpoint = "new configured model: no TC2 model reference to compare"
        if actual_cfg["run"]["seed"] != reference["run"]["seed"]:
            issues.append("Protocol seed differs")
        for name in ("nudenet", "clip-h14"):
            detector_settings[name] = {
                key: actual_cfg["detectors"][name][key] for key in ("threshold", "flag_labels", "label_thresholds")
            }
            for key in detector_settings[name]:
                if actual_cfg["detectors"][name][key] != reference["detectors"][name][key]:
                    issues.append(f"Detector setting differs: {name}.{key}")
        for key in ("combine", "alignment", "quality", "scope_category"):
            if actual_cfg["metrics"][key] != reference["metrics"][key]:
                issues.append(f"Evaluator setting differs: metrics.{key}")
        if not actual_cfg["metrics"]["quality"]["enabled"]:
            issues.append("Protocol quality evaluator is disabled")
    except (ImportError, KeyError, ValueError) as exc:
        issues.append(f"Cannot validate configuration: {exc}")

    # An ancestry check alone cannot establish that the experiment implementation
    # equals the frozen protocol. Compare the actual scientific source bytes too.
    git = os.environ.get("VS_ASPIRE_GIT", "git")
    code = {}
    for relative in (
        "config.py",
        "io.py",
        "schema.py",
        "activations.py",
        "representation.py",
        "backends/image.py",
        "backends/detectors.py",
        "stages/s05_generate.py",
        "stages/s06_nsfw.py",
        "stages/s07_metrics.py",
        "e1e6_report.py",
        "stats.py",
        "hook_validation.py",
    ):
        name = "src/visual_synthesis/" + relative
        try:
            frozen = subprocess.check_output(
                [git, "-C", str(ROOT), "show", f"{BASE_COMMIT}:{name}"], stderr=subprocess.DEVNULL
            )
            code[relative] = sha(ROOT / name) == hashlib.sha256(frozen).hexdigest()
        except (OSError, subprocess.SubprocessError):
            code[relative] = False
        if not code[relative]:
            issues.append(f"Scientific source differs from {BASE_COMMIT[:7]}: {relative}")
    matches = all(packages[p]["match"] for p in TC2_REQUIRED)
    runtime_matches = all(row["match"] for row in packages.values())
    report = {
        "model": paths["model"],
        "tier": paths["tier"],
        "reference_commit": BASE_COMMIT,
        "python": sys.version,
        "protocol_ok": not issues,
        "tc2_required_versions_match": matches,
        "tc2_runtime_versions_match": runtime_matches,
        "packages": packages,
        "inputs": inputs,
        "scientific_settings": science,
        "source_matches": code,
        "execution_differences": execution,
        "detectors": detector_settings,
        "checkpoint_revision_comparison": checkpoint,
        "issues": issues,
        "gpu_fit_checked": False,
        "next_step": "ready checks CUDA/BF16, caches, hooks and two evaluated images",
    }
    save(paths["work"] / "setup-check.json", report)
    if not quiet:
        print(f"Protocol/files/dependencies: {'PASS' if not issues else 'FAIL'}")
        print(f"TC2 required package versions: {'MATCH' if matches else 'DIFFERENT'}")
        for name in TC2_REQUIRED:
            row = packages[name]
            print(f"  {name}: TC2={row['tc2']}  here={row['here']}  {'OK' if row['match'] else 'DIFF'}")
        for name, row in packages.items():
            if name not in TC2_REQUIRED and not row["match"]:
                print(f"  {name}: TC2={row['tc2']}  here={row['here']}  DIFF")
        for issue in issues:
            print(f"  FIX: {issue}")
        print(f"Execution differences: {json.dumps(execution)}")
        print(f"Checkpoint revisions: {checkpoint}")
        print(f"Report: {paths['work'] / 'setup-check.json'}")
    return report


def configure(paths):
    from visual_synthesis.config import ENV_OVERRIDES, load_dotenv

    load_dotenv(ROOT)
    old_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser()
    if (old_home / "token").is_file():
        os.environ.setdefault("HF_TOKEN_PATH", str((old_home / "token").resolve()))
    # Stale .env stage/model/path overrides must not alter a frozen run.
    for key in ENV_OVERRIDES:
        os.environ.pop(key, None)
    storage = paths["storage"]
    os.environ.update(
        VISUAL_SYNTHESIS_ROOT=str(ROOT),
        HF_HOME=str(storage / "hf-home"),
        HF_HUB_CACHE=str(storage / "hf-home/hub"),
        HF_ASSETS_CACHE=str(storage / "hf-home/assets"),
        HF_XET_CACHE=str(storage / "hf-home/xet"),
        HF_HUB_DISABLE_XET="1",
        XDG_CACHE_HOME=str(storage / "cache"),
        TORCH_HOME=str(storage / "torch"),
    )
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    overlay = {
        "generate": {"model": paths["model"]},
        "paths": {
            "models": str(paths["models"]),
            "runs": str(paths["runs"]),
            "logs": str(paths["work"] / "pipeline-logs"),
            "cache": str(paths["work"] / "cache"),
        },
        "metric_models": {
            "clip-vit-base-patch32": {"repo_id": str(paths["models"] / "openai--clip-vit-base-patch32")},
            "clip-iqa-vit-base-patch16": {"repo_id": str(paths["models"] / "openai--clip-vit-base-patch16")},
        },
    }
    if paths["config"].exists() and read(paths["config"]) != overlay:
        raise ValueError("Workspace paths changed; use a new workspace, do not mix existing runs")
    save(paths["config"], overlay)
    # config.py loads .env again in every child. Explicit inherited values keep
    # that file from restoring stale TC2 paths or a different seed.
    os.environ.update(
        VS_MODELS_DIR=str(paths["models"]),
        VS_RUNS_DIR=str(paths["runs"]),
        VS_LOGS_DIR=str(paths["work"] / "pipeline-logs"),
        VS_SEED="42",
        VS_LOG_LEVEL="info",
    )


def disk_guard(storage, budget_gib, payload=0):
    if not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError("VS_ASPIRE_BUDGET_GIB must be positive and finite")
    # du counts symlinked shared checkpoints once and uses allocated filesystem blocks.
    used = int(subprocess.check_output(["du", "-sk", str(storage)], text=True).split()[0]) * 1024
    needed = int(payload * 1.2) + 4 * GIB
    if min(int(budget_gib * GIB) - used, shutil.disk_usage(storage).free) < needed:
        raise ValueError("Insufficient workspace budget/free space (includes 4 GiB headroom)")
    return {"allocated_bytes": used, "budget_gib": budget_gib, "filesystem_free": shutil.disk_usage(storage).free}


def snapshot_complete(destination, generator=False):
    required = ["model_index.json"] if generator else ["config.json", "preprocessor_config.json"]
    if not all((destination / name).is_file() for name in required):
        return False
    if generator:
        try:
            index = read(destination / "model_index.json")
        except (ValueError, OSError):
            return False
        components = [
            name
            for name, value in index.items()
            if not name.startswith("_")
            and isinstance(value, list)
            and value
            and value[0] is not None
            and name != "safety_checker"
        ]
        if not components:
            return False
        parts = []
        for name in components:
            part = destination / name
            if not part.is_dir() or not any(part.iterdir()):
                return False
            if not any(word in name for word in ("tokenizer", "scheduler", "processor", "feature_extractor")):
                parts.append(part)
    else:
        parts = [destination]
    for part in parts:
        indexes = [p for p in part.glob("*index*.json") if "weight_map" in read(p)]
        if indexes:
            for index in indexes:
                if not all(
                    (part / f).is_file() and (part / f).stat().st_size for f in set(read(index)["weight_map"].values())
                ):
                    return False
        elif not any(
            p.stat().st_size and not re.search(r"-\d+-of-\d+", p.name)
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for p in part.glob(pattern)
        ):
            return False
    return True


def prepare_resources(paths, budget, allow_download):
    # Hub settings are read at import time. This preparation child alone may
    # download; its parent and subsequent production children remain offline.
    offline = "0" if allow_download else "1"
    os.environ.update(HF_HUB_OFFLINE=offline, TRANSFORMERS_OFFLINE=offline)
    import torch

    from visual_synthesis.config import get_model, load_config, validate

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    cfg = load_config(configs(paths), use_env=False)
    problems = validate(cfg)
    if problems:
        raise ValueError(problems)
    model = get_model(cfg, paths["model"])
    if model["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This model requires BF16 support; no precision fallback is allowed")
    if model["quantization"]:
        raise ValueError("Native precision requires quantization=null; validate quantized models separately")
    if not model["hook"]["layers"]:
        raise ValueError("Configure this model's hook layers before its E1–E7 readiness check")
    plans = []
    choices = [
        (model["repo_id"], paths["models"] / (model["local_dir"] or paths["model"]), model["allow_patterns"], True),
        (
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            paths["models"] / "clip-h14",
            ["config.json", "preprocessor_config.json", "model.safetensors"],
            False,
        ),
        *[
            (
                f"openai/clip-vit-base-patch{size}",
                paths["models"] / f"openai--clip-vit-base-patch{size}",
                ["*.json", "*.txt", "model.safetensors", "pytorch_model.bin"],
                False,
            )
            for size in (32, 16)
        ],
    ]
    for repo, destination, patterns, generator in choices:
        complete = snapshot_complete(destination, generator)
        if not generator and repo.startswith("openai/"):
            complete = complete and all((destination / n).is_file() for n in ("vocab.json", "merges.txt"))
        plan = {"repo_id": repo, "path": str(destination), "reused": complete, "revision": "existing-cache"}
        if not complete:
            if not allow_download:
                raise FileNotFoundError(f"Incomplete cache: {destination}; run ready with VS_PROTOCOL_ALLOW_DOWNLOAD=1")
            from huggingface_hub import HfApi, snapshot_download

            info = HfApi().model_info(repo, files_metadata=True)
            files = select_snapshot_files(info.siblings, patterns)
            if not info.sha or not files or any(n is None for n in files.values()):
                raise ValueError(f"Missing immutable revision or file sizes for {repo}")
            disk_guard(paths["storage"], budget, sum(files.values()))
            snapshot_download(repo, revision=info.sha, local_dir=destination, allow_patterns=list(files))
            plan.update(revision=info.sha, selected_bytes=sum(files.values()))
            if not snapshot_complete(destination, generator):
                raise ValueError(f"Downloaded cache is incomplete: {destination}")
        plans.append(plan)
    head = paths["models"] / "clip-h14/h14_nsfw.pth"
    if not head.is_file():
        source = ROOT / "models/clip-h14/h14_nsfw.pth"
        if source.is_file():
            disk_guard(paths["storage"], budget, source.stat().st_size)
            head.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, head)
        elif allow_download:
            from visual_synthesis.download import download_url

            disk_guard(paths["storage"], budget, 32 * 1024**2)
            download_url(cfg["detectors"]["clip-h14"]["url"], head.parent)
        if not head.is_file():
            raise FileNotFoundError(f"Missing LAION head: {head}; prepare it in ready before production")
    # Ensure NudeNet's packaged ONNX checkpoint is present; don't install/download in production.
    import nudenet

    if not list(Path(nudenet.__file__).parent.glob("*.onnx")):
        raise FileNotFoundError("NudeNet's packaged ONNX checkpoint is missing")
    cache = {
        str(p): [p.stat().st_size, p.stat().st_mtime_ns]
        for _, destination, _, _ in choices
        for p in destination.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    }
    save(
        paths["work"] / "resources.json",
        {
            "plans": plans,
            "cache_files": cache,
            "head_sha256": sha(head),
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu_driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "storage": disk_guard(paths["storage"], budget),
        },
    )


def check_resources(paths):
    resource = read(paths["work"] / "resources.json")
    for name, expected in resource["cache_files"].items():
        p = Path(name)
        if not p.is_file() or [p.stat().st_size, p.stat().st_mtime_ns] != expected:
            raise ValueError(f"Cached resource changed after readiness: {p}")
    if sha(paths["models"] / "clip-h14/h14_nsfw.pth") != resource["head_sha256"]:
        raise ValueError("LAION head changed after readiness")
    return resource


def select_snapshot_files(siblings, patterns):
    """Use configured files, prefer safetensors, and exclude alternate binary weight copies."""
    files = {f.rfilename: f.size for f in siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in (patterns or ["*"]))}
    safe_dirs = {str(Path(name).parent) for name in files if name.endswith(".safetensors")}
    for name in list(files):
        # Diffusers does not need the repo's monolithic root checkpoint or
        # ONNX/Flax/OpenVINO exports when loading a component-based pipeline.
        if any(part in ("onnx", "openvino", "flax") for part in Path(name).parts):
            files.pop(name)
        elif name.endswith((".ckpt", ".msgpack", ".onnx", ".h5")):
            files.pop(name)
        elif "/" not in name and name.endswith((".safetensors", ".bin")) and "model_index.json" in files:
            files.pop(name)
        elif name.endswith(".bin") and str(Path(name).parent) in safe_dirs:
            files.pop(name)
    return files


def records(path):
    return (
        [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if Path(path).exists()
        else []
    )


def expected_table(name, validation=False, limit=None):
    import pandas as pd

    rows = pd.read_parquet(ROOT / "data/processed/e1e6" / f"{name}.parquet")
    if validation:
        rows = rows[rows.subset_split == "validation"]
    return rows.iloc[:limit] if limit else rows


def check_generations(path, expected, model, capture=False, condition=None):
    """Validate final journal records, coverage, fixed seeds, precision and saved artifacts."""
    from visual_synthesis.backends.image import precision

    rows = records(path / "generations.jsonl")
    latest = {r["prompt_id"]: r for r in rows}
    wanted = expected.set_index("prompt_id")
    if set(latest) != set(wanted.index):
        raise ValueError(f"{path.name}: generated coverage {len(latest)} != expected {len(wanted)}")
    for key, row in latest.items():
        if (
            row.get("generation_status") != "success"
            or row.get("seed") != 42
            or row.get("precision") != precision(model)
            or row.get("model") != model["id"]
        ):
            raise ValueError(f"{path.name}: failed generation, seed or precision mismatch for {key}")
        if row["prompt"] != wanted.loc[key, "prompt"]:
            raise ValueError(f"{path.name}: prompt mismatch for {key}")
        if (row.get("width"), row.get("height"), row.get("num_inference_steps")) != (
            model["width"],
            model["height"],
            model["num_inference_steps"],
        ):
            raise ValueError(f"{path.name}: generation settings differ from the protocol")
        if condition:
            mode, vector, alpha = condition
            actual = row.get("intervention_mode"), row.get("intervention_vector"), row.get("intervention_alpha")
            if actual != (mode, vector, alpha) or row.get("intervention_passes") != "all":
                raise ValueError(f"{path.name}: intervention condition mismatch")
        for field in ["image_path", "activation_path"] if capture else ["image_path"]:
            if not Path(row.get(field, "")).is_file():
                raise ValueError(f"{path.name}: missing {field} for {key}")
    return latest


def check_detections(path, expected):
    latest = {r["prompt_id"]: r for r in records(path / "detections.jsonl")}
    if set(latest) != set(expected.prompt_id):
        raise ValueError(f"{path.name}: incomplete detector coverage")
    if any(r.get("status") != "success" or r.get("flagged") is None for r in latest.values()):
        raise ValueError(f"{path.name}: failed or unknown verdict; cannot report as unflagged")


def check_metrics(path, expected, model):
    import numpy as np
    import pandas as pd

    from visual_synthesis.backends.image import precision

    table = pd.read_parquet(path / "image_metrics.parquet")
    if table.prompt_id.duplicated().any() or set(table.prompt_id) != set(expected.prompt_id):
        raise ValueError(f"{path.name}: incomplete/duplicate image metrics")
    for column in ("flagged", "flagged_nudenet", "flagged_clip-h14", "clip_score", "quality_score"):
        if column not in table or table[column].isna().any():
            raise ValueError(f"{path.name}: missing {column}; stop before report generation")
    if not np.isfinite(table[["clip_score", "quality_score"]].to_numpy(float)).all():
        raise ValueError(f"{path.name}: non-finite evaluation scores")
    if (
        not pd.to_numeric(table.seed, errors="coerce").eq(42).all()
        or not table.precision.eq(precision(model)).all()
        or not table.model.eq(model["id"]).all()
    ):
        raise ValueError(f"{path.name}: metrics seed/precision mismatch")


class Runner:
    def __init__(self, paths, seconds):
        self.paths = paths
        self.model = selected_model(paths)
        self.deadline = time.monotonic() + seconds - 180
        self.state_path = paths["work"] / "progress.json"
        self.state = read(self.state_path) if self.state_path.exists() else {"done": []}

    def step(self, name, command, check):
        if name in self.state["done"]:
            check()
            return
        remaining = self.deadline - time.monotonic()
        if remaining < 180:
            raise Checkpoint(name)
        self.state.update(status="running", active=name, job=os.environ.get("PBS_JOBID"))
        save(self.state_path, self.state)
        print(f"STEP {name}", flush=True)
        log = self.paths["logs"] / f"{name}.log"
        with log.open("a") as out:
            out.write(f"\nPBS {os.environ.get('PBS_JOBID')} {time.asctime()}\n")
            out.flush()
            child = subprocess.Popen(
                [str(x) for x in command], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = child.wait(timeout=remaining)
            except BaseException as exc:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise Checkpoint(name) from exc
                raise
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {log}")
        check()
        self.state["done"].append(name)
        save(self.state_path, self.state)

    def pipeline(self, action, *args):
        return [
            sys.executable,
            "-m",
            "visual_synthesis",
            action,
            *config_args(self.paths),
            "--mode",
            "incremental",
            *map(str, args),
        ]

    def script(self, name, *args):
        return [sys.executable, ROOT / "scripts" / name, *args]

    def generation(self, run, table, capture=False, condition=None, limit=None):
        expected = expected_table(table, validation=condition is not None, limit=limit)
        args = [
            "--model",
            self.paths["model"],
            "--input",
            ROOT / "data/processed/e1e6" / f"{table}.parquet",
            "--run-name",
            run,
        ]
        if limit:
            args += ["--limit", str(limit)]
        if capture:
            args += ["--set", "generate.capture.enabled=true"]
        if condition:
            mode, vector, alpha = condition
            args += [
                "--set",
                'stages.generate.filters={"subset_split":"validation"}',
                "--set",
                "generate.intervention.enabled=true",
                "--set",
                f"generate.intervention.e4_dir={self.paths['e4']}",
                "--set",
                f"generate.intervention.mode={mode}",
                "--set",
                f"generate.intervention.vector={vector}",
                "--set",
                f"generate.intervention.alpha={alpha}",
            ]
        path = self.paths["runs"] / "s05_generate" / run
        self.step(
            run + "-generate",
            self.pipeline("generate", *args),
            lambda: check_generations(path, expected, self.model, capture, condition),
        )
        self.evaluate(run, expected)

    def evaluate(self, run, expected):
        paths = self.paths
        gen = paths["runs"] / "s05_generate" / run
        for detector, suffix in (("nudenet", "nudenet"), ("clip-h14", "laion")):
            name = run + "-" + suffix
            target = paths["runs"] / "s06_nsfw" / name
            self.step(
                name,
                self.pipeline("nsfw", "--input", gen, "--run-name", name, "--detector", detector),
                lambda target=target: check_detections(target, expected),
            )
        target = paths["runs"] / "s07_metrics" / run
        self.step(
            run + "-metrics",
            self.pipeline(
                "metrics",
                "--input",
                paths["runs"] / "s06_nsfw" / (run + "-nudenet"),
                "--run-name",
                run,
                "--set",
                f"metrics.generations_input={gen}",
                "--set",
                f"metrics.secondary_detections={paths['runs'] / 's06_nsfw' / (run + '-laion')}",
            ),
            lambda: check_metrics(target, expected, self.model),
        )

    def report(self, label):
        self.step(
            label,
            self.script(
                "e1e6_report.py",
                "tables",
                "--model",
                self.paths["model"],
                "--runs",
                self.paths["runs"],
                "--output",
                self.paths["report"].parent,
                "--exclude",
                ROOT / "results/review/excluded_prompts.csv",
                *config_args(self.paths),
            ),
            lambda: read_mode(self.paths),
        )

    def phase(self, phase):
        model, e46 = self.paths["model"], self.paths["e46_table"]
        if phase == "e1":
            self.generation(f"e1-{model}", self.paths["core_table"])
        elif phase == "e7":
            self.generation(f"e7-{model}", "e7_v1")
        elif phase == "e4":
            self.generation(f"e4-{model}-capture", e46, capture=True)
            self.step(
                "e4-analysis",
                self.script(
                    "e4_analyze.py",
                    "--capture-run",
                    self.paths["runs"] / "s05_generate" / f"e4-{model}-capture",
                    "--output",
                    self.paths["e4"],
                ),
                lambda: check_analysis(self.paths),
            )
        elif phase in ("e5", "e6"):
            check_analysis(self.paths)
            if phase == "e6":
                if "e5-mode" not in self.state["done"]:
                    raise ValueError("E5 must finish before selecting E6")
                modes, alphas = [read_mode(self.paths)], (0.25, 0.75)
            else:
                modes, alphas = ("ablation", "steering"), (0.5, 1.0)
            for mode in modes:
                for vector in ("safety", "random:0"):
                    for alpha in alphas:
                        run = f"{phase}-{model}-{mode}-{vector.replace(':', '')}-a{alpha}"
                        self.generation(run, e46, condition=(mode, vector, alpha))
            if phase == "e5":
                self.report("e5-mode")
        elif phase == "report":
            self.report("final-report")


def check_analysis(paths):
    result = read(paths["e4"] / "statistics.json")
    if not result.get("layer") or not (paths["e4"] / "directions.npz").is_file():
        raise ValueError("E4 direction analysis is incomplete")


def read_mode(paths):
    import pandas as pd

    from visual_synthesis.e1e6_report import e6_mode

    table = pd.read_csv(paths["report"] / "e5.csv")
    modes = e6_mode(table)
    if paths["model"] not in modes:
        raise ValueError("No complete E5 results to select E6 mode")
    return modes[paths["model"]]


def run(paths, phase, budget, seconds):
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("GPU work must run inside PBS; submit aspire_protocol.pbs or use the submit command")
    if phase not in PHASES:
        raise ValueError(f"Unknown protocol phase {phase}")
    import fcntl

    for folder in (paths["work"], paths["logs"], paths["models"]):
        folder.mkdir(parents=True, exist_ok=True)
    with (paths["work"] / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_code()
        verify_inputs(paths)
        configure(paths)
        disk_guard(paths["storage"], budget)
        setup = setup_report(paths, quiet=phase != "check")
        if setup["issues"]:
            raise ValueError(f"Setup check failed; inspect {paths['work'] / 'setup-check.json'}")
        if phase == "check":
            return 0 if setup["tc2_runtime_versions_match"] else 2
        if not setup["tc2_runtime_versions_match"]:
            print(f"TC2 package versions DIFFER: inspect {paths['work'] / 'setup-check.json'}", flush=True)
        current = identity(paths)
        ident = paths["work"] / "identity.json"
        if ident.exists() and read(ident) != current:
            raise ValueError("Code, configuration or environment changed since readiness; use a new workspace")
        save(ident, current)
        runner = Runner(paths, seconds)
        if phase != "ready" and not (paths["work"] / "ready.json").exists():
            raise ValueError("Run and inspect the readiness job before production")
        if phase != "ready":
            ready = read(paths["work"] / "ready.json")
            if ready.get("passed") is not True or ready["identity"] != current:
                raise ValueError("Readiness did not pass with the current code/environment")
            check_resources(paths)
        try:
            if phase == "ready":
                runner.step(
                    "resources",
                    [sys.executable, ROOT / "scripts/aspire_protocol.py", "prepare"],
                    lambda: check_resources(paths),
                )
                hook = paths["work"] / "hook-validation" / f"{paths['model']}.json"

                def hook_check():
                    if not read(hook)["passed"]:
                        raise ValueError("The target-account main hook validation failed")

                runner.step(
                    "hook-validation",
                    runner.script(
                        "validate_hooks.py",
                        *config_args(paths),
                        "--model",
                        paths["model"],
                        "--steps",
                        "2",
                        "--output",
                        hook.parent,
                    ),
                    hook_check,
                )
                runner.generation(f"pilot-fit-{paths['model']}", paths["core_table"], limit=2)
                save(
                    paths["work"] / "ready.json",
                    {
                        "passed": True,
                        "job": os.environ["PBS_JOBID"],
                        "identity": current,
                        "tc2_required_versions_match": setup["tc2_required_versions_match"],
                        "tc2_runtime_versions_match": setup["tc2_runtime_versions_match"],
                    },
                )
            else:
                for part in ("e1", "e7", "e4", "e5", "e6", "report") if phase == "all" else [phase]:
                    runner.phase(part)
            runner.state.update(status="complete", phase=phase)
            save(runner.state_path, runner.state)
            print(f"COMPLETE {phase}; results in {paths['work']}", flush=True)
        except Checkpoint as exc:
            runner.state.update(status="checkpoint", active=str(exc), phase=phase)
            save(runner.state_path, runner.state)
            return 75
        except Exception as exc:
            runner.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            save(runner.state_path, runner.state)
            raise
    return 0


def submission(paths, phase, seconds, after=None, index=1, max_jobs=12):
    project = os.environ.get("VS_ASPIRE_PROJECT", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", project):
        raise ValueError("Set VS_ASPIRE_PROJECT to your NSCC project ID")
    if after and not JOB_ID.fullmatch(after):
        raise ValueError("Invalid dependency job ID")
    if seconds < 900 or seconds > 21600 or not 1 <= index <= max_jobs <= 24:
        raise ValueError("Use 900–21600 seconds/job and 1–24 total jobs")
    python = os.environ.get("VS_ASPIRE_PYTHON", sys.executable)
    git = os.environ.get("VS_ASPIRE_GIT", shutil.which("git") or "")
    values = {
        "VS_ASPIRE_REPO": str(ROOT),
        "VS_ASPIRE_PYTHON": python,
        "VS_ASPIRE_GIT": git,
        "VS_ASPIRE_STORAGE": str(paths["storage"]),
        "VS_ASPIRE_PROJECT": project,
        "VS_ASPIRE_BUDGET_GIB": os.environ.get("VS_ASPIRE_BUDGET_GIB", ""),
        "VS_PROTOCOL_PHASE": phase,
        "VS_PROTOCOL_MODEL": paths["model"],
        "VS_PROTOCOL_TIER": paths["tier"],
        "VS_PROTOCOL_WALL_SECONDS": str(seconds),
        "VS_PROTOCOL_JOB_INDEX": str(index),
        "VS_PROTOCOL_MAX_JOBS": str(max_jobs),
        "VS_PROTOCOL_ALLOW_DOWNLOAD": os.environ.get("VS_PROTOCOL_ALLOW_DOWNLOAD", "0"),
    }
    if os.environ.get("HF_TOKEN_PATH"):
        values["HF_TOKEN_PATH"] = os.environ["HF_TOKEN_PATH"]
    for key, value in values.items():
        if not value or any(c in value for c in ",\n\r"):
            raise ValueError(f"Missing/unsupported PBS environment value for {key}")
    log = paths["logs"] / f"pbs-{phase}-{time.time_ns()}.log"
    hours, remain = divmod(seconds, 3600)
    minutes, secs = divmod(remain, 60)
    command = [
        "qsub",
        "-P",
        project,
        "-q",
        "normal",
        "-N",
        f"v-{paths['model'][:7]}-{phase[:4]}",
        "-o",
        str(log),
        "-l",
        f"walltime={hours:02}:{minutes:02}:{secs:02}",
        "-v",
        ",".join(values),
    ]
    if after:
        command += ["-W", f"depend=afterok:{after}"]
    command.append(str(ROOT / "scripts/aspire_protocol.pbs"))
    return command, values


def submit(paths, phase, seconds, after=None, index=1, max_jobs=12, dry_run=False):
    verify_code()
    verify_inputs(paths)
    verify_model_name(paths)
    command, values = submission(paths, phase, seconds, after, index, max_jobs)
    if dry_run:
        # Do not print token paths or a complete environment; the command exports names only.
        print(
            json.dumps(
                {
                    "command": command,
                    "workspace": str(paths["work"]),
                    "phase": phase,
                    "model": paths["model"],
                    "tier": paths["tier"],
                    "job_index": index,
                    "max_jobs": max_jobs,
                },
                indent=2,
            )
        )
        return None
    if phase not in ("check", "ready") and not after:
        ready = paths["work"] / "ready.json"
        if not ready.exists() or read(ready).get("passed") is not True:
            raise ValueError("Complete this model's readiness job before submitting production")
    paths["logs"].mkdir(parents=True, exist_ok=True)
    result = subprocess.check_output(command, env={**os.environ, **values}, cwd=ROOT, text=True).strip()
    if not JOB_ID.fullmatch(result):
        raise RuntimeError(f"Unexpected qsub response: {result[:200]}")
    save(
        paths["work"] / f"submission-{result}.json",
        {"job": result, "phase": phase, "index": index, "max_jobs": max_jobs, "wall_seconds": seconds, "after": after},
    )
    print(f"Submitted {result}; model={paths['model']}; phase={phase}; workspace={paths['work']}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    s = sub.add_parser("submit")
    s.add_argument("phase", choices=PHASES)
    s.add_argument("--model", required=True, help="model ID from the shared config")
    s.add_argument("--tier", choices=("auto", "full", "reduced"), default="auto")
    s.add_argument("--after", help="PBS dependency, afterok")
    s.add_argument("--max-jobs", type=int, default=12, help="bounded all-phase continuations, including first job")
    s.add_argument("--wall-seconds", type=int, default=None)
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("run")
    sub.add_parser("prepare")
    status = sub.add_parser("status")
    status.add_argument("--model", required=True)
    status.add_argument("--tier", choices=("auto", "full", "reduced"), default="auto")
    check = sub.add_parser("check", help="compare setup with TC2 without loading model weights")
    check.add_argument("--model", required=True)
    check.add_argument("--tier", choices=("auto", "full", "reduced"), default="auto")
    check.add_argument(
        "--strict", action="store_true", help="also fail for differences in the recorded runtime packages"
    )
    args = parser.parse_args(argv)
    try:
        model = args.model if args.action in ("submit", "status", "check") else os.environ["VS_PROTOCOL_MODEL"]
        tier = args.tier if args.action in ("submit", "status", "check") else os.environ["VS_PROTOCOL_TIER"]
        paths = layout(storage_path(os.environ["VS_ASPIRE_STORAGE"]), model, tier)
        if args.action == "check":
            verify_code()
            verify_model_name(paths)
            configure(paths)
            report = setup_report(paths)
            return 2 if report["issues"] or (args.strict and not report["tc2_runtime_versions_match"]) else 0
        if args.action == "status":
            progress = paths["work"] / "progress.json"
            state = read(progress) if progress.exists() else {"status": "not started", "done": []}
            ready_path = paths["work"] / "ready.json"
            ready = read(ready_path) if ready_path.exists() else {}
            setup_path = paths["work"] / "setup-check.json"
            setup = read(setup_path) if setup_path.exists() else {}
            print(
                json.dumps(
                    {
                        "model": model,
                        "tier": paths["tier"],
                        "workspace": str(paths["work"]),
                        "ready": ready.get("passed") is True,
                        "tc2_versions_match": setup.get("tc2_runtime_versions_match"),
                        **{k: v for k, v in state.items() if k != "done"},
                        "completed_steps": len(state["done"]),
                    },
                    indent=2,
                )
            )
            return 0
        budget = float(os.environ["VS_ASPIRE_BUDGET_GIB"])
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("VS_ASPIRE_BUDGET_GIB must be positive and finite")
        if args.action == "submit":
            seconds = args.wall_seconds or (7200 if args.phase in ("check", "ready") else 21600)
            submit(paths, args.phase, seconds, args.after, max_jobs=args.max_jobs, dry_run=args.dry_run)
            return 0
        if args.action == "prepare":
            if not os.environ.get("PBS_JOBID"):
                raise RuntimeError("Resource preparation requires PBS")
            import fcntl

            configure(paths)
            # Different model workspaces can share the same evaluator caches.
            # Serialize preparation and its budget check without locking inference.
            with (paths["storage"] / ".protocol-cache.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                prepare_resources(paths, budget, os.environ.get("VS_PROTOCOL_ALLOW_DOWNLOAD") == "1")
            return 0
        phase = os.environ["VS_PROTOCOL_PHASE"]
        seconds = int(os.environ["VS_PROTOCOL_WALL_SECONDS"])
        result = run(paths, phase, budget, seconds)
        if result == 75 and phase == "all":
            index, maximum = int(os.environ["VS_PROTOCOL_JOB_INDEX"]), int(os.environ["VS_PROTOCOL_MAX_JOBS"])
            if index >= maximum:
                raise RuntimeError("Continuation job limit reached; inspect progress and explicitly resubmit")
            next_job = submit(paths, phase, seconds, os.environ["PBS_JOBID"], index + 1, maximum)
            save(paths["work"] / f"continuation-{os.environ['PBS_JOBID']}.json", {"next_job": next_job})
            return 0  # afterok continuation starts only after this checkpoint job exits successfully
        return result
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
