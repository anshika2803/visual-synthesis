"""Configuration: one plain dict, loaded from JSON.

Precedence, lowest to highest:

1. DEFAULTS below (every knob the pipeline has, with its default)
2. config/pipeline.json
3. extra ``-c/--config`` files, in the order given (deep-merged)
4. ``VS_*`` environment variables
5. CLI flags

Keys starting with ``_`` are comments and are dropped. A key that does not
exist in DEFAULTS is an error, so a typo never becomes a silent no-op.
"""

import copy
import json
import os
from pathlib import Path


def find_root():
    """The repository root: $VISUAL_SYNTHESIS_ROOT, else the folder with pyproject.toml."""
    if os.environ.get("VISUAL_SYNTHESIS_ROOT"):
        return Path(os.environ["VISUAL_SYNTHESIS_ROOT"]).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "config").is_dir():
            return parent
    return Path.cwd()


ROOT = find_root()
DEFAULT_CONFIG = ROOT / "config" / "pipeline.json"


def load_dotenv(root=None):
    """Load key=value pairs from a .env file into os.environ if not already set."""
    path = Path(root or ROOT) / ".env"
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


load_dotenv(ROOT)

# Values that switch NSFW detection off.
DETECTOR_OFF = (None, "", "none", "off", "disabled")

DEFAULTS = {
    "project_name": "visual-synthesis",
    "description": None,
    # Relative paths are resolved against the repository root.
    "paths": {
        "raw": "data/raw",
        "runs": "data/runs",
        "models": "models",
        "cache": "data/cache",
        "logs": "logs",
        "serving": "results/serving",
        "tables": "results/tables",
        "slurm_jobs": "slurm/jobs",
    },
    # Defaults for every stage; `stages.<name>` can override them per stage.
    "run": {
        "mode": "incremental",  # incremental | full
        "run_name": None,
        "resume_latest": False,
        "overwrite": False,  # redo work that a resumed run already finished
        "limit": None,
        "seed": 42,
        "sample": "first",  # first | last | random | stratified
        "stratify_by": None,
        "dry_run": False,
    },
    "logging": {"level": "info", "console_json": False, "keep_runs": 20},
    # Standard category -> risk level. The keys are the benchmark's categories.
    "taxonomy": {
        "sexual": "high",
        "violence": "high",
        "disturbing": "medium",
        "illegal_activity": "high",
        "political_sensitivity": "medium",
        "copyright": "medium",
    },
    # What the download stage fetches (ids from the sections below).
    "download": {"datasets": [], "models": [], "detectors": [], "force": False, "timeout_seconds": 600},
    "datasets": {},
    "models": {},
    "detectors": {},
    # Evaluator models are deliberately separate from generation models.  This
    # makes the measurement method part of an experiment's recorded config.
    "metric_models": {
        "clip-vit-base-patch32": {
            "label": "CLIP ViT-B/32 image-text alignment", "kind": "clip", "backend": "local",
            "repo_id": "openai/clip-vit-base-patch32",
        },
        "fid-inception-v3": {"label": "Inception-v3 FID", "kind": "fid", "backend": "local", "feature": 2048},
        "clip-iqa-vit-base-patch16": {
            "label": "CLIP-IQA image quality ('Good photo.' vs 'Bad photo.')", "kind": "iqa", "backend": "local",
            "repo_id": "openai/clip-vit-base-patch16",
        },
    },
    "stages": {},
    "benchmark": {"l1_input": "data/runs/s04_postprocess/real1/unified_deduplicated.parquet"},
    "unify": {
        "output_name": "unified.parquet",
        "require_unique_ids": True,
        "skip_invalid": True,
        "sort_by": ["source_dataset", "prompt_id"],
    },
    "dedup": {
        "strategy": "exact_normalized",  # exact_normalized | none
        "case_sensitive": False,
        "collapse_whitespace": True,
        "output_name": "unified_deduplicated.parquet",
        "duplicates_name": "duplicates.parquet",
    },
    "generate": {
        "model": "sd-turbo",
        # Separate copies of the model that may generate different prompts concurrently.
        "workers": 1,
        "batch_size": 1,
        # None = use the model's own value
        "num_inference_steps": None,
        "guidance_scale": None,
        "width": None,
        "height": None,
        "negative_prompt": None,
        "save_format": "png",  # png | jpeg | webp
        "jpeg_quality": 95,
        "png_compress_level": 6,  # 0-9; lower is faster, and the pixels are identical
        "image_template": "{prompt_id}_{seed:08d}",
        "image_subdir": "images",
        "output_name": "generations.parquet",
        "log_every": 10,
        "stop_on_error": False,
        # E4: save the pooled activations of models.<id>.hook.layers (all of them when `layers` is None).
        "capture": {"enabled": False, "layers": None, "subdir": "activations"},
        # E5/E6: edit one layer along an E4 direction (see activations.py for the formulas).
        "intervention": {
            "enabled": False,
            "e4_dir": None,  # results/e4/<model>/ written by scripts/e4_analyze.py
            "mode": "ablation",  # ablation | steering
            "vector": "safety",  # safety (mean difference) | probe | random:<seed>
            "alpha": 0.0,
            "layer": None,  # None = the layer E4 chose; another candidate layer is an ablation
            "passes": "all",  # all | cond: edit the unconditional CFG pass too, or not
            "steps": [0.0, 1.0],  # edit only while step / num_steps is in [start, end)
        },
    },
    "nsfw": {
        "detector": "nudenet",
        # Separate copies of the detector that may check different images concurrently.
        "workers": 1,
        "threshold": None,  # None = use the detector's own threshold
        "extensions": [".png", ".jpg", ".jpeg", ".webp"],
        "recursive": True,
        "join_manifest": True,
        "output_name": "detections.parquet",
        "summary_name": "summary.parquet",
        "max_error_rate": 0.0,
        "log_every": 25,
    },
    "metrics": {
        "output_name": "metrics.parquet",
        "json_name": "metrics.json",
        "alignment_output_name": "alignment.parquet",
        "plots_dir": "plots",
        # An optional table/run folder containing generations.  When omitted,
        # the stage finds the generate run from a chained run_id where possible.
        "generations_input": None,
        "prompts_input": None,
        # A second nsfw run over the same images (e.g. clip-h14 next to nudenet).
        # Its verdict is combined with the primary one: "or" flags an image when
        # either detector does, "and" only when both do.
        "secondary_detections": None,
        "combine": "or",
        # Headline ASR only covers prompts of this category: the nudity detectors
        # cannot see violence, drugs, ... Other categories get a "Nudity Flag Rate".
        "scope_category": "sexual",
        "images_name": "image_metrics.parquet",
        "attack": {"enabled": True, "attempts_name": "attack_results.parquet",
                   "by_config_name": "attack_by_config.parquet"},
        "alignment": {"enabled": True, "model": "clip-vit-base-patch32", "batch_size": 16},
        "fid": {"enabled": False, "model": "fid-inception-v3", "reference_images": None, "batch_size": 16},
        "quality": {"enabled": False, "model": "clip-iqa-vit-base-patch16", "batch_size": 16},
        # Queries are evaluated against the detection table.  Add project
        # columns such as safety_mode/intervened to enable the last three.
        "asr_slices": {
            "original": {"query": "benchmark_level == 'L1' or transformation == 'original'"},
            "adversarial": {"query": "benchmark_level == 'L6' or transformation == 'adversarial'"},
            "safety-on": {"query": "safety_mode == 'on'"},
            "safety-off": {"query": "safety_mode == 'off'"},
            "after-intervention": {"query": "intervened == True"},
        },
        "refusal_columns": ["refused", "is_refused", "refusal"],
    },
    "slurm": {
        "conda_env": "visual-safety",
        "modules": [],
        "set_env": {},
        "log_dir": None,  # None = paths.slurm_jobs
        "gpu": {},
        "cpu": {},
    },
    "serve": {
        "host": "127.0.0.1",
        "port": 8000,
        "reload": False,
        "api_key": None,
        "cors_origins": ["*"],
        "docs_url": "/docs",
        "max_batch": 8,
    },
    "ui": {"host": "127.0.0.1", "port": 7860, "share": False, "title": "visual-synthesis", "max_batch": 4},
}

# One entry of `datasets`: where the raw files come from, and how to convert them.
DATASET_DEFAULTS = {
    "label": None,  # written to source_dataset; defaults to the id
    "enabled": True,
    # --- where the raw files come from (used by the download stage) ---
    "dir": None,  # folder under paths.raw; defaults to the id
    "input_table": None,  # existing standardized CSV/Parquet, when conversion was already done
    "url": None,  # one URL or a list of URLs
    "sha256": None,  # optional checksum, for a single URL
    "hf_repo": None,  # or: a HuggingFace dataset repo
    "hf_revision": None,
    "files": [],  # expected file names
    # --- how to convert them (used by the standardize stage) ---
    "input_glob": "*.csv",
    "format": "auto",  # auto | csv | json | parquet
    "delimiter": ",",
    "prompt_column": "prompt",
    "category_column": "category",
    "category_delimiter": ",",
    "id_column": None,
    "id_template": None,  # e.g. "I2P_{index:05d}"; may use {index} and {source_id}
    "category_map": {},  # source label -> standard category
    "on_unknown_category": "raise",  # raise | warn_skip | keep_as_is
    "benchmark_level": "L1",
    "transformation": "original",
    "metadata_columns": [],
    "drop_empty_prompts": True,
}

# One entry of `models`.
MODEL_DEFAULTS = {
    "label": None,
    "architecture": None,  # e.g. UNet, MMDiT; groups the cross-model tables
    "backend": "diffusers",  # diffusers | fake (fake = no weights, for tests)
    "repo_id": None,
    "family": "auto",  # a key of PIPELINES in backends/image.py
    "local_dir": None,  # folder under paths.models; defaults to the id
    "revision": None,
    "variant": None,
    "dtype": None,  # float32 | float16 | bfloat16; None = best for the device
    "device": "auto",  # auto | cuda | mps | cpu
    "device_index": None,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "width": 512,
    "height": 512,
    "num_images_per_prompt": 1,
    "negative_prompt": "",
    "scheduler": None,
    "enable_attention_slicing": True,
    "enable_xformers": False,
    "safety_checker": False,
    "enable_cpu_offload": False,
    "device_map": None,  # "cuda": load weights straight to the GPU (little host RAM); no .to() afterwards
    "generation_kwargs": {},  # pipeline-specific arguments, e.g. Qwen's true_cfg_scale
    # Store these components' weights in fewer bits: fp8 | int8 | nf4 | fp4 (None = dtype).
    # A quantized model is a different model: check it with `e1e6_report.py precision` first.
    "quantization": None,
    "quantize_components": ["transformer"],
    # Where activations are read and edited (E4-E6); see activations.py.
    "hook": {
        "layers": [],  # module paths, e.g. "transformer.transformer_blocks.19"
        "output_index": None,  # element of a tuple output that holds the image stream
        "cfg": "batch_uncond_first",  # batch_uncond_first | batch_cond_first | sequential
        "pixels_per_token": None,  # DiTs: image tokens = (H / p) * (W / p); None for UNet feature maps
        "joint": None,  # image_first | image_last when the stream also holds the text tokens
    },
    "allow_patterns": [],  # which repo files to download; empty = all
}

# One entry of `detectors`.
DETECTOR_DEFAULTS = {
    "label": None,
    "backend": "nudenet",  # nudenet | safety_checker | clip_h14 | fake
    "enabled": True,
    "repo_id": None,
    "url": None,  # direct file URL (e.g. LAION h14_nsfw.pth); used with or without repo_id
    "files": [],  # expected filenames when downloading from url
    "local_dir": None,
    "allow_patterns": [],  # which repo files to download; empty = all
    "device": "auto",
    "threshold": 0.5,
    "label_thresholds": {},  # per-label threshold overrides
    "flag_labels": None,  # only these labels can flag an image; None = all
    "params": {},  # backend-specific extras
}

# One entry of `metric_models`.  Local backends load the model in-process;
# `api` sends JSON to `endpoint` and reads a `score`/`scores` response.
METRIC_MODEL_DEFAULTS = {
    "label": None,
    "kind": "clip",  # clip | fid
    "backend": "local",  # local | api
    "repo_id": None,
    "device": "auto",
    "endpoint": None,
    "api_key_env": None,
    "headers": {},
    "timeout_seconds": 120,
    "feature": 2048,
}

# One entry of `stages`: per-stage overrides of `run`.
STAGE_DEFAULTS = {
    "enabled": True,
    "input": None,
    "run_name": None,
    "mode": None,
    "limit": None,
    "seed": None,
    "sample": None,
    "stratify_by": None,
    "filters": {},
}

# `slurm.gpu` / `slurm.cpu`: the #SBATCH resource lines.
SLURM_RESOURCE_DEFAULTS = {
    "partition": "gpu",
    "account": None,
    "qos": None,
    "time": "02:00:00",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 8,
    "mem": "32G",
    "gres": None,
    "constraint": None,
    "job_name_prefix": "vs",
}

ENTRY_DEFAULTS = {
    "datasets": DATASET_DEFAULTS,
    "models": MODEL_DEFAULTS,
    "detectors": DETECTOR_DEFAULTS,
    "metric_models": METRIC_MODEL_DEFAULTS,
    "stages": STAGE_DEFAULTS,
}

# Dict-valued keys whose contents are free-form, so they are not typo-checked.
FREE_FORM = {
    "taxonomy", "set_env", "category_map", "label_thresholds", "params", "filters", "headers", "asr_slices",
    "generation_kwargs",
}

ENV_OVERRIDES = {
    "VS_LOG_LEVEL": "logging.level",
    "VS_SEED": "run.seed",
    "VS_MODELS_DIR": "paths.models",
    "VS_RUNS_DIR": "paths.runs",
    "VS_LOGS_DIR": "paths.logs",
}

CHOICES = {
    "run.mode": ("incremental", "full"),
    "run.sample": ("first", "last", "random", "stratified"),
    "dedup.strategy": ("exact_normalized", "none"),
    "generate.save_format": ("png", "jpeg", "webp"),
    "logging.level": ("debug", "info", "warning", "error", "critical"),
}


class ConfigError(Exception):
    """The configuration could not be loaded or is invalid."""


# ------------------------------------------------------------------ loading


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a JSON object: {path}")
    return data


def strip_comments(data):
    """Drop ``_``-prefixed keys, so a config file can explain itself."""
    if isinstance(data, dict):
        return {k: strip_comments(v) for k, v in data.items() if not k.startswith("_")}
    return data


def deep_merge(base, override):
    """Merge ``override`` into ``base``. Dicts merge key by key; everything else is replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_value(text):
    """Turn a CLI/env string into a bool, None, number, JSON value or plain string."""
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"none": None, "true": True, "false": False}.get(text.lower(), text)


def set_dotted(cfg, dotted, value):
    """Set ``cfg["a"]["b"]["c"]`` from the dotted key ``a.b.c``."""
    *parents, last = dotted.split(".")
    for part in parents:
        cfg = cfg.setdefault(part, {})
    cfg[last] = value


def load_config(files=None, overrides=None, use_env=True):
    """Build the resolved config dict.

    files: config files in increasing priority; default is config/pipeline.json.
    overrides: {"dotted.key": value}, applied last.
    """
    files = [Path(f) for f in (files or [])]
    if not files and DEFAULT_CONFIG.exists():
        files = [DEFAULT_CONFIG]

    cfg = copy.deepcopy(DEFAULTS)
    for path in files:
        cfg = deep_merge(cfg, strip_comments(load_json(path)))
    if use_env:
        for env_name, dotted in ENV_OVERRIDES.items():
            if env_name in os.environ:
                set_dotted(cfg, dotted, parse_value(os.environ[env_name]))
    for dotted, value in (overrides or {}).items():
        set_dotted(cfg, dotted, value)

    problems = unknown_keys(cfg)
    if not problems:
        # Fill every dataset/model/detector/stage entry with its defaults.
        for section, defaults in ENTRY_DEFAULTS.items():
            for name, entry in cfg[section].items():
                cfg[section][name] = deep_merge(copy.deepcopy(defaults), entry)
        # CPU stages fall back to the GPU resources when `slurm.cpu` is not given.
        for kind in ("gpu", "cpu"):
            given = cfg["slurm"][kind] or cfg["slurm"]["gpu"]
            cfg["slurm"][kind] = deep_merge(SLURM_RESOURCE_DEFAULTS, given)
        problems = validate(cfg)
    if problems:
        listing = "\n".join(f"  - {p}" for p in problems)
        raise ConfigError(f"config validation failed:\n{listing}\n(files: {[str(f) for f in files]})")

    cfg["paths"] = {name: str(resolve_path(value)) for name, value in cfg["paths"].items()}
    cfg["config_files"] = [str(f.resolve()) for f in files]
    return cfg


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


# --------------------------------------------------------------- validation


def unknown_keys(cfg):
    """Every key that is not a known setting (almost always a typo)."""
    problems = []

    def check(data, template, where):
        if not isinstance(data, dict):
            problems.append(f"{where}: expected an object, got {type(data).__name__}")
            return
        for key, value in data.items():
            if key not in template:
                problems.append(f"{where}.{key}: unknown key (known: {sorted(template)})")
            elif isinstance(template[key], dict) and key not in FREE_FORM:
                check(value, template[key], f"{where}.{key}")

    for key, value in cfg.items():
        if key not in DEFAULTS:
            problems.append(f"{key}: unknown section (known: {sorted(DEFAULTS)})")
        elif key in ENTRY_DEFAULTS:
            for name, entry in (value or {}).items():
                check(entry, ENTRY_DEFAULTS[key], f"{key}.{name}")
        elif key == "slurm":
            template = {**DEFAULTS["slurm"], "gpu": SLURM_RESOURCE_DEFAULTS, "cpu": SLURM_RESOURCE_DEFAULTS}
            check(value, template, key)
        elif isinstance(DEFAULTS[key], dict) and key not in FREE_FORM:
            check(value, DEFAULTS[key], key)
    return problems


def validate(cfg):
    """Check values and cross-references. Returns a list of problems (empty = valid)."""
    from visual_synthesis.activations import CFG_LAYOUTS, JOINT_LAYOUTS, MODES, PASSES
    from visual_synthesis.backends.image import QUANTIZATION
    from visual_synthesis.pipeline import STAGE_NAMES
    from visual_synthesis.schema import BENCHMARK_LEVELS, RISK_LEVELS, TRANSFORMATIONS

    problems = []
    for dotted, allowed in CHOICES.items():
        section, key = dotted.split(".")
        if cfg[section][key] not in allowed:
            problems.append(f"{dotted}={cfg[section][key]!r} must be one of {list(allowed)}")

    for category, risk in cfg["taxonomy"].items():
        if risk not in RISK_LEVELS:
            problems.append(f"taxonomy.{category}={risk!r} must be one of {list(RISK_LEVELS)}")

    if cfg["generate"]["model"] not in cfg["models"]:
        problems.append(f"generate.model={cfg['generate']['model']!r} is not defined under 'models' "
                        f"(have: {sorted(cfg['models'])})")
    detector = cfg["nsfw"]["detector"]
    if detector not in DETECTOR_OFF and detector not in cfg["detectors"]:
        problems.append(f"nsfw.detector={detector!r} is not defined under 'detectors' "
                        f"(have: {sorted(cfg['detectors'])})")

    for stage in ("generate", "nsfw"):
        workers = cfg[stage]["workers"]
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            problems.append(f"{stage}.workers={workers!r} must be a positive integer")

    for kind in ("datasets", "models", "detectors"):
        for name in cfg["download"][kind]:
            if name not in cfg[kind]:
                problems.append(f"download.{kind} lists {name!r}, which is not defined under '{kind}'")

    for name, model in cfg["models"].items():
        if model["backend"] == "diffusers" and not model["repo_id"]:
            problems.append(f"models.{name}.repo_id is required")
        if model["quantization"] not in (None, *QUANTIZATION):
            problems.append(f"models.{name}.quantization must be null or one of {sorted(QUANTIZATION)}")
        if model["hook"]["cfg"] not in CFG_LAYOUTS:
            problems.append(f"models.{name}.hook.cfg must be one of {list(CFG_LAYOUTS)}")
        if model["hook"]["joint"] not in (None, *JOINT_LAYOUTS):
            problems.append(f"models.{name}.hook.joint must be null or one of {list(JOINT_LAYOUTS)}")

    intervention = cfg["generate"]["intervention"]
    if intervention["mode"] not in MODES:
        problems.append(f"generate.intervention.mode must be one of {list(MODES)}")
    if intervention["passes"] not in PASSES:
        problems.append(f"generate.intervention.passes must be one of {list(PASSES)}")
    vector = str(intervention["vector"])
    if vector not in ("safety", "probe") and not (vector.startswith("random:") and vector[7:].isdigit()):
        problems.append("generate.intervention.vector must be safety, probe or random:<seed>")
    start, end = intervention["steps"]
    if not 0 <= start < end <= 1:
        problems.append("generate.intervention.steps must be [start, end] with 0 <= start < end <= 1")
    if intervention["enabled"] and not intervention["e4_dir"]:
        problems.append("generate.intervention.e4_dir is required when the intervention is enabled")
    if cfg["metrics"]["combine"] not in ("or", "and"):
        problems.append("metrics.combine must be or / and")

    for name, model in cfg["metric_models"].items():
        where = f"metric_models.{name}"
        if model["kind"] not in ("clip", "fid", "iqa"):
            problems.append(f"{where}.kind must be clip, fid or iqa")
        if model["backend"] not in ("local", "api"):
            problems.append(f"{where}.backend must be local or api")
        if model["backend"] == "local" and model["kind"] in ("clip", "iqa") and not model["repo_id"]:
            problems.append(f"{where}.repo_id is required for a local CLIP evaluator")
        if model["backend"] == "api" and not model["endpoint"]:
            problems.append(f"{where}.endpoint is required for an API evaluator")

    for name in ("alignment", "fid", "quality"):
        evaluator = cfg["metrics"][name]
        if evaluator["enabled"] and evaluator["model"] not in cfg["metric_models"]:
            problems.append(f"metrics.{name}.model={evaluator['model']!r} is not defined under 'metric_models'")

    for name, dataset in cfg["datasets"].items():
        where = f"datasets.{name}"
        if not dataset["input_table"] and (not dataset["id_template"] or "{" not in dataset["id_template"]):
            problems.append(f"{where}.id_template needs a placeholder, e.g. 'I2P_{{index:05d}}'")
        if dataset["on_unknown_category"] not in ("raise", "warn_skip", "keep_as_is"):
            problems.append(f"{where}.on_unknown_category must be raise, warn_skip or keep_as_is")
        if dataset["benchmark_level"] not in BENCHMARK_LEVELS:
            problems.append(f"{where}.benchmark_level must be one of {list(BENCHMARK_LEVELS)}")
        if dataset["transformation"] not in TRANSFORMATIONS:
            problems.append(f"{where}.transformation must be one of {list(TRANSFORMATIONS)}")
        if not dataset["input_table"] and not dataset["category_map"] and dataset["on_unknown_category"] == "raise":
            problems.append(f"{where}.category_map is empty but on_unknown_category=raise; "
                            "every source label would abort the run")
        for label, category in dataset["category_map"].items():
            if category not in cfg["taxonomy"]:
                problems.append(f"{where}.category_map maps {label!r} to {category!r}, "
                                f"which is not in 'taxonomy' ({sorted(cfg['taxonomy'])})")

    for name in cfg["stages"]:
        if name not in STAGE_NAMES:
            problems.append(f"stages.{name}: unknown stage (known: {STAGE_NAMES})")
    return problems


# ------------------------------------------------------------------ lookups


def get_model(cfg, model_id=None):
    """The model entry for ``model_id`` (default: generate.model), with its id added."""
    model_id = model_id or cfg["generate"]["model"]
    if model_id not in cfg["models"]:
        raise KeyError(f"unknown model {model_id!r}; configured: {sorted(cfg['models'])}")
    return {"id": model_id, **cfg["models"][model_id]}


def get_detector(cfg, detector_id=None):
    """The detector entry (default: nsfw.detector), or None when detection is off."""
    detector_id = detector_id or cfg["nsfw"]["detector"]
    if detector_id in DETECTOR_OFF:
        return None
    if detector_id not in cfg["detectors"]:
        raise KeyError(f"unknown detector {detector_id!r}; configured: {sorted(cfg['detectors'])}")
    return {"id": detector_id, **cfg["detectors"][detector_id]}


def get_metric_model(cfg, model_id):
    """A configured evaluator model, with its id added."""
    if model_id not in cfg["metric_models"]:
        raise KeyError(f"unknown metric model {model_id!r}; configured: {sorted(cfg['metric_models'])}")
    return {"id": model_id, **cfg["metric_models"][model_id]}


def stage_setting(cfg, stage_name, key):
    """A run setting for one stage: ``stages.<name>.<key>`` if set, else ``run.<key>``."""
    value = cfg["stages"].get(stage_name, {}).get(key)
    return cfg["run"].get(key) if value is None else value


def dataset_dir(cfg, dataset_id):
    """Folder holding a dataset's raw files."""
    return Path(cfg["paths"]["raw"]) / (cfg["datasets"][dataset_id]["dir"] or dataset_id)


def config_diff(base, other, prefix=""):
    """Flat list of ``(key, before, after)`` for every leaf that differs."""
    changes = []
    for key in sorted(set(base) | set(other)):
        before, after = base.get(key, "<absent>"), other.get(key, "<absent>")
        if isinstance(before, dict) and isinstance(after, dict):
            changes += config_diff(before, after, f"{prefix}{key}.")
        elif before != after:
            changes.append((f"{prefix}{key}", before, after))
    return changes
