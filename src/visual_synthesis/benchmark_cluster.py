"""Prepare existing inputs, prefetch weights, and check an allocated cluster node."""

import argparse
import json
from pathlib import Path

from visual_synthesis.backends.image import PIPELINES, DiffusersBackend
from visual_synthesis.benchmark_sampling import ATTACK_FIELDS, SAMPLING, load_prompts, sample_prompts
from visual_synthesis.config import ROOT, get_detector, get_model, load_config
from visual_synthesis.io import file_sha256, safe_name, write_json

DEFAULT_INPUT = ROOT / "data/processed/unified/unified_prompt_candidates.csv"
DEFAULT_CONFIGS = [ROOT / "config/pipeline.json", ROOT / "config/benchmark.models.json",
                   ROOT / "config/benchmark.hpc.json"]


def prepare_inputs(input_path, output, sample_size=60, seed=42, sampling="auto"):
    """Export source rows and metadata already present in the tracked manifest."""
    frame = load_prompts(input_path)
    sample, info = sample_prompts(frame, sample_size, seed, sampling)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sample.to_csv(output / "pilot_sample.csv", index=False)
    outputs = {"pilot_sample": str(output / "pilot_sample.csv")}
    sources = frame.get("source_dataset")
    if sources is not None:
        ring = frame[sources.eq("ring-a-bell")]
        if not ring.empty:
            if any(field not in ring or ring[field].isna().any() for field in ATTACK_FIELDS):
                raise ValueError("the manifest does not retain all Ring-A-Bell attack fields")
            path = output / "ring_a_bell_standardized.csv"
            ring.to_csv(path, index=False)
            outputs["ring_a_bell"] = str(path)
        l1 = frame[sources.isin(["I2P", "T2I-RiskyPrompt"]) & frame.benchmark_level.eq("L1")]
        if not l1.empty:
            path = output / "unified_benchmark_l1.csv"
            l1.to_csv(path, index=False)
            outputs["l1"] = str(path)
    report = {"source": str(Path(input_path).resolve()), "source_sha256": file_sha256(input_path),
              "sampling": info, "outputs": outputs,
              "note": "Existing source text only; no new attacks, transformations or human validation created."}
    write_json(report, output / "input_report.json")
    return report


def _check_model_files(model, root):
    index = root / "model_index.json"
    if not index.is_file():
        raise ValueError(f"missing {index}; run the prefetch command")
    for component, definition in json.loads(index.read_text()).items():
        if component.startswith("_") or not isinstance(definition, list) or not definition[0]:
            continue
        if component == "safety_checker" and not model["safety_checker"]:
            continue
        folder = root / component
        if not folder.is_dir() or not any(folder.iterdir()):
            raise ValueError(f"missing component {folder}; rerun prefetch to complete the download")
        kind = str(definition[-1])
        if any(word in kind for word in ("Tokenizer", "Scheduler", "Processor", "FeatureExtractor")):
            continue
        weights = [p for p in folder.iterdir() if p.suffix in (".safetensors", ".bin", ".pt")]
        variant = model["variant"]
        if variant:
            weights = [p for p in weights if f".{variant}." in p.name]
        if not weights:
            raise ValueError(f"no {variant or 'model'} weights in {folder}; rerun prefetch")
        indexes = list(folder.glob("*.index.json"))
        if variant:
            indexes = [p for p in indexes if f".{variant}." in p.name]
        # A repository can contain index files for several formats. A complete
        # matching format is sufficient; do not demand every optional variant.
        if indexes and not any(all((folder / name).is_file() for name in
                                   set(json.loads(p.read_text()).get("weight_map", {}).values())) for p in indexes):
            raise ValueError(f"incomplete weight shards in {folder}; rerun prefetch")


def check_environment(cfg, model_ids, input_path, device="cuda", require_cache=False,
                      sample_size=60, seed=42, sampling="auto", generation_only=False):
    """Inspect inputs, imports, CUDA and cached files without loading model weights."""
    checks = []

    def check(name, operation):
        try:
            detail = operation()
            checks.append({"name": name, "status": "ok", "detail": detail})
        except Exception as exc:
            checks.append({"name": name, "status": "error", "detail": f"{type(exc).__name__}: {exc}"})

    def inputs():
        sample, info = sample_prompts(load_prompts(input_path), sample_size, seed, sampling)
        return {**info, "sample_rows": len(sample), "input": str(input_path)}

    check("input and sampling", inputs)

    def hardware():
        import torch

        if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 6):
            raise RuntimeError("the pinned Transformers needs PyTorch 2.6+; activate a compatible CUDA environment")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; run this check inside an allocated GPU job with CUDA PyTorch")
            return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "bfloat16_supported": torch.cuda.is_bf16_supported()}
        return {"torch": torch.__version__, "device": "cpu (prefetch/import check only)"}

    check("PyTorch and device", hardware)
    for model_id in model_ids:
        model = get_model(cfg, model_id)

        def pipeline(model=model):
            if model["backend"] == "fake":
                return "synthetic backend"
            import diffusers

            if device == "cuda" and model["dtype"] == "bfloat16":
                import torch

                if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                    raise RuntimeError(
                        "this model requests bfloat16; use a supported GPU or a validated dtype override"
                    )
            return {"class": getattr(diffusers, PIPELINES[model["family"]]).__name__,
                    "diffusers": diffusers.__version__}

        check(f"pipeline {model_id}", pipeline)
        if require_cache and model["backend"] != "fake":
            def cached(model=model):
                from huggingface_hub import snapshot_download

                source = DiffusersBackend(model, cfg["paths"]["models"]).weights_source()
                root = Path(source) if Path(source).is_dir() else Path(snapshot_download(
                    model["repo_id"], revision=model["revision"], local_files_only=True
                ))
                _check_model_files(model, root)
                return str(root)

            check(f"cached weights {model_id}", cached)
    if not generation_only:
        detector = get_detector(cfg)

        def detection():
            if detector is None:
                raise ValueError("select a detector or use --generation-only")
            if detector["backend"] == "nudenet":
                import nudenet

                if not list(Path(nudenet.__file__).parent.glob("*.onnx")):
                    raise RuntimeError("NudeNet's bundled ONNX model is missing; reinstall nudenet")
            elif detector["backend"] != "fake":
                raise ValueError("cluster preflight currently supports nudenet; select it in the cluster profile")
            return detector["id"]

        check("detector", detection)
        if cfg["metrics"]["alignment"]["enabled"]:
            def clip():
                from transformers import CLIPModel, CLIPProcessor

                evaluator = cfg["metric_models"][cfg["metrics"]["alignment"]["model"]]
                if evaluator["backend"] != "local":
                    return "API evaluator configured; network health is not checked"
                if require_cache:
                    from huggingface_hub import snapshot_download

                    folder = Path(snapshot_download(evaluator["repo_id"], local_files_only=True))
                    if not any((folder / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
                        raise RuntimeError("CLIP evaluator weights are missing; run prefetch")
                return f"{CLIPModel.__name__} / {CLIPProcessor.__name__}"

            check("alignment evaluator", clip)
    return {"status": "ok" if all(c["status"] == "ok" for c in checks) else "error", "checks": checks}


def prefetch(cfg, model_ids, generation_only=False):
    """Fill selected model folders and the HF evaluator cache, including partial downloads."""
    from huggingface_hub import snapshot_download

    paths = {}
    for model_id in model_ids:
        model = get_model(cfg, model_id)
        if model["backend"] == "fake":
            continue
        root = Path(cfg["paths"]["models"]) / safe_name(model["local_dir"] or model_id)
        print(f"Prefetching {model_id} -> {root}", flush=True)
        snapshot_download(model["repo_id"], revision=model["revision"], local_dir=root,
                          allow_patterns=model["allow_patterns"] or None)
        _check_model_files(model, root)
        paths[model_id] = str(root)
    if not generation_only and cfg["metrics"]["alignment"]["enabled"]:
        evaluator = cfg["metric_models"][cfg["metrics"]["alignment"]["model"]]
        if evaluator["backend"] == "local":
            print(f"Prefetching evaluator {evaluator['repo_id']}", flush=True)
            paths["alignment"] = snapshot_download(evaluator["repo_id"], allow_patterns=[
                "*.json", "*.txt", "model.safetensors", "pytorch_model.bin"
            ])
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "prefetch", "prepare-inputs"])
    parser.add_argument("-c", "--config", action="append", dest="configs")
    parser.add_argument("-m", "--model", action="append", dest="models")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--sampling", choices=SAMPLING, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-inputs":
            report = prepare_inputs(args.input, args.output or ROOT / "results/benchmarks/cluster-inputs",
                                    args.sample_size, args.seed, args.sampling)
        else:
            cfg = load_config(args.configs or DEFAULT_CONFIGS)
            models = args.models or cfg["download"]["models"]
            if args.command == "prefetch":
                load_prompts(args.input)  # Detect missing/LFS-pointer input before costly downloads.
                report = {"prefetched": prefetch(cfg, models, args.generation_only)}
            else:
                report = check_environment(cfg, models, args.input, args.device, args.require_cache,
                                           args.sample_size, args.seed, args.sampling, args.generation_only)
            if args.output:
                write_json(report, args.output)
        print(json.dumps(report, indent=2))
        return 1 if report.get("status") == "error" else 0
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
