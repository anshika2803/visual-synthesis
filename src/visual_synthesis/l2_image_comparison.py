"""Matched-seed image comparison for original text and two paraphrasing instructions."""

from __future__ import annotations

import argparse
import gc
import html
import json
import sys
from pathlib import Path

from visual_synthesis.backends.detectors import build_detector, is_flagged
from visual_synthesis.backends.image import build_image_backend, save_image
from visual_synthesis.config import get_detector, get_model, load_config
from visual_synthesis.l2_pair_review import SceneJudge, combine
from visual_synthesis.l2_paraphrase import (
    OUTPUT_SCHEMA,
    ROOT,
    SYSTEM_PROMPT,
    OllamaClient,
    OllamaError,
    _atomic_text,
    _csv_text,
    _digest,
    _run_lock,
    _write_json,
    load_run,
    load_settings,
    parse_response,
)
from visual_synthesis.l2_verification import DebertaNLI
from visual_synthesis.stages.s07_metrics import feature_tensor

DEFAULT_PROMPT = ROOT / "config/prompts/l2_rephrase_v2.txt"
VARIANTS = ("original", "v1", "v2")


def load_sources(source_run, limit):
    manifest, parents, attempts = load_run(source_run)
    if manifest["run_kind"] not in {"benign_smoke", "generation_classification_smoke"}:
        raise ValueError("This initial image-comparison command accepts benign software-test runs only.")
    rows = [
        {
            "parent_prompt_id": p["parent_prompt_id"],
            "original": p["original_prompt"],
            "v1": attempts[p["parent_prompt_id"]]["paraphrased_prompt"],
        }
        for p in parents
        if attempts.get(p["parent_prompt_id"], {}).get("generation_status") == "produced"
    ]
    if not rows:
        raise ValueError("Source run has no generated pairs.")
    return rows[:limit]


def image_requests(pairs, seeds):
    return [
        {
            "parent_prompt_id": pair["parent_prompt_id"],
            "variant": variant,
            "prompt_id": f"{pair['parent_prompt_id']}_{variant}",
            "prompt": pair[variant],
            "reference_prompt": pair["original"],
            "seed": seed,
        }
        for pair in pairs
        for seed in seeds
        for variant in VARIANTS
        if pair.get(variant)
    ]


def clip_snapshot():
    base = Path.home() / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots"
    for folder in sorted(base.glob("*")):
        if (
            (folder / "config.json").exists()
            and (folder / "preprocessor_config.json").exists()
            and any((folder / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
        ):
            return folder
    raise ValueError("No complete cached CLIP ViT-B/32 snapshot found; no model is downloaded automatically.")


def score_images(rows, detector_config, models_root, clip_path):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    detector = build_detector(detector_config, models_root)
    detector.load()
    try:
        for row in rows:
            scores = detector.detect(row["image_path"])
            selected = {key: val for key, val in scores.items() if key in detector_config["flag_labels"]}
            row.update(
                nudity_flagged=is_flagged(scores, detector_config),
                max_nudity_detection_score=max(selected.values(), default=0.0),
                detector_scores=scores,
            )
    finally:
        detector.unload()
    processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).to("cpu").eval()
    try:
        with torch.inference_mode():
            for row in rows:
                with Image.open(row["image_path"]) as image:
                    # Every variant is scored against the ORIGINAL reference, not its own wording.
                    token_count = len(processor.tokenizer(row["reference_prompt"])["input_ids"])
                    if token_count > 77:
                        row.update(clip_alignment_to_original=None, alignment_status="reference_exceeds_77_tokens")
                        continue
                    inputs = processor(
                        text=[row["reference_prompt"]],
                        images=[image.convert("RGB")],
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                    )
                    image_features = feature_tensor(clip.get_image_features(pixel_values=inputs["pixel_values"]))
                    text_features = feature_tensor(
                        clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    )
                    score = torch.nn.functional.cosine_similarity(image_features, text_features)[0].item()
                    row.update(clip_alignment_to_original=float(score), alignment_status="ok")
    finally:
        del clip, processor
        gc.collect()
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["parent_prompt_id"], row["seed"]), {})[row["variant"]] = row
    comparisons = []
    for (parent_id, seed), group in groups.items():
        if not set(VARIANTS).issubset(group):
            continue
        a, b, c = (group[key].get("clip_alignment_to_original") for key in VARIANTS)
        comparisons.append(
            {
                "parent_prompt_id": parent_id,
                "seed": seed,
                "original_alignment": a,
                "v1_alignment": b,
                "v2_alignment": c,
                "v2_minus_original": c - a if c is not None and a is not None else None,
                "v2_minus_v1": c - b if c is not None and b is not None else None,
            }
        )
    differences = [row["v2_minus_v1"] for row in comparisons if row["v2_minus_v1"] is not None]
    variant_scores = {
        variant: [
            row["clip_alignment_to_original"]
            for row in rows
            if row["variant"] == variant and row.get("clip_alignment_to_original") is not None
        ]
        for variant in VARIANTS
    }
    return {
        "images_scored": len(rows),
        "complete_matched_comparisons": len(comparisons),
        "v2_higher_alignment_than_v1": sum(value > 0 for value in differences),
        "v2_lower_alignment_than_v1": sum(value < 0 for value in differences),
        "v2_equal_alignment_to_v1": sum(value == 0 for value in differences),
        "mean_v2_minus_v1": sum(differences) / len(differences) if differences else None,
        "nudity_flag_counts": {
            variant: sum(row["nudity_flagged"] for row in rows if row["variant"] == variant) for variant in VARIANTS
        },
        "mean_alignment_by_variant": {
            variant: sum(values) / len(values) if values else None for variant, values in variant_scores.items()
        },
        "general_visual_quality_measured": False,
        "human_validated": False,
        "scope": "Small benign comparison; CLIP alignment is not an aesthetic-quality score. "
        "NudeNet does not assess all six project risk categories. No harmful-category robustness conclusion.",
        "comparisons": comparisons,
    }


def write_report(output, rows, summary):
    columns = [
        "parent_prompt_id",
        "variant",
        "seed",
        "prompt",
        "reference_prompt",
        "text_is_l2",
        "image_path",
        "clip_alignment_to_original",
        "nudity_flagged",
        "max_nudity_detection_score",
        "alignment_status",
    ]
    _atomic_text(output / "image_results.csv", _csv_text(rows, columns))
    _write_json(output / "image_results.json", rows)
    _write_json(output / "summary.json", summary)
    _atomic_text(
        output / "paired_scores.csv",
        _csv_text(
            summary["comparisons"],
            [
                "parent_prompt_id",
                "seed",
                "original_alignment",
                "v1_alignment",
                "v2_alignment",
                "v2_minus_original",
                "v2_minus_v1",
            ],
        ),
    )
    cards = []
    for parent_id, seed in sorted({(row["parent_prompt_id"], row["seed"]) for row in rows}):
        group = {row["variant"]: row for row in rows if row["parent_prompt_id"] == parent_id and row["seed"] == seed}
        cells = []
        for variant in VARIANTS:
            if variant not in group:
                continue
            row = group[variant]
            alignment = row.get("clip_alignment_to_original")
            alignment_text = f"{alignment:.4f}" if alignment is not None else "unavailable"
            relative = Path(row["image_path"]).relative_to(output).as_posix()
            cells.append(
                f'<article><h3>{html.escape(variant)}</h3><img src="{html.escape(relative)}">'
                f"<p>{html.escape(row['prompt'])}</p><p>Text L2: {html.escape(row['text_is_l2'])}<br>"
                f"Alignment to original: {alignment_text}<br>Nudity flag: {row['nudity_flagged']}</p></article>"
            )
        cards.append(
            f'<section><h2>{html.escape(parent_id)} · image seed {seed}</h2><div class="grid">'
            + "".join(cells)
            + "</div></section>"
        )
    document = (
        '<!doctype html><meta charset="utf-8"><title>Original and paraphrased image comparison</title>'
        "<style>body{font:16px system-ui;margin:32px;background:#f5f6f8;color:#17283b;max-width:1500px}"
        ".grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}"
        "article{background:white;padding:16px;border-radius:8px}img{width:100%;height:auto}"
        "h2{margin-top:32px}p{line-height:1.5}</style><h1>Original and paraphrased image comparison</h1>"
        "<p>original = source caption; v1 = previous Qwen instruction; v2 = revised instruction. "
        "Same image model and seed within each row. Alignment uses the original caption for every image.</p>"
        f"<p>{html.escape(summary['scope'])}</p>" + "".join(cards)
    )
    _atomic_text(output / "report.html", document)


def make_contact_sheets(output, rows):
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    parent_ids = sorted({row["parent_prompt_id"] for row in rows})
    for seed in sorted({row["seed"] for row in rows}):
        fig, axes = plt.subplots(len(parent_ids), 3, figsize=(12, 4.7 * len(parent_ids)), squeeze=False)
        for i, parent_id in enumerate(parent_ids):
            group = {
                row["variant"]: row for row in rows if row["parent_prompt_id"] == parent_id and row["seed"] == seed
            }
            for j, variant in enumerate(VARIANTS):
                ax = axes[i][j]
                ax.set_axis_off()
                if variant not in group:
                    continue
                row = group[variant]
                with Image.open(row["image_path"]) as image:
                    ax.imshow(image.convert("RGB"))
                score = row.get("clip_alignment_to_original")
                score_text = f"{score:.3f}" if score is not None else "n/a"
                title = f"{parent_id} · {variant}\n" + "\n".join(textwrap.wrap(row["prompt"], 48))
                ax.set_title(title, fontsize=9, loc="left")
                ax.text(
                    0,
                    -0.06,
                    f"Alignment to original: {score_text} · nudity flag: {row['nudity_flagged']}",
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="left",
                    va="top",
                )
        fig.suptitle(f"Original / previous rewrite / revised rewrite · matched image seed {seed}", fontsize=12)
        fig.subplots_adjust(hspace=0.45, wspace=0.1, top=0.91, bottom=0.05)
        fig.savefig(output / f"comparison_seed_{seed}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def run_comparison(source_run, output, prompt_file, seeds, limit, image_model, device):
    pairs = load_sources(source_run, limit)
    revised_system = Path(prompt_file).read_text(encoding="utf-8")
    settings = load_settings()
    client = OllamaClient(settings)
    cfg = load_config(use_env=False)
    model = get_model(cfg, image_model)
    model.update(device=device, dtype="float16", enable_attention_slicing=True)
    backend = build_image_backend(model, cfg["paths"]["models"])
    if not Path(backend.weights_source()).is_dir():
        raise ValueError("Image model is not cached locally; this comparison never downloads image weights.")
    clip_path = clip_snapshot()
    detector_config = get_detector(cfg, "nudenet")
    output.mkdir(parents=True, exist_ok=True)
    with _run_lock(output):
        metadata = {
            "source_pairs": pairs,
            "source_pairs_sha256": _digest(pairs),
            "revised_prompt": revised_system,
            "previous_prompt": SYSTEM_PROMPT,
            "qwen": client.identity(),
            "qwen_options": settings["options"],
            "image_seeds": seeds,
            "image_model": model,
            "image_weights": str(Path(backend.weights_source()).resolve()),
            "clip_snapshot": str(clip_path),
            "detector": detector_config,
            "driver_sha256": _digest(Path(__file__).read_bytes()),
        }
        manifest_path = output / "experiment.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
            raise ValueError("Inputs or comparison settings changed. Select a new --output folder.")
        _write_json(manifest_path, metadata)
        verifier, judge = DebertaNLI(), SceneJudge(settings)
        try:
            for pair in pairs:
                path = output / "text" / f"{pair['parent_prompt_id']}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                else:
                    response = client.request(
                        "/api/chat",
                        {
                            "model": settings["model"],
                            "stream": False,
                            "format": OUTPUT_SCHEMA,
                            "options": settings["options"],
                            "keep_alive": "5m",
                            "messages": [
                                {"role": "system", "content": revised_system},
                                {"role": "user", "content": json.dumps({"description": pair["original"]})},
                            ],
                        },
                    )
                    record = {"response": response, **parse_response(response)}
                    record["v1_check"] = combine(pair["original"], pair["v1"], verifier, judge)
                    if record["generation_status"] == "produced":
                        record["v2_check"] = combine(pair["original"], record["paraphrased_prompt"], verifier, judge)
                    _write_json(path, record)
                pair["v2"] = record["paraphrased_prompt"] if record["generation_status"] == "produced" else ""
                pair["v1_check"] = record["v1_check"]["label"]
                pair["v2_check"] = record.get("v2_check", {}).get("label", record["generation_status"])
                print(f"Text {pair['parent_prompt_id']}: v1={pair['v1_check']}, v2={pair['v2_check']}", flush=True)
        finally:
            del verifier, judge
            gc.collect()
            client.request("/api/chat", {"model": settings["model"], "messages": [], "stream": False, "keep_alive": 0})
        _write_json(output / "text_pairs.json", pairs)
        requests = image_requests(pairs, seeds)
        rows = []
        try:
            for index, request in enumerate(requests, 1):
                key = f"{request['prompt_id']}_{request['seed']}"
                record_path = output / "image_records" / f"{key}.json"
                if record_path.exists():
                    row = json.loads(record_path.read_text())
                    if not Path(row["image_path"]).exists():
                        raise ValueError(f"Saved image is missing: {row['image_path']}")
                else:
                    produced = backend.generate([request])[0]
                    image_path = output / "images" / f"{key}.png"
                    save_image(produced["image"], image_path)
                    produced["image"].close()
                    pair = next(pair for pair in pairs if pair["parent_prompt_id"] == request["parent_prompt_id"])
                    row = {
                        **request,
                        "image_path": str(image_path),
                        "generation_seconds": produced["seconds"],
                        "text_is_l2": "original"
                        if request["variant"] == "original"
                        else pair[request["variant"] + "_check"],
                    }
                    _write_json(record_path, row)
                rows.append(row)
                print(f"Image [{index}/{len(requests)}] {key}", flush=True)
        finally:
            backend.unload()
        rows = score_images(rows, detector_config, cfg["paths"]["models"], clip_path)
        summary = summarize(rows)
        summary["text_pair_outcomes"] = [
            {"parent_prompt_id": pair["parent_prompt_id"], "v1": pair["v1_check"], "v2": pair["v2_check"]}
            for pair in pairs
        ]
        write_report(output, rows, summary)
        make_contact_sheets(output, rows)
        print(f"Scored {len(rows)} real images; report: {output / 'report.html'}")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "results/runs/local/l2-fresh-check")
    parser.add_argument("--output", type=Path, default=ROOT / "results/runs/local/l2-image-comparison-v2")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--image-model", default="sd-turbo")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args(argv)
    if args.limit <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--limit must be positive and --seeds must be distinct.")
    try:
        run_comparison(
            args.source_run.resolve(),
            args.output.resolve(),
            args.prompt_file,
            args.seeds,
            args.limit,
            args.image_model,
            args.device,
        )
        return 0
    except (ValueError, KeyError, OSError, ImportError, RuntimeError, OllamaError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped. Repeat the command with unchanged inputs to resume saved text and images.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
