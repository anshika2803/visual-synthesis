"""Gradio web UI: generate images and inspect them for NSFW content.

    $ visual-synthesis ui        # http://127.0.0.1:7860

The detector dropdown has an "off" option; skipping detection is faster.
Images are written under ``<paths.serving>/ui/``.
"""

import logging
from pathlib import Path

from visual_synthesis.config import DETECTOR_OFF, get_model
from visual_synthesis.log import setup_logging
from visual_synthesis.serving.engine import Engine

log = logging.getLogger(__name__)


def build_ui(cfg, engine=None):
    """Build the Gradio app (not launched yet)."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("the web UI needs the 'ui' extra: pip install -e '.[ui]'") from exc

    engine = engine or Engine(cfg)
    out_dir = Path(cfg["paths"]["serving"]) / "ui"
    default_model = get_model(cfg)
    default_detector = "off" if cfg["nsfw"]["detector"] in DETECTOR_OFF else cfg["nsfw"]["detector"]
    detector_ids = ["off"] + [name for name, d in cfg["detectors"].items() if d["enabled"]]

    def generate(prompt, negative, model_id, steps, guidance, width, height, seed, count, detector_id, threshold):
        if not prompt or not prompt.strip():
            return None, "Enter a prompt first.", ""
        try:
            result = engine.generate(
                prompt.strip(), out_dir, model=model_id, n=int(count),
                seed=None if seed is None or int(seed) < 0 else int(seed),  # -1 = random
                negative_prompt=negative.strip() or None, steps=int(steps), guidance_scale=float(guidance),
                width=int(width), height=int(height), detector=detector_id, threshold=float(threshold),
            )
        except Exception as exc:  # show the problem in the page instead of a traceback
            log.exception("generation failed")
            return None, f"Generation failed: {exc}", ""

        details = (f"model `{result['model']}` · {result['width']}×{result['height']} · steps {result['steps']} · "
                   f"seed `{result['seed']}` · {result['generation_time_s']}s")
        if result["detections"] is None:
            return result["images"], details, "**Detection off.**"
        rows = "\n".join(
            f"| {Path(d['image_path']).name} | {'🚩 yes' if d['flagged'] else 'no'} | {d['max_score']} | "
            f"{', '.join(d['labels'][:4])} |"
            for d in result["detections"]
        )
        verdicts = (f"**Detector** `{detector_id}` · **threshold** {threshold}\n\n"
                    f"| image | flagged | max score | labels |\n|---|---|---|---|\n{rows}")
        return result["images"], details, verdicts

    with gr.Blocks(title=cfg["ui"]["title"]) as app:
        gr.Markdown(f"# {cfg['ui']['title']}\nGenerate images and inspect them for NSFW content.")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a serene mountain lake at sunrise")
                negative = gr.Textbox(label="Negative prompt", placeholder="optional")
                model_id = gr.Dropdown(list(cfg["models"]), value=default_model["id"], label="Model")
                detector_id = gr.Dropdown(detector_ids, value=default_detector, label="NSFW detector",
                                          info="Choose 'off' to skip detection (faster).")
                button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                steps = gr.Slider(1, 100, value=default_model["num_inference_steps"], step=1, label="Steps")
                guidance = gr.Slider(0, 20, value=default_model["guidance_scale"], step=0.1, label="Guidance")
                width = gr.Slider(256, 1536, value=default_model["width"], step=64, label="Width")
                height = gr.Slider(256, 1536, value=default_model["height"], step=64, label="Height")
                seed = gr.Number(value=cfg["run"]["seed"], precision=0, label="Seed (-1 for random)")
                count = gr.Slider(1, cfg["ui"]["max_batch"], value=1, step=1, label="Images")
                threshold = gr.Slider(0, 1, value=0.5, step=0.05, label="Flag threshold")
        gallery = gr.Gallery(label="Result", columns=2, height=512)
        details = gr.Markdown()
        verdicts = gr.Markdown()

        def model_defaults(chosen):
            """Move the sliders to the chosen model's own settings."""
            model = get_model(cfg, chosen)
            return model["num_inference_steps"], model["guidance_scale"], model["width"], model["height"]

        model_id.change(model_defaults, inputs=model_id, outputs=[steps, guidance, width, height])
        inputs = [prompt, negative, model_id, steps, guidance, width, height, seed, count, detector_id, threshold]
        outputs = [gallery, details, verdicts]
        button.click(generate, inputs=inputs, outputs=outputs)
        prompt.submit(generate, inputs=inputs, outputs=outputs)
    return app


def run_ui(cfg, host=None, port=None, share=None):
    """Launch the UI. CLI values override the ``ui`` config."""
    setup_logging(cfg["logging"]["level"], console_json=cfg["logging"]["console_json"], stage="ui")
    engine = Engine(cfg)
    ui = cfg["ui"]
    try:
        build_ui(cfg, engine).launch(
            server_name=host or ui["host"],
            server_port=port or ui["port"],
            share=ui["share"] if share is None else share,
        )
    finally:
        engine.close()
