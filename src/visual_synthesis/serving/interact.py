"""Interactive terminal session: type a prompt, get an image and its NSFW verdict.

    $ visual-synthesis interact --model sd-turbo --detector off

Slash commands change the settings without restarting; ``/help`` lists them.
"""

import shutil
import uuid
from pathlib import Path

from visual_synthesis.config import DETECTOR_OFF
from visual_synthesis.log import setup_logging
from visual_synthesis.serving.engine import Engine

HELP = """\
  <any text>            generate an image for that prompt
  /model <id>           switch model          /models      list models
  /detector <id>|off    switch NSFW detector  /detectors   list detectors
  /threshold <0..1>     flagging threshold
  /steps <n>            inference steps
  /guidance <number>    guidance scale
  /size <w>x<h>         image size
  /seed <n>             base seed (each image uses the next one)
  /negative <text>      negative prompt (empty to clear)
  /save <path>          copy the last image somewhere
  /history              recent generations
  /help                 this text
  /quit                 exit"""


def run_interactive(cfg, model=None, detector=None, steps=None, width=None, height=None, seed=None,
                    threshold=None, save_dir=None):
    """Run the prompt loop until /quit or Ctrl-D. Returns the process exit code."""
    setup_logging("warning", stage="interact")
    engine = Engine(cfg)
    out_dir = Path(save_dir) if save_dir else Path(cfg["paths"]["serving"]) / f"interact_{uuid.uuid4().hex[:8]}"
    detector = detector or cfg["nsfw"]["detector"]
    # The session's current settings; slash commands edit this dict.
    state = {
        "model": model or cfg["generate"]["model"],
        "detector": "off" if detector in DETECTOR_OFF else detector,
        "threshold": threshold,
        "steps": steps,
        "guidance": None,
        "width": width,
        "height": height,
        "seed": cfg["run"]["seed"] if seed is None else seed,
        "negative": None,
    }
    history = []
    print(f"model: {state['model']}   detector: {state['detector']}   images: {out_dir}")
    print("type a prompt to generate, /help for commands, /quit to exit")

    try:
        while True:
            try:
                line = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                command, _, argument = line[1:].partition(" ")
                if command in ("quit", "exit", "q"):
                    break
                print(handle_command(command.lower(), argument.strip(), state, history, cfg))
                continue
            try:
                state["seed"] += 1
                result = engine.generate(
                    line, out_dir, model=state["model"], prompt_id=f"interact_{len(history) + 1:05d}",
                    seed=state["seed"], negative_prompt=state["negative"], steps=state["steps"],
                    guidance_scale=state["guidance"], width=state["width"], height=state["height"],
                    detector=state["detector"], threshold=state["threshold"],
                )
            except Exception as exc:  # a REPL should report, not crash
                print(f"error: {exc}")
                continue
            history.append({"prompt": line, "seed": result["seed"], "path": result["images"][0]})
            print(f"saved {result['images'][0]} ({result['width']}x{result['height']}, "
                  f"seed={result['seed']}, {result['generation_time_s']:.2f}s)")
            for detection in result["detections"] or []:
                print(f"NSFW detector={detection['detector']} threshold={detection['threshold']} "
                      f"flagged={detection['flagged']}")
                for label in detection["labels"][:6]:
                    print(f"    {label:28} {detection['per_class'][label]:.4f}")
    finally:
        engine.close()
    print("bye")
    return 0


def handle_command(command, argument, state, history, cfg):
    """Apply one slash command to ``state`` and return the line to print."""
    if command in ("help", "h", "?"):
        return HELP
    if command == "models":
        return "\n".join(f"  {name}  {m['repo_id'] or m['backend']}" for name, m in cfg["models"].items())
    if command == "detectors":
        return "\n".join(
            f"  {name}  {d['backend']} (threshold={d['threshold']})" for name, d in cfg["detectors"].items()
        )
    if command == "history":
        lines = [f"  seed={h['seed']}  {h['prompt'][:60]}  {h['path']}" for h in history[-10:]]
        return "\n".join(lines) or "no history yet"
    if command == "model":
        if argument not in cfg["models"]:
            return f"unknown model {argument!r}; try /models"
        state["model"] = argument
    elif command == "detector":
        if argument not in cfg["detectors"] and argument not in DETECTOR_OFF:
            return f"unknown detector {argument!r}; try /detectors"
        state["detector"] = "off" if argument in DETECTOR_OFF else argument
    elif command == "negative":
        state["negative"] = argument or None
    elif command == "save":
        if not history:
            return "nothing generated yet"
        if not argument or Path(argument).exists():
            return "usage: /save <path that does not exist yet>"
        Path(argument).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history[-1]["path"], argument)
        return f"saved {argument}"
    elif command in ("steps", "seed", "threshold", "guidance", "size"):
        try:
            if command == "size":
                state["width"], state["height"] = (int(part) for part in argument.lower().split("x"))
            else:
                state[command] = int(argument) if command in ("steps", "seed") else float(argument)
        except ValueError:
            return f"usage: /{command} " + ("512x512" if command == "size" else "<number>")
    else:
        return f"unknown command /{command}; try /help"
    return f"{command} -> {argument or '(cleared)'}"
