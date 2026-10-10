from pathlib import Path
from threading import Barrier, Lock
from time import sleep

import pandas as pd

from visual_synthesis.backends.image import FakeBackend
from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s05_generate


def prompts_table(tmp_path, count=3):
    path = tmp_path / "prompts.parquet"
    pd.DataFrame({"prompt_id": [f"P{i}" for i in range(count)], "prompt": ["a cat"] * count}).to_parquet(path)
    return path


def test_one_image_is_generated_per_prompt(cfg, tmp_path):
    result = run_stage(cfg, s05_generate, input_path=prompts_table(tmp_path))

    table = pd.read_parquet(f"{result['run_dir']}/generations.parquet")
    assert list(table["prompt_id"]) == ["P0", "P1", "P2"]
    assert all(Path(path).exists() for path in table["image_path"])


def test_resumed_run_only_generates_the_missing_images(cfg, tmp_path):
    prompts = prompts_table(tmp_path)
    cfg["run"]["limit"] = 1
    first = run_stage(cfg, s05_generate, run_name="t", input_path=prompts)
    cfg["run"]["limit"] = None
    second = run_stage(cfg, s05_generate, run_name="t", input_path=prompts)

    assert (first["rows_out"], second["rows_out"], second["rows_skipped"]) == (1, 2, 1)
    assert len(pd.read_parquet(f"{second['run_dir']}/generations.parquet")) == 3


def test_configured_workers_use_separate_generation_backends(cfg, tmp_path, monkeypatch):
    """Each generation worker owns a simultaneously-loaded backend instance."""
    cfg["generate"]["workers"] = 3
    state = {"active": 0, "maximum": 0}
    state_lock = Lock()
    barrier = Barrier(3)

    class SlowFakeBackend:
        def __init__(self, model):
            self.delegate = FakeBackend(model)

        def load(self):
            self.delegate.load()
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            barrier.wait()

        def generate(self, requests):
            sleep(0.01)
            return self.delegate.generate(requests)

        def unload(self):
            self.delegate.unload()
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(s05_generate, "build_image_backend", lambda model, models_root: SlowFakeBackend(model))
    result = run_stage(cfg, s05_generate, input_path=prompts_table(tmp_path, count=6))

    table = pd.read_parquet(f"{result['run_dir']}/generations.parquet")
    assert result["workers"] == 3
    assert state["maximum"] == 3
    assert set(table["prompt_id"]) == {f"P{i}" for i in range(6)}


def test_mps_diffusers_are_limited_to_one_worker(monkeypatch):
    monkeypatch.setattr(s05_generate, "pick_device", lambda preference, index: "mps")

    workers = s05_generate.effective_worker_count(
        {"backend": "diffusers", "device": "auto", "device_index": None}, 3, [1, 2, 3]
    )

    assert workers == 1
