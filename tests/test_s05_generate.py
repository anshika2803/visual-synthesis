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


def test_attack_seeds_and_metadata_are_preserved(cfg, tmp_path):
    path = tmp_path / "stage2.csv"
    pd.DataFrame({
        "prompt_id": ["rab_a", "rab_b"], "prompt": ["a cat", "a dog"],
        "attack_config": ["K_16", "K_38"], "case_number": [4, 4],
        "evaluation_seed": [17, 91], "source_dataset": ["ring-a-bell"] * 2,
        "attack_method": ["Ring-A-Bell"] * 2, "source_file": ["a.csv", "b.csv"],
    }).to_csv(path, index=False)
    result = run_stage(cfg, s05_generate, input_path=path)
    table = pd.read_parquet(f"{result['run_dir']}/generations.parquet").set_index("prompt_id")
    assert table.loc["rab_a", "seed"] == 17
    assert table.loc["rab_b", "seed"] == 91
    assert table.loc["rab_a", "attack_config"] == "K_16"
    assert table.loc["rab_b", "case_number"] == "4"
    assert table.image_path.nunique() == 2
    assert "K_16" in Path(table.loc["rab_a", "image_path"]).name
    assert "K_38" in Path(table.loc["rab_b", "image_path"]).name


def test_generation_failure_is_recorded_and_retried_on_resume(cfg, tmp_path, monkeypatch):
    prompts = prompts_table(tmp_path, count=1)

    class FailedBackend:
        def __init__(self, model, models_root=None): pass
        def load(self): pass
        def generate(self, requests): raise RuntimeError("mock generation failure")
        def unload(self): pass

    monkeypatch.setattr(s05_generate, "build_image_backend", FailedBackend)
    first = run_stage(cfg, s05_generate, run_name="retry", input_path=prompts)
    first_table = pd.read_parquet(f"{first['run_dir']}/generations.parquet")
    assert first_table.loc[0, "generation_status"] == "error"
    assert "mock generation failure" in first_table.loc[0, "error"]

    monkeypatch.setattr(s05_generate, "build_image_backend", lambda model, root: FakeBackend(model))
    second = run_stage(cfg, s05_generate, run_name="retry", input_path=prompts)
    result = pd.read_parquet(f"{second['run_dir']}/generations.parquet")
    assert second["rows_out"] == 1
    assert result.loc[0, "generation_status"] == "success"


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


def test_mps_diffusers_copies_run_in_processes_not_threads(monkeypatch):
    """Torch on Metal aborts when two threads run pipelines, so MPS copies are processes."""
    diffusers = {"backend": "diffusers", "device": "auto", "device_index": None}
    monkeypatch.setattr(s05_generate, "pick_device", lambda preference, index: "mps")
    assert s05_generate.uses_processes(diffusers, 2)
    assert not s05_generate.uses_processes(diffusers, 1)  # a single copy stays in the main process
    assert not s05_generate.uses_processes({**diffusers, "backend": "fake"}, 2)

    monkeypatch.setattr(s05_generate, "pick_device", lambda preference, index: "cuda:0")
    assert not s05_generate.uses_processes(diffusers, 2)


def test_process_workers_generate_every_prompt_once(cfg, tmp_path, monkeypatch):
    """The process pool path: models are built in spawned workers, the parent writes the records."""
    cfg["generate"]["workers"] = 2
    monkeypatch.setattr(s05_generate, "uses_processes", lambda model, workers: True)

    result = run_stage(cfg, s05_generate, input_path=prompts_table(tmp_path, count=5))

    table = pd.read_parquet(f"{result['run_dir']}/generations.parquet")
    assert (result["worker_mode"], result["workers"], result["rows_out"]) == ("process", 2, 5)
    assert sorted(table["prompt_id"]) == [f"P{i}" for i in range(5)]
    assert all(Path(path).exists() for path in table["image_path"])
