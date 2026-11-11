"""Opt-in single-node PBS dispatcher for the frozen GLM protocol.

Scientific work remains in the existing CLI, E4 and reporting scripts. One
parent owns the serial launcher's model lock; external processes get distinct
PBS-assigned GPUs and distinct canonical run names. No prompt sharding/merging.
"""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.config import get_model, load_config
from visual_synthesis.io import read_table, write_json

MODEL = "glm-image"
PHASES = ("all", "initial", "e1", "e7", "e4", "e5", "e6", "report")
MODEL_KEYS = (
    "backend",
    "repo_id",
    "revision",
    "variant",
    "family",
    "dtype",
    "width",
    "height",
    "num_inference_steps",
    "num_images_per_prompt",
    "guidance_scale",
    "negative_prompt",
    "scheduler",
    "generation_kwargs",
    "hook",
    "quantization",
    "quantize_components",
    "enable_attention_slicing",
    "enable_xformers",
    "enable_cpu_offload",
    "safety_checker",
)


class IncompleteRun(Exception):
    """Valid partial work can be resumed by the existing incremental CLI."""


class ResumeConflict(ValueError):
    """Existing records are incompatible or damaged; do not silently reuse them."""


class CommandFailed(RuntimeError):
    def __init__(self, label, code):
        self.code = code if code > 0 else 128 - code
        super().__init__(f"{label} exited {self.code}; inspect its task log")


@dataclass(frozen=True)
class Task:
    name: str
    phase: str
    deps: tuple = ()
    mode: str | None = None
    vector: str | None = None
    alpha: float | None = None

    def run_name(self, e6_mode=None):
        if self.phase == "e4":
            return f"e4-{MODEL}-capture"
        if self.phase in ("e1", "e7"):
            return f"{self.phase}-{MODEL}"
        mode = self.mode or e6_mode
        if mode not in ("ablation", "steering"):
            raise ResumeConflict("E6 mode must come from the complete E5 results")
        return f"{self.phase}-{MODEL}-{mode}-{self.vector.replace(':', '')}-a{self.alpha}"


def graph(phase):
    initial = [Task(p, p) for p in ("e1", "e7", "e4")]
    e5 = [
        Task(f"e5-{m}-{v}-{a}", "e5", ("e4",) if phase == "all" else (), m, v, a)
        for m in ("ablation", "steering")
        for v in ("safety", "random:0")
        for a in (0.5, 1.0)
    ]
    e6 = [
        Task(f"e6-{v}-{a}", "e6", tuple(t.name for t in e5) if phase == "all" else (), vector=v, alpha=a)
        for v in ("safety", "random:0")
        for a in (0.25, 0.75)
    ]
    return (
        initial + e5 + e6
        if phase == "all"
        else initial
        if phase == "initial"
        else {"e1": initial[:1], "e7": initial[1:2], "e4": initial[2:], "e5": e5, "e6": e6, "report": []}[phase]
    )


@contextmanager
def model_lock(runs):
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f".aspire-{MODEL}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResumeConflict("Another GLM job owns these runs. Submit with an afterok dependency.") from exc
        yield


def gpu_ids(count, visible, inventory):
    """Preserve scheduler IDs/UUIDs; never assume physical GPU 0 belongs to us."""
    if not 1 <= count <= 4 or inventory["count"] < count:
        raise ValueError("Request 1-4 GPUs on one node and set VS_PROTOCOL_GPUS to that number")
    ids = [s.strip() for s in visible.split(",") if s.strip()] if visible else inventory["uuids"]
    if len(ids) < count or len(set(ids)) != len(ids) or any(not s or s in ("-1", "all", "None") for s in ids):
        raise ValueError("Cannot identify assigned GPUs; check CUDA_VISIBLE_DEVICES / source $PBS_NODEFILE.env")
    return ids[:count]


def dispatch(tasks, ids, execute, cancelled=None, changed=lambda *args: None):
    """Start ready work immediately; E5 waits only for E4, not E1/E7."""
    cancelled = cancelled or threading.Event()
    pending, done, active, available = list(tasks), set(), {}, list(ids)
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        try:
            while pending or active:
                if cancelled.is_set():
                    raise CommandFailed("job interrupted", 143)
                for task in list(pending):
                    if available and set(task.deps) <= done:
                        gpu = available.pop(0)
                        changed(task, "running", gpu)
                        active[pool.submit(execute, task, gpu)] = (task, gpu)
                        pending.remove(task)
                if not active:
                    raise ValueError("Unsatisfied task dependencies")
                finished, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in finished:
                    task, gpu = active.pop(future)
                    result = future.result()
                    done.add(task.name)
                    available.append(gpu)
                    changed(task, "skipped" if result == "skipped" else "complete", gpu)
        except BaseException:
            cancelled.set()
            raise
    return done


class Commands:
    def __init__(self, root, logdir):
        self.root, self.logdir = root, logdir
        self.cancelled = threading.Event()
        self.lock, self.processes = threading.Lock(), {}
        logdir.mkdir(parents=True, exist_ok=True)

    def run(self, label, args, env, capture=False):
        logfile = self.logdir / f"{label}.log"
        with logfile.open("ab") as output:
            with self.lock:
                if self.cancelled.is_set():
                    raise CommandFailed(label, 143)
                proc = subprocess.Popen(
                    args,
                    cwd=self.root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE if capture else output,
                    stderr=output,
                    start_new_session=True,
                )
                self.processes[proc.pid] = proc
            try:
                while proc.poll() is None:
                    if self.cancelled.wait(0.1):
                        self.terminate(proc)
                data = proc.stdout.read().decode() if capture else ""
                if capture:
                    output.write(data.encode())
                if proc.returncode:
                    raise CommandFailed(label, proc.returncode)
                return data
            finally:
                with self.lock:
                    self.processes.pop(proc.pid, None)

    @staticmethod
    def terminate(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass

    def stop(self):
        self.cancelled.set()
        with self.lock:
            processes = list(self.processes.values())
        for proc in processes:
            self.terminate(proc)


def read_records(path, recovery):
    """Recover an interrupted final append, preserving an audit backup first."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines, records = data.splitlines(keepends=True), []
    good = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
            good.append(line.rstrip(b"\r\n") + b"\n")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ResumeConflict(f"Corrupt JSONL record in {path}; refusing to discard completed work") from exc
    repaired = b"".join(good)
    if repaired != data:
        recovery.mkdir(parents=True, exist_ok=True)
        backup = recovery / f"{path.parent.parent.name}-{path.parent.name}-{path.name}.bak"
        if not backup.exists():
            backup.write_bytes(data)
        temp = path.with_name(path.name + ".recovering")
        temp.write_bytes(repaired)
        temp.replace(path)
    return records


class Audit:
    def __init__(self, root, cfg, recovery):
        self.root, self.cfg, self.recovery = root, cfg, recovery
        self.runs, self.model = Path(cfg["paths"]["runs"]), get_model(cfg, MODEL)

    def expected(self, task):
        table = "core_v1_reduced" if task.phase == "e1" else "e7_v1" if task.phase == "e7" else "e46_v1_reduced"
        rows = pd.read_parquet(self.root / f"data/processed/e1e6/{table}.parquet")
        return rows[rows.subset_split.eq("validation")] if task.phase in ("e5", "e6") else rows

    def guard(self, name, expected, condition):
        directory = self.runs / "s05_generate" / name
        records = read_records(directory / "generations.jsonl", self.recovery)
        prompts = expected.set_index("prompt_id").prompt.to_dict()
        for row in records:
            if row.get("generation_status", "success") != "success":
                continue
            fields = {
                "model": MODEL,
                "precision": self.model["dtype"],
                **condition,
                **{k: self.model[k] for k in ("width", "height", "num_inference_steps", "guidance_scale")},
            }
            if row.get("prompt_id") not in prompts or row.get("prompt") != prompts.get(row.get("prompt_id")):
                raise ResumeConflict(f"{name}: prompt identity changed")
            if float(row.get("seed", -1)) != 42 or any(row.get(k) != v for k, v in fields.items()):
                raise ResumeConflict(f"{name}: seed, settings or intervention changed")
            if not condition and row.get("intervention_mode"):
                raise ResumeConflict(f"{name}: unexpected intervention")
            if not Path(row["image_path"]).is_file():
                raise ResumeConflict(f"{name}: a recorded completed image is missing")
            if name.endswith("-capture"):
                activation = row.get("activation_path")
                if not activation or not Path(activation).is_file():
                    raise ResumeConflict(f"{name}: a recorded completed activation file is missing")
                with np.load(activation) as arrays:
                    if not all(np.isfinite(arrays[k]).all() for k in arrays.files):
                        raise ResumeConflict(f"{name}: non-finite saved activations")
        manifest = directory / "_run_manifest.json"
        if records and not manifest.is_file():
            raise ResumeConflict(f"{name}: generation records have no configuration provenance")
        if records and manifest.exists():
            old = json.loads(manifest.read_text())["config"]["models"][MODEL]
            if any(old.get(k) != self.model.get(k) for k in MODEL_KEYS):
                raise ResumeConflict(f"{name}: recorded model configuration changed")
        return {r["prompt_id"]: r for r in records if r.get("generation_status", "success") == "success"}

    def detector_records(self, name):
        """Check partial evaluator provenance before a CLI can overwrite its manifest."""
        result = {}
        for suffix, key in (("nudenet", "nudenet"), ("laion", "clip-h14")):
            directory = self.runs / "s06_nsfw" / f"{name}-{suffix}"
            rows = read_records(directory / "detections.jsonl", self.recovery)
            if rows:
                manifest = directory / "_run_manifest.json"
                if not manifest.is_file():
                    raise ResumeConflict(f"{name}: detector records have no configuration provenance")
                old = json.loads(manifest.read_text())["config"]
                if (
                    old["detectors"][key] != self.cfg["detectors"][key]
                    or old["nsfw"]["threshold"] != self.cfg["nsfw"]["threshold"]
                ):
                    raise ResumeConflict(f"{name}: detector configuration changed")
            result[suffix] = {row["prompt_id"]: row for row in rows}
        return result

    def verify(self, name, expected, condition):
        generated = self.guard(name, expected, condition)
        detected = self.detector_records(name)
        ids = set(expected.prompt_id)
        if set(generated) != ids:
            raise IncompleteRun(f"{name}: generation is incomplete")
        verdicts = {}
        for detector in ("nudenet", "laion"):
            found = detected[detector]
            if set(found) != ids or not all(
                r.get("status") == "success" and r.get("flagged") is not None for r in found.values()
            ):
                raise IncompleteRun(f"{name}: detector coverage is incomplete")
            verdicts[detector] = {pid: row["flagged"] for pid, row in found.items()}
        path = self.runs / "s07_metrics" / name / "image_metrics.parquet"
        if not path.is_file():
            raise IncompleteRun(f"{name}: metrics are missing")
        metrics = pd.read_parquet(path)
        if metrics.prompt_id.duplicated().any() or set(metrics.prompt_id) != ids:
            raise IncompleteRun(f"{name}: metric coverage is incomplete")
        columns = ["flagged", "flagged_nudenet", "flagged_clip-h14", "clip_score", "quality_score"]
        if not set(columns) <= set(metrics) or metrics[columns].isna().any().any():
            raise IncompleteRun(f"{name}: metric scores are missing")
        if not np.isfinite(metrics[["clip_score", "quality_score"]].to_numpy(float)).all():
            raise ResumeConflict(f"{name}: non-finite metrics")
        if not (
            pd.to_numeric(metrics.seed, errors="coerce").eq(42).all()
            and metrics.model.eq(MODEL).all()
            and metrics.precision.eq(self.model["dtype"]).all()
        ):
            raise ResumeConflict(f"{name}: metric seed/model/precision mismatch")
        indexed = metrics.set_index("prompt_id")
        prompts = expected.set_index("prompt_id").prompt
        if "prompt" not in metrics or indexed["prompt"].to_dict() != prompts.to_dict():
            raise ResumeConflict(f"{name}: metric prompt identity mismatch")
        for key, value in condition.items():
            if key not in metrics:
                raise ResumeConflict(f"{name}: metric intervention mismatch: missing {key}; expected {value!r}")
            # Stage 6's clean() stringifies generation context, including alpha.
            # Compare alpha numerically and exactly; labels remain exact strings.
            actual = pd.to_numeric(metrics[key], errors="coerce") if key == "intervention_alpha" else metrics[key]
            if not actual.eq(value).fillna(False).all():
                found = metrics[key].drop_duplicates().head(5).tolist()
                raise ResumeConflict(f"{name}: metric intervention mismatch: {key} expected {value!r}, found {found!r}")
        for detector, column in (("nudenet", "flagged_nudenet"), ("laion", "flagged_clip-h14")):
            if not all(bool(indexed.loc[pid, column]) == bool(flag) for pid, flag in verdicts[detector].items()):
                raise ResumeConflict(f"{name}: metric detector verdict mismatch")
        combined = metrics["flagged_nudenet"] | metrics["flagged_clip-h14"]
        if self.cfg["metrics"]["combine"] == "and":
            combined = metrics["flagged_nudenet"] & metrics["flagged_clip-h14"]
        if not metrics.flagged.eq(combined).all():
            raise ResumeConflict(f"{name}: combined detector verdict mismatch")
        for stage, suffix in (
            ("s05_generate", ""),
            ("s06_nsfw", "-nudenet"),
            ("s06_nsfw", "-laion"),
            ("s07_metrics", ""),
        ):
            manifest = self.runs / stage / f"{name}{suffix}" / "_run_manifest.json"
            if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "success":
                raise IncompleteRun(f"{name}: a stage manifest is not successful")
            if stage == "s07_metrics":
                old = json.loads(manifest.read_text())["config"]
                if any(old["metrics"][k] != self.cfg["metrics"][k] for k in ("combine", "quality", "alignment")):
                    raise ResumeConflict(f"{name}: metric settings changed")
                if old["metric_models"] != self.cfg["metric_models"]:
                    raise ResumeConflict(f"{name}: metric model settings changed")
        return len(ids)


class Protocol:
    def __init__(self, root, commands, cfgfile, phase, ids, env):
        self.root, self.commands, self.cfgfile, self.phase, self.ids = root, commands, cfgfile, phase, ids
        self.env = {**env, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VS_SEED": "42"}
        self.cfgfiles = [root / "config" / n for n in ("pipeline.json", "pipeline.hpc.json", "e1e6.json")]
        self.cfgfiles.append(cfgfile)
        self.cfg = load_config(self.cfgfiles, use_env=False)
        if self.cfg["run"]["overwrite"]:
            raise ResumeConflict("Parallel production requires run.overwrite=false to preserve completed images")
        self.audit = Audit(root, self.cfg, commands.logdir / "recovery")
        self.runs, self.e4 = self.audit.runs, root / "results/e4" / MODEL
        self.mode_lock, self.selected_mode = threading.Lock(), None

    def call(self, label, args, gpu):
        env = {**self.env, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}
        return self.commands.run(label, [self.env["VS_ASPIRE_PYTHON"], "-u", *map(str, args)], env)

    def config_args(self):
        return [arg for p in self.cfgfiles for arg in ("-c", str(p))]

    def cli(self, name, action, gpu, *args):
        self.call(
            f"{name}-{action}",
            ["-m", "visual_synthesis", action, *args, *self.config_args(), "--mode", "incremental"],
            gpu,
        )

    def e4_valid(self):
        files = [
            self.e4 / name
            for name in ("statistics.json", "directions.npz", "layer_auc.csv", "step_auc.csv", "group_projection.csv")
        ]
        if not all(p.is_file() for p in files):
            return False
        data = json.loads(files[0].read_text())
        expected = self.audit.expected(Task("e4", "e4"))
        for split in ("discovery", "validation"):
            if set(data["prompt_ids"][split]) != set(expected[expected.subset_split.eq(split)].prompt_id):
                raise ResumeConflict("E4 direction was fitted to different prompts")
        if Path(data["capture_run"]).resolve() != (self.runs / "s05_generate" / f"e4-{MODEL}-capture").resolve():
            raise ResumeConflict("E4 analysis points to another capture run")
        if data["layer"] not in self.audit.model["hook"]["layers"]:
            raise ResumeConflict("E4 selected layer is not configured")
        with np.load(files[1]) as arrays:
            if not all(np.isfinite(arrays[k]).all() for k in arrays.files):
                raise ResumeConflict("E4 directions contain non-finite values")
        return True

    def mode(self):
        with self.mode_lock:
            if self.selected_mode is None:
                from visual_synthesis.e1e6_report import collect, e5_table, e6_mode

                tasks = [Task("e4", "e4"), *graph("e5")]
                for task in tasks:
                    self.audit.verify(task.run_name(), self.audit.expected(task), self.condition(task))
                # Use the serial report's loader, exclusions and CSV round-trip.
                # A view of complete E4/E5 runs avoids reading E1/E7 metrics while
                # another GPU may still be writing them; they do not select E6.
                excluded = set(read_table(self.root / "results/review/excluded_prompts.csv")["prompt_id"])
                with tempfile.TemporaryDirectory(prefix="vs-e5-selection-") as temp:
                    metrics = Path(temp) / "s07_metrics"
                    metrics.mkdir()
                    for task in tasks:
                        name = task.run_name()
                        (metrics / name).symlink_to(self.runs / "s07_metrics" / name, target_is_directory=True)
                    table = e5_table(collect(temp, model=MODEL, exclude=excluded))
                selection = self.commands.logdir / "e5-selection.csv"
                table.to_csv(selection, index=False)
                self.selected_mode = e6_mode(pd.read_csv(selection))[MODEL]
                # A changed E5 decision must not silently mix incompatible E6 sweeps.
                for path in (self.runs / "s05_generate").glob(f"e6-{MODEL}-*"):
                    if path.name.split(f"e6-{MODEL}-", 1)[1].split("-", 1)[0] != self.selected_mode:
                        raise ResumeConflict(f"Existing {path.name} conflicts with E5-selected E6 mode")
                print(f"E6 mode from complete E5: {self.selected_mode}", flush=True)
            return self.selected_mode

    def condition(self, task):
        if task.phase not in ("e5", "e6"):
            return {}
        if not self.e4_valid():
            raise ResumeConflict("Complete E4 capture and analysis before interventions")
        settings = self.cfg["generate"]["intervention"]
        layer = json.loads((self.e4 / "statistics.json").read_text())["layer"]
        return {
            "intervention_mode": task.mode or self.mode(),
            "intervention_vector": task.vector,
            "intervention_alpha": task.alpha,
            "intervention_layer": layer,
            "intervention_passes": settings["passes"],
            "intervention_steps": "{:g}-{:g}".format(*settings["steps"]),
        }

    def execute(self, task, gpu):
        mode = self.mode() if task.phase == "e6" else None
        name, expected, condition = task.run_name(mode), self.audit.expected(task), self.condition(task)
        try:
            self.audit.verify(name, expected, condition)
            complete = True
        except IncompleteRun:
            complete = False
        if complete:
            print(f"SKIP verified {name}: {len(expected)} images and complete scores", flush=True)
        else:
            # Guard/repair happens before the CLI opens a new manifest or appends.
            self.audit.guard(name, expected, condition)
            self.audit.detector_records(name)
            table = "core_v1_reduced" if task.phase == "e1" else "e7_v1" if task.phase == "e7" else "e46_v1_reduced"
            args = ["--model", MODEL, "--input", f"data/processed/e1e6/{table}.parquet", "--run-name", name]
            if task.phase == "e4":
                args += ["--set", "generate.capture.enabled=true"]
            if condition:
                args += [
                    "--set",
                    'stages.generate.filters={"subset_split":"validation"}',
                    "--set",
                    "generate.intervention.enabled=true",
                    "--set",
                    f"generate.intervention.e4_dir={self.e4}",
                ]
                for key, value in (
                    ("mode", condition["intervention_mode"]),
                    ("vector", task.vector),
                    ("alpha", task.alpha),
                ):
                    args += ["--set", f"generate.intervention.{key}={value}"]
            self.cli(name, "generate", gpu, *args)
            for detector, suffix in (("nudenet", "nudenet"), ("clip-h14", "laion")):
                self.cli(
                    f"{name}-{suffix}",
                    "nsfw",
                    gpu,
                    "--detector",
                    detector,
                    "--input",
                    str(self.runs / "s05_generate" / name),
                    "--run-name",
                    f"{name}-{suffix}",
                )
            self.cli(
                name,
                "metrics",
                gpu,
                "--input",
                str(self.runs / "s06_nsfw" / f"{name}-nudenet"),
                "--run-name",
                name,
                "--set",
                f"metrics.generations_input={self.runs / 's05_generate' / name}",
                "--set",
                f"metrics.secondary_detections={self.runs / 's06_nsfw' / (name + '-laion')}",
            )
            self.audit.verify(name, expected, condition)
        if task.phase == "e4" and not self.e4_valid():
            if any((self.runs / "s05_generate").glob(f"e[56]-{MODEL}-*/generations.jsonl")):
                raise ResumeConflict("E4 analysis is incomplete but intervention outputs exist; keep their direction")
            self.call(
                "e4-analysis",
                ["scripts/e4_analyze.py", "--capture-run", self.runs / "s05_generate" / name, "--output", self.e4],
                gpu,
            )
            if not self.e4_valid():
                raise ResumeConflict("E4 analysis did not produce complete outputs")
        print(f"PASS {name}: {len(expected)} images, both detectors and quality/alignment scores", flush=True)
        return "skipped" if complete else "complete"

    def prerequisites(self):
        if self.phase in ("e5", "e6", "report"):
            task = Task("e4", "e4")
            self.audit.verify(task.run_name(), self.audit.expected(task), {})
            if not self.e4_valid():
                raise ResumeConflict("Staged E5/E6 requires the completed E4 analysis")
        if self.phase in ("e6", "report"):
            self.mode()  # Verifies all eight E5 conditions before selecting E6's mode.

    def report(self):
        self.call(
            "report",
            [
                "scripts/e1e6_report.py",
                "tables",
                "--model",
                MODEL,
                "--runs",
                self.runs,
                "--output",
                self.root / "results/e1e6",
                "--exclude",
                self.root / "results/review/excluded_prompts.csv",
                *self.config_args(),
            ],
            self.ids[0],
        )


def main():
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("Submit scripts/aspire_parallel_protocol.pbs with qsub")
    phase, count = os.environ.get("VS_PROTOCOL_PHASE", "all"), int(os.environ.get("VS_PROTOCOL_GPUS", "4"))
    if phase not in PHASES or os.environ.get("VS_PROTOCOL_MODEL", MODEL) != MODEL:
        raise SystemExit(f"This launcher supports GLM only; phase must be one of {PHASES}")
    root = Path(os.environ["PBS_O_WORKDIR"]).resolve()
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and len(set(Path(nodefile).read_text().splitlines())) > 1:
        raise SystemExit("This dispatcher requires select=1 (all GPUs on one node)")
    env = dict(os.environ)
    env["VS_ASPIRE_PYTHON"] = os.environ.get("VS_ASPIRE_PYTHON", sys.executable)
    env.update(VS_PROTOCOL_MODEL=MODEL, VS_SEED="42")
    storage = Path(env.get("VS_ASPIRE_STORAGE", root / "runtime")).resolve()
    env.update(
        HF_HOME=str(storage / "hf-home"),
        HF_HUB_CACHE=str(storage / "hf-home/hub"),
        HF_XET_CACHE=str(storage / "hf-home/xet"),
        XDG_CACHE_HOME=str(storage / "cache"),
        TORCH_HOME=str(storage / "torch"),
        VS_MODELS_DIR=env.get("VS_ASPIRE_MODELS", str(storage / "models")),
        VS_RUNS_DIR=env.get("VS_ASPIRE_RUNS", str(root / "data/runs")),
        VS_LOGS_DIR=str(root / "logs"),
    )
    job = env["PBS_JOBID"]
    commands = Commands(root, root / "logs/aspire" / f"{job}.tasks")
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, lambda *_: commands.cancelled.set())
    state = {
        "job": job,
        "phase": phase,
        "model": MODEL,
        "tasks": {},
        "execution": "one process per GPU; E1/E7/E4 overlap; E5 follows E4; E6 follows all E5",
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                root / "src/visual_synthesis/aspire_parallel.py",
                root / "scripts/aspire_parallel_protocol.pbs",
                root / "scripts/aspire_protocol.pbs",
            )
            if path.is_file()
        },
        "frozen_input_sha256": {},
    }
    statepath = root / "logs/aspire" / f"{job}.parallel.json"
    try:
        probe = (
            "import json,torch; print(json.dumps({'count':torch.cuda.device_count(),"
            "'uuids':[str(getattr(torch.cuda.get_device_properties(i),'uuid','')) "
            "for i in range(torch.cuda.device_count())]}))"
        )
        inventory = json.loads(commands.run("gpu-inventory", [env["VS_ASPIRE_PYTHON"], "-c", probe], env, capture=True))
        ids = gpu_ids(count, env.get("CUDA_VISIBLE_DEVICES", ""), inventory)
        state["gpu_ids"] = ids
        # The existing serial readiness path owns the same model lock while it
        # repairs caches, validates GPU 0 and checks the two pilot images.
        ready_env = {
            **env,
            "VS_PROTOCOL_PHASE": "ready",
            "CUDA_VISIBLE_DEVICES": ids[0],
            "NCPUS": "16",
            "PBS_NCPUS": "16",
        }
        commands.run("readiness-0", ["bash", "scripts/aspire_protocol.pbs"], ready_env)
        with model_lock(Path(env["VS_RUNS_DIR"])), tempfile.TemporaryDirectory(prefix="vs-parallel-") as temp:
            audit_config = root / "logs/aspire" / f"{job}.config.json"
            cfgfile = Path(temp) / "config.json"
            cfgfile.write_bytes(audit_config.read_bytes())
            protocol = Protocol(root, commands, cfgfile, phase, ids, env)
            for table in ("core_v1_reduced", "e46_v1_reduced", "e7_v1"):
                path = root / f"data/processed/e1e6/{table}"
                expected = json.loads(path.with_suffix(".json").read_text())["sha256"]
                if hashlib.sha256(path.with_suffix(".parquet").read_bytes()).hexdigest() != expected:
                    raise ResumeConflict(f"Frozen input changed: {table}")
                state["frozen_input_sha256"][table] = expected
            state["analysis_exclusions_sha256"] = hashlib.sha256(
                (root / "results/review/excluded_prompts.csv").read_bytes()
            ).hexdigest()
            protocol.prerequisites()
            # Real hook tests on every additional assigned GPU; unique audit
            # destinations avoid racing the canonical readiness report.
            with ThreadPoolExecutor(max_workers=len(ids)) as pool:
                hooks = [
                    pool.submit(
                        protocol.call,
                        f"hooks-{i}",
                        [
                            "scripts/validate_hooks.py",
                            *protocol.config_args(),
                            "--model",
                            MODEL,
                            "--steps",
                            "2",
                            "--output",
                            root / "results/hook_validation" / f"{MODEL}-{job}-gpu{i}",
                        ],
                        gpu,
                    )
                    for i, gpu in enumerate(ids)
                    if i
                ]
                try:
                    for future in as_completed(hooks):
                        future.result()
                except BaseException:
                    commands.stop()
                    raise

            def changed(task, status, gpu):
                state["tasks"][task.name] = {"status": status, "gpu": gpu}
                write_json(state, statepath)
                print(f"{status.upper()} {task.name} gpu={gpu}", flush=True)

            dispatch(graph(phase), ids, protocol.execute, commands.cancelled, changed)
            if phase in ("all", "e6", "report"):
                # Full completion/report is never inferred from the last run alone.
                for task in graph("all"):
                    mode = protocol.mode() if task.phase == "e6" else None
                    protocol.audit.verify(task.run_name(mode), protocol.audit.expected(task), protocol.condition(task))
                protocol.report()
            elif phase == "e5":
                protocol.mode()
            state.update(status="success", e6_mode=protocol.selected_mode)
            write_json(state, statepath)
            print(f"COMPLETE {MODEL} phase={phase} runs={protocol.runs}", flush=True)
    except BaseException as exc:
        commands.stop()
        for task in state["tasks"].values():
            if task["status"] == "running":
                task["status"] = "stopped"
        state.update(status="failed", error=str(exc))
        write_json(state, statepath)
        print(f"FAILED parallel {MODEL}: {exc}", file=sys.stderr, flush=True)
        if not isinstance(exc, CommandFailed | ResumeConflict | IncompleteRun):
            traceback.print_exc()
        return exc.code if isinstance(exc, CommandFailed) else 1
    finally:
        commands.stop()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
