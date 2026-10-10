"""Slurm: write (and optionally submit) a job script that runs one stage.

The script is generated from the ``slurm`` section of the config, so a cluster
run is reproducible from the repository alone:

    visual-synthesis slurm render generate --model sd-xl --limit 100
    visual-synthesis slurm render generate --array        # one task per model

GPU stages use ``slurm.gpu`` resources, the others ``slurm.cpu``.
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from visual_synthesis.config import ROOT

log = logging.getLogger(__name__)

# slurm.gpu / slurm.cpu key -> #SBATCH option. Unset (None) values are left out.
SBATCH_OPTIONS = {
    "partition": "partition",
    "account": "account",
    "qos": "qos",
    "time": "time",
    "nodes": "nodes",
    "ntasks": "ntasks",
    "cpus_per_task": "cpus-per-task",
    "mem": "mem",
    "gres": "gres",
    "constraint": "constraint",
}


def render_job(cfg, stage, model=None, limit=None, run_name=None, job_name=None, array_models=None,
               overwrite=False):
    """Write the ``.sbatch`` file for a stage and return its path.

    ``array_models``: a list of model ids turns the job into a Slurm array with
    one task per model, each writing to its own run folder.
    """
    slurm = cfg["slurm"]
    resources = slurm["gpu" if stage.NEEDS_GPU or array_models else "cpu"]
    jobs_dir = Path(cfg["paths"]["slurm_jobs"])
    log_dir = Path(slurm["log_dir"]) if slurm["log_dir"] else jobs_dir
    name = f"{stage.SLUG}_array" if array_models else stage.SLUG
    path = jobs_dir / f"{name}.sbatch"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")

    job_name = job_name or f"{resources['job_name_prefix']}-{name}"
    run_name = run_name or f"{stage.SLUG}_{datetime.now():%Y%m%d-%H%M%S}"
    log_file = log_dir / (f"{stage.SLUG}-%A_%a" if array_models else f"{stage.SLUG}-%j")

    lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}"]
    lines += [f"#SBATCH --{option}={resources[key]}" for key, option in SBATCH_OPTIONS.items()
              if resources[key] is not None]
    lines += [f"#SBATCH --output={log_file}.out", f"#SBATCH --error={log_file}.err"]
    if array_models:
        lines.append(f"#SBATCH --array=0-{len(array_models) - 1}")

    lines += ["", "set -euo pipefail", f'cd "{ROOT}"']
    lines += [f"module load {module}" for module in slurm["modules"]]
    # The conda environment is built once on the cluster and only activated here.
    lines += ['source "$(conda info --base)/etc/profile.d/conda.sh"', f'conda activate "{slurm["conda_env"]}"']
    lines += [f'export {key}="{value}"' for key, value in slurm["set_env"].items()]

    command = [f"srun visual-synthesis run {stage.NAME}"]
    command += [f'--config "{file}"' for file in cfg["config_files"]]
    if array_models:
        lines += ["", "MODELS=(" + " ".join(f'"{m}"' for m in array_models) + ")",
                  'MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"']
        command += [f'--run-name "{run_name}_${{MODEL}}"', '--model "${MODEL}"']
    else:
        command.append(f'--run-name "{run_name}"')
        if model:
            command.append(f'--model "{model}"')
    if limit:
        command.append(f"--limit {limit}")
    lines += ["", " \\\n  ".join(command), ""]

    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("rendered %s", path)
    return path


def submit_job(path):
    """Submit a rendered job with ``sbatch`` and return the job id."""
    try:
        result = subprocess.run(["sbatch", str(path)], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not run sbatch: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    # sbatch prints "Submitted batch job 12345".
    return result.stdout.strip().split()[-1]
