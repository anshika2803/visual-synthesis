#!/bin/bash
# One job on TC2 for any pipeline command or E1-E7 script (docs/e1_e6_research_plan.md section 10):
#
#   sbatch scripts/slurm_run.sh generate -c config/pipeline.json -c config/pipeline.hpc.json -c config/e1e6.json \
#     --model sd-xl --input data/processed/e1e6/core_v1.parquet --run-name e1-sd-xl
#   sbatch scripts/slurm_run.sh scripts/validate_hooks.py -c config/pipeline.json -c config/e1e6.json --model sd-xl
#
# The QoS "normal" allows 1 GPU, 10 CPUs, 30 GB and 2 jobs per user in total, 6 h each.
# A CPU-only job next to a GPU job: sbatch --gres=gpu:0 --cpus-per-task=3 --mem=8G ... (and lower the GPU job's).
# fp8 weights need an L40S: sbatch --gres=gpu:nvidia_l40s:1 ...
# Pipeline commands always resume (--mode incremental): a job that hit the time
# limit is re-submitted with the identical command. Submit from the repository root.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=vs-e1e6
#SBATCH --output=slurm/jobs/%x-%j.out
#SBATCH --error=slurm/jobs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# conda's activation scripts read unset variables, so -u is off while it runs.
set +u
module load anaconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-visual-safety}"
set -u
export HF_HOME="${HF_HOME:-$PWD/data/cache/huggingface}" TOKENIZERS_PARALLELISM=false
# hf_xet downloads stalled without an error on TC2 (job 50766); plain HTTP resumes partial files.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is): $*"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>/dev/null || true
# GPU utilization and memory every 30 s, next to the job's log (how busy the GPU really is).
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 \
  > "slurm/jobs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-0}-gpu.csv" 2>/dev/null &

run_one() {
  case "$1" in
    *.py) python "$@" ;;
    # evaluate RUN: both detectors and the metrics of one generate run
    evaluate)
      local run=$2; shift 2
      local cfg=(-c config/pipeline.json -c config/pipeline.hpc.json -c config/e1e6.json)
      visual-synthesis nsfw "${cfg[@]}" --detector nudenet --input "data/runs/s05_generate/$run" \
        --run-name "$run-nudenet" --mode incremental "$@"
      visual-synthesis nsfw "${cfg[@]}" --detector clip-h14 --input "data/runs/s05_generate/$run" \
        --run-name "$run-laion" --mode incremental "$@"
      visual-synthesis metrics "${cfg[@]}" --input "data/runs/s06_nsfw/$run-nudenet" --run-name "$run" \
        --mode incremental --set "metrics.generations_input=data/runs/s05_generate/$run" \
        --set "metrics.secondary_detections=data/runs/s06_nsfw/$run-laion" "$@" ;;
    # after RUN: wait until a generate run (of another job) has finished; fails if that run failed
    after)
      local manifest="data/runs/s05_generate/$2/_run_manifest.json"
      until grep -qsE '"status": "(success|failed)"' "$manifest"; do sleep 60; done
      grep -qs '"status": "success"' "$manifest" ;;
    *) visual-synthesis "$@" --mode incremental ;;
  esac
}

# chain 'STEP 1' 'STEP 2' ...: several steps in one job, stopping at the first failure
# (the QoS allows only 2 queued jobs, and every new job waits in the queue again).
# A CPU job can follow a GPU job's runs: chain 'after R1' 'evaluate R1' 'after R2' 'evaluate R2' ...
if [ "$1" = chain ]; then
  shift
  for step in "$@"; do
    echo "== step: $step"
    eval "run_one $step"
  done
else
  run_one "$@"
fi
