#!/usr/bin/env bash
#SBATCH --job-name=vs-resource-fit
#SBATCH --account=stud
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Submit this script with sbatch; model computation must not run on the head node." >&2
  exit 2
fi
if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch scripts/tc2_resource_trial.sh ONE_MODEL_ID" >&2
  exit 2
fi
case "$1" in
  sd-xl|sd-3.5-large|flux-dev|qwen-image|z-image-turbo|glm-image|pixart-sigma|sd-1.5) ;;
  *) echo "Unknown model ID: $1" >&2; exit 2 ;;
esac
vs_trial_repo=${VS_TRIAL_REPO:-/home/msai/swarangi001/visual-synthesis}
cd "$vs_trial_repo"
if [ "$(git branch --show-current)" != swara/experiments ]; then
  echo "Expected swara/experiments; refusing to switch branches or alter existing changes." >&2
  exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate visual-safety
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
mkdir -p results/benchmarks
vs_trial_out=$(mktemp -d "results/benchmarks/tc2-${SLURM_JOB_ID}-$1-XXXXXX")
vs_trial_step_pid=""

vs_trial_finish() {
  vs_trial_code=$?
  trap - EXIT
  python scripts/tc2_trial_support.py finish --output "$vs_trial_out" \
    --job-id "$SLURM_JOB_ID" --exit-code "$vs_trial_code" || true
  echo "Trial artifacts: $vs_trial_repo/$vs_trial_out"
  exit "$vs_trial_code"
}
vs_trial_stop() {
  if [ -n "$vs_trial_step_pid" ]; then
    kill -TERM "$vs_trial_step_pid" 2>/dev/null || true
    wait "$vs_trial_step_pid" 2>/dev/null || true
  fi
  exit 124
}
trap vs_trial_finish EXIT
trap vs_trial_stop USR1 TERM INT
vs_trial_run() {
  "$@" &
  vs_trial_step_pid=$!
  if wait "$vs_trial_step_pid"; then
    vs_trial_run_code=0
  else
    vs_trial_run_code=$?
  fi
  vs_trial_step_pid=""
  return "$vs_trial_run_code"
}
git rev-parse HEAD > "$vs_trial_out/git-commit.txt"
git status --short > "$vs_trial_out/git-status.txt"
vs_trial_prepare=(--model "$1" --output "$vs_trial_out")
if [ "${VS_TRIAL_ALLOW_DOWNLOAD:-0}" = 1 ]; then
  vs_trial_prepare+=(--allow-download)
fi
if [ -n "${VS_TRIAL_BUDGET_GIB:-}" ]; then
  vs_trial_prepare+=(--budget-gib "$VS_TRIAL_BUDGET_GIB")
fi
if [ -n "${VS_TRIAL_ATTEMPT_CAP_GIB:-}" ]; then
  vs_trial_prepare+=(--attempt-cap-gib "$VS_TRIAL_ATTEMPT_CAP_GIB")
fi
python scripts/tc2_trial_support.py prepare "${vs_trial_prepare[@]}"
vs_trial_state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$vs_trial_out/preparation.json")
# Fail on missing imports, unsupported GPU precision or bad input before fetching weights.
python scripts/benchmark_cluster.py check --model "$1" --device cuda --generation-only \
  --output "$vs_trial_out/environment-preflight.json"
if [ "$vs_trial_state" = download_permitted ]; then
  # One pinned model/file selection; record any quota failure and stop before inference.
  echo "Prefetching only $1; progress and errors: $vs_trial_out/prefetch.log"
  if vs_trial_run python scripts/benchmark_cluster.py prefetch \
      -c config/pipeline.json -c config/benchmark.models.json -c config/benchmark.hpc.json \
      -c "$vs_trial_out/download-selection.json" --model "$1" --generation-only \
      > "$vs_trial_out/prefetch.log" 2>&1; then
    touch "$vs_trial_out/prefetch-completed"
    echo "Prefetch completed for $1"
  else
    vs_trial_prefetch_code=$?
    tail -n 30 "$vs_trial_out/prefetch.log" >&2
    exit "$vs_trial_prefetch_code"
  fi
fi
export VS_BENCH_MODE=fit
export VS_BENCH_SAMPLE_SIZE=2
export VS_BENCH_TIMEOUT=${VS_TRIAL_TIMEOUT:-18000}
export VS_BENCH_OUTPUT="$vs_trial_out/benchmark"
export VS_BENCH_PYTHON="$(command -v python)"
# Keep allocation/accounting enforcement around the existing benchmark wrapper.
vs_trial_run srun --ntasks=1 --cpus-per-task=10 bash scripts/benchmark_tc2.sh "$1"
