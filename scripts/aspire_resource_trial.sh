#!/usr/bin/env bash
# PBS Pro launcher for ASPIRE 2A/2A+. Supply your NSCC project with qsub -P.
#PBS -N vs-eight-fit
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=06:00:00
#PBS -j oe
set -euo pipefail

if [ -z "${PBS_JOBID:-}" ]; then
  echo "Submit with qsub; GPU trials and downloads must run on a compute node." >&2
  exit 2
fi
cd "${VS_ASPIRE_REPO:-${PBS_O_WORKDIR:?PBS working directory missing}}"
if [ "$(git branch --show-current)" != swara/experiments ]; then
  echo "Expected swara/experiments. Refusing to switch branches or alter changes." >&2
  exit 2
fi
: "${VS_ASPIRE_STORAGE:?Set an absolute dedicated directory on assigned project storage or scratch}"
: "${VS_ASPIRE_BUDGET_GIB:?Set the verified capacity budget for that directory in GiB}"

if [ -n "${VS_ASPIRE_PYTHON:-}" ]; then
  vs_aspire_python=$VS_ASPIRE_PYTHON
else
  if ! command -v conda >/dev/null; then
    echo "Load Conda first, or export VS_ASPIRE_PYTHON=/absolute/env/bin/python." >&2
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${VS_ASPIRE_CONDA_ENV:-visual-safety}"
  vs_aspire_python=$(command -v python)
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${NCPUS:-${PBS_NCPUS:-14}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=180
# The Python driver fixes all weight/cache paths before importing the HF libraries.
exec "$vs_aspire_python" -u scripts/aspire_trial_support.py run \
  --storage "$VS_ASPIRE_STORAGE" --budget-gib "$VS_ASPIRE_BUDGET_GIB"
