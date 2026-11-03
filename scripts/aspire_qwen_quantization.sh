#!/bin/bash
#PBS -N vq-qwen-paired
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:45:00
#PBS -j oe
set -euo pipefail
ulimit -c 0
if [ -z "${PBS_JOBID:-}" ]; then
  echo 'Submit with scripts/submit_aspire_qwen_quantization.sh; computation requires PBS.' >&2
  exit 2
fi
export PATH="${PBS_O_PATH:-${PATH:-/usr/bin:/bin}}"
cd "${VS_ASPIRE_REPO:?Missing repository}"
vs_python=${VS_ASPIRE_PYTHON:?Missing shared Python}
vs_git=${VS_ASPIRE_GIT:?Missing shared Git}
if [ "$("$vs_git" branch --show-current)" != swara/experiments ]; then
  echo 'Expected swara/experiments; no branch switch performed.' >&2
  exit 2
fi
export PATH="${vs_python%/*}:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${NCPUS:-${PBS_NCPUS:-16}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS TOKENIZERS_PARALLELISM=false
exec "$vs_python" -u scripts/aspire_qwen_quantization.py run
