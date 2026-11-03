#!/bin/bash
# Lightweight login-node submission. No active environment/model-cache mutation.
set -euo pipefail
if [ "$#" -ne 0 ]; then
  echo 'Usage: bash scripts/submit_aspire_qwen_quantization.sh (settings through VS_QWEN_* variables)' >&2
  exit 2
fi
export VS_ASPIRE_REPO
VS_ASPIRE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
export VS_ASPIRE_PYTHON=${VS_ASPIRE_PYTHON:-$HOME/.conda/envs/visual-safety/bin/python}
export VS_ASPIRE_GIT=${VS_ASPIRE_GIT:-${VS_ASPIRE_PYTHON%/*}/git}
if [ ! -x "$VS_ASPIRE_PYTHON" ] || [ ! -x "$VS_ASPIRE_GIT" ]; then
  echo 'Set shared VS_ASPIRE_PYTHON and VS_ASPIRE_GIT executables; see SETUP.md.' >&2
  exit 2
fi
if [ "$("$VS_ASPIRE_GIT" -C "$VS_ASPIRE_REPO" branch --show-current)" != swara/experiments ]; then
  echo 'Expected swara/experiments; no branch switch performed.' >&2
  exit 2
fi
: "${VS_ASPIRE_PROJECT:?Set your project ID from myprojects}"
: "${VS_ASPIRE_BUDGET_GIB:?Set your verified total runtime workspace capacity in GiB}"
export VS_ASPIRE_STORAGE=${VS_ASPIRE_STORAGE:-$VS_ASPIRE_REPO/runtime}
VS_ASPIRE_STORAGE=$("$VS_ASPIRE_PYTHON" -c \
  'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert not p.is_relative_to(Path.home().resolve()), "Use scratch, not home"; print(p)' \
  "$VS_ASPIRE_STORAGE")
export VS_QWEN_SOURCE=${VS_QWEN_SOURCE:-$VS_ASPIRE_STORAGE/models/qwen-image}
if [ ! -r "$VS_QWEN_SOURCE/model_index.json" ]; then
  echo 'Cached Qwen weights are required; this comparison never downloads model checkpoints.' >&2
  exit 2
fi
export VS_QWEN_SAMPLE_SIZE=${VS_QWEN_SAMPLE_SIZE:-12}
export VS_QWEN_ORDER=${VS_QWEN_ORDER:-bf16,nf4}
# qsub -v uses commas as delimiters; encode the order with a safe name instead.
case "$VS_QWEN_ORDER" in
  bf16-first|bf16,nf4) export VS_QWEN_ORDER=bf16-first ;;
  nf4-first|nf4,bf16) export VS_QWEN_ORDER=nf4-first ;;
  *) echo 'VS_QWEN_ORDER must be bf16-first or nf4-first.' >&2; exit 2 ;;
esac
export VS_QWEN_WALL_SECONDS=${VS_QWEN_WALL_SECONDS:-6300}
walltime=$("$VS_ASPIRE_PYTHON" -c \
  'import sys; s=int(sys.argv[1]); n=int(sys.argv[2]); assert 600 <= s <= 7200 and 1 <= n <= 24, "Use 600-7200 seconds and 1-24 prompts"; print(f"{s//3600:02}:{s%3600//60:02}:{s%60:02}")' \
  "$VS_QWEN_WALL_SECONDS" "$VS_QWEN_SAMPLE_SIZE")
mkdir -p "$VS_ASPIRE_STORAGE/outputs" "$VS_ASPIRE_STORAGE/logs"
export VS_QWEN_OUTPUT
VS_QWEN_OUTPUT=$(mktemp -d "$VS_ASPIRE_STORAGE/outputs/qwen-quant-XXXXXX")
run_tag=${VS_QWEN_OUTPUT##*/}
vs_export=VS_ASPIRE_REPO,VS_ASPIRE_PROJECT,VS_ASPIRE_PYTHON,VS_ASPIRE_GIT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_QWEN_SOURCE,VS_QWEN_SAMPLE_SIZE,VS_QWEN_ORDER,VS_QWEN_WALL_SECONDS,VS_QWEN_OUTPUT
if [ -n "${VS_QWEN_INPUT:-}" ]; then
  export VS_QWEN_INPUT
  vs_export=$vs_export,VS_QWEN_INPUT
fi
cd "$VS_ASPIRE_REPO"
# Always normal + <=2h: PBS routes to gdev. No dependency on the long g1 pilots.
job_id=$(qsub -P "$VS_ASPIRE_PROJECT" -q normal -l "walltime=$walltime" \
  -o "$VS_ASPIRE_STORAGE/logs/$run_tag.pbs.log" -v "$vs_export" scripts/aspire_qwen_quantization.sh)
printf 'Outputs: %s\nLogs: %s/logs/%s.*.log\n' "$VS_QWEN_OUTPUT" "$VS_ASPIRE_STORAGE" "$run_tag" >&2
printf '%s\n' "$job_id"
