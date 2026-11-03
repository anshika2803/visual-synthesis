#!/bin/bash
# Login-node submission only; models and evaluators run inside PBS.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash scripts/submit_aspire_qwen_calibration.sh COMPLETED_PILOT_DIRECTORY' >&2
  exit 2
fi
export VS_ASPIRE_REPO
VS_ASPIRE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
export VS_ASPIRE_PYTHON=${VS_ASPIRE_PYTHON:-$HOME/.conda/envs/visual-safety/bin/python}
export VS_ASPIRE_GIT=${VS_ASPIRE_GIT:-${VS_ASPIRE_PYTHON%/*}/git}
: "${VS_ASPIRE_PROJECT:?Set your NSCC project ID}"
: "${VS_ASPIRE_BUDGET_GIB:?Set your verified runtime workspace budget in GiB}"
if [ ! -x "$VS_ASPIRE_PYTHON" ] || [ ! -x "$VS_ASPIRE_GIT" ]; then
  echo 'Shared Python and Git executables are required; see SETUP.md.' >&2
  exit 2
fi
if [ "$("$VS_ASPIRE_GIT" -C "$VS_ASPIRE_REPO" branch --show-current)" != swara/experiments ]; then
  echo 'Expected swara/experiments; no branch changes are performed.' >&2
  exit 2
fi
export VS_ASPIRE_STORAGE=${VS_ASPIRE_STORAGE:-$VS_ASPIRE_REPO/runtime}
export VS_CALIBRATION_SOURCE
VS_CALIBRATION_SOURCE=$("$VS_ASPIRE_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1")
VS_ASPIRE_STORAGE=$("$VS_ASPIRE_PYTHON" -c \
  'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert p.is_absolute() and not p.is_relative_to(Path.home().resolve()), "Use scratch storage"; print(p)' \
  "$VS_ASPIRE_STORAGE")
if [ ! -r "$VS_CALIBRATION_SOURCE/pilot_summary.json" ] || [ ! -r "$VS_CALIBRATION_SOURCE/learn/direction.npz" ]; then
  echo 'A completed pilot with its learned direction is required.' >&2
  exit 2
fi
mkdir -p "$VS_ASPIRE_STORAGE/outputs" "$VS_ASPIRE_STORAGE/logs"
export VS_CALIBRATION_OUTPUT
VS_CALIBRATION_OUTPUT=$(mktemp -d "$VS_ASPIRE_STORAGE/outputs/qwen-calibration-XXXXXX")
run_tag=${VS_CALIBRATION_OUTPUT##*/}
cd "$VS_ASPIRE_REPO"
job_id=$(qsub -P "$VS_ASPIRE_PROJECT" -q normal -N vq-calibration \
  -o "$VS_ASPIRE_STORAGE/logs/$run_tag.pbs.log" \
  -v VS_ASPIRE_REPO,VS_ASPIRE_PROJECT,VS_ASPIRE_PYTHON,VS_ASPIRE_GIT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_CALIBRATION_SOURCE,VS_CALIBRATION_OUTPUT \
  scripts/aspire_qwen_calibration.sh)
printf 'Outputs: %s\nLogs: %s/logs/%s.*\n' "$VS_CALIBRATION_OUTPUT" "$VS_ASPIRE_STORAGE" "$run_tag" >&2
printf '%s\n' "$job_id"
