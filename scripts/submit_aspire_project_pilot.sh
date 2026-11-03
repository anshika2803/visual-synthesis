#!/bin/bash
# Login-node submission only: no model imports, recursive scans or downloads.
set -euo pipefail
if [ "$#" -gt 1 ]; then
  echo "Usage: bash scripts/submit_aspire_project_pilot.sh [all|model-id]" >&2
  exit 2
fi
export VS_PILOT_MODEL=${1:-all}
case "$VS_PILOT_MODEL" in
  all|sd-xl|sd-3.5-large|flux-dev|qwen-image|z-image-turbo|glm-image|pixart-sigma|sd-1.5) ;;
  *) echo "Unknown model: $VS_PILOT_MODEL" >&2; exit 2 ;;
esac
export VS_ASPIRE_REPO
VS_ASPIRE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
export VS_ASPIRE_PYTHON=${VS_ASPIRE_PYTHON:-$HOME/.conda/envs/visual-safety/bin/python}
export VS_ASPIRE_GIT=${VS_ASPIRE_GIT:-${VS_ASPIRE_PYTHON%/*}/git}
if [ ! -x "$VS_ASPIRE_PYTHON" ] || [ ! -x "$VS_ASPIRE_GIT" ]; then
  echo "Set VS_ASPIRE_PYTHON and VS_ASPIRE_GIT to shared Conda executables; see SETUP.md." >&2
  exit 2
fi
if [ "$("$VS_ASPIRE_GIT" -C "$VS_ASPIRE_REPO" branch --show-current)" != swara/experiments ]; then
  echo "Expected swara/experiments; no branch switch performed." >&2
  exit 2
fi
: "${VS_ASPIRE_PROJECT:?Set your own NSCC project ID from myprojects}"
: "${VS_ASPIRE_BUDGET_GIB:?Set verified total workspace capacity in GiB, e.g. 500}"
export VS_ASPIRE_STORAGE=${VS_ASPIRE_STORAGE:-$VS_ASPIRE_REPO/runtime}
VS_ASPIRE_STORAGE=$("$VS_ASPIRE_PYTHON" -c \
  'import pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); assert p.is_absolute() and not p.is_relative_to(pathlib.Path.home().resolve()), "Use scratch, not home"; print(p)' \
  "$VS_ASPIRE_STORAGE")
export VS_PILOT_ALLOW_DOWNLOAD=${VS_PILOT_ALLOW_DOWNLOAD:-1}
export VS_PILOT_BASELINE=${VS_PILOT_BASELINE:-60}
export VS_PILOT_DISCOVERY=${VS_PILOT_DISCOVERY:-10}
export VS_PILOT_VALIDATION=${VS_PILOT_VALIDATION:-6}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$VS_ASPIRE_STORAGE/hf-home/token}
if [[ "${HF_TOKEN_PATH##*/}" == hf_* ]]; then
  echo 'HF_TOKEN_PATH must name a token file, e.g. $HF_HOME/token; do not put a token value in its filename.' >&2
  exit 2
fi
if [[ "$VS_PILOT_MODEL" == all || "$VS_PILOT_MODEL" == flux-dev || "$VS_PILOT_MODEL" == sd-3.5-large ]] && [ ! -r "$HF_TOKEN_PATH" ]; then
  echo 'Gated models require a readable HF_TOKEN_PATH file. Run hf auth login first; see SETUP.md.' >&2
  exit 2
fi
if [ "$VS_PILOT_MODEL" = all ]; then
  previous_job=${VS_PILOT_AFTER:-}
  for model in qwen-image glm-image flux-dev sd-3.5-large z-image-turbo pixart-sigma sd-xl sd-1.5; do
    previous_job=$(VS_PILOT_AFTER="$previous_job" bash "${BASH_SOURCE[0]}" "$model")
    printf '%s %s\n' "$model" "$previous_job"
  done
  exit 0
fi
mkdir -p "$VS_ASPIRE_STORAGE/runs"
export VS_PILOT_OUTPUT
VS_PILOT_OUTPUT=$(mktemp -d "$VS_ASPIRE_STORAGE/runs/$VS_PILOT_MODEL-XXXXXX")
mkdir -p "$VS_PILOT_OUTPUT/logs"
pbs_log="$VS_PILOT_OUTPUT/logs/pbs.log"
vs_export=VS_ASPIRE_REPO,VS_ASPIRE_PROJECT,VS_ASPIRE_PYTHON,VS_ASPIRE_GIT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_PILOT_MODEL,VS_PILOT_OUTPUT,VS_PILOT_ALLOW_DOWNLOAD,VS_PILOT_BASELINE,VS_PILOT_DISCOVERY,VS_PILOT_VALIDATION,HF_TOKEN_PATH
if [ -n "${VS_ASPIRE_INPUT:-}" ]; then
  export VS_ASPIRE_INPUT
  vs_export=$vs_export,VS_ASPIRE_INPUT
fi
set --
if [ -n "${VS_PILOT_AFTER:-}" ]; then
  if [[ ! "$VS_PILOT_AFTER" =~ ^[0-9]+(\.[A-Za-z0-9_.-]+)?$ ]]; then
    echo 'Invalid VS_PILOT_AFTER job ID' >&2
    exit 2
  fi
  set -- -W "depend=afterany:$VS_PILOT_AFTER"
fi
cd "$VS_ASPIRE_REPO"
job_id=$(qsub -P "$VS_ASPIRE_PROJECT" -N "vp-${VS_PILOT_MODEL:0:12}" -o "$pbs_log" \
  "$@" -v "$vs_export" scripts/aspire_project_pilot.sh)
printf 'PBS output (usually delivered at job end): %s\n' "$pbs_log" >&2
printf 'Run folder: %s\nLive logs: %s/logs/\n' "$VS_PILOT_OUTPUT" "$VS_PILOT_OUTPUT" >&2
printf '%s\n' "$job_id"
