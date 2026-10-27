#!/usr/bin/env bash
# Run in an activated CUDA PyTorch environment on an allocated GPU node.
# Example: sbatch <your resource flags> scripts/benchmark_tc2.sh sd-xl sd-1.5
set -euo pipefail

vs_bench_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$vs_bench_root"
if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/benchmark_tc2.sh MODEL [MODEL ...]" >&2
  echo "Set VS_BENCH_MODE=fit for a tiny fit test; the default is a 60-prompt, 3-repeat pilot." >&2
  exit 2
fi

vs_bench_python=${VS_BENCH_PYTHON:-python}
vs_bench_mode=${VS_BENCH_MODE:-pilot}
vs_bench_input=${VS_BENCH_INPUT:-data/processed/unified/unified_prompt_candidates.csv}
vs_bench_output=${VS_BENCH_OUTPUT:-results/benchmarks/tc2-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}-${vs_bench_mode}}
vs_bench_config=(-c config/pipeline.json -c config/benchmark.models.json -c config/benchmark.hpc.json)
vs_bench_models=()
for vs_bench_model in "$@"; do
  vs_bench_models+=(--model "$vs_bench_model")
done
vs_bench_extra=(--passes 1)
case "$vs_bench_mode" in
  fit)
    vs_bench_sample=${VS_BENCH_SAMPLE_SIZE:-2}
    vs_bench_warmup=1
    vs_bench_repeats=1
    vs_bench_extra+=(--generation-only)
    ;;
  pilot)
    vs_bench_sample=${VS_BENCH_SAMPLE_SIZE:-60}
    vs_bench_warmup=2
    vs_bench_repeats=3
    ;;
  *) echo "VS_BENCH_MODE must be fit or pilot" >&2; exit 2 ;;
esac
vs_bench_check_extra=(--sampling auto)
if [ "$vs_bench_mode" = fit ]; then
  vs_bench_check_extra+=(--generation-only)
fi
if [ "${VS_BENCH_OFFLOAD:-0}" = 1 ]; then
  vs_bench_extra+=(--cpu-offload)
fi
vs_bench_check=$(mktemp)
trap 'rm -f "$vs_bench_check"' EXIT

"$vs_bench_python" scripts/benchmark_cluster.py check "${vs_bench_config[@]}" "${vs_bench_models[@]}" \
  --input "$vs_bench_input" --sample-size "$vs_bench_sample" --device cuda --require-cache \
  --output "$vs_bench_check" "${vs_bench_check_extra[@]}"

if "$vs_bench_python" scripts/benchmark_resources.py "${vs_bench_config[@]}" "${vs_bench_models[@]}" \
  --input "$vs_bench_input" --output "$vs_bench_output" --device cuda \
  --sampling auto --sample-size "$vs_bench_sample" --warmup "$vs_bench_warmup" \
  --repeats "$vs_bench_repeats" --target-prompts "${VS_BENCH_TARGET_PROMPTS:-6000}" \
  --timeout "${VS_BENCH_TIMEOUT:-7200}" "${vs_bench_extra[@]}"; then
  vs_bench_status=0
else
  vs_bench_status=$?
fi
if [ -d "$vs_bench_output" ]; then
  cp "$vs_bench_check" "$vs_bench_output/preflight.json"
fi
exit "$vs_bench_status"
