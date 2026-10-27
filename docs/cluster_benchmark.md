# Pull and run the resource pilot on TC2

The checkout contains the eight-model profile, level-aware pilot, cluster
preflight and GPU-job wrapper. Download only your assigned pair. This workflow
profiles pretrained evaluation, with no training job.

## Pull and install

From your cluster checkout:

```bash
git fetch origin
git switch swara/experiments
git pull --ff-only origin swara/experiments
git lfs install
git lfs pull --include="data/processed/unified/unified_prompt_candidates.csv"
```

Use your working Python 3.11/3.12 environment with **CUDA PyTorch 2.6 or newer**.
The pinned [Transformers release requires PyTorch 2.6+](https://pypi.org/project/transformers/5.20.1/).
For example, if your environment is
named `visual-synthesis`:

```bash
conda activate visual-synthesis
python -m pip install -e '.[gpu,nsfw,metrics,benchmark]' \
  -c requirements/benchmark-cluster.txt
```

The constraint file pins the Diffusers/Transformers/Accelerate APIs checked
locally. It does not select a CUDA wheel or install a GPU driver; retain the
PyTorch/torchvision CUDA build supported by TC2. Full inference of the eight
checkpoints still needs validation on the allocated GPU.

Check imports and inputs on a login/download node, without loading model weights:

```bash
python scripts/benchmark_cluster.py check --device cpu --model sd-xl --model sd-1.5
```

Replace those IDs throughout with your assigned pair. Available IDs: `sd-xl`,
`sd-3.5-large`, `flux-dev`, `qwen-image`, `z-image-turbo`, `glm-image`,
`pixart-sigma`, `sd-1.5`. A CPU check does not establish CUDA availability.
The helper detects missing inputs and undownloaded Git LFS pointers.

## Prefetch your pair and evaluator

Run where network downloads are allowed, ahead of the GPU job:

```bash
python scripts/benchmark_cluster.py prefetch --model sd-xl --model sd-1.5
```

This completes the selected model folders, checks their components/shards and
caches CLIP alignment. NudeNet uses its bundled ONNX file. Gated models require
your existing Hugging Face access, configured through normal HF login. Tokens
are not benchmark arguments. If using `HF_HOME` or `VS_MODELS_DIR` for scratch
storage, set them consistently for both prefetch and the GPU job.

## Fit test and timing pilot

Inside an allocated GPU node, with your environment active:

```bash
VS_BENCH_MODE=fit bash scripts/benchmark_tc2.sh sd-xl sd-1.5
bash scripts/benchmark_tc2.sh sd-xl sd-1.5
```

The fit test uses two sampled prompts, one warmup, one repeat and generation
only. It is a fit/load check, not a representative timing estimate. The pilot
uses 60 prompts, two warmups, three repeats, NudeNet and CLIP; FID is disabled.
Both run CUDA/import/input/cache preflight checks before loading a generator.
Models run sequentially in fresh processes, with one model copy per allocation.
HF downloads are blocked during timing.

To submit instead, supply the resources your TC2 account permits. Example flags
for the lighter example pair, with the partition placeholder replaced:

```bash
sbatch --partition=YOUR_PARTITION --export=ALL --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --time=04:00:00 \
  scripts/benchmark_tc2.sh sd-xl sd-1.5
```

Add account/QoS flags if required and choose RAM/GPU/time for the selected
models. The wrapper uses the activated environment; it does not guess modules
or switch environments. Submit the job yourself after choosing those resources.

| Environment variable | Default / purpose |
|---|---|
| `VS_BENCH_MODE` | `pilot` or `fit` |
| `VS_BENCH_INPUT` | Tracked unified candidate CSV |
| `VS_BENCH_OUTPUT` | New `results/benchmarks/tc2-<job-or-timestamp>-<mode>` folder |
| `VS_BENCH_SAMPLE_SIZE` | `60` for pilot, `2` for fit |
| `VS_BENCH_TARGET_PROMPTS` | `6000` per model per pass |
| `VS_BENCH_TIMEOUT` | `7200` seconds per model pilot |
| `VS_BENCH_OFFLOAD` | `1` enables CPU offload; default `0` |
| `VS_BENCH_PYTHON` | `python` from the activated environment |

Use new output folders for independent runs. CPU-offloaded and resident runs
need separate measurements. For more options, use
`python scripts/benchmark_resources.py --help` directly, layering `pipeline.json`,
`benchmark.models.json`, then `benchmark.hpc.json`.

## Input readiness and projections

The default tracked manifest has 14,223 candidates and zero human-validated rows.
It contains L0/L1/L3/L4/L6 and no L2. The 60-row pilot currently takes 12 prompts
per available level, with category coverage inside levels where possible.
Unknown levels are excluded. Missing levels make extrapolations **PROVISIONAL**
and suppress complete project totals. This profiles candidate text; it does not
produce final scientific benchmark scores or invent missing prompts/reviews.
Set `VS_BENCH_INPUT` to the reviewed six-level manifest once available.

Complete level pilots weight per-level timings equally, matching the planned
1,000 prompts per level. Per-level timings and sample coverage are retained.
Explicit attack seeds, including seed zero, are preserved across models and
repeats. Other prompts use deterministic fallback seeds.

## Optional source exports for the merged workflows

The tracked manifest retains the text and provenance for the legacy L1 and
Ring-A-Bell CSVs that are absent from the checkout. Restore those exact rows:

```bash
python scripts/benchmark_cluster.py prepare-inputs \
  --output results/benchmarks/cluster-inputs
```

This exports a pilot sample, the 285-row Ring-A-Bell CSV, the original
10,892-row I2P/T2I-RiskyPrompt L1 pool and a checksum/provenance report. It
does not generate attacks, transform text, validate candidates or download data.
Outputs are ignored local artifacts; choose another folder if it already exists.

For a Stage 2-only resource pilot:

```bash
VS_BENCH_INPUT=results/benchmarks/cluster-inputs/ring_a_bell_standardized.csv \
VS_BENCH_TARGET_PROMPTS=285 \
bash scripts/benchmark_tc2.sh sd-xl sd-1.5
```

Automatic sampling keeps complete cases across K_16/K_38/K_77. Sample sizes
round down to three-prompt cases, with a minimum of one case. This report is an
attack-only projection, not the balanced six-level workload. Avoid counting
these prompts twice if they already form part of the final L6 allocation.
The existing `stage2` command can also use this CSV and the now-present
`config/stage2.hpc.json`; pass `--input` explicitly. Original source paths are
not silently created or replaced.

## Collect results

The run root has `report.json`, `summary.csv`, `sample.parquet` and
`preflight.json`. Each model has a report, `worker.log`, images and manifests.
Reports retain GPU/RAM peaks, storage, failures, seeds, per-level timings and
provisional status. GPU allocator peaks exclude other jobs/context memory;
leave headroom. Queue time, manual review and report writing are additional.

Merge members' reports with `visual-synthesis benchmark-summary ... --output
results/benchmarks/team`. Use identical input files, sampling strategy, seed and
target workload. Provisional pilots do not produce a complete project total.
See [the estimation guide](benchmarking.md) for formulas and limits.
