# Ring-A-Bell evaluation in the main pipeline

The acquired Ring-A-Bell raw CSVs live in `data/raw/ring-a-bell/`. The existing
285-row standardized table is `data/processed/standardized/ring-a-bell.parquet`.
Its `attack_config`, `case_number`, `evaluation_seed`, and `source_file` fields
are retained by stages 2 through 7. The raw prompt data's redistribution
terms still need confirmation before publishing this branch.

## What was integrated

- Stage 2 accepts the existing standardized Ring-A-Bell table through
  `datasets.ring-a-bell.input_table` in `config/pipeline.json`. The CPU chain
  `standardize -> unify -> postprocess` produced
  `data/runs/{s02_standardize,s03_unify,s04_postprocess}/integrated-benchmark/`.
  All 285 Ring-A-Bell prompts survive postprocessing.
- Stages 5 and 6 already handled image generation and NSFW detection. The
  separate `stage2` wrapper and its two extra config files were removed.
  Both completed 285-image runs now live under `data/runs/s05_generate/` and
  `data/runs/s06_nsfw/`, with image paths updated. Their original manifests
  record `relocated_from` and retain the historical experiment configuration.
- Stage 7 now computes attempt-aware prompt ASR, case ASR, configuration ASR,
  and generation/detector failure counts when `attack_config` is present.
  It writes `attack_results.parquet` and `attack_by_config.parquet` beside
  `metrics.parquet` and `metrics.json`. Saved full-run outputs are under
  `data/runs/s07_metrics/ring-a-bell-sd-turbo/` and
  `data/runs/s07_metrics/ring-a-bell-sdxl/`.

## Results from saved runs

| Model | Prompt ASR | Case ASR | Generation failures | Detector failures |
| --- | ---: | ---: | ---: | ---: |
| SD-Turbo | 243/285 (85.26%) | 94/95 (98.95%) | 0 | 0 |
| SDXL Base 1.0 | 271/285 (95.09%) | 95/95 (100%) | 0 | 0 |

These are NudeNet nudity-related positive rates with threshold 0.3 and the
configured exposure labels. A positive is not a human-confirmed policy
violation or a measure of other safety categories. All attempted prompts are
in the denominator; failed generation or detection attempts count separately.
The saved summaries and recomputed stage 7 metrics agree. No model inference
was rerun during this integration.

## Commands

Activate `visual-safety` and install the project as described in the README.
Use only the existing base and HPC configs:

```bash
visual-synthesis run all --from standardize --to postprocess \
  --run-name integrated-benchmark --mode incremental \
  -c config/pipeline.json
```

To recalculate SD-Turbo metrics from saved runs (no model inference):

```bash
visual-synthesis metrics --run-name ring-a-bell-sd-turbo --mode incremental \
  --input data/runs/s06_nsfw/sd-turbo-ring-a-bell-sd-turbo \
  -c config/pipeline.json -c config/pipeline.hpc.json \
  --set metrics.generations_input=data/runs/s05_generate/sd-turbo-ring-a-bell-sd-turbo \
  --set metrics.prompts_input=data/processed/standardized/ring-a-bell.parquet \
  --set metrics.alignment.enabled=false
```

For SDXL, substitute `sd-xl-ring-a-bell-sdxl` for the generation and detection
run names and `ring-a-bell-sdxl` for the stage 7 run name. Stage 7's optional
CLIP alignment remains disabled in these saved evaluations, so this command
performs only CPU table aggregation and plotting. For a future evaluation,
pass the stage 2 `ring-a-bell.parquet` directly to `visual-synthesis generate`
with a chosen model and run name, then pass that run to `nsfw` and its result
to `metrics`. The generated table carries each row's original evaluation seed.
