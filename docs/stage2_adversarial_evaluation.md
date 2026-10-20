# Stage 2 adversarial prompt evaluation

Stage 2 evaluates standardized Ring-A-Bell prompts through the existing Diffusers
generator and Stage 6 detector. Results are **nudity-related ASR**: a positive
means a detector score crossed its configured threshold. It is not a human
confirmed policy violation or a measure of overall safety, violence, illegal
activity, or copyright compliance.

## Dataset

The input is `data/processed/stage2/ring_a_bell_standardized.csv`, derived from
the Ring-A-Bell `Nudity_eta_3_K_16.csv`, `...K_38.csv`, and `...K_77.csv` files.
Its required columns are `prompt_id`, `prompt`, `source_dataset`,
`attack_method`, `attack_config`, `case_number`, `evaluation_seed`, and
`source_file`. The available file has 285 unique prompt IDs, 95 cases and 95
prompts per configuration. Preserve the original standardized and raw CSVs.

SneakyPrompt vocabulary and benign controls are not adversarial examples and
are excluded. MMA-Diffusion is not included until benchmark access is available.

## HPC setup and commands

```bash
conda activate visual-synthesis
pip install -e '.[gpu,nsfw]'
```

Run the matched-case smoke evaluation first. It selects three cases containing
all three configurations (nine prompts total) and uses their original seeds:

```bash
visual-synthesis stage2 -c config/pipeline.json -c config/stage2.hpc.json --smoke --run-name smoke
```

After reviewing its outputs, run all 285 prompts:

```bash
visual-synthesis stage2 -c config/pipeline.json -c config/stage2.hpc.json --full --run-name ring-a-bell-sd-turbo
```

SD-Turbo is the default. SDXL can use the existing `sd-xl` model entry:

```bash
visual-synthesis stage2 -c config/pipeline.json -c config/stage2.hpc.json --full --model sd-xl --run-name ring-a-bell-sd-xl \
  --set generate.num_inference_steps=30 --set generate.guidance_scale=7.5 \
  --set generate.width=1024 --set generate.height=1024
```

Run on an allocated GPU node with weights and NudeNet available. Adapt the
partition/account and environment to your cluster; for example:

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00
conda activate visual-synthesis
visual-synthesis stage2 -c config/pipeline.json -c config/stage2.hpc.json --smoke --run-name smoke
```

Do not launch the full evaluation as part of setup or tests. The Stage 2
command performs actual generation and detector inference.

## Outputs and resume

Artifacts are isolated at `results/stage2/<model-id>/<run-name>/`. Its `runs/`
folder contains the regular Stage 5 and Stage 6 manifests, images, detector
scores and failure records. The run root contains:

- `generation_results.csv`: all per-prompt generation records, including failures.
- `nsfw_results.csv`: per-image raw detector results, threshold and failures.
- `attack_results.csv`: one row per attempt with prompt and attack metadata,
  generation status, image path, detector status, raw max score and flag.
- `metrics_summary.json`: prompt and case ASR, attempts, failures, threshold,
  generation times and total command wall time.
- `metrics_by_config.csv`: separate results for K_16, K_38 and K_77.

Repeating a command with the same run name resumes it. Successful generations
are skipped; generation failures stay in the manifest and are retried. Use a
new run name for a separate run. Stage 1 defaults and artifacts are unchanged.

## Definitions and limitations

- Prompt ASR = positive detector results / all attempted adversarial prompts.
  Generation and detector failures remain in the denominator and are counted
  separately; they do not count as positives.
- Configuration ASR applies the same definition to each K setting.
- Case ASR is the proportion of underlying cases with at least one positive
  result among their three configurations.
- Generation failure rate is failed generations / attempts. Successful image
  count is also reported separately.
- Per-image generation time is backend-reported. The summary's total generation
  runtime is the sum of per-image times; `total_runtime_s` is command wall time.

The original `evaluation_seed`, model, detector threshold, raw scores, image
paths and attack metadata are retained. Results can still vary with GPU and
PyTorch/Diffusers versions and deterministic-kernel support.
