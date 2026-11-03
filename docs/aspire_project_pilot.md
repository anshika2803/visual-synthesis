# Larger ASPIRE 2A resource pilot with evaluation and interventions

Branch: `swara/experiments`. All eight models are supported on one A100 40GB per job, with CPU offload,
16 CPUs and 110GB RAM. The existing eight-model [fit measurements](aspire_trial_results.md)
establish baseline execution. This pilot adds real capture/intervention hooks
and evaluator timing. GPU behavior remains to be verified on the cluster; local
checks exercise tensor mathematics, sampling, projections and PBS submission.

## Per-model protocol

| Phase | Measured images | Purpose |
|---|---:|---|
| Baseline | 60 | More representative generation timing, including held-out validation rows |
| Capture | 32 | 10 benign + 10 sexual-harmful discovery rows, 6+6 disjoint validation rows; five candidate layers |
| Learn | — | Mean direction and small CPU logistic probe per layer; intervention layer fixed to middle candidate |
| Alpha-zero | 2 | Same prompt/seed as baseline; exact saved-image hash equivalence |
| Safety ablation, alpha 1 | 12 | Centered removal of learned direction |
| Random ablation, alpha 1 | 12 | Deterministic unit random direction, seed 0 |
| Safety steering, alpha 1 | 12 | Shift by learned mean-difference magnitude |
| Random steering, alpha 1 | 12 | Random direction with matched magnitude |
| Safety ablation, alpha 2 | 12 | Profile an additional strength |
| Evaluation | 154 existing images | NudeNet, LAION CLIP-H/14, CLIP alignment, CLIP-IQA B/32 quality variant |

Each generation phase loads the model in a fresh process and does one unhooked
warmup. Load, warmup, per-image generation/writing, GPU allocated/reserved peaks,
worker RSS and disk changes are separate. Evaluation uses fresh processes after
generation; NudeNet is CPU, other evaluators use the GPU. A CPU learning phase
records observed time rather than pretending it is generator fine-tuning.
The default fits the six-hour request as a planning expectation, not a measured
runtime guarantee. Baseline GLM alone is expected to take roughly 90 minutes.
Per-phase timeout is two hours; PBS imposes the six-hour whole-job limit.

## Capture/intervention implementation

`src/visual_synthesis/pilot_interventions.py` defines checked image-stream
adapters for all eight models. Captures immediately pool spatial/token dimensions
and copy only channel vectors to CPU, averaging conditional calls across steps.
The stored `activations.npz` contains one vector per image/layer, not every full
token map. This intentionally profiles pooled capture; it does not predict the
cost of dumping full token/spatial activations.

The current adapters were inspected against local Diffusers 0.41.0 source:

- SD 1.5/SDXL: representative down/mid/up blocks; capture conditional CFG half.
- SD 3.5/PixArt: five transformer blocks; SD 3.5 uses image output index 1.
- FLUX: five blocks spanning dual/single lists; image output index 1.
- Qwen: five transformer blocks, image output index 1; capture the first of each
  sequential conditional/unconditional pair, as in the pinned pipeline loop.
- Z-Image: five main layers, basic text-to-image image-token prefix only.
- GLM: five **DiT decoder** blocks, image output index 0, sequential CFG.
  This is the plan's decoder fallback, not an AR-language-model ablation.

Runtime output ranks, image-token range, batch size, hook execution and
sequential CFG call parity are checked. Hook paths, shapes and call counts are
written into reports. Offload stays enabled. Interventions affect the image
stream in both conditional and unconditional passes; text streams are preserved.

For benign center mu, unit learned/random direction d, difference magnitude m:

```text
ablation: h' = h - alpha * ((h - mu) dot d) * d
steering: h' = h - alpha * m * d
```

Alpha zero returns the original tensor exactly. A failed image equivalence check
stops dependent intervention phases and leaves their costs unknown. GPU OOM in
capture/intervention is recorded; baseline images can still be evaluated.
Random controls use the same prompts, seeds, center and magnitude. Discovery
rows do not overlap the baseline/validation rows. The fixed middle intervention
layer avoids selecting a direction using the same held-out validation effects.

The CLIP-IQA variant uses the same OpenAI ViT-B/32 checkpoint as alignment with
normalized image/text features and `softmax(100 * similarity)` for `Good photo.` /
`Bad photo.`. This follows the two-anchor formulation in
[TorchMetrics CLIP-IQA](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html),
with feature-output handling compatible with Transformers 5.20.1. It is not the
separate PIQ `clip_iqa` checkpoint. LAION scoring uses the repository's existing
head/preprocessing; calibration is still required before final safety claims.

## Submit all eight or selected models

Run on the login node after following [SETUP.md](../SETUP.md), using your own project:

```bash
cd /path/to/your/visual-synthesis
export VS_ASPIRE_PROJECT=YOUR_PROJECT_ID
export VS_ASPIRE_STORAGE="$PWD/runtime"
export VS_ASPIRE_BUDGET_GIB=500
export HF_TOKEN_PATH="$VS_ASPIRE_STORAGE/hf-home/token"

bash scripts/submit_aspire_project_pilot.sh all
```

The helper issues the exact `qsub -P ... -N ... -o ... -v ...` submission,
using `scripts/aspire_project_pilot.sh`. It exports explicit shared Git/Python
paths and the credential **file path**, restores PBS's submission PATH and
verifies `swara/experiments`. It never switches branches or deletes files.
Login work is limited to path/branch checks, directory creation and submission.

`all` queues all eight as separate jobs in the existing heavy-first order,
with Qwen first for early feedback on its tight memory headroom. No argument
also defaults to `all`. Inspect hook fit/identity results before production.
To profile a particular assigned pair instead of all eight, for example:

```bash
first_job=$(bash scripts/submit_aspire_project_pilot.sh sd-xl)
VS_PILOT_AFTER="$first_job" bash scripts/submit_aspire_project_pilot.sh sd-1.5
```

`afterany` allows the next model to be profiled even if Qwen cannot fit capture.
One workspace lock is shared with the old fit runner, so do not run simultaneous
jobs against that same workspace. Teammates use their own workspace/account.
For one model after a previous job has already left scheduler history, omit
`VS_PILOT_AFTER` and submit its helper command directly.

Optional size overrides, exported before submission:

```bash
export VS_PILOT_BASELINE=60       # Must exceed twice validation-per-class
export VS_PILOT_DISCOVERY=10      # Per class; >=2
export VS_PILOT_VALIDATION=6      # Per class; >=2
export VS_PILOT_ALLOW_DOWNLOAD=1 # Default: prepare missing resources individually
```

Use identical source input/counts for all models. `VS_ASPIRE_INPUT` can select an
approved shared CSV/Parquet; explicit row seeds are preserved and missing seeds
use stable prompt-ID hashes. All generation RNGs are reset per prompt, including
the global RNG used by some autoregressive prior samplers, in addition to the
latent generator. Default inputs remain candidates, lack L2 and have no
reported human validation. This pilot measures resources; direction/probe
accuracy, detector flags and image scores cannot establish the final study's
safety conclusions. Near-boundary E6 prompts are not provided by this pilot.

## Results and whole-project forecasting

All new logs, for every model and run, share `<checkout>/runtime/logs/`.
Names such as `<model>-<unique-id>.baseline.log` avoid overwriting older runs.
All output files and reports are under
`<checkout>/runtime/outputs/<model>-<unique-id>/`. See the folder map in `SETUP.md`.
Each phase's `spec.json`, `report.json`, images and scores live in `<phase>/`.
Image-producing phases also have PNGs and `images.jsonl`. Capture adds pooled activations;
learning adds `direction.npz`. `pilot_summary.json` is updated between phases.

The forecast includes:

- Original baseline: 6,000 images/model.
- E4 capture: 500/model.
- E5: four profiled mode/vector combinations × two strengths × 300 images.
- E6: two extra strengths × two ablation vectors × 300 images.
- All four evaluator loads and measured per-image inference/writing.
- Repeated generator/evaluator setup for estimated job chunks, allowing ten
  minutes of each six-hour job for other startup/exit work.

This is 10,100 images/model for the extended scope. Core (5,300/model) and
reduced-core (2,650/model) forecasts are saved too. E5 strength costs use the
profiled mode at alpha 1; E6 safety uses alpha 2, random uses its alpha-1 timing.
Those strength assumptions, queue time, retries and actual CPU probe learning
at the final discovery size remain limitations. Failed or unmeasured phases
produce `incomplete` forecasts, never zero-cost estimates. Only 20 discovery
images train the pilot probes; CPU learning is observed, not linearly scaled.

After all eight pilots, aggregate the most recent report for each model:

```bash
"$HOME/.conda/envs/visual-safety/bin/python" scripts/aspire_project_pilot.py aggregate \
  --storage "$VS_ASPIRE_STORAGE" \
  --output "$VS_ASPIRE_STORAGE/project_estimate.json"
```

Missing/failed models remain explicit. A newest failure is not silently replaced
with an older success. The aggregate checks whether prompt-table hashes match;
cross-model scientific comparisons require matching reviewed inputs. Final PBS
accounting is attached after each job using the `collect` command in `SETUP.md`.
