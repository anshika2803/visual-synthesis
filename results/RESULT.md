# ASPIRE results: PixArt, FLUX, Qwen and GLM

## Current reports

The completed ASPIRE result tables and plots on `swara/runs` are:

- `results/e1e6/pixart-sigma/`
- `results/e1e6/flux-dev/`
- `results/e1e6/qwen-image/`
- `results/e1e6/glm-image/`

The four `e3.csv` files use the same 438 reduced-tier prompts before filtering
into the reported levels and sexual/nudity scope.

E1 and E2 retain each model's original full/reduced cohort. PixArt E5/E6 use 209
validation prompts; FLUX, Qwen and GLM use 106. Do not compare those populations
as if identical. Central cross-model ASR, alignment and quality comparisons
should use shared prompt IDs and report sample sizes and confidence intervals.

## Interpretation

No tested intervention demonstrated a statistically supported nudity-related
safety improvement while preserving benign and near-boundary image quality.
This conclusion applies to the tested layers, strengths, prompts and seed.

| Model | Main finding |
|---|---|
| PixArt | Large reductions in nudity flags coincide with substantial quality loss and suppression. E6 reports `not reached`. |
| FLUX | Quality is largely maintained at weak settings, but the safety reductions are statistically uncertain and no E5 direction-specific benefit is established. |
| Qwen | Strong steering reduces nudity flags with severe quality loss. E6's selected alpha 0.25 increases harmful ASR by 3.85 percentage points. |
| GLM | Strong steering reduces harmful-prompt flags with substantial quality loss. E6's selected alpha 0.25 gives zero harmful-ASR reduction. |

E6 `status=ok` means a tested alpha satisfied the benign-suppression constraint
(at most 10%). It does not establish a positive ASR reduction, statistical
significance, an advantage over the random direction, or preserved quality.
Do not describe FLUX, Qwen or GLM's selected E6 row as a successful trade-off.

E4's strong probe accuracy shows that prompt-group information is decodable;
it does not by itself identify a safety mechanism. E7 reductions in flags that
coincide with reduced prompt alignment may reflect semantic/comprehension loss.
Safety measurements cover nudity detected automatically, with built-in safety
checkers disabled. Quality is measured with automatic metrics. One seed per
prompt was used; confidence intervals do not measure variation across seeds.

## Supporting evidence and retention

Available stage manifests, resolved configurations and execution-time package
versions belong in `results/provenance/aspire/`. Raw logs and large run artifacts
remain on ASPIRE. Exported evidence and any missing records are listed in
`inventory.json`; checked retention paths and counts are listed in
`retention.json` when those metadata files are supplied.

The root on ASPIRE is `/scratch/users/ntu/swarangi/visual-synthesis`:

| Evidence | Retention location relative to that root |
|---|---|
| Generated images and generation records | `data/runs/s05_generate/e*-<model>*/` |
| E4 activation captures | `data/runs/s05_generate/e4-<model>-capture/activations/` (or the manifest's configured capture subdirectory) |
| Detector outputs | `data/runs/s06_nsfw/e*-<model>*/` |
| Per-image metrics for central analysis | `data/runs/s07_metrics/e*-<model>*/image_metrics.parquet` |
| Job/task logs and saved accounting | `logs/aspire/` |

Keep these originals on ASPIRE until Jay confirms submission and retention is
no longer needed. Generated images and activation arrays are excluded from the
repository handoff. Per-image metrics remain available for a separate data
handoff or central analysis on ASPIRE.

## Central follow-up

Jay coordinates reviewer-B calibration/agreement and accepted evaluator
settings, any rescoring, A1 exclusion confirmation, A3 translation spot-check
records, and compatibility with the other models. Conclusions remain
conditional on those central checks.
