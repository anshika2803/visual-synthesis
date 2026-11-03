# Qwen: diagnose steering collapse before scaling the experiment

The completed BF16 pilot (`qwen-image-1fICtV`) produced nearly blank images for
11/12 prompts under safety steering at alpha 1. Lower detector flags therefore
do not establish useful safety improvement. Ablation at alpha 1 left the union
flag count unchanged; alpha 2 introduced more explicit nudity in some examples.

The follow-up changes **steering strength only**, preserving the direction,
transformer block 30, all conditional/unconditional passes, BF16, offload,
50 steps, CFG, resolution, original prompts and seeds. It does not silently
normalize vectors differently or change which denoising steps receive the edit.
Per-call diagnostics measure the requested edit relative to incoming activation
RMS so that a later normalization/schedule change can be motivated by evidence.

## Initial diagnostic

- Reuse the stored baseline images and `learn/direction.npz`; do not repeat capture or learning.
- Use four previously examined prompts: two ordinary scenes (man/truck, bus),
  and two explicit adult-nudity examples (`TRP_02373`, `TRP_02490`).
- Check alpha zero on one prompt from each group. Any hash mismatch with the
  original baseline stops the experiment before the strength sweep.
- Test alpha **0.001, 0.01, 0.03, 0.1** for both learned and random directions.
  The single random direction uses the original pilot's seed 0 and matched magnitude.
- Generate **34 new images** and reuse four baselines. Load Qwen once, then
  unload it before running the same four evaluators on all 38 records.
- Expect about **37 minutes of generation** at 66 seconds/image, plus load,
  warmup, diagnostics and evaluation. Approximately **45–60 minutes** is a
  planning estimate; the PBS request is 90 minutes, excluding queue delays.

This is diagnostic tuning data, not a fresh held-out evaluation. Its two risk
examples were deliberately selected after examining baseline images. Do not
report their flag rate as an unbiased performance estimate.

## Inspect the plan locally

```bash
python scripts/aspire_qwen_calibration.py plan \
  --source runtime/outputs/qwen-image-1fICtV
```

This checks input-table hashes, baseline image hashes, original prompts/seeds,
the learned direction, fixed intervention layer and discovery/validation split.
It does not load model weights or require a GPU. Default prompt IDs target the
reviewed pilot; `--prompt-ids` and `--alphas` can override the diagnostic plan
when invoking the Python script directly.

## Transfer the three new scripts from the Mac

Use the same repository directory as the existing pilots. Run this in Mac Terminal:

```bash
rsync -av \
  scripts/aspire_qwen_calibration.py \
  scripts/aspire_qwen_calibration.sh \
  scripts/submit_aspire_qwen_calibration.sh \
  nscc-aspire2a:/home/users/ntu/swarangi/scratch/visual-synthesis/scripts/
```

The runner reuses existing project modules. No shared environment packages or
model checkpoints are installed or downloaded. Original pilot outputs are read-only.

## Submit from the cluster login node

Use the existing shared Python/Git executables and verified storage budget from
the original pilot setup. Supply your allocated project with `VS_ASPIRE_PROJECT`.

```bash
cd /home/users/ntu/swarangi/scratch/visual-synthesis
export VS_ASPIRE_STORAGE="$PWD/runtime"
# Keep your existing VS_ASPIRE_PROJECT and verified VS_ASPIRE_BUDGET_GIB values.
bash scripts/submit_aspire_qwen_calibration.sh \
  "$VS_ASPIRE_STORAGE/outputs/qwen-image-1fICtV"
```

The submission helper checks these settings and requires `swara/experiments`;
it does not change branches. It uses a unique `qwen-calibration-*` output folder
and run-prefixed logs under `runtime/logs`. All generation and scoring run inside PBS.
Cached generator, CLIP models, detector head and bundled NudeNet weights must
already be present. A final checkpoint metadata comparison detects concurrent
changes to the shared cache; it is not a full tensor-content hash.

## How to assess the results

Inspect `condition_summary.csv`, `paired_scores.csv`, original PNGs and
`hook_diagnostics.jsonl` together. The summary keeps benign and sexual-risk
groups separate and includes quality, prompt alignment, both detectors and
near-blank counts. Baseline blank-image heuristics remain unknown rather than
being silently assumed zero. The heuristic is a diagnostic, not a safety label.

For each learned strength, compare the **same strength's random control** and
the saved baseline. A candidate needs visibly reduced explicit content while
retaining recognizable benign scenes and reasonable quality/alignment. A white
image or a detector score barely crossing a threshold is insufficient evidence.
The runner deliberately does not select a winning strength automatically.

If no smaller strength preserves quality while changing risky content, inspect
the per-step shift ratios before changing normalization, conditional-pass scope,
denoising schedule, direction construction or layer. Change one factor at a time
and retain matching random controls. Any layer/strength chosen using these
prompts must later be evaluated on **new, reviewed prompts**, multiple seeds,
and multiple independent random directions. Discriminative probe accuracy alone
does not identify the best causal intervention layer.
