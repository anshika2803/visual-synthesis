# TC2 results: SD 1.5, SDXL, Z-Image Turbo and SD 3.5 Large

These four models were run on the TC2 Slurm cluster at tag `protocol-v1` (`a0ee1af`). The
other four (PixArt-Sigma, FLUX, Qwen, GLM) are in `RESULT.md`, from ASPIRE.

## Reports

| Model | Tier | Tables and E6 plot | E4 direction | Hook check |
|---|---|---|---|---|
| SD 1.5 | full | `results/e1e6/sd-1.5/` | `results/e4/sd-1.5/` | 21/21 PASS |
| SDXL | full | `results/e1e6/sd-xl/` | `results/e4/sd-xl/` | 21/21 PASS |
| Z-Image Turbo | full | `results/e1e6/z-image-turbo/` | `results/e4/z-image-turbo/` | 16/16 PASS |
| SD 3.5 Large | reduced | `results/e1e6/sd-3.5-large/` | `results/e4/sd-3.5-large/` | 21/21 PASS |

- Every report was made with `scripts/e1e6_report.py tables --exclude results/review/excluded_prompts.csv`.
- E3 compares models on the 438 reduced-tier prompts.
- E5/E6 use the validation split: 209 prompts for the full tier, 106 for SD 3.5.
- E6 ran in ablation mode for all four, chosen by the frozen rule.

## Notes for the central analysis

- **SD 1.5 E1 images** come from run `pilot-e1-sd-1.5`. It was generated on 11-03 on the
  provisional 930-prompt table, before the freeze. The excluded prompts are dropped, so the
  sexual-scope denominator is 341, the same as the other full-tier models. All other SD 1.5 runs use
  the frozen tables.
- **SD 1.5 E1 was re-scored with the calibrated detector setting on 11-12 (job 53708).**
  - Its first scoring predates the calibration and used LAION raw `nsfw` at 0.5.
  - The same 930 images were re-scored with NudeNet 0.3 and LAION `nsfw_l2` at 0.55, into runs
    `e1-sd-1.5-nudenet` and `e1-sd-1.5-laion`, with metrics run `e1-sd-1.5`.
  - Every scoring run of all 8 models now uses the same detector setting.
  - Effect: ASR 0.739 → 0.733, benign flag rate 9.5% → 3.5%. The LAION verdict changed for 51 of
    the 930 images. E5/E6 are unchanged.
- **SDXL** was rerun from scratch on the frozen tables. The earlier pre-freeze SDXL runs are not
  part of these results.
- **cuDNN fix on fresh environments (TC2 users 2 and 3, SDXL and SD 3.5):**
  - Installing from `results/environment/tc2-visual-safety-pip-freeze.txt` puts both
    `nvidia-cudnn-cu12` 9.19 and `-cu13` 9.24 into the same folder, and the CUDA 13 copy can win.
    That fails on TC2's driver 570 with `CUDNN_STATUS_NOT_INITIALIZED`.
  - Fixed with `pip install --no-deps --force-reinstall nvidia-cudnn-cu12==9.19.0.56`, then
    verified: `torch.backends.cudnn.version()` = 91900 and a test convolution runs.
  - Package versions are unchanged, so the freezes in `provenance/tc2/versions/` still apply.
- Run manifests for every stage are in `results/provenance/tc2/manifests/`. Package versions per
  account are in `results/provenance/tc2/versions/`.

## Main findings

| | SD 1.5 | SDXL | Z-Image | SD 3.5 Large |
|---|---|---|---|---|
| E1 ASR (n) | 0.73 (341) | 0.73 (341) | 0.69 (341) | 0.56 (191) |
| Benign flag rate | 3.5% | 4.5% | 4.5% | 6.0% |
| E3 ASR L1 / L3 / L4 / L6 | 0.65 / 0.75 / 0.46 / 0.95 | 0.81 / 0.31 / 0.36 / 1.00 | 0.87 / 0.38 / 0.32 / 0.93 | 0.81 / 0.44 / 0.14 / 0.76 |
| E5 direction-specific conditions | 0 of 8 | 0 of 8 | 0 of 8 | 0 of 8 |
| E7 full translation fr / sw / zh | 0.65 / 0.19 / 0.10 | 0.74 / 0.23 / 0.13 | 0.71 / 0.26 / 0.68 | 0.74 / 0.16 / 0.06 |

- **E5/E6:** no intervention meets the pre-registered direction-specific criterion.
- **SD 3.5 E6:**
  - Ablating the safety direction lowers harmful ASR step by step, from 0.69 to 0.50 at α = 1.
  - The random direction gives 0.77, a gap of −0.27 [−0.46, −0.08].
  - But the drop from baseline is not significant (Holm p = 1.0, n = 26).
  - Report it as a trend, not an effect. Benign suppression stays at or below 10%.
- **E6 `status=ok`** means only that the suppression limit holds, not that the trade-off succeeded
  (same reading as `RESULT.md`).
- **E7:** with a full Swahili or Chinese translation, ASR drops for the CLIP-encoder models. Prompt
  alignment drops with it, which suggests the model doesn't understand the prompt, not that it is
  safer. Z-Image, with a multilingual encoder, stays at 0.68 for Chinese.
- **E4:** high probe AUC shows the prompt groups can be decoded from the activations. It does not
  by itself identify a safety mechanism.

## Retention

Generated images, E4 activation captures, detector outputs and per-image metrics stay on TC2 under
each account's project directory, in `data/runs/s05_generate`, `s06_nsfw` and `s07_metrics`.
Keep them until submission. Reviewer B's 50 labels (`results/calibration/labels_reviewer_B.csv`)
are still missing. If their agreement check changes the detector setting, every run gets re-scored
there. Images are never committed.
