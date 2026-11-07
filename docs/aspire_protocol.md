# ASPIRE: one PBS launcher for every model

`scripts/aspire_protocol.pbs` calls the existing `visual_synthesis` CLI,
`validate_hooks.py`, `e4_analyze.py`, and `e1e6_report.py`. There is no separate
ASPIRE Python controller, persistent path overlay, readiness marker, or automatic
continuation job. The default allocation is one A100 GPU, 16 CPUs, 110 GB RAM,
and 72 hours. Submit to `normal`; PBS routes the long request to `glong`.

## Submit Qwen

From the repository on Aspire, after any previous job has finished:

```bash
GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin main
git lfs pull --include='data/processed/e1e6/*' --exclude=''
mkdir -p logs/aspire
qsub -P personal-swarangi -v VS_PROTOCOL_MODEL=qwen-image scripts/aspire_protocol.pbs
```

Use `qstat -u "$USER"` to see active jobs. PBS stdout/stderr goes under
`logs/aspire/`, not into the repository root. Inspect the newest job log with:

```bash
ls -lt logs/aspire
tail -n 100 logs/aspire/JOB_ID.OU
qstat -xf JOB_ID | grep -E 'job_state =|Exit_status =|comment =|admin_comment ='
```

`F` means finished, including failures; check the exit status and final log
message. A successful whole pipeline ends with `COMPLETE MODEL phase=all`.
On failure, the log ends with `FAILED: step=... exit=...`, after the original
Python exception. PBS history may expire; keep the log.

Change `VS_PROTOCOL_MODEL` to another configured model ID. The four expensive
models (SD 3.5 Large, FLUX, Qwen and GLM) use the frozen reduced tables; the other
four use the full tables. E7 always uses all 282 variants. For a future model,
set `VS_PROTOCOL_TIER=reduced` if needed and configure its hooks in the normal
model config. No new launcher is needed.

An optional phase runs readiness and then only that experiment:

```bash
qsub -P personal-swarangi -v VS_PROTOCOL_MODEL=qwen-image,VS_PROTOCOL_PHASE=ready scripts/aspire_protocol.pbs
qsub -P personal-swarangi -v VS_PROTOCOL_MODEL=qwen-image,VS_PROTOCOL_PHASE=e1 scripts/aspire_protocol.pbs
```

Default `all`: downloads/cache repair → hook validation → two evaluated pilot
images → E1 → E7 → E4 capture/analysis → eight E5 conditions → E5-based mode
selection → four E6 conditions → final report. E2/E3 are derived from E1.
Readiness runs in the same allocation as production. It is never inferred from
an old marker file. Failed hooks, incomplete detections, missing/non-finite
metrics or non-finite saved activations stop the job.

## Paths and resumption

| Location | Purpose |
|---|---|
| `data/processed/e1e6/` | Existing frozen prompt tables; never rebuilt |
| `data/runs/s01_download/`, `s05_generate/`, `s06_nsfw/`, `s07_metrics/` | Normal pipeline outputs and manifests |
| `results/hook_validation/`, `results/e4/`, `results/e1e6/` | Normal analysis outputs |
| `logs/aspire/` | PBS output, job config and package versions in the log |
| `runtime/models/`, `runtime/hf-home/` | Reused weight/token caches from the earlier Aspire setup |
| Node-local temporary directory | Live job config, removed when the job exits |

The per-job config only changes execution: native model precision, no
quantization, CPU offload, one worker/batch, paths, and local evaluator caches.
Prompt tables, seed 42, resolutions, steps, guidance, hooks, detector thresholds
and E5/E6 rules remain in the existing configs. Qwen stays BF16; its TC2 FP8
setting is a recorded cluster difference. Package versions are printed by
`pip freeze` into the job log. The packages are not upgraded by the launcher.

After a timeout or corrected error, submit the identical command again.
`--mode incremental` reuses completed images and detector records. Readiness
and metric/report calculations rerun. No job resubmits itself. Do not move,
rename, delete, or edit the active run directories while a job is queued,
running or exiting.

To resume valid outputs in the previous controller's workspace without moving
anything, explicitly select that runs directory:

```bash
qsub -P personal-swarangi \
  -v VS_PROTOCOL_MODEL=qwen-image,VS_ASPIRE_RUNS="$PWD/runtime/protocol-v1/qwen-image-reduced/runs" \
  scripts/aspire_protocol.pbs
```

Optional `VS_ASPIRE_PYTHON`, `VS_ASPIRE_STORAGE`, `VS_ASPIRE_MODELS`, and
`VS_ASPIRE_RUNS` select an existing Python environment, cache root, model cache,
and run directory. Defaults are the previously used
`~/.conda/envs/visual-safety/bin/python`, `runtime`, `runtime/models`, and
`data/runs`. Keep Hugging Face credentials in `.env` or the existing token file.

## What was removed and what stays

The old `aspire_protocol.py` and `e1e6.aspire.json` are retired. Their
`runtime/protocol-v1/*` checkpoints and archives are not needed by this launcher.
Earlier `runtime/outputs/*` folders contain pilot, quantization and calibration
results, not redundant scripts. Preserve those experiments and the model caches.
The older PBS `.o`/`.e` files are ignored by Git; new logs stay in `logs/aspire`.
No generated images, cached weights, or old run folders are deleted by this change.

After all previous jobs have finished, move only the old root-level PBS logs
into one archive folder (the logs are preserved):

```bash
mkdir -p logs/aspire/legacy
find . -maxdepth 1 -type f \( -name '*.o[0-9]*' -o -name '*.e[0-9]*' \) \
  -exec mv -n -t "$PWD/logs/aspire/legacy" -- {} +
```

The FP32 activation-storage correction remains in the core hook code. BF16
model activations can exceed FP16's finite range; saving steps in FP32 avoids
that data loss. Existing captures containing infinities must be regenerated.
