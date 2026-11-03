# Observed TC2 resource trials

Evidence received through 2026-10-28 (Asia/Singapore). These are user-supplied
Slurm logs, not runs performed on the local Mac. Raw imports and derived
observations are kept locally under the ignored
`results/benchmarks/tc2-imported-<JOBID>/` folders. No results are inferred for
models that have not run.

## Progress across the eight models

Each completed fit trial used one warmup and two measured prompts, one image
per prompt, generation only. The allocation was one GPU, ten CPUs, 30G host
RAM and six hours. GiB values below use 2^30 bytes.

| Model | Trial / GPU | Load | Inference per prompt | Prompt including I/O | GPU allocated / reserved peak | Worker RAM peak | 6,000-prompt generation estimate |
|---|---|---:|---:|---:|---:|---:|---:|
| SDXL | 47773 / L40S | 95.23 s | 3.49 s | 4.06 s | 10.49 / 14.12 GiB | 6.50 GiB | 6.80 h |
| SD 1.5 | 47836 / A40 | 17.28 s | 1.26 s | 1.39 s | 2.61 / 3.18 GiB | 2.57 GiB | 2.32 h |
| PixArt-Σ | Pending | — | — | — | — | — | — |
| Z-Image-Turbo | Pending | — | — | — | — | — | — |
| SD 3.5 Large | Pending | — | — | — | — | — | — |
| FLUX.1-dev | Pending | — | — | — | — | — | — |
| Qwen-Image | Pending | — | — | — | — | — | — |
| GLM-Image | Pending | — | — | — | — | — | — |

Inference and I/O columns are means of the two measured requests. GPU peaks
cover the PyTorch allocator, not total board memory. Worker RAM is its lifetime
RSS high-water mark; Slurm step MaxRSS has a different sampling scope.

The projections include model setup and mean prompt time with I/O for one pass
over 6,000 prompts. They exclude evaluators, downloads, queue time, and actual
training. The current project evaluates pretrained models and has no training
stage. Full-project time remains unavailable with six models unmeasured.

These are provisional capacity and latency observations. SDXL used 1024×1024
images on an L40S; SD 1.5 used 512×512 on an A40, both at 30 steps. They do not
establish a controlled speed ranking. Neither trial covers all prompt levels;
the source candidates lack L2 and human validation. A larger pilot is needed
before scheduling full runs. The SDXL projection exceeds a single six-hour job.

## SD 1.5: job 47836

The download and benchmark succeeded with exit code zero and no CPU offload
hooks. The A40 reported 44.43 GiB GPU memory. Model loading took 17.28 seconds,
warmup took 7.18 seconds, and the two inference calls took 1.254 and 1.258
seconds. The approximate 6,000-prompt generation time is 2 hours 19 minutes,
before evaluation. The initial 60-row preflight only validated input; the
actual benchmark measured two prompts.

The pinned model revision was
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. The selected file payload was
2,742,217,630 bytes. Observed allocated growth of the models directory was
2,745,204,736 bytes (2.56 GiB), with zero growth in the measured Hugging Face
cache. Trial outputs occupied about 1.20 MiB. Repository and models totals
overlap and must not be added together. Personal quota remains unknown; this
successful write does not establish capacity for another model.

Final accounting supplied by the user confirms job, batch and compute step
`COMPLETED`, all with exit code `0:0`. Total job elapsed time was 1:33 and the
compute step was 44 seconds. Slurm compute-step MaxRSS was 2,794,180 KiB.
The earlier exit-handler snapshot showing `RUNNING` has been superseded by
this final accounting.

The subsequent attempt to save final accounting failed with ENOSPC while
writing `accounting-final.json.tmp`. The user subsequently retrieved final
accounting directly with `sacct`; it is now recorded locally. The successful
benchmark measurements remain valid;
the original trial summary was not replaced by that failed write. Resolve
storage capacity before preparing another download. See the
[storage-failure checks](tc2_resource_trials.md#if-collecting-results-reports-no-space-left).

Node totals of 104 CPUs and approximately 377 GiB RAM in machine metadata do
not increase the job's ten-CPU, 30G RAM allocation. Download duration cannot be
separated from other job setup overhead using these logs.

For SDXL, final accounting reported `COMPLETED`, exit code zero and elapsed
time 2:57. Its original cache-growth calculation compared different folders
and is invalid; see the correction in [the trial runbook](tc2_resource_trials.md#cache-accounting-correction-for-job-47773).

## Shared filesystem capacity: downloads deferred

The user's subsequent checks showed repository and home on the same NFS mount,
`10.128.10.41:/msai` at `/home/msai`. `df -h` displayed 10T total, 10T used,
20G available and 100% use. `df -i` displayed 48,085,744 total inodes,
47,849,811 used and 235,933 free (about 0.49% free), also rounded to 100% use.
These are shared filesystem statistics, not the user's personal quota. Some
space and inodes remained at query time, so these snapshots do not establish
the exact cause of the earlier ENOSPC write failure.

Reported directory usage was 2.6G for SD 1.5, 2.5G for SD-Turbo, 6.7G for SDXL,
582M for the project Hugging Face cache, 16G for the home Hugging Face cache,
and 4.7M for benchmark outputs. Preserve the measured outputs. No automatic
cache deletion or shared-storage assignment is assumed.

Before the next download, obtain a confirmed project/scratch location, its
remaining byte and file quotas, and access from MGPU-TC2 compute nodes. Point
future model downloads at that verified location using `VS_MODELS_DIR`; retain
the existing caches. Trial outputs also need a writable destination; changing
the models directory alone does not relocate the job's default output folder
under the repository. Stop new model downloads while storage capacity is
unresolved.

## Cleanup completed; benchmarking remains paused

Benchmark trials are paused at the user's request while cleanup proceeds.
The user removed only `models/sd-turbo`, which is outside the selected eight
benchmark models. A subsequent check showed only SD 1.5 (2.6G) and SDXL (6.7G)
under `models/`, and 9.9G for the complete project directory. The shared mount
then reported 71G available and still displayed 100% use. The change from 20G
to 71G cannot be attributed entirely to this deletion because the filesystem
is shared and the checks occurred at different times. Personal quota remains
unconfirmed. These are disk usage checks, not new model benchmark results.

The broader home audit before cleanup showed 99G total use, including 32G in
`.conda` (30G environments, 1.9G package cache), 16G in `.local`, 16G in
`.cache/huggingface`, 15G in `hf_cache`, 4.4G in `flow-guided-peft`, and 4.1G in
`.vscode-server`. Counts included 442,961 inodes in home and 283,063 in `.conda`.
Nested directory totals overlap. Environment and user-package deletion needs
an inventory of which installations remain in use by the user's projects;
environment names alone do not establish that they are disposable.

The user subsequently ran Conda's tarball/index-cache cleanup (27 tarballs
totalling 49.0 MB and one index cache reported). The environment inventory
showed `visual-safety` active at 8.0G, `llava` at 6.3G, `tabm` at 8.5G,
`escnet` at 7.1G, and `llm-chat` at 16K. The shared base installation is under
`/apps/anaconda3-2025.06-0`. User Python package directories measured 9.0G for
Python 3.13 and 6.7G for Python 3.9.

The user explicitly authorized removal of `tabm`, `escnet`, and `llm-chat`.
Subsequent terminal output confirms those three environments are gone, leaving
base, `llava` and `visual-safety`. After unused-package-cache cleanup, `.conda`
occupied 15G (145M package cache), and home occupied 80G. Keep `visual-safety`,
`llava` and base. The authorized environment
directories account for roughly 15.6G before deletion; shared hard links and
remaining package-cache copies can reduce actual reclaimed storage. No broader
environment or model-cache removal is inferred from this authorization.

The user confirmed all project code runs in `llava` (Python 3.10.22) or
`visual-safety` (Python 3.11.18), with base (Python 3.13.5) used only for Conda
management, and explicitly requested removal of the Python 3.9 and 3.13 user
package directories under `~/.local/lib`. The shared base installation under
`/apps` was outside that removal scope. Conda startup and declared dependency
checks were run with user-site loading disabled (`PYTHONNOUSERSITE=1`).

The guarded user-package cleanup stopped before deletion: Conda 25.5.1 startup,
environment listing and the base Python dependency check succeeded with
user-site loading disabled, then the LLaVA environment's pip check reported
`decord 0.6.0 is not supported on this platform`. Visual-safety's dependency
check was not reached in that first attempt. A subsequent normal LLaVA check
produced the same Decord warning, and visual-safety's dependency check with
user-site loading disabled passed. The Decord issue predates this cleanup;
no LLaVA packages were changed to address it.

The user then removed both authorized user-package directories. Post-removal
output confirmed Conda 25.5.1 still starts and lists only base, `llava`, and
`visual-safety`. `.local` measured 272K, `.conda` measured 15G, and home measured
64G, down from the initial approximately 99G. The overall measured reduction
was about 35G. The shared base installation, retained environments, project
code, data, results and selected model weights were outside the deletion
targets. Requested cleanup is complete; benchmark trials remain paused until
the user resumes them. Personal disk quota is still unconfirmed.

## Deferred next individual storage plan

The following plan remains deferred until the user resumes benchmarking and
current storage capacity has been checked.

After resolving the reported storage limit, check current storage and the
PixArt-Σ file selection before any more weights
are downloaded. The existing `prepare` action queries metadata and writes a
plan; it does not fetch weights or run a model and is suitable on the head node:

```bash
python scripts/tc2_trial_support.py prepare \
  --model pixart-sigma --allow-download \
  --output results/benchmarks/pixart-storage-plan
```

If quota is still unknown, this reports `blocked` together with the download
size. Inspect that plan before selecting an explicit attempt cap or obtaining
verified storage. The SD 1.5 3 GiB cap is specific to its selected files. Keep
actual downloads and computation inside the one-model Slurm job described in
[the trial runbook](tc2_resource_trials.md).
