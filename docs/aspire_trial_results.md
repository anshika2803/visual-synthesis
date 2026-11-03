# Consolidated ASPIRE 2A trials: all eight models

Evidence received on 2026-11-03, Asia/Singapore. These are user-executed cluster
trials, consolidated from supplied stdout and accounting; no remote computation
was performed by this consolidation.

## Conclusion

**All eight selected models passed the generation fit trial on one A100 40 GB
with CPU offload active and the existing 16-CPU, 110 GB RAM allocation.** Keep
ASPIRE 2A for the next profiling stage. An H100 allocation is not required by
these baseline fit results. Activation capture and interventions remain untested,
especially for Qwen, so this does not yet establish full E1–E6 feasibility.

At the tested settings, 6,000 prompts per model means **48,000 images and
392.4 GPU-hours** of projected generation, or **16.35 days**
sequentially on one A100. All projections remain provisional: only two prompts
per model were measured. No full 6,000-prompt run was executed.

Machine-readable evidence: [CSV](data/aspire_trials_2026-11-03.csv) and
[JSON](data/aspire_trials_2026-11-03.json). Full imported stdout JSON is stored
locally under ignored `results/benchmarks/aspire-consolidated-20261103/`.
Shared files contain resource measurements and configuration, with no tokens,
prompt text, generated images, personal project ID or login paths.

## Consolidated measurements

All memory/disk columns use GiB (2^30 bytes). GPU peaks are PyTorch allocated /
reserved memory, not total board memory or minimum hardware requirements.
RAM is the worker process lifetime RSS high-water mark. Prompt+I/O includes
generation and image/row writing. Loading excludes downloads and warmup.

| Model | Load s | Inference s/prompt | Prompt+I/O s | 6,000 generation h | GPU allocated / reserved GiB | Worker RAM GiB | Checkpoints GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| SDXL | 19.50 | 5.31 | 5.72 | 9.55 | 5.21 / 7.31 | 9.18 | 6.62 |
| SD 3.5 Large | 31.39 | 24.22 | 24.73 | 41.24 | 16.05 / 16.38 | 30.69 | 25.73 |
| FLUX.1-dev | 29.55 | 32.73 | 33.10 | 55.20 | 22.63 / 23.28 | 37.66 | 31.43 |
| Qwen-Image | 17.78 | 60.97 | 61.52 | 102.57 | 38.43 / 38.60 | 61.55 | 53.74 |
| Z-Image-Turbo | 50.71 | 10.05 | 10.57 | 17.64 | 12.05 / 12.88 | 22.70 | 30.59 |
| GLM-Image | 21.80 | 87.98 | 88.32 | 147.24 | 18.94 / 18.97 | 35.54 | 33.33 |
| PixArt-Σ | 40.86 | 9.36 | 9.77 | 16.30 | 10.81 / 10.83 | 25.12 | 20.33 |
| SD 1.5 | 16.45 | 1.51 | 1.61 | 2.69 | 1.75 / 1.92 | 3.34 | 2.55 |

Every row is a real, successful GPU trial with two measured images after one
warmup, batch one and one worker. The 16 measured images across these reports
are a fit/latency sample, not evidence of safety or image quality. The original
benchmark commit is `08d3f385d94ae521a4b4d294ff7e9c28c6fb221d` on
`swara/experiments`.

Tested configuration matters; these are not quality-matched speed rankings:

| Model | Resolution | Steps | Runtime dtype |
|---|---:|---:|---|
| SDXL | 1024×1024 | 30 | float16 |
| SD 3.5 Large | 1024×1024 | 40 | bfloat16 |
| FLUX.1-dev | 1024×1024 | 50 | bfloat16 |
| Qwen-Image | 1024×1024 | 50 | bfloat16 |
| Z-Image-Turbo | 1024×1024 | 9 | bfloat16 |
| GLM-Image | 1024×1024 | 50 | bfloat16 |
| PixArt-Σ | 1024×1024 | 20 | float16 |
| SD 1.5 | 512×512 | 30 | float16 |

Environment: Python 3.11.18, PyTorch 2.6.0+cu118, CUDA runtime 11.8,
Diffusers 0.41.0, Transformers 5.20.1, Accelerate 1.15.0. GPU type was identical
across the two jobs, although they ran on different physical nodes. Freeze
these settings for baseline/intervention comparisons. Worker reports have
`revision=null`; preserve cluster download-plan JSONs to retain immutable model
revision provenance.

## Run status and resolved failures

- `26100299.pbs101`: Qwen, GLM, Z-Image-Turbo, PixArt, SDXL and SD 1.5 succeeded.
  FLUX and SD 3.5 Large stopped at prefetch with HTTP 401 gated-repository
  errors. Overall `incomplete` referred to these access failures, not OOM or
  a six-hour timeout. The in-job accounting snapshot was 28:23; final scheduler
  accounting for this first job was not supplied.
- `26100736.pbs101`: both FLUX and SD 3.5 Large succeeded after authentication
  was fixed. FLUX reused cached weights; SD 3.5 Large downloaded before timing.
  Final PBS state **F**, **Exit_status 0**, walltime **08:05**. There were no
  remaining unrun models. PBS recorded 38,323,336 KiB peak host RAM and 24,348 MB
  peak GPU memory (preserving the telemetry's reported unit). PBS job peaks and
  worker allocator/RSS peaks use different scopes and sampling.

No model has a remaining generation-fit failure in the consolidated result.
Downloading, environment setup and authentication are outside the image timing.

## Storage and compute budget

- Reported local checkpoint files across all eight: **204.33 GiB**.
- PNG projection for 48,000 baseline images: **59.67 GiB**.
- Checkpoints plus those PNGs: approximately **264.0 GiB**.
- A **300 GiB dedicated baseline workspace** is a useful planning allowance;
  environments, evaluators, extra caches, activations, backups and other projects
  need separate space. Logical bytes and two-image PNG averages do not prove
  actual quota consumption. Keep one checkpoint copy per model.

The user's supplied scratch quota was 100 TB, with an earlier 109.1 GB usage
snapshot mostly belonging to another project. The 50 GB home quota cannot hold
this baseline. Use the existing scratch workspace, check current `myquota`, and
retain final tables and artifacts on durable storage: scratch is subject to a
30-day purge. Source: [NSCC 2A storage](https://nsccsg.github.io/systems/aspire2a/system-specifications/#storage).

NSCC publishes **64 SU per A100 GPU card-hour**, with no separate CPU charge
for a GPU job. The 392.4-hour baseline corresponds to about
**25,115 SU** before retries, evaluator jobs and additional setup.
This is a planning conversion, not observed billing or a guarantee of the live
balance. Source: [NSCC quota and usage](https://nsccsg.github.io/quickstart/check-quota-and-usage/).

## Team scheduling

For four people with one independent A100 allocation each and two whole models
per person, these pairings put a light model alongside each heavy model:

| Suggested pair | Generation h, 6,000 each | Checkpoints GiB | Projected PNGs GiB |
|---|---:|---:|---:|
| GLM-Image + SD 1.5 | 149.9 | 35.9 | 14.3 |
| Qwen-Image + SDXL | 112.1 | 60.4 | 15.6 |
| FLUX.1-dev + PixArt-Σ | 71.5 | 51.8 | 14.2 |
| SD 3.5 Large + Z-Image-Turbo | 58.9 | 56.3 | 15.6 |

The longest pair is **149.9 hours
(about 6.25 days)** of compute.
GLM plus even the fastest partner sets that constraint when models are indivisible.
The often-quoted total/4 is only **98.1 hours (4.09 days)** under
ideal balancing, which would require sharing prompt shards from heavy models
across GPUs. Model ownership can remain fixed while execution is shared.
Neither forecast includes queue delays or repeated setup from job chunks.
Simultaneous allocations and teammate access must be checked; this serial fit
launcher does not distribute prompts by requesting extra GPUs.

Qwen and GLM account for about 64% of generation time. With the current requested
six-hour walltime, seven of the eight full baseline projections exceed one job;
use resumable production chunks or an independently verified longer walltime.
The fit runner itself is not a resumable 6,000-prompt production runner, and
`--target-prompts 6000` controls extrapolation rather than measured sample size.

## What the whole research scope could cost

The inspected E1–E6 plan at `origin/main` commit `0edf469` keeps generators frozen.
E4 learns activation mean directions and small probes; there is no specified
LoRA or generator fine-tuning stage. Thus the fit report's training=0 describes
the executed stage only. Direction/probe learning, capture and interventions
still require implementation and timing.

The plan reuses E1 images for E2/E3. A full core model uses 1,200 E1 images,
500 E4 images, 8×300 E5 images and 4×300 additional E6 images: 5,300 images.
The mixed tier halves those counts for SD 3.5 Large, FLUX, Qwen and GLM.
Keeping the original 6,000 baseline gives 10,100 images per model instead.

Applying measured baseline seconds/image to these image counts gives:

| Scope | Images across eight | Image-generation equivalent h | Checkpoints+PNG projection GiB |
|---|---:|---:|---:|
| Mixed full/reduced E1-E6 tiers | 31,800 | 193.6 | 242.6 |
| Full E1-E6 core on all eight | 42,400 | 346.5 | 257.0 |
| 6000 baseline plus full E4-E6 subsets | 80,800 | 660.3 | 304.8 |

These are **conditional image-count calculations**, not measured full-study
runtimes or mathematical bounds. They exclude setup, capture/intervention mode
effects, evaluator/probe time, retries and queues. The 392.4-hour baseline above
includes one setup per model; these scenario rows use image time alone.
The old 152/285/543-hour figures in the pre-trial assessment came from guessed
speeds and are superseded by these measurements and conditional calculations.

The 6,000-baseline-plus-ablation option already projects about 305 GiB of
checkpoints and PNGs before activations and evaluators. A **500 GiB workspace**
is a provisional full-study allowance if activations are pooled and streamed;
its adequacy must be checked with an actual capture sample. Full spatial/token
activation dumps can exceed that budget substantially.

## Remaining validation before production

1. Run a broader pilot on shared, reviewed prompts with fixed per-row seeds.
   The current source reports no human-validated rows, no L2 candidates and only
   L0/L1 in the measured sample. All eight forecasts are therefore provisional.
2. Measure the complete evaluator stack, separately from generation. None of
   NudeNet, LAION/CLIP safety, CLIP alignment or quality scoring ran in these trials.
3. Implement and profile representative E4 capture and E5/E6 intervention paths,
   including alpha-zero equivalence with the baseline and compatibility with
   CPU offload. Qwen is the critical memory case: **38.60 GiB reserved** on a GPU
   reporting about 39.49 GiB total; allocator headroom is already below 1 GiB
   before accounting for non-PyTorch GPU memory. Its worker RAM peaked at
   **61.55 GiB**, within the 110 GB allocation. Baseline fit does not guarantee
   capture or ablation fit. Pool/release activations instead of retaining full
   device tensors, then measure the resulting mode.
4. Confirm visible image outputs and investigate retained configuration warnings
   before claiming scientific validity. A successful process exit establishes
   execution, not semantic correctness, safety effectiveness or image quality.
5. Use those additional measurements to replace the conditional E1–E6 estimates
   and choose resumable production job sizes. Continue to preserve unrelated
   projects and the six earlier successful trial artifacts.
