# ASPIRE feasibility for the eight-model E1–E6 study

Checked on 2026-11-03 (Asia/Singapore). Remote branch references were refreshed
with `git fetch origin`. No branches were switched or merged, no jobs or model
downloads were started, and existing working changes were preserved.

## Assessment

**ASPIRE 2A+ is a suitable target for the study**, with an 80 GB H100 per model
job as the starting allocation. This is an engineering assessment, not an
eight-model fit result. Qwen and especially GLM still need full pipeline and
activation-hook trials. ASPIRE 2B offers more memory headroom if the project has
access. ASPIRE 2A can support much of the study, but its 40 GB A100s require a
more careful memory plan for the largest models.

Hardware capacity, a project's permitted queues/SU balance, and implemented
experiment code are separate requirements. The user's verified TC2 allocation
does not establish an NSCC allocation.

## Relevant branch changes

| Reference after fetch | Commit | Relevance |
|---|---|---|
| `origin/main` | `0edf469`, 2026-11-02 | Adds `docs/e1_e6_research_plan.md`: baseline evaluation, representation capture, ablation, steering, random controls and strength sweeps. This commit adds a specification, not the implementation. |
| `origin/anshika/integrate-model-implementation` | `921b88c`, 2026-10-31 | Already included in `origin/main`; integrated prompt preparation, Ring-A-Bell evaluation, Parquet/LFS outputs and TC2 partition settings. |
| `swara/experiments`, `origin/swara/experiments` | `dca0348`, 2026-10-27 | Contains all eight model configurations and pipeline mappings, resource benchmarks and TC2 fit scripts. The local checkout also contains uncommitted storage fixes and L2 pilot work. |

Other fetched teammate branch tips are older. No ASPIRE-specific launcher or
activation/intervention implementation was found in the inspected branch tips.
The local L2 work adds prompt preparation, verification and small image
comparisons; it does not implement E4–E6 or supply fit results for the remaining
six selected image models.

The [new research plan](https://github.com/anshika2803/visual-synthesis/blob/0edf46975169a878274d2e87c274ebed99120272/docs/e1_e6_research_plan.md)
calls for selectively porting the eight-model support into integrated `main`.
A blanket merge would need careful reconciliation of the divergent dataset and
attack-metric configuration. No merge was attempted for this assessment.

## Published ASPIRE configurations

These are node capacities, not resources automatically granted to each job.

| System / node type | GPUs per node | VRAM per GPU | CPU cores per node | Host RAM per node | Local storage |
|---|---:|---:|---:|---:|---|
| ASPIRE 2A, Cray EX GPU | 4 × A100 SXM | 40 GB | 64, AMD EPYC 7713 | 512 GB | No local NVMe listed |
| ASPIRE 2A, AI | 4 × A100 SXM | 40 GB | 64, AMD EPYC 7713 | 512 GB | 12 TB NVMe |
| ASPIRE 2A, larger AI | 8 × A100 SXM | 40 GB | 128, dual EPYC 7713 | 1 TB | 14 TB NVMe |
| ASPIRE 2A+ | 8 × H100 SXM | 80 GB | 112, dual Xeon Platinum 8480C | 2 TB | About 28 TB usable NVMe |
| ASPIRE 2B, GPU | 8 × H200 | 141 GB | 192, dual EPYC 9655 | 2.25 TB | 15.36 TB NVMe |

Sources: official [ASPIRE 2A specifications](https://nsccsg.github.io/systems/aspire2a/system-specifications/),
[ASPIRE 2A+ specifications](https://nsccsg.github.io/systems/aspire2a%2B/system-specifications/)
and [ASPIRE 2B specifications](https://nsccsg.github.io/systems/aspire2b/system-specifications/).

For a one-GPU job, the current NSCC guide lists **16 CPU cores and 110 GB host
RAM on 2A**, or **14 CPU cores and 235 GB host RAM on 2A+**. This substantially
improves on TC2's 30G host RAM limit. A 2A A100 has less per-card VRAM than the
TC2 A40/L40S cards observed in the first trials. Source:
[GPU resource allocation](https://nsccsg.github.io/quickstart/launch-an-interactive-job/).

Eight cards in a node do not provide one automatic shared VRAM pool. The current
backend uses one device, with optional CPU offload; allocating extra GPUs alone
will not shard a large model or distribute prompts to them.

## Model-by-model fit assessment

Assumptions: batch one, the configured fp16/bf16 settings, 512px SD 1.5 and
1024px other models. Entries below are expectations requiring pilots, except
for the explicitly recorded TC2 measurements. They do not include full-model
training or fine-tuning.

| Model | One A100 40 GB, ASPIRE 2A | One H100 80 GB, ASPIRE 2A+ | Evidence / remaining check |
|---|---|---|---|
| SD 1.5 | Expected comfortable fit | Expected comfortable fit | TC2 measured 2.61 GiB CUDA allocated, 3.18 GiB reserved. Hooks remain untested. |
| SDXL | Expected comfortable fit | Expected comfortable fit | TC2 measured 10.49 GiB allocated, 14.12 GiB reserved. Hooks remain untested. |
| PixArt-Σ | Expected fit | Expected fit | No local GPU timing/peak. Include T5 encoder in the load trial. |
| Z-Image-Turbo | Expected fit | Expected fit | Official card describes a 16G deployment; its setup is not our measured configuration. |
| SD 3.5 Large | Expected fit; check loading/workspace | Expected fit with more headroom | Larger backbone and several text encoders; no measured local peak. |
| FLUX.1-dev | Plan CPU offload/component placement | Expected fit; validate | Diffusers documents roughly 50 GB RAM/VRAM for all components; do not rely on the plan's approximate 33 GB full-pipeline figure. |
| Qwen-Image | Full pipeline on one card is not a sound baseline; test block offload or explicit sharding | Preferred starting target; validate | 20B diffusion backbone: roughly 40 GB of bf16 weights before its encoder and inference workspace. Quantization changes the experiment configuration. |
| GLM-Image | Offload candidate; substantial runtime/fit uncertainty | Suitable starting target with offload fallback; not a guaranteed full-residency fit | Hybrid 9B AR + 7B decoder. Check AR/DiT hook choice as well as memory. |

Primary references: [PixArt-Σ](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS),
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
[SD 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large),
[FLUX memory guidance](https://huggingface.co/docs/diffusers/api/pipelines/flux#optimize),
[Qwen architecture](https://github.com/QwenLM/Qwen-Image), and
[GLM model card](https://huggingface.co/zai-org/GLM-Image).

GLM's official materials give different setup-dependent memory claims: the model
card describes about 23 GB with CPU offload; its GitHub README recommends
more than 80 GB or multiple GPUs, yet reports a 37.82 GB peak for a particular
1024px, batch-one H100 test. Its 64.34 s/image in that test is an external
reference, not our measured throughput. Source:
[GLM inference-cost table](https://github.com/zai-org/GLM-Image#inference-cost).

The backend first constructs the pipeline on CPU before moving it to the GPU.
Additional ASPIRE host RAM helps this load path, but does not establish runtime
fit. Record the actual offload mode and peak RAM/VRAM for every trial.

## How much work do the ablations add?

The new plan proposes a core study before expanding the original 6,000-prompt
baseline. It reuses images rather than generating separate E1, E2 and E3 sets.

| Stage | Full tier, images per model | Reuse |
|---|---:|---|
| E1 baseline / E2 robustness / E3 comparison | Up to 1,200 | E2 and E3 analyze E1 outputs |
| E4 activation capture | 500 | Also the alpha-zero baseline for E5/E6 |
| E5: two modes × two direction types × two strengths | 8 × 300 = 2,400 | Validation prompts only |
| E6: two new strengths × two direction types | 4 × 300 = 1,200 | Other strengths reused from E5/E4 |
| **Total** | **Up to 5,300** | Optional random seeds excluded |

Halving all groups gives up to 2,650 per model. The plan assigns this reduced
tier to SD 3.5 Large, FLUX, Qwen and GLM, and the full tier to the other four.

| Scope | Total images across eight models | Generation hours using the plan's unmeasured speed assumptions |
|---|---:|---:|
| Proposed mixture of full/reduced tiers | 31,800 | 152.4 |
| Full core study on every model | 42,400 | 284.9 |
| Original 6,000 baseline per model + full E4–E6 subsets | 80,800 | 542.9 |

The last row replaces 1,200 baseline images with 6,000, retaining the same
500/300-prompt E4–E6 subsets: `(6000 + 500 + 8*300 + 4*300) * 8`.
Running *each* intervention on all 6,000 prompts would be a different, much
larger scope.

These hours are **arithmetic from the plan's guesses**, not an ASPIRE forecast
or bound. The assumed seconds/image are SD 1.5 1.5, SDXL 5, PixArt 4, Z-Image 3,
SD 3.5 20, FLUX 25, Qwen 60 and GLM 75. They exclude hook overhead, model setup,
evaluation, downloads, retries and queues. The two TC2 fit tests do not calibrate
the other six architectures or H100/A100 performance. Profile baseline, capture
and intervention separately before accepting a project runtime estimate.

Four concurrent GPUs reduce elapsed time only when work is scheduled and
balanced. The proposed two-model ownership pairs are uneven; dividing total
hours by four is only an ideal lower bound. If reduced tiers are used, freeze
nested shared subsets and report sample sizes; cross-model paired comparisons
must use prompts common to the models being compared.

E4 learns mean directions and small logistic-regression probes on CPU. The
generators remain frozen; the plan does not call for LoRA or generator training.
Inference hooks can operate without retaining gradient graphs. Pool activations
immediately and store them on CPU: an illustrative 500 prompts × 5 layers ×
50 steps × 4096 channels × fp16 is about 0.95 GiB per model. That is not a
measured shape, and does not apply directly to GLM's variable-length AR path.
Keeping complete token/spatial outputs at every step would need much more
memory and storage.

## Storage, scheduling and allocation

NSCC lists a **50 GB home quota** on these systems. Use the assigned project
storage for retained checkpoints/results and the approved scratch/job-local
paths for temporary work. The 2A/2A+ specifications list 100 TB user scratch,
subject to a 30-day purge; local NVMe is removed at job end. These advertised
defaults do not replace checking the actual account with `myquota`.
Sources: [2A storage](https://nsccsg.github.io/systems/aspire2a/system-specifications/#storage),
[2A+ storage](https://nsccsg.github.io/systems/aspire2a%2B/system-specifications/#storage).

A **500 GB–1 TB project-storage planning target** is reasonable if retaining
all eight checkpoints and the core study, with exact payload sizes and a small
output sample checked first. This is a recommended budget, not a verified
requirement or entitlement. For scale, 42,400 PNGs at an assumed 1–3 MiB each
occupy about 41–124 GiB before weights, environments, activations or extra
copies. Avoid duplicate checkpoint caches; downloading models individually
remains appropriate.

2A and 2A+ use **PBS Pro (`qsub`)**. The TC2 Slurm wrapper, `sbatch`, `srun`, job
IDs and accounting cannot be copied unchanged. The Python benchmark engine can
be reused behind a PBS launcher. Source:
[NSCC batch submission](https://nsccsg.github.io/quickstart/submit-a-batch-job/).

The published 2A+ routing policy gives small batch jobs (up to seven GPUs) up
to 24 hours, and the 120-hour queue requires at least eight GPUs. Verify current
queue properties and project access before writing the launcher. Do not reserve
eight GPUs merely to obtain a longer walltime for one serial worker. Source:
[2A+ queue policy](https://help.nscc.sg/aspire2aplus/urgent-scheduled-maintenance-for-aspire-2a-on-22-january-2025-9am-to-6-30pm-2-2-2-2-2-3-3-2/).

NSCC's published conversion rates are 64 SU/A100 card-hour on 2A, 128 SU/H100
card-hour on 2A+, and 235 SU/H200 card-hour on 2B. For example, 500 allocated
H100 card-hours require 64,000 SU; this is not a runtime estimate. Check the
project balance and expiry as well as its eligible system. Source:
[quota and usage guide](https://nsccsg.github.io/quickstart/check-quota-and-usage/).

Read-only checks on an authenticated **ASPIRE 2A/2A+** session:

```bash
hostname
myprojects
myquota
myusage
qstat -Qf
```

Run `nvidia-smi` and any model computations inside an allocated compute job.
An NSCC project ID is needed for submission; TC2's `stud` account is not that ID.

## Work still needed before the full study

1. Confirm the accessible ASPIRE version, project SU balance, storage and queues.
2. Reconcile eight-model support with integrated `main`; implement the planned
   capture, interventions, probes, detector union, quality and reporting pieces.
3. Add a PBS launcher and adapt resource/accounting collection, preserving
   Slurm support for TC2. Explicit multi-GPU dispatch/sharding is separate work.
4. Fit-test the remaining models individually, followed by a representative
   pilot for all eight. Check the chosen hook with offload enabled, alpha-zero
   equivalence, pooled activation size and capture/intervention throughput.
5. Freeze reviewed prompt subsets and model/environment versions. Local L2
   pilots have not produced a human-validated final benchmark. Use the same
   model settings across baseline and intervention conditions.

The hardware recommendation is **2A+ first, or 2B if already available**. The
full project remains conditional on those software, data and allocation checks;
benchmarking has not been resumed by this assessment.
