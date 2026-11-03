# Project runtime planning after the first two GPU trials

Prepared 2026-10-29 (Asia/Singapore). No new cluster jobs, model downloads or
package changes were performed. Benchmark execution remains paused.

## What has actually been measured

All eight configured pipelines passed import checks. Two pipelines have GPU
timings, each from one warmup and two measured prompts:

| Model | Recorded GPU | Prompt including image/output writing | Generation projection for 6,000 prompts |
|---|---|---:|---:|
| SDXL | NVIDIA L40S | 4.0601 s | 6.80 h |
| SD 1.5 | NVIDIA A40 | 1.3876 s | 2.32 h |

The measured-model subtotal is approximately 9.12 GPU-job hours for one pass,
one image per prompt. These projections include the setup recorded by their
pilots and remain provisional. They exclude evaluation and queuing. The two
GPUs and image resolutions differ, so their latency ratio does not isolate
model architecture performance.

## Profiles of the six remaining models

These settings come from `config/benchmark.models.json`. They are not measured
runtime or memory results.

| Model | Resolution | Configured steps | Planning consideration |
|---|---|---:|---|
| PixArt-Σ | 1024×1024 | 20 | Different DiT backbone and T5 text encoder; SDXL throughput is not directly transferable. |
| Z-Image-Turbo | 1024×1024 | 9 | Distilled low-step model; nine configured scheduler steps correspond to the official eight-NFE example. A useful early fit trial. |
| SD 3.5 Large | 1024×1024 | 40 | Larger diffusion network and multiple text encoders; host RAM fit needs checking. |
| FLUX.1-dev | 1024×1024 | 50 | Large transformer and text encoders; loading memory is a material concern. |
| Qwen-Image | 1024×1024 | 50 | 20B diffusion model plus its text encoder; full-precision pipeline fit is unproven. |
| GLM-Image | 1024×1024 | 50 plus autoregressive generation | 9B autoregressive generator plus 7B diffusion decoder; decoder steps alone do not describe total computation. |

Primary references:

- [PixArt-Σ model card](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
- [Z-Image-Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [SD 3.5 Large model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [FLUX pipeline documentation](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [Qwen-Image model card](https://huggingface.co/Qwen/Qwen-Image)
- [GLM-Image architecture and inference-cost table](https://github.com/zai-org/GLM-Image#inference-cost)

GLM's published reference is 64.34 seconds end-to-end at 1024×1024, batch one,
on an H100 for its documented fixed prompt. Multiplying by 6,000 gives 107.23
hours on that reference setup. Its reported peak VRAM for that case is 37.82 GB.
This is an external reference, not a TC2 prediction: hardware, prompt length,
library version, loading/offloading and implementation can change results.
The same README contains a broader >80GB/multi-GPU recommendation inconsistent
with that particular peak-memory table, while the model card describes an
offload option. No single quoted VRAM number establishes fit in our allocation.

## Explicit what-if budget for the full project

There is no calibrated per-model timing model for the remaining architectures.
The following are editable assumptions about their **average** prompt time,
not predictions, confidence intervals, or measured lower/upper bounds. They
assume all six can be made runnable on the chosen hardware.

```text
Remaining images = 6 models × 6,000 prompts = 36,000
Remaining generation hours = 36,000 × assumed mean seconds / 3,600
Combined hours = 9.12 measured-model projection + remaining generation hours
```

| Assumed average for each remaining image, including output writing | Remaining six models | All eight: generation budget | Ideal four-GPU lower bound |
|---|---:|---:|---:|
| 20 s | 200 h | 209.12 GPU-job h | 52.28 h |
| 40 s | 400 h | 409.12 GPU-job h | 102.28 h |
| 80 s | 800 h | 809.12 GPU-job h | 202.28 h |

The four-GPU figures divide work by four and require four independent GPUs,
perfect load balancing, and no queue delays. Assigning two models to each member
does not guarantee equal work. Total completion time is determined by the
slowest member's pair and actual scheduling. Jobs must respect the six-hour
wall limit, so longer production runs need resumable chunks.

These figures exclude remaining-model load/warmup costs, downloads, evaluation
with NudeNet/CLIP, failed attempts, additional experiment passes, and queuing.
Additional image copies/seeds and passes multiply generation work. The current
repository evaluates pretrained models and has no training stage; actual
fine-tuning would require a separate estimate from its method, data and steps.
The prompt source is still missing L2 and has no human-validated source rows.

## Current RAM/loading limitation

The allocation is one GPU, ten CPUs, 30G host RAM and six hours. The recorded
A40/L40S GPUs exposed roughly 44.4 GiB usable VRAM. Host RAM and VRAM are
separate limits.

The current image backend calls `pipeline_class.from_pretrained(...)` without a
device map, then moves the completed pipeline to the device or enables CPU
offload. Therefore, a pipeline can exceed the host RAM allocation before GPU
placement or its offload fallback runs. Successful pipeline-class imports do
not test this load path. Large FLUX, Qwen and GLM pipelines need explicit fit
validation; increased disk free space does not increase their RAM allowance.

Quantization, component-wise/device-aware loading or a different allocation may
be necessary. These would create a new performance configuration requiring its
own pilot. They have not been applied or authorized as implementation changes
by this planning request. A model that fails to load has no finite successful
runtime estimate under the unchanged setup.

## Replacing assumptions with evidence

When the user resumes trials, run each remaining model individually with a
small fit sample on its intended hardware, then use a larger level-aware pilot
to stabilize latency and time the evaluators. Disk metadata planning should
precede each download, with no simultaneous prefetch of all models. Import
success and third-party benchmarks do not replace these fit checks.
