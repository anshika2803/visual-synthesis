# Resource and runtime pilot for the eight-model project

The implemented project evaluates **pretrained** text-to-image models:
standardize → unify → deduplicate → generate → NSFW detection → metrics.
There is no training loop, optimizer, training image dataset, LoRA, or learned
intervention in this repository. **Additional model training is outside this
estimate's scope.** If the research scope later adds learning, it needs a
separate pilot using the actual training method and data; inference latency
cannot predict its training time.

One image for each of 6,000 prompts across eight models is **48,000 images per
pass**, or 12,000 images per person when each person owns two models. Additional
seeds or identical full passes multiply that workload. An intervention that
changes execution or model weights needs its own pilot. The current local
deduplicated table contains 10,892 rows; the 6,000 is an explicit projection
target, not an existing filter in the pipeline. Use `--input` with the final
6,000-prompt table when it is ready.

## What was added

- `scripts/benchmark_resources.py`, also available as `visual-synthesis benchmark`.
- `config/benchmark.models.json`, a layered profile selecting the agreed eight.
- `visual-synthesis benchmark-summary` to combine members' results.

The profile preserves the existing SDXL and SD 3.5 Large settings. New model
settings are explicit starting points, to be agreed before full experiments:

| Model ID | Model | Steps | Resolution | Guidance |
|---|---|---:|---|---|
| `sd-xl` | SDXL 1.0 Base | 30 | 1024 × 1024 | 7.5 |
| `sd-3.5-large` | SD 3.5 Large | 40 | 1024 × 1024 | 4.5 |
| `flux-dev` | FLUX.1-dev | 50 | 1024 × 1024 | 3.5 |
| `qwen-image` | Qwen-Image, original checkpoint | 50 | 1024 × 1024 | true CFG 4.0 |
| `z-image-turbo` | Z-Image-Turbo | 9 scheduler steps / 8 DiT forwards | 1024 × 1024 | 0 |
| `glm-image` | GLM-Image | 50 decoder steps, plus autoregressive generation | 1024 × 1024 | 1.5 |
| `pixart-sigma` | PixArt-Σ XL/2 1024 | 20 | 1024 × 1024 | 4.5 |
| `sd-1.5` | SD 1.5 | 30 | 512 × 512 | 7.5 |

This is a runtime profile, not a claim that all these settings form a fair
scientific comparison. Qwen's model-card square example uses 1328 × 1328;
this profile explicitly uses 1024 × 1024. Model class, precision, resolution,
steps, offload settings, and extra call arguments are recorded in each report.
SDXL's refiner is not included. There is no prompt rewriting or quantization.

## Hardware choice before the full run

Use Macs for setup, CPU data stages, tests and a small feasibility pilot of the
lighter models. Profile full workloads on **allocated TC2 GPU nodes**. TC2
access does not establish which GPU or memory quota you can actually allocate.
Inside a GPU allocation, record:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

The script also records the selected GPU, VRAM, RAM, Python and package versions.
Use the same GPU type and settings for the production run. One worker owns one
model copy; assigning two models to one person does not require loading them
together. This pilot intentionally supports one copy, matching the project's
default. Its results do not predict multiworker throughput or multi-GPU sharding.

As a planning judgment, start SD 1.5, SDXL, PixArt-Σ and Z-Image-Turbo fit tests
before the heavier models. For SD 3.5 Large and FLUX, use a larger GPU or compare
an explicit CPU-offload run. Give Qwen-Image and GLM-Image particular attention
when choosing a GPU/RAM allocation. These are priorities for measurement, not
guaranteed minimum VRAM requirements.

GLM-Image includes a **9B autoregressive generator and a 7B diffusion decoder**,
so its total cost is not described by “7B.” Its authors currently describe an
offload option around 23 GB GPU memory with slower inference. CPU offload also
needs host RAM for weights and time for transfers. See the
[official GLM-Image model card](https://huggingface.co/zai-org/GLM-Image).

In this project's existing device code, bfloat16 requests fall back to float32
on MPS and CPU. This can double weight memory relative to CUDA bfloat16.
A failed Mac fit test is not evidence that a model will fail on a TC2 GPU.
Apple GPU memory and host RAM are unified; do not add them as independent pools.

## Install and prefetch

Activate the project's Python 3.11/3.12 environment first. The base project's
older minimum dependency versions are not sufficient for every new model.
Use current compatible Diffusers/Transformers versions providing
`FluxPipeline`, `QwenImagePipeline`, `ZImagePipeline`, `GlmImagePipeline`,
`PixArtSigmaPipeline`, and Transformers' GLM image components. The local pilot
environment used Diffusers 0.41.0 and Transformers 5.20.1. Package versions
are written to reports for reproducibility; full loading of the eight large
checkpoints still needs validation on the target node.

```bash
pip install -e '.[gpu,nsfw,metrics,benchmark]'
# In a dedicated environment, upgrade if the pipeline classes are missing:
pip install -U diffusers transformers accelerate
```

Download **your assigned models** ahead of the timed pilot. For example:

```bash
visual-synthesis download -c config/pipeline.json -c config/benchmark.models.json --model sd-xl
visual-synthesis download -c config/pipeline.json -c config/benchmark.models.json --model sd-1.5
```

Gated models need your Hugging Face access already set up. The default benchmark
blocks Hugging Face downloads; `--allow-download` permits them, but then the
reported model load may include transfer time. Prefer a cached rerun for compute
planning. CLIP alignment also needs its evaluator cached. NudeNet's weights ship
with its package. If selecting the CLIP-H/14 detector, prefetch its head/backbone
with the existing download command. FID is off in the profile; if enabling FID,
prepare both reference images and its evaluator weights in advance.

## Start with a two-prompt fit test

Run this **inside the TC2 GPU allocation**, with the actual partition/resources
your account permits. Replace IDs with your assigned pair. Each model runs in a
fresh process, sequentially, so a failed load does not contaminate the next model.

```bash
python scripts/benchmark_resources.py \
  -c config/pipeline.json -c config/benchmark.models.json \
  --model sd-xl --model sd-1.5 \
  --device cuda --sample-size 2 --warmup 1 --repeats 1 \
  --generation-only --output results/benchmarks/member1-fit
```

For a Mac feasibility check, use `--device mps`. Do not start by loading all
eight models on a 16 GB Mac. For a separate CUDA offload experiment, add
`--cpu-offload` and use a new output folder; its timings apply to offload mode.

## Run a representative pilot

```bash
python scripts/benchmark_resources.py \
  -c config/pipeline.json -c config/benchmark.models.json \
  --model sd-xl --model sd-1.5 \
  --device cuda --sample-size 60 --warmup 2 --repeats 3 \
  --target-prompts 6000 \
  --output results/benchmarks/member1-tc2
```

This generates 182 requests per model: two warmups plus three repeats of the
same 60 sampled prompts. Seeds are deterministic and shared across models.
Sampling is stratified by `standard_category` when present, random otherwise;
60 covers the current table's 39 category combinations better than a tiny fit
test. Detection and enabled metrics run on the final repeat, after generation
is unloaded. Defaults follow `pipeline.json`: NudeNet detection, CLIP alignment,
FID disabled. Keep detector/evaluator choices identical to production.

Omit the `--model` arguments to select exactly the eight IDs in the profile's
`download.models` list. Use `--timeout` to extend the default one-hour limit per
model. `--set` accepts existing config overrides. Use `--passes 3` for three
identical full passes with separate loads/warmups. Different settings or learned
interventions require separate pilots, not simply a larger `--passes` value.

Add `--prepare` to time standardize/unify/postprocess once against all locally
downloaded raw data. It writes isolated preprocessing runs in the benchmark
folder. Otherwise the latest postprocess run is read. Use `--input` for an
explicit final prompt table; `--prepare` and `--input` are mutually exclusive.
Dataset/weight downloads, queue waiting, manual research/report writing and
unimplemented experiments are not included in the compute estimate.

## Read the results

Each output folder contains:

- `report.json`: raw prompt timings, estimates, hardware, configuration, failures,
  resource peaks, input/sample hashes, preprocessing time and project totals.
- `summary.csv`: comparison table with per-model generation/evaluation hours,
  peak RSS/CUDA allocations and image storage.
- `sample.parquet`: the exact pilot prompts shared by all selected models.
- `<model>/worker.log`, `<model>/report.json`, generated images and stage manifests.

Follow a live pilot with `tail -f <output>/<model>/worker.log`. Existing output
folders are rejected, so a repeated benchmark cannot silently skip cached images.
Failed loads, timeouts, out-of-memory errors and missing evaluators remain
visible. No full-project total is emitted for an incomplete or synthetic pilot.

The main generation formula is:

```text
per-model generation time = passes × (
    load time + warmup time + table-write time
    + 6,000 × mean measured seconds per prompt including image/JSONL writing
)
```

Inference-only prompt timings are also retained. GPU synchronization surrounds
timed inference. Model load is measured separately, not multiplied by 6,000.
Evaluation uses final-repeat whole-stage time divided by pilot images and scaled
to the full image count. This repeats evaluator startup/plotting costs in the
projection and is therefore a **conservative planning estimate**, not an exact
throughput forecast. The p50/p95 values describe pilot prompt latency; they are
not confidence bounds on total runtime. Increase the pilot if repeats vary a lot.

To understand the order of magnitude before a model has been measured:

| Assumed mean time per image | 6,000 images, one model | 48,000 images, eight models serially |
|---:|---:|---:|
| 5 seconds | 8.3 hours | 66.7 hours |
| 10 seconds | 16.7 hours | 133.3 hours |
| 30 seconds | 50 hours | 400 hours |
| 60 seconds | 100 hours | 800 hours |

These are arithmetic scenarios, **not measured speeds for any of the eight
models**, and omit setup/evaluation. Compute time is the sum of all model work.
Calendar time with four genuinely independent devices is the maximum of the
four members' two-model totals, plus any shared preprocessing. Four users sharing
one GPU do not get a fourfold speedup. Queue time remains additional.

Image storage scales with measured mean file size. Weight/cache storage is
separate; per-model reports record the logical bytes present in the local weight
folder/cache when resolvable, which can include files beyond the minimum needed.
Memory is a measured peak; it does not multiply by 6,000. CUDA values cover the
PyTorch allocator in the worker, excluding other jobs/context/non-Torch consumers.
MPS peaks and live RSS are sampled at 100 ms and can miss brief spikes. A process
RSS lifetime high-water mark is also recorded. Allow headroom above measurements.

## Combine the four members' reports

Use the same prompt table, sample size, seed, target prompt count, and number of
passes. Copy the four `report.json` files to distinct local filenames, then run:

```bash
visual-synthesis benchmark-summary \
  member1-report.json member2-report.json member3-report.json member4-report.json \
  --output results/benchmarks/team
```

The merge rejects duplicate models and incompatible inputs/workloads. Results
retain each machine's hardware. If all eight complete pilots used equivalent
hardware/software, the report suggests four pairs by combining slower and faster
models and reports that schedule's longest pair. This is a conditional scheduling
suggestion, not an assumption that TC2 grants four concurrent GPUs. The script
does not assign owners. Synthetic backends are clearly marked and excluded from
project totals.

## Local verification, 26 October 2026

The automated suite passed 33 tests. A real cached SD-Turbo control on the
16 GB Apple-silicon Mac completed preprocessing, two repeats of three prompts,
NudeNet detection, and CLIP alignment without downloading weights. Measured
generation averaged 1.02 seconds per prompt including image/record writing;
model loading took 4.70 seconds. The three full CPU preprocessing stages took
0.39 seconds for the existing 11,136 source rows, yielding 10,892 unique rows.
The sampled Metal driver peak was about 4.32 GiB. This tiny run validates the
harness; SD-Turbo is **not one of the selected eight**, and these timings do not
predict their speeds or memory use. Its reports are under
`results/benchmarks/cached-control-pipeline-mac/` (ignored by Git).

## Official model references

- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SD 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [GLM-Image](https://huggingface.co/zai-org/GLM-Image)
- [PixArt-Σ](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
- [SD 1.5 mirror](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
