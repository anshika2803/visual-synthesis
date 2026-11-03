# Qwen BF16 versus NF4 on ASPIRE 2A

Use branch `swara/experiments` and the environment from [SETUP.md](../SETUP.md).
This separate short resource comparison can be queued while the larger
eight-model pilots continue. It reuses existing Qwen checkpoint files and never
downloads, replaces or saves generator weights. GPU execution still needs to be
verified on ASPIRE; local tests cover pairing, configuration APIs and submission.

## Matched protocol

| Setting | BF16 | NF4 |
|---|---|---|
| Diffusion transformer | Original BF16 | bitsandbytes NF4, double quantization, BF16 compute |
| Text encoder / VAE | BF16 | BF16 |
| Placement | Model CPU offload | Same model CPU offload |
| Images | Same 12 prompts and evaluation seeds | Same 12 prompts and evaluation seeds |
| Warmup | One unmeasured image | One unmeasured image |
| Generation | 1024×1024, 50 steps, true CFG 4.0 | Identical |

Both arms use the same cached checkpoint, scheduler, negative prompt, attention
backend, row order and model settings from the eight-model overlay. NumPy,
Python and PyTorch RNGs reset per image, including the CPU latent generator.
Existing evaluation seeds are preserved; missing seeds get a stable prompt-ID
hash. The two arms run sequentially in fresh processes inside **one allocated
GPU job**, keeping the physical GPU fixed and making memory peaks comparable.
Seeds do not imply identical outputs after quantization.

Only the transformer is quantized; this is not a comparison of completely
four-bit pipelines. Keeping offload identical isolates precision more closely.
Testing a GPU-resident NF4 pipeline would be a separate deployment comparison.
Activation hooks, probes and ablations are not part of this short comparison.

The loading timer includes runtime NF4 conversion and offload setup. One warmup
is timed separately, then each image records synchronized inference and PNG
writing times. The report includes peak Torch CUDA allocated/reserved memory,
process RAM, actual NF4 layer count, output disk usage and a provisional
6,000-image generation forecast. That forecast excludes repeated loading across
job chunks, queueing, evaluators and ablations. PNG sizes and weight-file storage
are different quantities; runtime quantization does not shrink the cached BF16
checkpoint on disk.

## Submit on the login node

After the eight-model setup and Qwen cache are ready:

```bash
git pull --ff-only
export VS_ASPIRE_PROJECT=YOUR_PROJECT_ID
export VS_ASPIRE_STORAGE="$PWD/runtime"
export VS_ASPIRE_BUDGET_GIB=500  # Your verified total workspace budget

bash scripts/submit_aspire_qwen_quantization.sh
```

The helper prints the job ID and run prefix. Its submission requests
`-q normal -l select=1:ngpus=1 -l walltime=01:45:00`, with the selection in
the PBS script and the walltime explicitly passed by `qsub`. ASPIRE assigns
16 CPUs and 110GB RAM per GPU. It adds **no dependency on the six-hour pilots**
and does not inherit `VS_PILOT_AFTER` or an `ai` queue selection.

The [NSCC queue guide](https://help.nscc.sg/wp-content/uploads/ASPIRE2A-ADVANCED-JOB-MANAGEMENT-TRAINING_GUIDE.pdf)
documents routing short GPU jobs to `gdev` for requests of two hours or less.
Submit to `normal`, since `gdev` accepts routed jobs only. The user's live queue
inspection on 2026-11-03 confirmed a two-hour maximum, ten running jobs per user
and a queue-wide 16-GPU limit. All 16 were occupied at that snapshot, so waiting
is possible. PBS assigns an available A100; a shorter request changes queue
eligibility, not GPU model. Do not change `CUDA_VISIBLE_DEVICES` manually.

```bash
qstat -u "$USER"
qstat -Qf gdev
```

Extra site/project limits and availability can still prevent concurrency. A job
remaining queued does not mean its model has failed. Inspect `qstat -ans` for
scheduler comments.

## Concurrency and dependencies

The comparison does not acquire the large pilot's `.aspire-trial.lock` because
the shared generator files are read-only. Each comparison has its own output
lock and writable HF/CUDA/Triton/pip caches. Checkpoint file metadata and config
hashes must remain unchanged across both arms; tensor payloads are not fully
hashed. Do not remove/re-download Qwen while the comparison runs.

It installs only `bitsandbytes==0.48.2` into the run's private `dependencies/`
directory using `pip --target --no-deps`. Existing Conda packages are not
upgraded. That release supports the existing PyTorch/CUDA setup according to
the [bitsandbytes installation matrix](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation).
Only this small package installation needs network access; generation is
offline. The GPU preflight tests the actual NF4 CUDA kernel and the pinned
quantization API before either Qwen load. Granular transformer configuration
avoids the differing Diffusers/Transformers quantization-config signatures.

Allow roughly 1 GiB of additional working space for the dependency, run-local
caches and images; the script checks conservative storage headroom first.
There is no additional 54 GiB checkpoint copy and no serialized NF4 checkpoint.

## Inputs, order and duration overrides

To use a subset of the current larger pilot's frozen baseline CSV:

```bash
export VS_QWEN_INPUT="$PWD/runtime/outputs/qwen-image-YOUR_RUN/inputs/baseline.csv"
bash scripts/submit_aspire_qwen_quantization.sh
```

The selected rows and seeds are frozen once in the new comparison's
`prompts.csv`, used unchanged by both arms. With no override, the tracked
candidate input is sampled across available levels. Missing L2 and unreviewed
candidate status remain explicit in `input_manifest.json`.

```bash
export VS_QWEN_SAMPLE_SIZE=12       # 1-24; default 12
export VS_QWEN_WALL_SECONDS=6300    # 600-7200; default 1h45m
export VS_QWEN_ORDER=nf4-first      # Default bf16-first
bash scripts/submit_aspire_qwen_quantization.sh
```

The earlier BF16 fit measured about 61.5 seconds/image: 12 images plus warmup
would take about 13 minutes in that arm, excluding loading. NF4 speed is unknown.
The parent divides the remaining allocation between arms and records timeouts
as failures. A 1h45m request is a planning allowance, not a completion guarantee.
Filesystem/OS caching can favor the second load; repeat with reversed order
before treating loading-time differences as robust. Use the same input/count
for repetitions. GPU clock variation and other host jobs can also affect timing.

## Logs and results

All logs remain in `runtime/logs/`, prefixed `qwen-quant-XXXXXX`. All results
remain in `runtime/outputs/qwen-quant-XXXXXX/`, including:

- `comparison_summary.json` and `.csv`: both arms, memory, timings and speedup.
- `prompts.csv`, input manifest and checkpoint fingerprint: pairing evidence.
- `bf16/` and `nf4/`: PNGs, per-image timings/seeds and worker reports.
- `paired_images.csv`: per-prompt speedup, pixel MAE and PSNR.
- `comparison.html`: local side-by-side output gallery using the PNG files.

```bash
RUN_TAG=qwen-quant-YOUR_RUN  # Use the prefix printed by submission
tail -n 30 -F "runtime/logs/$RUN_TAG.bf16.log"
# NF4 comes next by default:
tail -n 30 -F "runtime/logs/$RUN_TAG.nf4.log"
cat "runtime/outputs/$RUN_TAG/comparison_summary.csv"
```

A speedup above 1 means NF4 was faster. Failed or incomplete arms never yield
a speedup estimate. Pixel differences describe output changes, not semantic
quality or safety; review images and run the study's evaluators before adopting
quantization for baseline/ablation work.

After exit, attach final accounting using the actual printed job ID:

```bash
"$HOME/.conda/envs/visual-safety/bin/python" scripts/aspire_qwen_quantization.py collect \
  --output "runtime/outputs/$RUN_TAG" --job-id YOUR_JOB_ID
```

`resources_used.mem` is job-level host RAM; the worker reports provide separate
arm memory peaks. Runtime directories are Git-ignored. Cached credentials and
token paths are not passed by this launcher.
