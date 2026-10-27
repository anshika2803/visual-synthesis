# TC2: one model at a time under the verified allocation

This adds two scripts alongside the existing benchmark scripts. It does not
replace project files, install packages, change branches, remove caches, or push
Git changes. The job activates **visual-safety** and uses the existing two-prompt
fit mode: one warmup and two measured requests, generation only.

Verified allocation supplied by the user: account `stud`, partition `MGPU-TC2`,
QoS `normal`, one GPU, ten CPUs, `30G` host RAM and `06:00:00` wall time. These
are encoded in `scripts/tc2_resource_trial.sh`. GPU VRAM is a separate quantity;
the allocated GPU's name and total memory are recorded by preflight.

## Copy these additions without overwriting existing changes

The companion bundle contains only:

- `scripts/tc2_resource_trial.sh`
- `scripts/tc2_trial_support.py`
- `docs/tc2_resource_trials.md`

Copy the bundle to your cluster checkout using your normal authenticated file
transfer. From the TC2 terminal:

```bash
cd /home/msai/swarangi001/visual-synthesis
tar --keep-old-files -xzf tc2-resource-trial-bundle.tar.gz
bash -n scripts/tc2_resource_trial.sh
conda activate visual-safety
```

`--keep-old-files` aborts on existing filenames rather than replacing them.
No `git pull`, checkout, reset, package installation or push is required for
these additions. The existing branch must already be `swara/experiments`; the
job checks that condition and refuses to switch it automatically.

## Inspect storage first

These are metadata checks, with no model loading or inference:

```bash
python scripts/tc2_trial_support.py storage \
  --output results/benchmarks/tc2-storage-check.json
mytcinfo
quota -s
```

The JSON records project/model/Hugging Face cache usage, filesystem free space,
raw quota output and your `mytcinfo` output. Cache/project paths may overlap;
do not sum their totals. Recovered 9.2 GB is not assumed to be your current
quota headroom. No shared-storage path is guessed or searched for.

If `mytcinfo` lists an assigned storage folder, verify that assignment and its
available quota before using it. `VS_MODELS_DIR` can point new model downloads
at an explicitly chosen models folder there. Do not change `HF_HOME` merely
to run cached SDXL: that could hide the already cached CLIP/SDXL assets.
The scripts do not migrate or delete either cache.

## Validate and submit cached SDXL first

The following is the exact initial submission path. `--test-only` asks the
scheduler to validate the request without running the job, as documented in
[Slurm's sbatch manual](https://slurm.schedmd.com/sbatch.html).

```bash
cd /home/msai/swarangi001/visual-synthesis
VS_TRIAL_ALLOW_DOWNLOAD=0 sbatch --test-only --export=ALL \
  --job-name=vs-sdxl-fit scripts/tc2_resource_trial.sh sd-xl

VS_TRIAL_ALLOW_DOWNLOAD=0 sbatch --parsable --export=ALL \
  --job-name=vs-sdxl-fit scripts/tc2_resource_trial.sh sd-xl
```

The Slurm directives already request the exact allocation above; no additional
resource flags or larger RAM/GPU allocation are needed. Keep the returned job
ID. Logs are `slurm-vs-sdxl-fit-<JOBID>.out` and `.err` in the submission folder.

The job checks storage and the existing cache, validates CUDA/imports/precision
and input, then uses `srun` to invoke `scripts/benchmark_tc2.sh sd-xl` in fit
mode. No model or evaluator is downloaded for the cached SDXL trial. If its
cache is incomplete, the attempt is recorded as blocked rather than fetched.

The script refuses direct execution outside a Slurm allocation. It runs one
model copy and accepts exactly one model argument. It also records the current
commit and dirty-file list without modifying those files.

## Inspect each result before preparing the next model

Artifacts are isolated under
`results/benchmarks/tc2-<JOBID>-<MODEL>-<unique-suffix>/`:

| Artifact | What it records |
|---|---|
| `preparation.json` | Cache status, download decisions, failures and size/budget calculations |
| `storage-before.json`, `storage-after.json` | Disk use, quota and assigned-folder information |
| `environment-preflight.json` | GPU type/VRAM, CUDA, pipeline imports and precision support |
| `benchmark/report.json` | Model load time, warmup, two prompt timings and worker resource peaks |
| `trial_summary.json` | Measured fields, observed disk changes, exit code and available Slurm accounting |
| `benchmark/<MODEL>/worker.log` | Per-prompt progress and model exceptions |

After the job completes, set `trial` to that exact output folder and collect
final job/step accounting. This is a read-only accounting query and is suitable
on the head node; it does not run a model:

```bash
trial=results/benchmarks/tc2-JOBID-sd-xl-UNIQUE_SUFFIX
python scripts/tc2_trial_support.py collect --output "$trial" --job-id JOBID
sacct -j JOBID --units=G \
  --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS,AllocTRES,Reason
```

Replace `JOBID` and the suffix with the actual values. Include job steps in
`sacct`; the allocation row alone may have zero memory statistics. Accounting
can be delayed, so rerun the query if still incomplete. See the
[Slurm accounting manual](https://slurm.schedmd.com/sacct.html).

The collector can create a failure summary even when an OOM kill or wall-time
termination prevented the job's exit handler from finishing.

## Remaining models: individual guarded downloads and trials

All eight supported IDs are:

| Suggested order | Model ID |
|---:|---|
| 1 | `sd-xl` — cached; run first |
| 2 | `sd-1.5` |
| 3 | `pixart-sigma` |
| 4 | `z-image-turbo` |
| 5 | `sd-3.5-large` |
| 6 | `flux-dev` |
| 7 | `qwen-image` |
| 8 | `glm-image` |

Do not prefetch the entire list or queue it as a model array. After reviewing
SDXL and storage, permit only the next selected model's download, for example:

```bash
VS_TRIAL_ALLOW_DOWNLOAD=1 sbatch --parsable --export=ALL \
  --job-name=vs-sd15-fit scripts/tc2_resource_trial.sh sd-1.5
```

For each subsequent ID, repeat that one-model command after inspecting the
previous job. Already cached models are reused. A new download first queries
file-size metadata, avoids duplicate binary weights when safetensors are
available, and checks the remaining download plus 10% and 1 GiB headroom against
both filesystem space and recognized numeric user quota. Inode/file quota is
checked when reported. Only then does the job invoke the existing prefetch
script with that model's exact file selection, without fetching an evaluator.

If user quota cannot be matched to the selected filesystem, the download is
blocked. After obtaining a verified remaining budget for an assigned folder,
you may pass `VS_TRIAL_BUDGET_GIB` as an additional cap. It represents verified
**remaining bytes available for this attempt**, not total quota, model size,
or the amount previously recovered. The budget never overrides a smaller known
quota or filesystem limit. No automatic cleanup/eviction is performed.

## Interpreting limits and failures

- GPU peak values are PyTorch allocated/reserved peaks for the worker. They
  exclude CUDA context and other non-PyTorch allocations; they are not total
  board memory. GPU total VRAM is recorded separately.
- RAM includes the worker RSS lifetime high-water mark and sampled RSS. Final
  Slurm `MaxRSS` is retained separately, with its job/step scope. Missing data
  after a kill is unavailable, never a measured zero.
- The 30G limit is host RAM. CPU offloading still uses host RAM, so it may not
  make a large model feasible. The existing backend can fall back to offload
  after GPU placement fails; its active-hook flag is retained in the report.
- `OUT_OF_MEMORY`, CUDA OOM, unsupported bfloat16, quota exhaustion, missing
  cache, gated-model access failures and `TIMEOUT` are distinct findings in
  logs/accounting. Do not extrapolate a failed model's latency.
- A pre-wall-time signal stops the current step and attempts to preserve a
  summary. Final accounting is still required for kills/cancellations.
- Two measured prompts establish a fit/rough latency point, not a reliable
  6,000-prompt runtime. Broader representative pilots remain provisional while
  L2 is absent from the candidate manifest.

No authentication, live quota inspection, scheduler validation or GPU results
are claimed from local preparation. Those checks occur when these commands run
on TC2. The job does not request resources outside the supplied limits.
