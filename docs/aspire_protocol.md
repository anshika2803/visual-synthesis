# Run the main E1–E7 protocol on ASPIRE

One submission script handles every configured model. Select `--model`; it
chooses the frozen full/reduced tier and gives each model its own workspace.
The adapter calls main's normal generate, nsfw, metrics, hook validation,
representation analysis and report code. It does not use the older resource
pilot's samples or intervention implementation.

Only three files are needed at runtime:

| File | Purpose |
|---|---|
| `scripts/aspire_protocol.pbs` | PBS runner submitted with `qsub` or the ASPIRE portal |
| `scripts/aspire_protocol.py` | Setup check, submission, checkpointing and experiment orchestration |
| `config/e1e6.aspire.json` | Shared model execution overrides |

Tests and this guide are development/support files. The earlier separate shell
submission wrapper, Python entry shim and package controller have been consolidated.

## Models and precision

| Model | Tier | Precision | Images in E1 + E7 + E4 + E5 + E6 |
|---|---|---|---:|
| sd-1.5 | full | FP16 | 4,056 |
| sd-xl | full | FP16 | 4,056 |
| pixart-sigma | full | FP16 | 4,056 |
| z-image-turbo | full | BF16 | 4,056 |
| sd-3.5-large | reduced | BF16 | 2,190 |
| flux-dev | reduced | BF16 | 2,190 |
| qwen-image | reduced | BF16 | 2,190 |
| glm-image | reduced | BF16 | 2,190 |

The shared override is `config/e1e6.aspire.json`. It enables model CPU offload
on the A100 / 110 GB host-memory allocation. The target-account readiness check
must pass for each model; earlier resource-pilot fit results do not establish
that main's hooks fit. There is no automatic precision fallback.

Qwen's main requirements are 1024×1024, 50 steps, `true_cfg_scale=4.0`, a
single-space negative prompt, seed 42, and the configured five hook layers.
Main's default FP8 transformer is a TC2 memory exception. The ASPIRE override
uses BF16 weights, `quantization=null`, `device_map=null`, and CPU offload.
The earlier NF4 experiment saved memory but degraded quality; its 12 pairs do
not establish an equivalent replacement. FP8/NF4 adoption needs a separate
paired precision check on the frozen `quant_v1` set. This runner is for native
precision; it does not silently adopt quantized weights or reuse pilot directions.
The earlier weak-steering calibration also does not replace main's fixed E5/E6
strengths.

## Pull the runner on ASPIRE

Use the main checkout intended for this protocol. Keep its code and
environment fixed while jobs are queued or running. If another job is using a
different branch in that directory, create a separate checkout first.

```bash
cd "$HOME/scratch/visual-synthesis"
export PATH="$HOME/.conda/envs/visual-safety/bin:$PATH"
git switch --no-overwrite-ignore main
git pull --ff-only origin main
git lfs pull --include='data/processed/e1e6/*' --exclude=''
```

The generic runner and its configuration arrive through Git; no patch files or
additional submission wrappers are needed.

## Environment and storage

Use the previously working ASPIRE environment. No packages are installed or
upgraded by the scripts. Main additionally needs scikit-learn for the E4 probe;
its absence produces an explicit preflight error. If needed, add dependencies
to a separate copy of the working environment, then select its Python below.
Do not install the TC2 CUDA package freeze blindly over the ASPIRE environment.
The readiness record freezes the actual Python/package versions and source
hashes; changing them later stops resumption. ASPIRE/BF16 and the environment
are recorded execution deviations from the TC2 setup.

Pooled activation steps are stored in FP32: BF16 activations can exceed FP16's
finite range, corrupting E4's per-step analysis when saved in FP16. Capture now
rejects non-finite values during readiness. This storage fix keeps model BF16
precision, the FP32 pooled means, hook locations and intervention rules intact.
The setup report records it under `approved_source_fixes`, while `source_matches`
still shows the difference from the original TC2 source. Only the exact audited
file hash is accepted; other scientific source changes fail the check. Apply
the fix before a fresh readiness run; existing overflowed captures must be
regenerated, since casting them back to FP32 cannot recover their values.

Paste these in each new cluster login. The 500 GiB value is the earlier planning
cap for the entire runtime workspace, not extra storage requested for this run;
check current usage/quota before selecting it.

```bash
export VS_ASPIRE_PROJECT=YOUR_PROJECT_ID
export VS_ASPIRE_STORAGE="$(readlink -f "$PWD/runtime")"
export VS_ASPIRE_BUDGET_GIB=500
export VS_ASPIRE_PYTHON="$HOME/.conda/envs/visual-safety/bin/python"
export VS_ASPIRE_GIT="$HOME/.conda/envs/visual-safety/bin/git"
export HF_TOKEN_PATH="$VS_ASPIRE_STORAGE/hf-home/token"
```

If your project ID or environment differs, set those paths accordingly. For
gated models, your own token must have the model's access granted. Token values
are not printed or put into submission arguments.

Cached models under `runtime/models/<model-id>` are reused. The alignment cache
is `runtime/models/openai--clip-vit-base-patch32`; main's quality evaluator is
the separate `openai--clip-vit-base-patch16` cache. The LAION backbone/head and
NudeNet ONNX weights are also checked. Optional downloads happen only inside
the readiness PBS job, after file-size/budget checks. Production is offline.
Preparation uses a shared cache lock, so two model jobs cannot modify the
shared caches or perform their download-budget checks concurrently.

## Compare your setup with TC2 first

This command checks installed package metadata, frozen prompt/exclusion hashes,
main's scientific source, model parameters, hook configuration and detector
thresholds. It does not load model weights or prove GPU fit.

```bash
"$VS_ASPIRE_PYTHON" scripts/aspire_protocol.py check --model qwen-image --strict
```

It compares against the frozen TC2 files at commit `16671d1`. The five versions
required by the TC2 setup guide are:

| Package | TC2 reference |
|---|---|
| torch | 2.11.0+cu128 |
| diffusers | 0.40.0 |
| transformers | 5.16.1 |
| nudenet | 3.4.2 |
| torchmetrics | 1.9.0 |

The result is saved to
`runtime/protocol-v1/qwen-image-reduced/setup-check.json`. `--strict` exits
nonzero if these or the other recorded runtime-package versions differ.
Missing dependencies, altered scientific
settings/source, and changed prompt/exclusion files also fail the check.
The report lists other runtime-package versions, deliberate offload/precision
differences, and whether a checkpoint revision can actually be compared.

For the same protocol, keep the frozen tables, seeds, generation parameters,
hook layers, E5/E6 conditions, detector labels/thresholds, and B/32 alignment
plus B/16 quality evaluators unchanged. Qwen's ASPIRE BF16/offload exception is
explicitly reported. Hardware and BF16 versus TC2 FP8 can change pixels even
with the same seed; this is a consistent protocol, not a promise of identical
images across the two executions.

Your earlier ASPIRE trials used Torch 2.6/cu118, Diffusers 0.41 and Transformers
5.20.1, which differ from this reference. Those are earlier measurements, not
a live check of your current environment. If the check reports differences,
prepare a separate environment to align versions where supported and check its
CUDA/driver compatibility in the readiness job. No environment changes are made
automatically. If different versions are retained, preserve the report as an
environment deviation and do not describe the environments as identical.

The TC2 config does not pin immutable Hugging Face model revisions, so a matching
repository name alone cannot prove equal checkpoint revisions. Compare the
recorded cache provenance with the teammate's model records before making that
claim. A successful readiness result verifies execution and is separate from
the strict TC2-version comparison.

## Submit Qwen first

Inspect the command without submitting:

```bash
"$VS_ASPIRE_PYTHON" scripts/aspire_protocol.py submit ready --model qwen-image --dry-run
```

The readiness job requests one GPU, 16 CPUs, 110 GB host memory and two hours.
It prepares missing caches when explicitly allowed, checks main's hooks, and
runs two full-setting images through both detectors, alignment and quality.

```bash
VS_PROTOCOL_ALLOW_DOWNLOAD=1 \
  "$VS_ASPIRE_PYTHON" scripts/aspire_protocol.py submit ready --model qwen-image
```

After PBS reports the job finished successfully, inspect its logs and readiness
status:

```bash
"$VS_ASPIRE_PYTHON" scripts/aspire_protocol.py status --model qwen-image
grep -E '^(PASS|FAIL)' runtime/protocol-v1/qwen-image-reduced/logs/hook-validation.log
qstat -xf YOUR_READY_JOB_ID
```

`ready` must be true and the hook checks must pass. A failed/unfinished readiness
job cannot start production. Re-submit the same readiness command to resume
an interrupted preparation; real errors need inspection first.

Then submit the complete protocol:

```bash
"$VS_ASPIRE_PYTHON" scripts/aspire_protocol.py submit all --model qwen-image --max-jobs 12
```

The controller submits `scripts/aspire_protocol.pbs`. If you prefer native
`qsub` or the portal, the same `.pbs` file can be submitted directly. For example,
with the environment/storage variables above already exported:

```bash
export VS_PROTOCOL_MODEL=qwen-image
export VS_PROTOCOL_PHASE=ready
export VS_PROTOCOL_WALL_SECONDS=7200
export VS_PROTOCOL_ALLOW_DOWNLOAD=1
qsub -P "$VS_ASPIRE_PROJECT" -q normal -l walltime=02:00:00 \
  -v VS_ASPIRE_PROJECT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_ASPIRE_PYTHON,VS_ASPIRE_GIT,VS_PROTOCOL_MODEL,VS_PROTOCOL_PHASE,VS_PROTOCOL_WALL_SECONDS,VS_PROTOCOL_ALLOW_DOWNLOAD,HF_TOKEN_PATH \
  scripts/aspire_protocol.pbs
```

For a direct production submission, use phase `all` and 21,600 seconds alongside
`walltime=06:00:00`. Keep the walltime variable and PBS limit consistent. In the
portal, select the same project, GPU resources and environment variables. The
Python submission command supplies these values and log paths automatically.
Set `VS_ASPIRE_REPO` to the absolute checkout path in the portal, since its
submission directory may be different from the repository.

The `submit all` command schedules one six-hour PBS job. Near its time budget the adapter terminates
the current command, retains incremental records, and submits one `afterok`
continuation. At most 12 jobs (72 allocated GPU-hours) are allowed per submitted
chain, including the first job. Completed commands are checked and skipped.
The bound limits retries; it does not mean the experiment measures 72 hours of
work. You can choose a smaller `--max-jobs` limit. Queue delays are additional.

A Python/CUDA failure, failed hook, incomplete detector result, missing quality
score, altered source/environment/cache, or exhausted job bound stops the chain.
Such errors do not auto-resubmit. If PBS kills the job before a continuation is
submitted, inspect the logs and submit the identical `all` command to resume.
For planned checkpoints, continuation submission requires `qsub` to be available
on the compute node; if it is unavailable, the checkpoint remains resumable
through the login-node helper.

## Use the same script for another model

Only the model argument changes, for example:

```bash
VS_PROTOCOL_ALLOW_DOWNLOAD=1 \
  "$VS_ASPIRE_PYTHON" scripts/aspire_protocol.py submit ready --model pixart-sigma

# After that model's readiness job passes:
"$VS_ASPIRE_PYTHON" scripts/aspire_protocol.py submit all --model pixart-sigma
```

Known models always use their frozen tier. A future model supported by the
existing image backend can be added to the shared `models` configuration with
its repository, dtype, generation settings and hook specification. No new
launcher is needed. Its default tier is full; `--tier reduced` can select the
reduced tables for a newly configured model. Changing the tier of one of the
eight frozen models is rejected. A new backend architecture still needs an
adapter/hook definition before the readiness gate can pass.

## Outputs and experiment order

For each model:

```text
runtime/protocol-v1/<model>-<tier>/
  ready.json, identity.json, resources.json, progress.json
  logs/                  per-command logs and PBS stdout
  runs/s05_generate/     images, generation records, pooled activations
  runs/s06_nsfw/         both detector runs
  runs/s07_metrics/      per-image alignment, quality and detector tables
  results/e4/<model>/    learned directions and layer analysis
  results/e1e6/<model>/  report tables and E6 plot
```

The sequence is E1 → E7 → E4 capture/analysis → eight E5 conditions → the
report's E6 mode selection → four E6 conditions → final report. E2/E3 reuse
E1; there is no separate image generation for them. The E4 capture run is the
zero-strength reference. E5/E6 generation is restricted to the validation split.

Individual `e1`, `e7`, `e4`, `e5`, `e6`, or `report` phases are available through
the same helper. Automatic continuation is enabled for `all` only. Qwen's counts
are 438 E1, 282 E7, 198 E4, eight × 106 E5 and four × 106 E6.
Full-tier counts are 874, 282, 392, eight × 209 and four × 209 respectively.

The adapter rejects incomplete generation/detector/quality coverage before
reporting, preventing main's missing-verdict coercion from becoming a result.
Cross-model E3 still requires the shared reduced cohort in the central analysis;
main's per-model E3 CSV uses that model's available E1 cohort. The unfinished
second calibration review may require rescoring later, so retain all images on
ASPIRE. The protocol-v1 tag was absent at the initial review; the required base
commit is `16671d1`, and actual code/config is frozen at each model's readiness.

## Validation scope

Local automated checks cover model/tier selection, preservation of scientific
settings, scheduling arguments, continuation bounds, checkpoint resumption,
validation filtering, incomplete cache shards, failed verdicts and seed mismatch.
They use local files and mocked PBS calls, with no GPU or job submission.
The real ASPIRE fit/hook/evaluator gate remains the first submitted job.
