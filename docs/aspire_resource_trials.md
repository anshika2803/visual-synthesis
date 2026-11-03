# ASPIRE trials from swara/experiments

The PBS launcher `scripts/aspire_resource_trial.sh` runs the existing benchmark
engine on **one GPU, one model at a time**. It supports ASPIRE 2A (A100 40 GB)
and ASPIRE 2A+ (H100 80 GB). Start on ASPIRE 2A using the offload trial below.
No branch switch, merge,
package installation, weight deletion or Git push is performed by these scripts.

Default order:

```text
qwen-image → glm-image → flux-dev → sd-3.5-large → z-image-turbo
           → pixart-sigma → sd-xl → sd-1.5
```

Each model gets one warmup and two measured prompts at the existing benchmark
settings. The runner records model loading, per-prompt inference and I/O times,
worker RAM and CUDA allocation/reservation peaks, actual GPU, versions, disk
changes and failures. Download time is separate. A failed model does not prevent
the next model from being tested, unless storage exhaustion is detected.

This tests pretrained generation fit. It does **not** implement or time the
planned E4–E6 activation capture/ablation code, or download/time evaluators.
Two-prompt results are provisional; use a representative pilot after fit works.

## Access

An NTU jump host provides a network route; an NSCC account/project determines
the resources you can use. NSCC documents personal projects for NTU users on
**ASPIRE 2A only**. Access to **2A+ or 2B** requires an allocated project and
membership. See [account and project setup](https://nsccsg.github.io/quickstart/get-an-account-and-a-project/).

From the NTU jump host, use your **NSCC user ID**:

```bash
# ASPIRE 2A: the system to try first
ssh YOUR_NSCC_USER_ID@aspire2antu.nscc.sg

# ASPIRE 2A+: requires a project allocated on the H100 system
ssh YOUR_NSCC_USER_ID@aspire2pntu.nscc.sg
```

Official access guides: [2A+](https://nsccsg.github.io/systems/aspire2a%2B/access/)
and [2A](https://nsccsg.github.io/systems/aspire2a/access/). Your TC2 username,
`stud` Slurm account and TC2 environment do not automatically create an NSCC
account, project or environment.

## Storage requirement

File metadata was queried on **2026-11-03**, using the current branch's component
patterns and weight variants. No weights were downloaded for this calculation.
Sizes include text encoders and VAE, exclude alternative binary weights where
safetensors exist, and reflect **on-disk files**, not runtime VRAM.

| Model | Selected payload, GiB | Metadata revision (prefix) |
|---|---:|---|
| Qwen-Image | 53.74 | `75e0b4be04f6` |
| GLM-Image | 33.33 | `2c433cc0cbc2` |
| FLUX.1-dev | 31.43 | `3de623fc3c33` |
| SD 3.5 Large | 25.73 | `ceddf0a7fdf2` |
| Z-Image-Turbo | 30.59 | `f332072aa78b` |
| PixArt-Σ | 20.33 | `e102b3591cc8` |
| SDXL | 6.62 | `462165984030` |
| SD 1.5 | 2.55 | `451f4fe16113` |
| **All eight** | **204.33 GiB (219.40 GB decimal)** | |

Plan for **300 GiB available** for retaining all eight in this dedicated trial
directory. For Qwen alone, plan for **100 GiB available**. These include working
headroom and tiny trial outputs; the Conda environment is additional. Some
checkpoints are stored in a larger dtype than their runtime bf16/fp16 setting.

Sequential downloads reduce simultaneous transfers; **they do not reduce the
eventual 204.33 GiB if all checkpoints are retained**. No checkpoint is
automatically removed. The full image/ablation study needs additional storage;
the earlier 500 GB–1 TB project-storage recommendation includes that larger scope.

The published home quota is 50 GB. Use approved project storage or your existing
scratch allocation. Scratch is temporary and subject to purge; preserve final
results in project storage. See [2A storage](https://nsccsg.github.io/systems/aspire2a/system-specifications/#storage)
and [2A+ storage](https://nsccsg.github.io/systems/aspire2a%2B/system-specifications/#storage).

On the login node, inspect allocation without recursive scans:

```bash
myprojects
myquota
```

The budget below is the **total capacity available to this dedicated directory**,
not the cluster filesystem's free space. Use 300 only after checking your
allocation has that much available. The guard subtracts existing workspace
usage before every new download, checks filesystem bytes/inodes and reserves
20% of the full selected payload plus 2 GiB. `myquota` output is saved for
inspection; it is not automatically parsed as proof of a particular quota.
Do not share this budget with concurrent unrelated downloads.

## Environment and branch

Run from an existing checkout of `swara/experiments` containing the two new
ASPIRE scripts and the original TC2 helper/benchmark files. The launcher refuses
another branch and never modifies tracked source files.

Use a cluster CUDA environment with Python 3.11/3.12, CUDA PyTorch 2.6 or newer,
and the benchmark dependencies. Default environment name: `visual-safety`.
The existing `requirements/benchmark-cluster.txt` pins the tested Diffusers,
Transformers and Accelerate APIs. If provisioning an environment, install a
cluster-supported CUDA PyTorch build first, then:

```bash
conda activate visual-safety
python -m pip install -e '.[gpu,benchmark]' -c requirements/benchmark-cluster.txt
```

This installation is a separate setup step, never executed by the job script.
Use an approved compute/interactive session for substantial installations or transfers.
Accept the gated SD 3.5 Large and FLUX.1-dev model terms with your own HF
account. An existing local HF login or `HF_TOKEN` in the repo's untracked `.env`
can supply credentials. Metadata visibility does not prove gated weight access.
When moving caches, the driver preserves an existing token file through
`HF_TOKEN_PATH`; it does not copy or print the token.
Do not place tokens in job commands, tracked config or logs.

If Conda is unavailable in the batch shell, export
`VS_ASPIRE_PYTHON=/absolute/path/to/visual-safety/bin/python` and include that
variable in `qsub -v`. This bypasses activation and uses that interpreter.

The launcher restores `PBS_O_PATH` (the submission shell's saved search path)
and adds the selected Python environment's `bin` directory before checking Git.
For Git installed through a module, export `VS_ASPIRE_GIT=$(command -v git)` on
the login node and include it in `qsub -v`. The same executable is used for
branch verification and Git provenance in the Python driver.

If the job reports **Git executable unavailable**, this is a batch environment
problem, not evidence of a different branch. No model downloads or trial output
directories are created at that stage. If the captured executable is absent on
compute nodes too, install Git into `visual-safety` in an allocated CPU session,
then export that environment's Git path. A real branch mismatch remains an
error; the launcher never switches branches.

## First ASPIRE 2A trial: Qwen with CPU offload

The 2A login hostname is **`aspire2antu.nscc.sg`**. The hostname
`aspire2pntu.nscc.sg` selects 2A+ instead. In an SSH configuration, use the 2A
hostname with your existing `ProxyJump ntu-jump` and NSCC user ID.

ASPIRE 2A's one-GPU allocation normally provides an **A100 40 GB, 16 CPU cores
and 110 GB host RAM**. See [NSCC GPU allocation](https://nsccsg.github.io/quickstart/launch-an-interactive-job/).
Start with Qwen alone and explicit CPU offload so its text encoder and generator
do not all need to remain on the GPU together. The Qwen transformer plus
workspace can still exceed 40 GB; this is a fit test, not a promise of fit.
If it fails, preserve the result and try GLM/FLUX next. Layer/group offload or
an H100 allocation may be needed for Qwen; quantization is not applied silently.

After pulling this branch, run on the **2A login node**:

```bash
cd /absolute/path/to/visual-synthesis
myprojects
myquota

export VS_ASPIRE_REPO="$PWD"
export VS_ASPIRE_PROJECT="YOUR_NSCC_PROJECT_ID"
export VS_ASPIRE_PYTHON="$(command -v python)"
export VS_ASPIRE_GIT="$(command -v git)"
export VS_ASPIRE_STORAGE="$HOME/scratch/visual-synthesis-trials"
export VS_ASPIRE_BUDGET_GIB=100
export VS_ASPIRE_ALLOW_DOWNLOAD=1
export VS_ASPIRE_MODELS=qwen-image
export VS_ASPIRE_CPU_OFFLOAD=1

qsub -P "$VS_ASPIRE_PROJECT" \
  -v VS_ASPIRE_REPO,VS_ASPIRE_PROJECT,VS_ASPIRE_PYTHON,VS_ASPIRE_GIT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_ASPIRE_ALLOW_DOWNLOAD,VS_ASPIRE_MODELS,VS_ASPIRE_CPU_OFFLOAD \
  scripts/aspire_resource_trial.sh
```

Activate `visual-safety` before capturing the interpreter path above.
Use the actual project ID shown by `myprojects`. A personal project is usable
only if it is listed for your account with sufficient GPU allocation.
The 100 GiB workspace budget requires that much available in the assigned
scratch/project allocation. Both the CUDA environment and accepted gated-model
licenses must already be available on ASPIRE.

For the full eight-model sequence, retain offload and raise the budget after
checking space. Leave `VS_ASPIRE_MODELS` out of the exported variable list:

```bash
unset VS_ASPIRE_MODELS
export VS_ASPIRE_BUDGET_GIB=300
qsub -P "$VS_ASPIRE_PROJECT" \
  -v VS_ASPIRE_REPO,VS_ASPIRE_PROJECT,VS_ASPIRE_PYTHON,VS_ASPIRE_GIT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_ASPIRE_ALLOW_DOWNLOAD,VS_ASPIRE_CPU_OFFLOAD \
  scripts/aspire_resource_trial.sh
```

This reuses cached weights and runs fresh measurements for all eight, heavier
models first. Keep offload settings recorded when comparing 2A and 2A+ results.

## Submit all eight, heavier models first

On the **ASPIRE login node**, in your repo:

```bash
cd /absolute/path/to/visual-synthesis
git branch --show-current

export VS_ASPIRE_REPO="$PWD"
export VS_ASPIRE_PROJECT="YOUR_NSCC_PROJECT_ID"
export VS_ASPIRE_STORAGE="$HOME/scratch/visual-synthesis-trials"
export VS_ASPIRE_BUDGET_GIB=300
export VS_ASPIRE_ALLOW_DOWNLOAD=1

qsub -P "$VS_ASPIRE_PROJECT" \
  -v VS_ASPIRE_REPO,VS_ASPIRE_PROJECT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_ASPIRE_ALLOW_DOWNLOAD \
  scripts/aspire_resource_trial.sh
```

Replace the repo path and project ID. The scratch example requires `$HOME/scratch`
to already resolve to your assigned scratch; otherwise specify the actual
approved absolute path. The runner refuses storage resolving inside your home.
For project storage use a dedicated subdirectory, e.g.
`/home/project/YOUR_NSCC_PROJECT_ID/visual-synthesis-trials` on 2A or
`/data/projects/YOUR_NSCC_PROJECT_ID/visual-synthesis-trials` on 2A+.
This command does not export `VS_ASPIRE_CPU_OFFLOAD`; use the 2A-specific
commands above for explicit offload.

The job requests **one GPU for six hours**; CPU/RAM are assigned using NSCC's GPU
ratios. It profiles all models serially in one allocation, retaining completed
results if walltime is reached. Slow transfers can require another job.
The internal model-worker timeout is 30 minutes; each download stage has a
90-minute timeout. No quantization, step reduction or resolution reduction is
automatically applied. The backend's existing CPU-offload fallback is recorded.
For a larger follow-up pilot, export `VS_ASPIRE_SAMPLE_SIZE=60` and include it
in `qsub -v`; it still uses one warmup and one measured pass.

To test **Qwen only first**, add this variable to the command:

```bash
export VS_ASPIRE_MODELS=qwen-image
qsub -P "$VS_ASPIRE_PROJECT" \
  -v VS_ASPIRE_REPO,VS_ASPIRE_PROJECT,VS_ASPIRE_STORAGE,VS_ASPIRE_BUDGET_GIB,VS_ASPIRE_ALLOW_DOWNLOAD,VS_ASPIRE_MODELS \
  scripts/aspire_resource_trial.sh
```

Unset `VS_ASPIRE_MODELS` before the all-eight command. A space-separated subset
can also be passed through that variable. Models already cached in this
workspace are reused. Every submission creates fresh measurement output.

For ASPIRE 2A or an explicit offload comparison, set `VS_ASPIRE_CPU_OFFLOAD=1`
and add it to `qsub -v`. Offloading can slow inference and may still be
insufficient for Qwen on a 40 GB card; the failure is recorded. It is a distinct
performance configuration. Do not request extra GPUs without implementing
explicit model sharding or independent workers.

## Read results and recover failures

```bash
qstat -u "$USER"
```

PBS stdout prints the exact output directory:
`$VS_ASPIRE_STORAGE/trials/pbs-JOBID-RANDOM/`. It contains:

- `run_summary.json`: completed results, model order, not-run models and Git state.
- `MODEL/trial_summary.json`: measurements/failure, offload request and disk deltas.
- `MODEL/benchmark/report.json` and `MODEL/benchmark/MODEL/report.json`: existing benchmark results.
- `MODEL/environment-preflight.json`, `preflight.log`, `prefetch.log`, `benchmark.log`
  and the benchmark worker log.
- `MODEL/download-plan.json`: exact metadata, revision, selected files and capacity guard.
- `pbs-allocation.json`: requested GPU/CPU/RAM/walltime, where PBS reports them.

After the job ends, attach final job accounting; use the **full PBS job ID**:

```bash
conda activate visual-safety
python scripts/aspire_trial_support.py collect \
  --job-id 12345.pbs101 \
  --output "$VS_ASPIRE_STORAGE/trials/pbs-12345.pbs101-REPLACE_SUFFIX"
```

PBS accounting gives job-level peak host RAM, exit status and resource limits.
Worker reports give per-model RAM/CUDA peaks. `collect` recovers records for a
model interrupted before its exit handler could run, without inventing timings.
If disk writes fail, query accounting without writing:

```bash
python scripts/aspire_trial_support.py collect --job-id 12345.pbs101 --stdout-only
```

Failure logs distinguish download access, storage exhaustion, GPU OOM and
timeouts where the error supplies that evidence. SIGKILL/host-RAM-limit details
may be available only from final PBS accounting. Job history can expire, so
collect it promptly. A lock prevents two jobs from downloading into the same
trial workspace concurrently. No GPU computation or large download is allowed
by the launcher outside a PBS job.

## Refresh the metadata-only storage estimate

This makes small API requests, without downloading model weights or scanning
large directories; it can be run before submission:

```bash
python scripts/aspire_trial_support.py plan \
  --output results/benchmarks/aspire-storage-plan.json
```

Gated/unavailable metadata is retained as an error. A partial total must not be
treated as the total for all eight. All execution paths use the dedicated
storage directory for models and HF caches, overriding stale TC2 `.env` paths.

## Pull the trial runner onto the cluster

In an existing `swara/experiments` checkout:

```bash
git branch --show-current
git pull --ff-only origin swara/experiments
```

Resolve any reported local conflicts without resetting or deleting cluster
changes. The runner does not require the separate uncommitted local L2 work,
benchmark result files or later TC2 accounting fixes.
