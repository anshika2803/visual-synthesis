# Cluster setup — ASPIRE 2A / PBS

This guide is for each teammate's own ASPIRE 2A account. It uses no personal
project IDs or usernames. Use branch **`swara/experiments`** and replace the
single `YOUR_PROJECT_ID` placeholder with a project listed for your account.

The pilot supports **all eight selected models**. `all` queues one GPU job per
model in sequence, starting with Qwen because it has the least memory headroom.
Each job measures generation, evaluation, pooled capture and interventions.
The eight-model two-prompt fit stage is already recorded in the
[consolidated results](docs/aspire_trial_results.md). The larger pilot is a new
measurement stage for evaluator and intervention costs; it reuses cached weights.

## 1. Login, project and storage

Connect to ASPIRE 2A using your institution's approved SSH/jump-host route and
NSCC username. The NTU 2A endpoint is `aspire2antu.nscc.sg`; the `aspire2pntu`
endpoint is 2A+, a separate system/allocation.

```bash
hostname
myprojects
myquota
readlink -f "$HOME/scratch"
export VS_ASPIRE_PROJECT=YOUR_PROJECT_ID
```

Use approved scratch/project storage for the checkout. The tested one-GPU
allocation provides an A100 40GB, 16 CPUs and 110GB RAM. The scripts request
six hours per job. Heavy work, installations and weight downloads run on compute
nodes. Login-node commands below only inspect, authenticate, update source or
submit jobs. Sources: [GPU allocation](https://nsccsg.github.io/quickstart/launch-an-interactive-job/),
[PBS submission](https://nsccsg.github.io/quickstart/submit-a-batch-job/).

Check your own quota: filesystem free space is not your quota. NSCC's published
home quota is 50GB and scratch is temporary, subject to a 30-day purge. Keep
final research artifacts backed up. [Storage policy](https://nsccsg.github.io/systems/aspire2a/system-specifications/#storage).

## 2. Checkout and one runtime folder

For a fresh checkout on scratch:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch swara/experiments \
  https://github.com/anshika2803/visual-synthesis.git \
  "$HOME/scratch/visual-synthesis"
cd "$HOME/scratch/visual-synthesis"
```

Use your site's Git module if `git` is unavailable on the login node. Existing
checkouts should use `git pull --ff-only` after preserving their changes; do not
reset a working checkout. For a different approved location, use that checkout
directory throughout this guide.

All new runtime files default to **`<checkout>/runtime/`**, which Git ignores:

```text
visual-synthesis/
  SETUP.md, config/, scripts/, src/, tests/      Source and shared instructions
  runtime/                                     All local cluster runtime data
    setup/                                     Installation caches
    models/                                    Reusable generator/evaluator weights
    hf-home/                                   HF cache and private token file
    logs/                                      Every model/run log in one folder
      <run>.pbs.log                            Scheduler stdout/stderr
      <run>.pilot.log                          Live driver progress
      <run>.preflight.log, <run>.preparation.log
      <run>.baseline.log, <run>.capture.log, <run>.learn.log
      <run>.alpha-zero.log, <run>.safety-ablation.log, ...
      <run>.nudenet.log, <run>.clip-h14.log, ...
    outputs/<model>-<unique-id>/                Every run's outputs and reports
      inputs/                                  Frozen sample tables and hashes
      <phase>/                                 Images, scores, pooled features
      pilot_summary.json                       Status, resources and forecasts
      phase_summary.csv                        Per-phase timings/memory
      image_metrics.csv                        Per-image evaluator scores
```

There is no need for a separate setup/trial directory in a fresh installation.
The Conda environment itself remains in `$HOME/.conda/envs/visual-safety` so it
is available from compute nodes; its installation caches go in `runtime/setup`.

If adopting an **existing external trial workspace**, move that workspace into
`runtime/` before creating a new runtime directory, after its jobs have ended.
For example, set `OLD_RUNTIME` to your own old workspace:

```bash
OLD_RUNTIME=/absolute/path/to/your/previous/workspace
(
  set -e
  OLD_RUNTIME=$(readlink -f "$OLD_RUNTIME")
  test -d "$OLD_RUNTIME"
  test ! -e runtime
  test "$(stat -c %d "$OLD_RUNTIME")" = "$(stat -c %d .)"
  mv -- "$OLD_RUNTIME" ./runtime
  ln -s -- "$PWD/runtime" "$OLD_RUNTIME"
)
```

Use a same-filesystem move; a cross-filesystem bulk transfer belongs on a compute
node. The compatibility symlink preserves old absolute image/cache references;
all data then physically lives in `runtime/`. If `runtime/` already exists,
stop and inspect rather than merging/overwriting caches. New teammates skip
this migration entirely.

## 3. Create the shared Conda environment

From the checkout on the login node:

```bash
qsub -I -P "$VS_ASPIRE_PROJECT" -q normal \
  -l select=1:ncpus=4:mem=16gb -l walltime=01:00:00
```

Wait for the compute-node prompt. Load your available Conda module (`module
avail` lists choices; `miniforge3` is a common option):

```bash
if ! command -v conda >/dev/null 2>&1; then
  module load miniforge3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$PBS_O_WORKDIR"

export CONDA_PKGS_DIRS="$PWD/runtime/setup/conda-pkgs"
export PIP_CACHE_DIR="$PWD/runtime/setup/pip-cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

conda create --prefix "$HOME/.conda/envs/visual-safety" \
  --override-channels -c conda-forge python=3.11 pip setuptools git git-lfs -y
conda activate "$HOME/.conda/envs/visual-safety"

python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-build-isolation -e '.[gpu,nsfw,metrics,benchmark]' \
  -c requirements/benchmark-cluster.txt

git lfs install
git lfs pull --include="data/processed/unified/unified_prompt_candidates.csv" --exclude=""
python -m pip check
git --version
visual-synthesis config validate -c config/pipeline.json \
  -c config/benchmark.models.json -c config/benchmark.hpc.json
python scripts/benchmark_cluster.py check --device cpu --generation-only
exit
```

Skip environment creation if a working `visual-safety` environment already
exists. Do not upgrade it while jobs are running. The tested CUDA build is
PyTorch 2.6.0+cu118, with Diffusers/Transformers/Accelerate pinned by the
constraints file. The generic Conda YAML is a different installation route.
Installing **Git inside the environment** prevents the compute-node
`git: command not found` failure. The launcher uses its absolute executable path.

After moving a checkout, reinstall its editable package on a compute node:
`python -m pip install --no-deps --no-build-isolation -e .`.
CUDA unavailable or `nvidia-smi` missing on a login/CPU node is expected. GPU
validation is performed by the submitted pilot.

## 4. Hugging Face access

On the login node, from the checkout:

```bash
export HF_HOME="$PWD/runtime/hf-home"
export HF_TOKEN_PATH="$HF_HOME/token"
"$HOME/.conda/envs/visual-safety/bin/hf" auth login
chmod 600 "$HF_TOKEN_PATH"
"$HOME/.conda/envs/visual-safety/bin/hf" auth whoami
```

Accept/request access to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
and [SD 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
with the same account. Enter a read token only at the login prompt, or follow
the supported browser flow. `HF_TOKEN_PATH` names a **file**, not a token value.
The submission helper passes only its path, never its contents. Keep credentials
out of screenshots, Git and shared `.env` files. Exposed tokens should be revoked.

If an older `HF_TOKEN_PATH` names a readable file beginning with `hf_`, the
submission helper securely copies its contents to `runtime/hf-home/token`.
It preserves the original file, prefers an existing standard token file, and
passes only the standard path to PBS. No second login is needed for a valid
cached credential. If neither file is usable, repeat the login commands above.

Verify gated-file access without downloading weights:

```bash
"$HOME/.conda/envs/visual-safety/bin/python" - <<'PY'
from huggingface_hub import get_hf_file_metadata, hf_hub_url
for repo in ("black-forest-labs/FLUX.1-dev", "stabilityai/stable-diffusion-3.5-large"):
    get_hf_file_metadata(hf_hub_url(repo, "model_index.json"), token=True)
    print(repo, "ACCESS OK")
PY
```

Cached file credentials take precedence over stale `.env` token values in the
new pilot. Public metadata access alone does not establish gated weight access.

## 5. Submit all eight, or your assigned models

From the checkout on the login node:

```bash
export VS_ASPIRE_PROJECT=YOUR_PROJECT_ID
export VS_ASPIRE_STORAGE="$PWD/runtime"
export VS_ASPIRE_BUDGET_GIB=500
export HF_TOKEN_PATH="$VS_ASPIRE_STORAGE/hf-home/token"

bash scripts/submit_aspire_project_pilot.sh all
```

Set the budget only after checking your allocation. It is total workspace
capacity in GiB, not a new quota. All eight generator files total about 204 GiB;
the 48,000-image baseline projects to another 60 GiB. A 500 GiB planning
allowance adds space for evaluators, pooled activations and study outputs.

`all` queues **eight separate jobs**, connected by `afterany` dependencies so
they run one at a time and reuse caches. Order: Qwen, GLM, FLUX, SD 3.5 Large,
Z-Image-Turbo, PixArt, SDXL, SD 1.5. A failure is recorded and the next model
can still run. This profiles all eight, not just Qwen. Teammates assigned two
models can submit those two instead:

```bash
first_job=$(bash scripts/submit_aspire_project_pilot.sh sd-xl)
VS_PILOT_AFTER="$first_job" bash scripts/submit_aspire_project_pilot.sh sd-1.5
```

If Qwen's larger pilot is already running, queue the other seven with one command:

```bash
qstat -u "$USER"  # Find the current Qwen job ID, including its .pbs101 suffix
VS_PILOT_AFTER=YOUR_CURRENT_QWEN_JOB_ID \
  bash scripts/submit_aspire_project_pilot.sh remaining
```

`remaining` always means GLM, FLUX, SD 3.5 Large, Z-Image-Turbo, PixArt,
SDXL and SD 1.5; it does not inspect previous results or submit Qwen. Each job
requests its own six-hour allocation. The default routing queue is `normal`.
`VS_ASPIRE_QUEUE=ai` selects the other ASPIRE 2A A100 pool only for projects
with approved AI-queue access. See [GPU choices and limits](docs/aspire_project_pilot.md#gpu-choices-and-concurrency)
before changing queues; H100 GPUs are on the separate ASPIRE 2A+ system.

Do not run simultaneous jobs against the same runtime directory; it has a
shared cache/budget lock. Each teammate has their own account and runtime.
Weights/evaluators are downloaded individually on compute nodes, before timing;
cached resources are reused. Bare `visual-synthesis download` uses legacy model
defaults, so use these scripts and the eight-model config overlay.

Each model's default pilot generates 154 measured images plus warmups, with
60 baseline prompts, five-layer pooled capture, CPU probes, identity checking,
learned/random ablation and steering, and four evaluators. See the
[protocol and limitations](docs/aspire_project_pilot.md). These are resource
pilots on candidate prompts, not validated safety-effectiveness studies.

For a separate short Qwen BF16/NF4 comparison while these jobs continue, use
`bash scripts/submit_aspire_qwen_quantization.sh`. It requests one GPU for
1h45m through `normal`, targeting development-queue routing, reuses Qwen's
cached weights and isolates dependency/cache writes from active pilots.
See the [paired comparison protocol](docs/aspire_qwen_quantization.md).

## 6. Logs, outputs and final accounting

Every new log is in `<checkout>/runtime/logs/`. Every output/report is under
`<checkout>/runtime/outputs/`. Filenames use a unique model/run prefix so runs
cannot overwrite each other. The submission command prints that prefix and
the output directory immediately. From the checkout, for example:

```bash
ls runtime/logs
ls runtime/outputs
RUN_TAG=qwen-image-ABC123  # Replace with the exact printed run prefix
cat "runtime/outputs/$RUN_TAG/pilot_summary.json"
tail -n 40 -F "runtime/logs/$RUN_TAG.pilot.log"
tail -n 40 -F "runtime/logs/$RUN_TAG.baseline.log"
# Later:
tail -n 40 -F "runtime/logs/$RUN_TAG.capture.log"
tail -n 40 -F "runtime/logs/$RUN_TAG.safety-ablation.log"
```

`-F` waits for logs not created yet; Ctrl+C only stops watching. Scheduler
`<run>.pbs.log` may be delivered only at job end; `<run>.pilot.log` and phase
logs are live. All models' logs share the same directory. New jobs create
new run folders, so an old 401 log remains unchanged after fixing authentication.
Cached weights produce no prefetch log.

From the checkout, after each job ends:

```bash
qstat -u "$USER"
"$HOME/.conda/envs/visual-safety/bin/python" scripts/aspire_project_pilot.py collect \
  --job-id YOUR_JOB_ID --output /absolute/path/to/the/printed/run/folder
```

PBS state `F` means finished; exit status `0` means success. Also inspect phase
statuses. Attach final accounting because an in-job snapshot can still say `R`.
On storage errors inspect `myquota`, `df -h` and `df -i` before retrying.

After all assigned models, or all eight, aggregate results from the checkout:

```bash
"$HOME/.conda/envs/visual-safety/bin/python" scripts/aspire_project_pilot.py aggregate \
  --storage "$PWD/runtime" --output "$PWD/runtime/project_estimate.json"
```

Missing/failed models remain explicit. Final reports and frozen reviewed inputs
must be retained outside scratch's purge lifecycle. Old `data/runs` manifests
and root-level PBS files belong to previous pipeline runs; preserve them as
needed. New pilot data is confined to the ignored runtime folder.

## Evaluator-preparation error in an earlier pilot version

If all models stop with `model metadata does not contain a complete configured
Diffusers weight selection`, update this branch before retrying. The earlier
pilot incorrectly applied the generator's `model_index.json` requirement to
flat Transformers CLIP repositories. The corrected evaluator selector validates
their config/processor/tokenizer files and root weights separately, preferring
safetensors when available. Generator validation and download-capacity checks
remain strict. This failure happens before timed generation; existing model
caches and completed two-prompt fit results remain usable. Retry one model first
to confirm evaluator preparation completes, then submit the remaining pilots.
