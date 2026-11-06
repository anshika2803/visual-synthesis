# TC2 run guide: E1–E7 per model

As of 2026-11-06. Protocol: `docs/e1_e6_research_plan.md` §0 (tag `protocol-v1`).

## Go / no-go

**Yes, all 8 models can run, once Jay has tagged `protocol-v1`.** Prompts, seeds, model
settings and detector thresholds are frozen in that tag. The second calibration reviewer
does not block generation: if their check changes anything, Jay re-scores the finished
runs centrally. Keep your images on TC2 until the project is submitted.

- [ ] Jay: the team confirms the 78 prompt exclusions; then Jay commits, tags `protocol-v1` and pushes, including the Git LFS files.
- [ ] You: Step 1 (setup) and Step 2 (readiness check) pass on your own TC2 account before E1.

| Model | Owner | Tier | E1 prompts | E4–E6 prompts | Extra sbatch flag | GPU-hours (rough) |
|---|---|---|---|---|---|---|
| sd-xl | Jay | full | done | done | — | done |
| sd-1.5 | Jay | full | done | `e46_v1` | — | 2 |
| z-image-turbo | Jay | full | `core_v1` | `e46_v1` | — | 4 |
| pixart-sigma | | full | `core_v1` | `e46_v1` | — | 5 |
| sd-3.5-large | | reduced | `core_v1_reduced` | `e46_v1_reduced` | — | 20 |
| flux-dev | | reduced | `core_v1_reduced` | `e46_v1_reduced` | — | 27 |
| qwen-image | | reduced | `core_v1_reduced` | `e46_v1_reduced` | `--gres=gpu:nvidia_l40s:1` | 45 |
| glm-image | | reduced | `core_v1_reduced` | `e46_v1_reduced` | — | 65 |

GPU-hours are scaled from Swara's A100 timings; your Step 2 fit test replaces them. Each
person gets 1 GPU at a time, so Qwen and GLM each need their own person.

- Qwen and GLM start E1 by **2026-11-07**.
- Generation stops at **23:59 on 2026-11-11**. A run not finished by then is reported as missing, never as a partial result.
- Send your results to Jay by **2026-11-12**.

## Step 1 — One-time setup on TC2 (about 30 minutes)

You end up with the frozen code, the reference package versions and verified prompt
tables. Run everything in `bash`; type `bash` first if your shell is zsh. Only `git`,
`pip`, `sbatch`, `squeue`, `tar` and `scp` run on the login node.

**1a. Get the code at the frozen tag.**

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/anshika2803/visual-synthesis.git
cd visual-synthesis
module load anaconda 2>/dev/null || true
conda create -y -n visual-safety -c conda-forge python=3.11 git-lfs
conda activate visual-safety
git lfs install
git checkout protocol-v1
git lfs pull
mkdir -p slurm/jobs
git describe --tags        # must print: protocol-v1
```

**1b. Install the exact reference versions.** These are the packages of the environment
that produced the SDXL runs.

```bash
grep -v -e '^#' -e '^-e ' results/environment/tc2-visual-safety-pip-freeze.txt > ~/vs-requirements.txt
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r ~/vs-requirements.txt
pip install --no-cache-dir --no-deps -e .
pip freeze > results/environment/$USER-pip-freeze.txt
grep -E '^(torch|diffusers|transformers|nudenet|torchmetrics)==' results/environment/$USER-pip-freeze.txt
```

The last command must print `torch==2.11.0+cu128`, `diffusers==0.40.0`,
`transformers==5.16.1`, `nudenet==3.4.2` and `torchmetrics==1.9.0`. If any differs, stop
and tell Jay.

**1c. Hugging Face token.** Use your own token; never commit `.env`. For `flux-dev` or
`sd-3.5-large`, first accept the model's licence on its huggingface.co page while logged in.

```bash
echo "HF_TOKEN=hf_your_token_here" > .env && chmod 600 .env
```

**1d. Check the frozen prompt tables.** Every line must end in `OK`. A `FAILED` usually
means `git lfs pull` did not run.

```bash
sha256sum -c <<'EOF'
bf1457da64d4b498bced4c3be3dbf614b88ddd919d7af40cb1e20ad39d83f839  data/processed/e1e6/core_v1.parquet
558dcce8c2585ea0f1e2329681adac1a89b64c7693dcdeb40009c82199c3fe4d  data/processed/e1e6/core_v1_reduced.parquet
1f5438fc278e1b618b320e66bda16fd0cd3a080d51ecab376dbe8bf6e7c19a33  data/processed/e1e6/e46_v1.parquet
82fec053db40d63d4e4d2afc0dbf9047fa76acdcd204051b0999641cbc8ac370  data/processed/e1e6/e46_v1_reduced.parquet
d0f0ce7b2c033eb27a0aa0224a7f571e468f00ae31d091717002a6be21ccd912  data/processed/e1e6/e7_v1.parquet
eb2a668efa1c0d95277228ff105614998fa471401bff5d36ec1392257c26e24f  data/processed/e1e6/quant_v1.parquet
2d24b18e78f16dab1f711cc2716415b17c3b173a4633da6b25c524ea22d682f1  results/review/excluded_prompts.csv
EOF
```

**1e. Variables used by every later step.** Paste these at the start of every new login.
Set `M` to your model id from the table above.

```bash
cd ~/projects/visual-synthesis && module load anaconda 2>/dev/null; conda activate visual-safety
M=pixart-sigma
CFG="-c config/pipeline.json -c config/pipeline.hpc.json -c config/e1e6.json"
GPU=""                       # qwen-image only: GPU="--gres=gpu:nvidia_l40s:1"
CORE=core_v1; E46=e46_v1     # sd-3.5-large, flux-dev, qwen-image, glm-image: CORE=core_v1_reduced; E46=e46_v1_reduced
```

## Step 2 — Readiness check (one job, under 1 hour)

This must pass on your account before your E1 counts (protocol amendment A6). One GPU job
downloads the model and the LAION detector, then runs the hook check. Last, it makes a
2-image fit test.

```bash
sbatch $GPU --job-name=vs-ready-$M scripts/slurm_run.sh chain \
  "download $CFG --model $M" \
  "download $CFG --detector clip-h14" \
  "scripts/validate_hooks.py $CFG --model $M --steps 2" \
  "generate $CFG --model $M --input data/processed/e1e6/$CORE.parquet --run-name pilot-fit-$M --limit 2"
```

When `squeue -u $USER` no longer lists the job, check it:

```bash
sacct -n -S 2026-11-06 --name=vs-ready-$M --format=JobID,State,Elapsed,MaxRSS
grep -E '^(PASS|FAIL)' slurm/jobs/vs-ready-$M-*.out
ls data/runs/s05_generate/pilot-fit-$M/images | wc -l
grep -o '"generation_time_s": [0-9.]*' data/runs/s05_generate/pilot-fit-$M/generations.jsonl
sort -t, -k3 -n slurm/jobs/vs-ready-$M-*-gpu.csv | tail -1
```

It passes when all of these hold:

1. State is `COMPLETED` and MaxRSS is below 30 GB.
2. Every line of the hook check says `PASS`, none says `FAIL`.
3. There are 2 images.

Send Jay the second `generation_time_s` (seconds per image) and the peak GPU memory from
the last line. Jay updates the time estimates from these.

**If anything fails, stop and send Jay the `.out` and `.err` files from `slurm/jobs/`.**
Failures include out-of-memory, a `FAIL` line or a Python error. Do not change the config,
steps, resolution, precision or hook layers to make it fit; any change is a protocol
deviation that Jay records.

One exception is already recorded: `qwen-image` runs its transformer in fp8 on an L40S,
because bf16 does not fit on TC2. That is why it needs the `GPU` flag.

## Step 3 — E1 (gives E1, E2 and your share of E3)

One job generates one image per prompt, then scores them with both detectors and computes
the metrics. E2 and E3 need no runs of their own: Jay computes them from every model's E1.

```bash
sbatch $GPU --job-name=vs-e1-$M scripts/slurm_run.sh chain \
  "generate $CFG --model $M --input data/processed/e1e6/$CORE.parquet --run-name e1-$M" \
  "evaluate e1-$M"
```

A job lasts at most 6 hours. If `sacct` shows `TIMEOUT`, submit **the identical command**
again; it resumes where it stopped. Repeat until the job ends `COMPLETED`.

Done when both hold:

- `wc -l < data/runs/s05_generate/e1-$M/generations.jsonl` prints **874** (full tier) or **438** (reduced tier).
- `data/runs/s07_metrics/e1-$M/image_metrics.parquet` exists.

## Step 4 — E7 multilingual prompts (282 images, every model)

E7 runs straight after E1 because it is cheap and it feeds the cross-model comparison.
Every model uses all 282 variants; E7 is never reduced. Use the committed `e7_v1.parquet`
as is. Never rebuild it with `make_multilingual_prompts.py`.

```bash
sbatch $GPU --job-name=vs-e7-$M scripts/slurm_run.sh chain \
  "generate $CFG --model $M --input data/processed/e1e6/e7_v1.parquet --run-name e7-$M" \
  "evaluate e7-$M"
```

You may queue this while E1 is still running. It waits until your GPU is free; TC2 allows
2 jobs per person, running or queued.

Done when `wc -l < data/runs/s05_generate/e7-$M/generations.jsonl` prints **282** and
`data/runs/s07_metrics/e7-$M/image_metrics.parquet` exists.

## Step 5 — E4 safety direction, then E5 interventions

One chained job does all of E4 and E5. It first generates the E4 prompts while recording
activations; that run is also the α = 0 baseline for E5 and E6. Next, it finds the safety
direction and picks the layer by the frozen rule. Last, it makes 8 E5 runs on the
validation prompts: 2 modes × 2 directions × 2 strengths, each followed by scoring.

```bash
steps=(
  "generate $CFG --model $M --input data/processed/e1e6/$E46.parquet --run-name e4-$M-capture --set generate.capture.enabled=true"
  "evaluate e4-$M-capture"
  "scripts/e4_analyze.py --capture-run data/runs/s05_generate/e4-$M-capture --output results/e4/$M"
)
for mode in ablation steering; do for vec in safety random:0; do for a in 0.5 1.0; do
  r=e5-$M-$mode-${vec/:/}-a$a
  steps+=("generate $CFG --model $M --input data/processed/e1e6/$E46.parquet --run-name $r --set 'stages.generate.filters={\"subset_split\":\"validation\"}' --set generate.intervention.enabled=true --set generate.intervention.e4_dir=results/e4/$M --set generate.intervention.mode=$mode --set generate.intervention.vector=$vec --set generate.intervention.alpha=$a" "evaluate $r")
done; done; done
sbatch $GPU --job-name=vs-e4e5-$M scripts/slurm_run.sh chain "${steps[@]}"
```

On `TIMEOUT`, paste the same block again; finished runs are skipped. The direction step
reruns each time and gives the same result.

Check the image counts:

```bash
for r in e4-$M-capture $(ls data/runs/s05_generate | grep "^e5-$M-"); do
  echo "$r $(wc -l < data/runs/s05_generate/$r/generations.jsonl)"
done
```

| Run | Full tier | Reduced tier |
|---|---|---|
| `e4-$M-capture` | 392 | 198 |
| each of the 8 `e5-$M-…` runs | 209 | 106 |

**If the first finished E5 run has 392 or 198 images, the validation filter did not
apply.** Cancel with `scancel <job id>` and tell Jay.

Done when `results/e4/$M/` holds the E4 tables and all 8 E5 runs have a folder under
`data/runs/s07_metrics/`.

## Step 6 — E6 safety–utility sweep

E6 adds α = 0.25 and 0.75 for one mode, with the safety direction and the random one. The
mode comes from E5 by a fixed rule: the mode with the larger direction-specific drop, or
ablation if neither has one. The report computes it, so nobody picks it by hand.

**6a. Get the mode** (CPU job, a few minutes):

```bash
sbatch --gres=gpu:0 --cpus-per-task=4 --mem=16G --time=00:30:00 --job-name=vs-report-$M \
  scripts/slurm_run.sh scripts/e1e6_report.py tables --model $M \
  -c config/pipeline.json -c config/e1e6.json --exclude results/review/excluded_prompts.csv
grep -h '"e6_mode"' slurm/jobs/vs-report-$M-*.out | tail -1
```

**6b. Run the 4 E6 conditions.** Set `MODE` to what 6a printed (`ablation` or `steering`).

```bash
MODE=ablation
steps=()
for vec in safety random:0; do for a in 0.25 0.75; do
  r=e6-$M-$MODE-${vec/:/}-a$a
  steps+=("generate $CFG --model $M --input data/processed/e1e6/$E46.parquet --run-name $r --set 'stages.generate.filters={\"subset_split\":\"validation\"}' --set generate.intervention.enabled=true --set generate.intervention.e4_dir=results/e4/$M --set generate.intervention.mode=$MODE --set generate.intervention.vector=$vec --set generate.intervention.alpha=$a" "evaluate $r")
done; done
sbatch $GPU --job-name=vs-e6-$M scripts/slurm_run.sh chain "${steps[@]}"
```

Done when each of the 4 `e6-$M-…` runs has **209** images (full tier) or **106** (reduced
tier) and a folder under `data/runs/s07_metrics/`.

## Step 7 — Check, send results to Jay, free disk

**7a. Final tables.** Run the 6a command again, now that E6 is in.

**7b. Check that every run is complete.** It should print 15 lines, each ending
`metrics-ok`, with the image counts from Steps 3–6.

```bash
for r in e1-$M e7-$M e4-$M-capture $(ls data/runs/s05_generate | grep -E "^e[56]-$M-"); do
  n=$(wc -l < data/runs/s05_generate/$r/generations.jsonl)
  test -f data/runs/s07_metrics/$r/image_metrics.parquet && s=metrics-ok || s=METRICS-MISSING
  echo "$r $n $s"
done
```

**7c. Pack the small result files, never the images.**

```bash
tar czf ~/e1e7-$M-$USER.tgz \
  data/runs/s07_metrics/e?-$M* data/runs/s0[56]_*/e?-$M*/_run_manifest.json \
  results/e4/$M results/e1e6/$M results/hook_validation/$M.json \
  results/environment/$USER-pip-freeze.txt slurm/jobs/vs-ready-$M-*.out
tar tzf ~/e1e7-$M-$USER.tgz | grep -c '/images/'     # must print 0
```

**7d. Copy it to your laptop** (run this on the laptop, with the address you use for
`ssh`), then send the `.tgz` file to Jay.

```bash
scp your_user@tc2_address:e1e7-*.tgz .
```

**7e. Free disk.** Wait until Jay confirms the results arrived. Then delete only this
model's downloaded weights, before starting your next model. The home quota is about 100 GB.

```bash
du -sh data/cache/huggingface/hub/models--*
rm -rf data/cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev   # example: use your model's folder
```

Keep the generated images in `data/runs/` until Jay says the project is submitted. Jay may
need them to re-score. They never leave TC2.

## Rules and troubleshooting

| Situation | What to do |
|---|---|
| Is my job running? | `squeue -u $USER`. For history: `sacct -u $USER -S today --format=JobID,JobName%22,State,Elapsed,MaxRSS` |
| Watch a job | `tail -f slurm/jobs/<job name>-<job id>.out`. Errors go to the `.err` file |
| `TIMEOUT` | Submit the identical command again; it resumes |
| `PENDING` with a `QOSMax…` reason | Normal. Your other job holds the GPU, or you have 2 jobs already; it starts when that one ends |
| `OUT_OF_MEMORY`, CUDA out of memory, a `FAIL` line, a Python error, `METRICS-MISSING` | Stop. Send the `.out` and `.err` files to Jay. Change nothing |
| Disk nearly full | Check with `quota -s` or `mytcinfo`. Ask Jay before deleting anything other than a finished model's weights |

These rules keep all 8 models comparable:

- **Stay on `protocol-v1`.** No `git pull`, no other branch (including `swara/experiments`) and no edits to tracked files. `git status --short` should list only new files under `data/`, `results/` and `slurm/`.
- **Use the run names exactly as written.** A test run you add yourself must start with `pilot-`; otherwise the report counts it as a result.
- **Submit all work through `sbatch`.** Never run Python on the login node.
- **Images never leave TC2.** Never commit, push, copy or share them. If a prompt or image seems to involve a minor, do not open it further and tell Jay.
- **One model at a time.** Keep one large model's weights on disk at a time.
