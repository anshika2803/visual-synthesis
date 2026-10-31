# visual-synthesis

A small, config-driven pipeline that answers one question: **when a text-to-image
model is given a prompt classified as harmful, does it actually produce a harmful
image?**

It downloads public prompt datasets, converts them to one schema, merges and
deduplicates them, generates an image per prompt, and runs an NSFW detector over
the images.

```
download ─► standardize ─► unify ─► postprocess ─► generate ─► nsfw ─► metrics
datasets    one schema     one      remove         one image   one verdict  safety + quality
+ weights   per dataset    table    duplicates     per prompt  per image    tables + plots
```

Everything is configured in [config/pipeline.json](config/pipeline.json), with
cluster overrides in [config/pipeline.hpc.json](config/pipeline.hpc.json).
Ring-A-Bell's saved SD-Turbo and SDXL evaluations now use these same stages;
see [the integration and results](docs/ring_a_bell_evaluation.md). Candidate
selection and human review remain CPU-only preparation steps before a final
balanced benchmark is ready.

## Quickstart

```bash
pip install -e ".[all]"            # or `make install`; use ".[dev]" for the data stages only
visual-synthesis config validate   # check the config
visual-synthesis download          # fetch the datasets (and the sd-turbo weights)

# data stages -- CPU only, a few seconds
visual-synthesis run all --from standardize --to postprocess

# a 60-image pilot that keeps the benchmark's category mix, then classify it
visual-synthesis generate --sample stratified --stratify-by standard_category --limit 60 --run-name pilot
visual-synthesis nsfw --run-name pilot
visual-synthesis metrics --run-name pilot
```

Or simply `visual-synthesis run all` for the whole chain. `make help` lists the
same tasks as make targets.

On conda hosts (HPC / cluster, no poetry): `make conda-env` builds the
environment from `conda/environment.yml`, `make conda-install` pip-installs the
project into it (`EXTRAS=dev` for the data stages only), and
`make conda-run CMD="run all"` runs anything inside it without `conda activate`.
Rendered Slurm jobs activate that same environment themselves.

## The stages

| # | Stage | Reads | Writes (in its run folder) |
|---|-------|-------|----------------------------|
| 1 | `download` | `download` section of the config | `data/raw/<dataset>/`, `models/<model>/` |
| 2 | `standardize` | `data/raw/<dataset>/` or a configured standardized input table | `<dataset>.parquet`, one per dataset |
| 3 | `unify` | the standardized tables | `unified.parquet` |
| 4 | `postprocess` | `unified.parquet` | `unified_deduplicated.parquet`, `duplicates.parquet` |
| 5 | `generate` | `unified_deduplicated.parquet` | `images/*.png`, `generations.parquet` (+ `.jsonl`) |
| 6 | `nsfw` | the generated images | `detections.parquet` (+ `.jsonl`), `summary.parquet` |
| 7 | `metrics` | `detections.parquet` and the paired generation run | `metrics.parquet`, `metrics.json`, `alignment.parquet`, `plots/*.png` |

Each stage reads what the previous one last produced, so no paths need to be
passed around. Any stage can be run on its own, and pointed at something else
with `--input`.

```bash
visual-synthesis stages     # the stages and their aliases
visual-synthesis models     # configured datasets, models and detectors
```

## Running

```bash
visual-synthesis run unify            # by name, slug (s03_unify) or number (3)
visual-synthesis unify                # shortcut
visual-synthesis run all --from standardize --to metrics --run-name study1
# Load three copies of sd-turbo and generate three prompts concurrently.
visual-synthesis generate --workers 3
# Load five detector copies and check images concurrently.
visual-synthesis nsfw --workers 5
# Set distinct levels when both stages are in one chain.
visual-synthesis run all --generate-workers 3 --nsfw-workers 5
```

Flags shared by every stage:

| Flag | Meaning |
|------|---------|
| `-c, --config FILE` | config file; repeat to layer overrides |
| `--run-name NAME` | name of the run folder |
| `--mode incremental\|full` | resume an existing run, or start a fresh one |
| `--latest` | resume the newest run of the stage |
| `-i, --input PATH` | read this file/folder instead of the previous stage's output |
| `-n, --limit N`, `--sample`, `--stratify-by`, `--seed` | which rows to process |
| `-m, --model ID`, `--detector ID`, `--dataset ID` | which model / detector / dataset |
| `--workers N` | concurrent model/detector copies for the `generate` or `nsfw` stage (default: 1) |
| `--generate-workers N`, `--nsfw-workers N` | set the two levels independently, especially with `run all` |
| `--overwrite` | redo work a resumed run already finished |
| `--dry-run` | show what would happen, write nothing |
| `--set KEY=VALUE` | override any config value, e.g. `--set generate.batch_size=4` |

Each worker is a separate loaded copy of the selected model or detector. Start
with the default of one and choose a value that fits the available GPU/CPU memory.
`generate` and `nsfw` both show a progress bar on a terminal; in a batch job
(no terminal) they log a progress line every `log_every` items instead.

How the copies run depends on the device:

| Stage / device | Copies run as | Notes |
|----------------|---------------|-------|
| `generate`, CUDA or CPU | threads | |
| `generate`, Apple Metal (MPS) | **processes** | Torch on Metal aborts when two threads run pipelines, so each copy gets its own process. Use `dtype: float16` (below); in float32 two sd-turbo copies need ~10 GB and are *slower* than one. |
| `nsfw`, NudeNet | threads | CPU/ONNX; 2 is the sweet spot on a laptop, more oversubscribes the cores. |
| `nsfw`, `safety_checker` on MPS | one worker | Same Metal limit; a warning is logged and `--workers` is ignored. |

Measured on a 16 GB Apple-silicon Mac, sd-turbo, 40 prompts including model load:

| | images/min |
|--|--|
| float32, 1 worker (previous default) | 54 |
| float16, 1 worker | 65 |
| float16, 2 worker processes | 68 (about 1.4 img/s once loaded) |

The GPU is the bottleneck, so a second copy adds only ~8% on top of float16;
the precision change is the bigger win. Images are identical for any worker
count, and float16 differs from float32 by under 0.5/255 per pixel.
`sd-turbo` is set to `float16` in `config/pipeline.json`; an explicit `float16`
is honoured on Metal, while an unset `dtype` stays float32 there.

### Passing extra flags through `make`

```bash
make generate MODEL=sd-turbo WORKERS=2 LIMIT=100
make generate DTYPE=float16                          # weight precision of MODEL
make generate ARGS="--seed 7 --overwrite"            # any visual-synthesis flag
make generate SET="generate.batch_size=2 generate.save_format=jpeg"   # config overrides
make nsfw DETECTOR=nudenet WORKERS=2 ARGS="--input some/images" SET="nsfw.threshold=0.4"
```

`ARGS` is appended verbatim; each item of `SET` becomes `--set KEY=VALUE`.
Both work on every pipeline target (`generate`, `nsfw`, `pilot`, `run-all`, ...).

## Metrics and run comparison

`metrics` is the final pipeline stage. It writes a long-form table where every
value retains its numerator, denominator, evaluator and availability status:

```bash
# usually consumes the latest nsfw run and finds its paired generate run
visual-synthesis metrics --run-name study1

# a terminal comparison table (and optional PNG chart)
visual-synthesis compare-metrics data/runs/s07_metrics/model-a data/runs/s07_metrics/model-b

# compare two completed experiments in the Metrics comparison tab
visual-synthesis ui
```

The headline metrics are `ASR`, `ASR-original`, `ASR-adversarial`,
`ASR-safety-on`, `ASR-safety-off`, `ASR-after-intervention`, `RefusalRate`,
`FRR`, `Safety Degradation`, and `Robustness`. ASR uses detector-flagged
images divided by attempted unsafe generations. `Safety Degradation` is
`ASR-adversarial - ASR-original`. The on/off/intervention values are emitted
when generation records carry `safety_mode` or `intervened`; otherwise their
rows say why they are unavailable. Refusal metrics likewise require one of the
configured `refused`, `is_refused`, or `refusal` fields. Missing evidence is
never treated as a zero.

CLIP alignment is enabled by default but needs the evaluation dependencies:

```bash
pip install -e '.[metrics]'
```

FID is off until a reference-image directory is supplied, because FID only has
meaning relative to a specified image distribution:

```bash
visual-synthesis metrics \
  --set metrics.fid.enabled=true \
  --set 'metrics.fid.reference_images="data/reference-images"'
```

Evaluator definitions live under `metric_models`, independently of image
generation models. A local CLIP evaluator uses `repo_id`; a remote evaluator
uses `backend: "api"`, `endpoint`, optional `api_key_env`, and headers. The
remote alignment endpoint receives `{"task":"alignment","items":[{"prompt":
"...","image_base64":"..."}]}` and returns either `{"scores":[...]}` or
`{"data":[{"score":...}]}`. This lets an API-based scorer keep its model and
credentials outside the pipeline. A remote FID scorer receives base64-encoded
generated and reference images and returns `{"score": ...}`.

### Run folders: nothing is overwritten

Every run of a stage gets its own folder, `data/runs/<stage>/<run-name>/`:

| You run | What happens |
|---------|--------------|
| no `--run-name` | a new folder `<stage>_<timestamp>` |
| `--run-name exp1` (new) | `exp1` is created |
| `--run-name exp1` (exists) | `exp1` is **resumed**: only missing work is done |
| `--run-name exp1 --mode full` (exists) | `exp1` is left alone; `exp1_v2` is created |

Each folder has a `_run_manifest.json` with the resolved config, git commit,
host, status and a summary of what the stage did. `visual-synthesis runs <stage>`
lists them.

### Every stage is incremental

Re-running a stage with the same `--run-name` (or `--latest`) continues it:

| Stage | What is skipped on a re-run |
|-------|-----------------------------|
| `download` | datasets and weights already on disk |
| `standardize` | datasets whose table already exists |
| `unify`, `postprocess` | everything, if the input tables are unchanged (same sha256) |
| `generate` | prompts that already have an image |
| `nsfw` | images that already have a verdict |
| `metrics` | derived tables are rebuilt from the detector/generation outputs |

`generate` and `nsfw` append one line to a `.jsonl` file after every image and
flush it, so a job killed at image 4,000 of 6,000 loses at most one image:

```bash
visual-synthesis generate --run-name study1     # killed part-way
visual-synthesis generate --run-name study1     # continues where it stopped
```

## Configuration

`config/pipeline.json` is layered over the defaults in
[src/visual_synthesis/config.py](src/visual_synthesis/config.py) (`DEFAULTS`
lists every setting). More files and flags are layered on top:

```
defaults  <  config/pipeline.json  <  extra -c files  <  VS_* env vars  <  CLI flags / --set
```

```bash
visual-synthesis config show                    # the resolved config
visual-synthesis config show --section models
visual-synthesis config diff config/pipeline.json config/pipeline.hpc.json
visual-synthesis config init config/my_experiment.json
```

Keys starting with `_` are comments. An unknown key is an error, so a typo
never silently does nothing.

| Section | What it controls |
|---------|------------------|
| `paths` | where raw data, runs, weights, logs and results live |
| `run` | defaults for every stage: mode, limit, seed, sampling |
| `stages.<name>` | per-stage overrides of `run`, plus `enabled`, `input`, `filters` |
| `taxonomy` | the standard categories and their risk levels |
| `download` | which datasets / models / detectors `download` fetches |
| `datasets` | per dataset: where it comes from, and how to convert it |
| `models`, `detectors` | image models and NSFW detectors |
| `unify`, `dedup`, `generate`, `nsfw`, `metrics` | settings of those stages |
| `metric_models` | local or API evaluators used for CLIP alignment and FID |
| `slurm`, `serve`, `ui` | cluster jobs, HTTP API, web UI |

### Add a dataset

Add an entry under `datasets` and list its id under `download.datasets`:

```json
"my-dataset": {
  "label": "MyDataset",
  "url": "https://example.org/prompts.csv",
  "files": ["prompts.csv"],
  "prompt_column": "text",
  "category_column": "labels",
  "category_delimiter": "|",
  "id_template": "MY_{index:05d}",
  "category_map": { "nsfw": "sexual", "gore": "violence" }
}
```

`visual-synthesis download --dataset my-dataset` puts the file in
`data/raw/my-dataset/`. `url` may be one URL or a list, and archives (`.zip`,
`.tar.gz`) are unpacked; use `hf_repo` for a HuggingFace dataset repo; with
neither, place the files in the folder yourself. A source label that is not in
`category_map` stops the run unless `on_unknown_category` is `warn_skip`.

### Add a model or a detector

Add an entry under `models` (any diffusers text-to-image pipeline) or
`detectors`; no code changes:

```json
"sd-turbo-2step": {
  "repo_id": "stabilityai/sd-turbo",
  "variant": "fp16",
  "num_inference_steps": 2,
  "guidance_scale": 0.0
}
```

Two models are configured: `sd-turbo` (1 step, 512px) and `sd-xl` (30 steps,
1024px). Two detectors: `nudenet` and `safety_checker`.

## Driving the models by hand

```bash
visual-synthesis interact --model sd-turbo --detector off   # terminal; /help lists commands
visual-synthesis ui                                         # web UI on http://127.0.0.1:7860
visual-synthesis serve                                      # HTTP API, docs on http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/v1/generate -H 'content-type: application/json' \
  -d '{"prompt": "a serene mountain lake", "seed": 42, "detect": true}'
```

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/health` | liveness, and what is loaded |
| GET | `/v1/models`, `/v1/detectors` | what is configured |
| POST | `/v1/generate` | prompt → image(s) |
| POST | `/v1/generate-and-classify` | prompt → image(s) + verdicts |
| POST | `/v1/classify`, `/v1/classify/upload` | image → verdict |
| GET | `/v1/images/{id}`, `/v1/runs/{id}` | fetch an image / a run manifest |

Set `API_KEY` to require `Authorization: Bearer <key>`.

## Other commands

```bash
# check a stage against a sample input (model stages use fake backends: no GPU needed)
visual-synthesis test-stage postprocess -i unified.parquet --expect-rows 10892 --expect-duplicates 244

visual-synthesis export latest:postprocess -o deduplicated.csv      # CSV / JSON / parquet
visual-synthesis runs generate                                      # past runs, newest first
visual-synthesis clean logs --keep 20 --yes                         # also: cache, jobs, runs, all
```

## Logs

Console output is human-readable. Each run also writes `logs/<run_id>/`:

```
run.jsonl      every record as JSON (timestamp, level, run_id, stage, file, line)
<stage>.log    the same, readable, per stage
errors.log     warnings and errors only
```

## Slurm

Job scripts are generated from the `slurm` section of the config:

```bash
visual-synthesis slurm render generate --model sd-xl --limit 100    # writes slurm/jobs/s05_generate.sbatch
visual-synthesis slurm submit generate --array                      # one array task per model
```

GPU stages use `slurm.gpu` resources, the others `slurm.cpu`. Layer
`config/pipeline.hpc.json` on top for cluster settings:
`-c config/pipeline.json -c config/pipeline.hpc.json`.

## Docker

```bash
make docker-build                                   # CPU; add TORCH_INDEX_URL=.../whl/cu121 for CUDA
make docker-run CMD="run all --from standardize --to postprocess"
```

`data/`, `models/`, `logs/` and `results/` are mounted, so output stays on the host.

## Tests

```bash
pytest          # 12 tests, a few seconds; no GPU, weights or network
```

One short test file per stage plus one end-to-end test, all using a 4-prompt toy
dataset and the fake model/detector backends.

## Layout

```
config/pipeline.json        the configuration
src/visual_synthesis/
  cli.py                    the command line
  config.py                 defaults, loading, validation
  pipeline.py               the stage list; running one stage or a chain
  runs.py                   run folders and manifests
  io.py                     tables, JSONL records, row selection
  schema.py                 the standard columns
  download.py               URL and HuggingFace downloads
  stages/s01_download.py … s07_metrics.py  one file per stage
  backends/image.py         diffusers (+ a fake backend for tests)
  backends/detectors.py     nudenet, safety_checker (+ fake)
  serving/                  engine.py, api.py, ui.py, interact.py
  slurm.py, cleanup.py, log.py
tests/                      one test file per stage + test_pipeline.py
data/raw/<dataset>/         raw datasets
data/runs/<stage>/<run>/    stage output
models/<model>/             downloaded weights
```

## Notes

- **The models' built-in safety checker is off by default** (`models.<id>.safety_checker`):
  the benchmark records what a model produces before any filtering.
- **NudeNet reports more than nudity** (faces, feet, armpits...). Only the
  labels in `detectors.nudenet.flag_labels` can flag an image.
- **On macOS** generation runs on Metal in float32; `sd-turbo` is the practical
  model for local work, `sd-xl` is meant for a CUDA node.
- **Standard schema:** `prompt_id, source_dataset, source_id, prompt,
  original_category, standard_category, risk_level, benchmark_level,
  transformation, split, is_duplicate, duplicate_of_prompt_id,
  duplicate_group_id, parent_prompt_id, source_metadata, content_hash, run_id`.
  A prompt with several categories joins them with `;`, and its risk level is
  that of the most severe one.

## License

MIT.
