# Local L2 paraphrase workflow

This is a component of the existing `visual-synthesis` repository. It reads the
team's unified CSV without changing it. No separate repository, paid API key,
Python package installation, or image-generation model is required.

It selects original T2I-RiskyPrompt L1 parents, calls a local text model, records
outcomes, and exports linked L2 candidates. The first pilot has 30 distinct
parents: five assigned to each of the six sampling categories. Multi-label
parents retain every source category and count toward just one sampling quota.

## First run

Use the terminal in this repository. Activate the environment you already use:

```bash
conda activate visual-synthesis
```

Finish the download already in progress:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Keep the Ollama application running. If you use only its CLI, start
`ollama serve` in another terminal; do not start a second server if one already
runs. The workflow does not start a server or download weights on your behalf.

Check that the exact model is available:

```bash
python scripts/l2_paraphrase.py doctor
```

Run three harmless software checks:

```bash
python scripts/l2_paraphrase.py smoke
```

Read `results/runs/local/l2-smoke/summary.md`. This is a software test of loading,
JSON output, logging and basic screens. It is not a test of risky-prompt coverage
or a guarantee of semantic correctness. Smoke results are separate from L2.

Prepare the real parents without making any model requests:

```bash
python scripts/l2_paraphrase.py prepare
```

Read `results/runs/local/l2-pilot/parents.csv`. Coordinate these IDs with the
team's L1 comparison set. If preparation has already been done for you, reuse
that frozen folder: preparation deliberately refuses to overwrite it.

Start with three of the selected parents, then continue the same run:

```bash
python scripts/l2_paraphrase.py run --limit 3
python scripts/l2_paraphrase.py run
```

The second command processes only parents without a saved outcome. Saved
refusals, malformed responses and technical errors are also skipped on resume;
they are evidence, not successful paraphrases. Ctrl+C stops the process; rerun
the same command to continue. A request interrupted before its outcome was
saved may be submitted again on resume.

Inspect `summary.md` and `pairs.csv` before scaling. A high refusal rate or
meaning changes justify stopping the pilot, not automatically cycling models
or retrying refused prompts.

## Files and interpretation

| File within a run folder | Purpose |
|---|---|
| `parents.csv` | Frozen selected parent IDs, original text and inherited labels. |
| `manifest.json` | Source checksum, sampling seed, quota and eligibility audit. |
| `generation.json` | Exact model digest, Ollama version, template, settings and driver checksum. |
| `attempts/*.json` | One durable record per completed parent, including raw responses and request errors. |
| `pairs.csv` | Every selected parent, including pending, refused, malformed, truncated and failed outcomes. |
| `candidates.csv` | Only outputs that passed structural checks; canonical schema first, followed by audit fields. |
| `summary.json` | Complete outcome counts and category distributions, including overlapping source labels. |
| `summary.md` | Readable overview by the nonoverlapping sampling categories. |

All default outputs are under the existing Git-ignored `results/runs/` area.
Keep code/config/docs in Git; share research data through the team's agreed
storage and contribution process. The workflow does not merge its export into
the unified CSV or final benchmark.

Candidate IDs depend on the frozen experiment, parent and candidate text.
`parent_prompt_id` comes from the driver, never from an ID invented by the model.
Source categories are inherited intended labels. They have not been independently
confirmed for the generated wording. `human_validated` remains false,
validated fields stay blank, and semantic equivalence remains unverified.

## What is automatic, and what is not

Automatic: balanced sampling, source provenance, local requests, one retry for
a transient technical failure, response logging, resume, refusal recording,
schema checks, empty-output checks, exact-original checks, duplicate checks,
lexical similarity measurement, and coverage reports.

The local API uses structured JSON output. Refusals reported by the model and
some common refusal prefixes are detected; detection is imperfect. Truncation
reported by Ollama is excluded. Oversized input is recorded as `input_too_long`
instead of silently shortened. Character length and API token counts cannot
guarantee that the runner never truncated its input internally.

**Passing structural checks does not establish that meaning, risk, or category
was preserved.** Do not call the export validated L2 or use its row count as a
count of confirmed paraphrases. An independent semantic-screening pass or the
team's review procedure can be added after measuring pilot yield. The initial
generator does not run an NLI check. The separate
[generation-and-classification test](l2_generation_classification.md) now
exercises a real independent NLI model, including known control pairs, and can
classify the existing pilot outputs.

## Configuration and separate experiments

Edit `config/l2_paraphrase.json` before a new experiment. Paths in this file are
relative to the repository root. `--run-dir` paths are relative to the current
working directory. The model API is restricted to the local machine and ignores
HTTP proxy environment variables for its local calls.

Defaults: one request at a time, 4,096-token context, 256 output tokens,
temperature 0.5, seed 42, at most one transient technical retry, and a conservative
1,600-character input limit. These are provisional pilot settings, not proven
quality thresholds. Seed recording does not guarantee identical outputs across
hardware or runner versions.

Freeze the model, settings and driver once generation begins. A changed model
digest, Ollama version, template, driver or generation setting blocks resume.
Prepare a new run folder for a changed experiment. If you want exactly the same
parent IDs in the new experiment, export the `parent_prompt_id` column as a text
file with one ID per line, no header, and pass `--parent-ids` during preparation.
The selected count and category quotas must be feasible within that list.

For a later 100-parent experiment (quotas differ by at most one):

```bash
python scripts/l2_paraphrase.py prepare --count 100 --run-dir results/runs/local/l2-100
python scripts/l2_paraphrase.py run --run-dir results/runs/local/l2-100
```

For 1,000 parents, six equal integer quotas are impossible: four sampling
categories get 167 and two get 166. Retain all multi-label annotations and report
the actual category coverage. Do not launch the larger run until pilot results
support it and the team agrees on parent selection and screening.

Rebuild reports offline, without invoking Ollama:

```bash
python scripts/l2_paraphrase.py report
```

## Verification

Setup checked on 29 October 2026: Ollama 0.24.0 reports the expected Qwen tag
and digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`.
All three benign smoke requests produced structurally acceptable outputs.
The 30-parent pilot has been frozen with five parents per sampling category;
no risky pilot requests were made during setup. The focused and related
schema/standardization tests passed (26 tests). These checks establish software
readiness, not risky-prompt yield or semantic validity.

```bash
PYTHONPATH=src python -m pytest tests/test_l2_paraphrase.py
python -m ruff check src/visual_synthesis/l2_paraphrase.py scripts/l2_paraphrase.py tests/test_l2_paraphrase.py
```

Tests use harmless synthetic text and a fake local-model client. They verify
sampling quotas, provenance, resume integrity, refusal/error handling and
structural exclusion; they do not measure the real model's semantic quality.

API reference: [Ollama chat](https://docs.ollama.com/api/chat),
[local model inventory](https://docs.ollama.com/api/tags).
