# Phase D2 review operations

Phase D2 adds small review batches, a merge operation that preserves source
evidence, a read-only progress report, and a pilot plan. It does not construct
the final benchmark or transform prompts.

## Review commands

From the repository root, set `PYTHONPATH=src` for the commands below. The
review master is separate from standardized datasets and should remain the
canonical editable decision file.

To include the preselected standardized L1 (1,000) and Ring-A-Bell L6 (285)
source candidates in the review master, append them before opening batches:

```bash
PYTHONPATH=src python scripts/phase_d2_review.py extend-master \
  --master data/processed/benchmark_prep/phase_d2/review_master.parquet \
  --standardized-candidates data/processed/unified/benchmark_6000_candidates.parquet
```

This adds rows to the separate review master, recomputes duplicate diagnostics,
and preserves existing decisions. It does not alter the input candidate manifest
or treat those source examples as human-reviewed.

```bash
PYTHONPATH=src python scripts/phase_d2_review.py select-batch \
  --master data/processed/benchmark_prep/phase_d2/review_master.parquet \
  --output data/processed/benchmark_prep/phase_d2/batches/l1_batch_001.parquet \
  --batch-size 75 --level L1
```

Add `--source COCO_Captions_2017` or `--source Adversarial_Nibbler`, and/or
`--category violence` (one of the six standard risk categories) to filter.
Use `--offset 75` for the next page of the same stable candidate-ID ordering.
Uncertain rows are excluded from normal batches; `--include-uncertain` makes
them eligible for follow-up. The batch has all source and duplicate evidence
columns. Fill `decision` (`accept`, `reject`, `uncertain`) and the reviewer
fields, then merge:

For a spreadsheet editor, export the selected Parquet batch to a temporary
CSV, edit it, and pass the temporary CSV as `--batch`. The merge writes the
canonical review master as Parquet.

```bash
PYTHONPATH=src python scripts/phase_d2_review.py merge-batch \
  --master data/processed/benchmark_prep/phase_d2/review_master.parquet \
  --batch data/processed/benchmark_prep/phase_d2/batches/l1_batch_001.parquet
```

Nonempty reviewer fields replace the corresponding master fields; blank batch
cells never erase decisions. When a value is changed, the old/new value is
appended to `review_evidence`. Source evidence columns are never merged from
the batch. Accepted decisions enter a validated manifest only after D1 checks:

```bash
PYTHONPATH=src python scripts/phase_d1_review.py validate \
  --review-csv data/processed/benchmark_prep/phase_d2/review_master.parquet \
  --output-dir data/processed/benchmark_prep/phase_d2/validated \
  --l1 data/runs/s04_postprocess/real1/unified_deduplicated.parquet
```

Read-only Markdown progress (include the D1 export to count validated rows):

```bash
PYTHONPATH=src python scripts/phase_d2_review.py dashboard \
  --master data/processed/benchmark_prep/phase_d2/review_master.parquet \
  --validated data/processed/benchmark_prep/phase_d2/validated/validated_candidates.parquet
```

## Review and transformation notes

The selector's proposed level is only a triage value. Reviewers can set
`validated_level`, category, confidence in `[0,1]`, reviewer ID, and notes.
For transformed L2/L3/L4 examples, enter the original L1 `parent_prompt_id`,
method, and parameters; the D1 exporter checks the parent and risk-category
preservation. Exact duplicates are validation errors; near matches are shown
for human adjudication. COCO caption and image IDs and Nibbler mode, validator,
failure-type, submitted prompt, and participant rewrite evidence remain visible.

Suggested pilot plan (planning only, no prompts generated):

```bash
PYTHONPATH=src python scripts/phase_d2_review.py pilot-plan \
  --l1 data/runs/s04_postprocess/real1/unified_deduplicated.parquet \
  --output data/processed/benchmark_prep/phase_d2/transformation_pilot_plan.parquet \
  --per-level 25 --seed 42
```

The plan records 25 L1 parents for each missing level, with blank variant text,
parent IDs, preserved category, planned method and acceptance criteria. The
methods are based on prior public work: human meaning-preserving paraphrases
for L2; seeded character/word perturbations with edit logs for L3; human
indirect/contextual framing for L4; and human adversarial rewrites informed by
Ring-A-Bell for L6. Ring-A-Bell's existing 285 rows remain unchanged. The
PromptBench character-level attack taxonomy is a methodology reference, not an
additional downloaded dataset or a direct benchmark source. Nibbler's submitted
prompt is the candidate prompt; its `rewritten_prompt` is annotator evidence
describing observed image harm. Validator attack outcomes do not establish the
transformation level.

Pilot quality review should verify: intended level is present; risk category
matches its parent; no new unintended risk/category was added; text is clear and
usable as a T2I prompt; transformations have reproducible provenance; prompt
duplicates are resolved; and reviewers record uncertainty or rejection rather
than auto-accepting. An L6 attack's success is a separate downstream outcome
measure and must never be inferred from its text label.
