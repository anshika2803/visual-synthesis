# Phase D1 reviewer workflow

Phase D1 reads the Phase C candidate manifests and writes separate review and
accepted-candidate artifacts. It does not modify the Phase C inputs, existing
standardized datasets, the Ring-A-Bell results, or the final unified benchmark.

For the current review, use the Phase D2 review master, which includes COCO,
corrected Nibbler submitted prompts, and preselected L1/L6 source candidates.
The initial Phase C Nibbler output is explicitly superseded because it used
participant rewrite evidence as prompt text. Do not recreate the active review
master from that superseded Parquet table.

## Prepare the editable review batch

From the repository root:

```bash
PYTHONPATH=src python scripts/phase_d1_review.py prepare \
  --coco data/processed/benchmark_prep/phase_c/coco_l0_candidates.parquet \
  --nibbler data/processed/benchmark_prep/phase_c/nibbler_submitted_candidates.parquet \
  --review-csv data/processed/benchmark_prep/phase_d1/review_batch.parquet
```

For spreadsheet review, export `review_batch.parquet` as CSV to a scratch
location, edit it, then import the edited rows into the Parquet review batch.
The validator also accepts the scratch CSV directly via `--review-csv`. Set
`decision` to `accept`,
`reject`, or `uncertain`; fill `validated_level`, taxonomy `standard_category`
(semicolon-separated for multiple categories), reviewer ID, confidence from 0
to 1, and notes. L0 should have a blank risk category. For L2/L3/L4, also enter
the L1 `parent_prompt_id`, method, and parameters. `near_duplicate_matches`
shows token-set Jaccard neighbors for reviewer consideration; exact duplicates
are explicitly grouped and cannot both be exported as accepted.

Nibbler rows include the actual submitted prompt, the participant's rewrite
describing the observed harm, attack mode, text- and image-safety validator
vote totals, image-failure annotation, and provenance. COCO rows show
caption ID, image ID, and split. Proposed levels are triage hints and are
editable; all decisions begin blank.

To refresh evidence after candidate manifests change, rerun the same `prepare`
command with the existing `review_batch.parquet`. It automatically reads that file
first and preserves all review fields by `candidate_id`. Rows whose candidate
is no longer present remain in the file with `candidate_available=false`; they
are excluded from validation export. A duplicate candidate ID in the saved
review file stops the refresh rather than guessing which decision to keep.

## Validate decisions and export accepted examples

```bash
PYTHONPATH=src python scripts/phase_d1_review.py validate \
  --review-csv data/processed/benchmark_prep/phase_d1/review_batch.parquet \
  --l1 data/runs/s04_postprocess/real1/unified_deduplicated.parquet \
  --output-dir data/processed/benchmark_prep/phase_d1/validated
```

Only complete `accept` decisions enter `validated_candidates.parquet`. `reject`
and `uncertain` remain in the review batch and are not exported. The validator
checks levels, categories, reviewer identity/confidence, parent L1 references,
category preservation, and exact duplicate acceptance conflicts. Near
duplicates are review warnings, not automatic rejections. Counts by level,
source, and category are written to `accepted_counts.json`; validation
problems go to `review_validation_issues.parquet`.

The accepted file remains a validated candidate manifest. The final 6,000-row
benchmark must not be assembled until every requested level has 1,000 valid
examples.
