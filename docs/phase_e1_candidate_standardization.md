# Phase E1 unified prompt candidates

Phase E1 writes `data/processed/unified/unified_prompt_candidates.csv` using the
canonical columns in `visual_synthesis.schema.STANDARD_COLUMNS`, followed by
explicit candidate/review extensions. The canonical field `prompt` holds the
prompt text. `proposed_benchmark_level` and `proposed_standard_category` retain
source proposals; `validated_level` and `validated_standard_category` are filled
only from the D1 validated export. `human_validated` is false for all other rows.

The unified manifest retains exact duplicate source rows. It marks one row as
the group anchor and links other rows to it using `duplicate_of_prompt_id` and
`duplicate_group_id`. Near duplicates are reported separately by token-set
Jaccard similarity and are never removed automatically.

## Build and verify

Run from the repository root:

```bash
PYTHONPATH=src python scripts/standardize_prompt_candidates.py
```

The command reads existing local files only. It writes the requested candidate
manifest and a separate report directory at
`data/processed/unified/unified_prompt_candidates_reports/`. Inputs can be
overridden with `--l1`, `--coco`, `--nibbler`, `--ring-a-bell`,
`--review-master`, `--validated`, and `--selected`. Use `--near-threshold 0.9`
to select the near-duplicate Jaccard threshold. The audit records input
checksums, input row counts, source/review merges, and output paths.

Focused tests:

```bash
PYTHONPATH=src pytest -q tests/test_prompt_candidate_standardization.py
```

## Contributor submission format

Contributors should provide one CSV row per source prompt. Use the canonical
field names and conventions below; keep a stable source identifier and never
reuse an existing prompt ID.

| Field | Contributor guidance |
|---|---|
| `prompt_id` | Globally unique and stable; prefix with the source name, such as `MYSET_00001`. |
| `source_dataset`, `source_id` | Dataset label and original record ID. Do not replace the original source ID with a generated row number when a source ID exists. |
| `prompt` | Exact text submitted or authored as the candidate prompt. For Nibbler, use its submitted prompt, not the participant's rewritten harm description. |
| `original_category` | Source label(s), joined with `;` when there are several. |
| `standard_category` | Known taxonomy labels only: `sexual`, `violence`, `disturbing`, `illegal_activity`, `political_sensitivity`, or `copyright`. Leave blank when unknown; do not guess. |
| `risk_level` | `low`, `medium`, or `high`, using the configured taxonomy. Leave blank when the category is unresolved. |
| `benchmark_level`, `transformation` | Use canonical L0–L6 and transformation values from `schema.py`. Keep a noncanonical proposal such as `unclassified` in `source_metadata` until reviewed. |
| `split`, `parent_prompt_id` | Preserve source split and link transformed prompts to the parent `prompt_id`; leave parent blank for independent original prompts. |
| `source_metadata` | Semicolon-separated `key=value` details. Reserved delimiters inside values are percent-escaped. Preserve provenance, source annotations, Nibbler attack modes/validator evidence, COCO caption/image IDs, and Ring-A-Bell configuration here. |

Do not set `human_validated` in submitted data. Review decisions belong in the
Phase D2 review master; only the D1 validated export establishes human
validation. The E1 CLI merges those fields into supplemental columns without
changing the canonical source files. Exact/near duplicates are retained or
reported with provenance for adjudication.
