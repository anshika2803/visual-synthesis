# Paired L2 text-generator comparison

This development experiment compares Qwen 3 4B Instruct with Gemma 2 2B using
the same revised instruction, structured response schema, seed, temperature,
context limit and output limit. The models differ in family, size and training;
this is a practical model comparison, not an isolated test of architecture.

Gemma 3 4B was the initial candidate. Its generic, explicit Q4_K_M and QAT tags
all returned registry HTTP 412 (newer Ollama required) with the installed Ollama
0.24.0. Gemma 2 2B was selected without upgrading the installed application.
No Gemma 3 inference took place.

## Sample and interpretation

There are 30 identical inputs per generator: the previous pilot's 25 non-sexual
parents, three previously inspected adult non-sexual nudity descriptions, and
two benign controls. Five source parents per category are retained except for
sexual, which has three. The two controls are not benchmark candidates.
Source IDs and labels remain provenance, not validated ground truth. This
reuses development examples and is not a held-out evaluation.

Each model receives one generation attempt per parent. Refusals, malformed
responses and technical errors remain distinct saved outcomes. No refusal is
automatically retried or routed to another model. Both models receive the full
same sample, rather than only the other model's failures.

The common diagnostic screen uses the pinned offline DeBERTa NLI model in both
directions, with unchanged exploratory thresholds, plus the existing literal
spatial-consistency check. Neither generator serves as the judge. The screen's
19 control predictions are retained, including failures. It can miss omitted
details and does not establish L2 validity. Lexical similarity measures wording
overlap and is not a quality score. All human-validation fields remain false.

## Run and inspect

```bash
conda activate visual-synthesis
python scripts/l2_model_compare.py
```

Default outputs: `results/runs/local/l2-model-comparison-v1/`.

- `experiment.json`: frozen inputs, instruction, settings and model digests.
- `attempts/`: raw responses, parsed outcomes and timings, one file per input.
- `comparison.csv`: original and generated text with diagnostic predictions.
- `summary.json`: category counts and screening limitations.
- `screening_controls.json`: checker control results.
- `report.html`: original text and both models' responses side by side.

Completed attempts are reused on resume. Inputs, generation code and model
identities must match the frozen manifest; changes require a new output folder.
`--phase screen` repeats only offline screening of existing outputs.

Use this experiment to propose content-specific generator assignments, then
test those assignments on fresh examples. Do not choose a routing rule solely
to maximize response rate or describe diagnostic positives as accepted labels.
No images are generated and no final benchmark rows are modified by this script.

## Completed development result

Both generators completed all 30 requests without technical errors. Qwen
produced 25 candidates and refused five; Gemma produced 30. The common diagnostic
screen marked 24 Qwen and 19 Gemma responses positive, including the benign
controls. Its 19 controls matched expected labels. This is not a measured
accuracy estimate: inspection identified new false positives in both pools.

Qwen's TRP_00899 changes both participants scanning to only the buyer scanning.
Gemma's spatial control changes behind a wall to against it. Gemma's only
source case with a positive screen where Qwen had no positive, TRP_03412, adds
a threatening demeanor and changes holding to brandishing. These observations
are retained in `assistant_diagnostic_audit.json`, separately from unchanged
raw predictions; they are not human validation. The report includes this audit.

Qwen remains the stronger development baseline in this comparison. Gemma's
greater response rate does not establish better coverage of faithful L2 pairs.
No reliable category-specific routing rule is demonstrated by this sample.
