# Generation and classification checks

Run the current check from the repository terminal:

    conda activate visual-synthesis
    python scripts/l2_verify.py self-test

The first run generates three harmless paraphrases through Ollama. Repeated
runs reuse their saved outcomes and classifier records. To generate a fresh
set in another folder:

    python scripts/l2_verify.py self-test --run-dir results/runs/local/l2-fresh-check

The generator and the original 30-parent pilot remain unchanged.

## What is checked

The default checker combines three sources of evidence:

1. DeBERTa NLI compares original-to-created and created-to-original meaning.
2. An explicit spatial check compares recognized relations attached to the same
   literal object phrase. Behind/inside changes are rejected, while
   beneath/under and beside/next-to are normalized as synonyms.
3. A fresh Qwen judging request compares subjects, roles, actions, quantities,
   positions, negation, objects, attributes and style. Changed-detail reports
   must include literal evidence copied from the source descriptions.

The detail judge uses the same Qwen checkpoint as the generator in a separate
request. It is not an independent second model. The NLI model is separate.
Expected control labels are not supplied to either inference model.

High-confidence NLI rejections remain No. An explicit spatial mismatch or
supported detail-change judgment is also No. Yes requires both NLI acceptance
and an equivalent-scene judgment. Incomplete, malformed, unsupported or
ambiguous judging output stays Uncertain. These are provisional screening
labels; human_validated remains false.

Wording variety is kept separate from meaning preservation. A light rewrite
can still be L2. A reworded prompt that changes the scene cannot pass merely
because it looks similar.

## Meaning of PASS

The checker evaluates 19 author-defined diagnostic control pairs: the original
12 simple controls, the observed pilot error, four additional examples and two
spatial-synonym/new-object cases. These are development and regression checks,
not a held-out human-validated benchmark.

The pipeline test passes when generation outcomes are complete, all three
benign inputs produced candidates, classification ran, and every control
matches its expected label. A correct No for a generated pair does not fail
the software test.

Generation quality is reported separately. For the current saved three-prompt
test, two generated pairs receive Yes and one receives No: sleeping was
changed to resting. Thus a passing pipeline does not mean every generated
candidate is an acceptable paraphrase.

Exit code 0 means the pipeline checks passed. Exit code 3 means the checks
need review. Exit code 2 means a setup/input error. The macOS redirect note and
Torch deprecation warning seen during these runs are dependency messages;
they did not prevent inference or the successful test.

## Results

Current outputs are separate from previous experiments:

    results/runs/local/l2-generation-classification-test/
      classification/nli-detail-v2/
        classification.csv
        controls.csv
        classifier.json
        records/
        summary.json

The first three columns are original_prompt, created_prompt and is_l2
(Yes, No or Uncertain). Remaining columns retain the original ID, reasons,
detail evidence, rewording assessment and whether controls passed.
The summary separates pipeline_test_passed from all_generated_pairs_accepted.

Classify the existing real pilot without regenerating refused prompts:

    python scripts/l2_verify.py classify

Results go under results/runs/local/l2-pilot/classification/nli-detail-v2/.
The original generation, refusal records and prior classifier experiments
remain available. A changed classifier configuration or input set requires
a new --output-dir; saved results are not silently mixed with another experiment.

## Limitations and previous failure

The NLI-only checker accepted the known incorrect pilot pair TRP_00899, where
behind a store became inside it. A whole-pair Qwen judgment also initially
missed this change. The explicit spatial check addresses this error across
recognized relations and common literal anchors. Unknown anchors, pronouns,
and more complex relations remain with the model checks.

Passing these controls does not prove accuracy across the six risk categories
or all benchmark levels. The detail judge can make mistakes and shares biases
with the generator. Subtle role, attribute, style or risk changes may be missed.
No result is automatically merged into a validated benchmark, and
automatic_benchmark_validation_established stays false.

The NLI-only experiment remains reproducible:

    python scripts/l2_verify.py self-test --backend nli

It retains the previous 12/13 outcome in the DeBERTa experiment folders; the
current workflow does not change its model, threshold or expected labels to
erase that failure.

NLI model: cross-encoder/nli-deberta-v3-base, pinned to revision
6c749ce3425cd33b46d187e45b92bbf96ee12ec7. The CPU inference loads only the
already-downloaded snapshot under models/l2-nli/. The exploratory 0.8
probability threshold and 512-token limit are unchanged. Overlong input is
flagged rather than silently shortened.

Reference: [classifier model card](https://huggingface.co/cross-encoder/nli-deberta-v3-base).
