# L2 paraphrase pilot plan

Plan date: 29 October 2026, Asia/Singapore.

## Objective and current obstacle

Create a derived L2 candidate set linked to selected T2I-RiskyPrompt L1
parents. Change wording while preserving intended meaning and source risk
categories. The teammate's previously reported obstacle was a guardrail refusal
described as a privacy violation. No exact provider response or reproducible
request has been supplied, so provider-specific diagnosis remains unavailable.

This plan selects a practical inference setup. It does not establish that the
generator will accept every source prompt or that automatic checks prove
semantic equivalence. Refusals remain recorded outcomes.

## Recommended first setup

| Component | Selection | Location |
|---|---|---|
| Text generator | `qwen3:4b-instruct-2507-q4_K_M` through Ollama | Local Apple-silicon Mac |
| Driver and structural checks | A small Python script, using the local Ollama HTTP API | Existing local `visual-synthesis` environment |
| Optional independent meaning screen | `cross-encoder/nli-deberta-v3-base` using Transformers | Separate local CPU pass after unloading the generator |
| Full pilot artifacts | New folder under `results/runs/local/l2-pilot/` | Local ignored results area |

Observed local prerequisites: Apple-silicon architecture, 16 GiB installed
memory, Ollama installed, and approximately 18.5 GiB free disk at inspection.
Only a DeepSeek-R1 1.5B model manifest was found in Ollama's default model
inventory. The proposed Qwen checkpoint has not been downloaded or tested by
this planning task.

The official Ollama tag lists a 2.5 GB Q4_K_M model. Its download size is not
its full runtime-memory requirement. A single short-context session is a
reasonable first fit test on this Mac, not a measured memory guarantee.
The underlying Qwen 4B Instruct checkpoint supports non-thinking output, which
avoids having to separate reasoning blocks from short paraphrase outputs.

Official sources:

- [Ollama checkpoint](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M)
- [Underlying Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Ollama chat API and structured response parameter](https://docs.ollama.com/api/chat)
- [NLI verifier model card](https://huggingface.co/cross-encoder/nli-deberta-v3-base)

## Why start locally

The local setup can test feasibility without a Slurm queue or a new server
framework. The repo's TC2 notes record ENOSPC failures, an unconfirmed personal
quota and deferred downloads. Confirm an appropriate writable storage location
before using TC2 for any new model download. Those notes are historical evidence,
not a live inspection of current cluster capacity.

If the local pilot works and local throughput becomes a problem, the same
underlying 4B checkpoint can be tested with Transformers on an allocated TC2
GPU. The existing A40/L40S observations concern other models and do not establish
Qwen's measured resource needs. Use batch size one first, limited context, and
a separate output folder. Observe site rules for downloads and GPU jobs. Do not
add vLLM or a larger model just to run a small pilot.

Moving to TC2 addresses a performance or memory problem. It does not guarantee
different safety-refusal behaviour.

## Stage 1: confirm the runner works

Check the Ollama version and whether the local server is running. Start the
Ollama application if needed. The proposed download command is:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

First exercise the runner on a few ordinary benign paraphrase pairs to test
model loading, response parsing and logging. This is a software check, not part
of the harmful L2 benchmark. Record the resolved model digest and runner version.

For the initial experiment, use one request at a time, a 4,096-token context,
and a short output limit such as 256 tokens. A provisional temperature of 0.5
is an experiment setting, not an established paraphrase-quality optimum.
Record all options and detect truncated outputs. Use structured responses where
supported, then validate their contents in code.

## Stage 2: select and freeze 30 parents

Select 30 distinct source records, aiming for five per risk category. Preserve
all category labels for multi-label records. Assign a documented primary
sampling category only for quota accounting if needed; do not duplicate one
parent merely to increase the row total.

Coordinate which parents will be present in the corresponding L1 comparison
set. Do not generate L2 for unrelated parents and later claim a matched L1/L2
comparison. Save parent IDs, original text, source record IDs, the source-file
checksum, and the deterministic sample seed in a frozen input manifest.

## Stage 3: attempt one candidate per parent

Attempt the supported paraphrasing task once per parent using a consistent,
recorded instruction template. No candidate is human-validated by this step.
The source ID and parent ID should be assigned by the driver from the input
record, rather than trusted merely because a model echoed an ID in JSON.

Record one outcome for every attempt:

- candidate produced;
- refusal;
- technical error;
- empty, malformed, or truncated response.

A refusal is an ordinary completed response expressing unwillingness to perform
the request. Technical failures include model-loading errors, connection
failures, out-of-memory errors, or a failed HTTP request. A provider may also
return an explicit policy-related error code. Preserve the actual response or
error so these cases are not conflated. Retry a transient technical failure
at most once in the pilot; do not repeatedly retry safety refusals or treat
them as successful paraphrases.

## Stage 4: screen candidates

1. Reject empty outputs, identical normalized text, invalid parent references,
   and duplicate candidate text. Flag malformed and truncated responses.
2. Record lexical differences. Do not impose an unexplained universal edit-
   distance cutoff or reward unnecessary changes to the requested scene.
3. Optionally run the independent NLI model in both directions. Record scores
   and contradiction/neutral flags. NLI predictions are screening evidence,
   not a proof of equivalent meaning, especially on an untested domain.
4. Inherit the original category as the intended source label and record its
   origin. Do not describe it as independently revalidated just because the
   same label was copied or emitted by the generator. Keep possible changes
   of subjects, actions, objects, negation, and risk category visible as flags.
5. Retain a distinct uncertain/rejected pool and the original failed attempts.

Do not use a fixed CLIP text-similarity threshold as a mandatory gate in the
first pilot. It has not been calibrated for this task and could bias a study
of wording/encoder sensitivity toward pairs that already share embeddings.

Recommended record fields:

```text
prompt_id, parent_prompt_id, source_dataset, source_id,
original_prompt, paraphrased_prompt, intended_source_categories,
source_file_sha256, generation_model, generation_model_digest,
generation_options, generation_status, error_or_refusal,
screening_model, screening_scores, screening_flags,
screening_status, human_validated
```

`human_validated` remains false unless a person actually reviews that record
through the agreed workflow.

## Stage 5: make a feasibility decision before scaling

Summarize all 30 attempts and outcomes by category. Determine whether the
generator produces useful candidates across the required categories, whether
the structural checks work, and whether the available evidence supports the
screening choices. Record time and resource use rather than promising a full-
dataset runtime in advance.

If the pilot is workable, expand to 100 parents, then consider the 1,000-pair
target. A second candidate for an unsuccessful parent is an optional measured
experiment, not a guarantee. Freeze selection and screening rules before the
main run. Report exclusions and acceptance/refusal rates by category so that
selection based on model compliance does not disappear from the study.

If refusals or meaning changes dominate, stop expansion and bring the evidence
to the team. Options are approved research sources, a smaller clearly scoped
L2 study, or an explicit revision of the dataset requirement. More GPU memory
does not resolve a policy refusal. Do not label an incomplete or automatically
screened collection as 1,000 human-validated paraphrases.

Suggested reporting label: **synthetically generated L2 candidates derived
from T2I-RiskyPrompt and screened with automated checks**. State separately
which records, if any, received human review.
