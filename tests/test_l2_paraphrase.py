import csv
import io
import json
from collections import Counter
from copy import deepcopy

import pytest

from visual_synthesis.l2_paraphrase import (
    CATEGORIES,
    OllamaClient,
    OllamaError,
    _smoke_parents,
    freeze_run,
    generate,
    load_run,
    load_settings,
    main,
    parse_response,
    select_parents,
    write_report,
)
from visual_synthesis.schema import STANDARD_COLUMNS


def write_source(path, rows):
    columns = [
        "prompt_id",
        "source_dataset",
        "source_id",
        "prompt",
        "standard_category",
        "benchmark_level",
        "is_duplicate",
        "human_validated",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def source_row(index, categories):
    return {
        "prompt_id": f"TRP_{index}",
        "source_dataset": "T2I-RiskyPrompt",
        "source_id": str(index),
        "prompt": f"Ordinary synthetic test scene number {index}.",
        "standard_category": categories,
        "benchmark_level": "L1",
        "is_duplicate": "False",
        "human_validated": "False",
    }


def response(text="A bicycle coloured red stands next to a wall made of brick.", *, status="produced", reason=""):
    return {
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "message": {"content": json.dumps({"status": status, "paraphrase": text, "reason": reason})},
    }


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []
        self.digest = "fixed-model-digest"

    def identity(self):
        return {
            "model": "test-model",
            "model_digest": self.digest,
            "ollama_version": "test-runner",
            "model_size_bytes": 1,
        }

    def request(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def frozen(tmp_path):
    run_dir = tmp_path / "run"
    freeze_run(run_dir, _smoke_parents(), {"run_kind": "benign_smoke"})
    return run_dir, load_settings()


def test_sampler_meets_quotas_preserves_multilabels_and_is_reproducible(tmp_path):
    source = tmp_path / "source.csv"
    rows = [source_row(index, category) for index, category in enumerate(CATEGORIES)]
    rows += [source_row(index + 100, ";".join(CATEGORIES)) for index in range(6)]
    write_source(source, rows)
    parents, audit = select_parents(source, source_dataset="T2I-RiskyPrompt", count=12, seed=42)
    assert len({row["parent_prompt_id"] for row in parents}) == 12
    assert Counter(row["sampling_category"] for row in parents) == dict.fromkeys(CATEGORIES, 2)
    assert sum(";" in row["intended_source_categories"] for row in parents) == 6
    assert all(row["source_human_validated"] == "False" for row in parents)
    assert audit["eligible_distinct_parents"] == 12
    write_source(source, list(reversed(rows)))
    again, _ = select_parents(source, source_dataset="T2I-RiskyPrompt", count=12, seed=42)
    assert again == parents


def test_sampler_excludes_invalid_parents_and_respects_team_allowlist(tmp_path):
    source = tmp_path / "source.csv"
    good = [source_row(index, category) for index, category in enumerate(CATEGORIES)]
    extra = [
        dict(source_row(50, "sexual"), is_duplicate="True"),
        dict(source_row(51, "sexual"), benchmark_level="L2"),
        source_row(52, "unknown"),
    ]
    write_source(source, good + extra)
    allowed = {row["prompt_id"] for row in good}
    parents, _ = select_parents(source, source_dataset="T2I-RiskyPrompt", count=6, seed=1, allowed_ids=allowed)
    assert {row["parent_prompt_id"] for row in parents} == allowed
    _, audit = select_parents(source, source_dataset="T2I-RiskyPrompt", count=6, seed=1)
    assert audit["source_exclusions"] == {"marked_duplicate": 1, "not_L1": 1, "unknown_category": 1}
    with pytest.raises(ValueError, match="not eligible"):
        select_parents(source, source_dataset="T2I-RiskyPrompt", count=6, seed=1, allowed_ids={"missing"})
    with pytest.raises(ValueError, match="Cannot fill"):
        select_parents(source, source_dataset="T2I-RiskyPrompt", count=12, seed=1)


def test_frozen_inputs_cannot_be_overwritten_or_edited(frozen):
    run_dir, _ = frozen
    with pytest.raises(ValueError, match="never overwritten"):
        freeze_run(run_dir, _smoke_parents(), {"run_kind": "benign_smoke"})
    with (run_dir / "parents.csv").open("a") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="changed"):
        load_run(run_dir)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (response(), "produced"),
        (response("", status="refused", reason="I cannot perform this request."), "refused"),
        (response(""), "empty"),
        (response(reason="unexpected reason"), "malformed"),
        ({"done": True, "message": {"content": "I cannot help with this request."}}, "refused"),
        ({"done": True, "message": {"content": "not JSON"}}, "malformed"),
        ({"done": True, "message": {"content": "[]"}}, "malformed"),
        ({"done": True, "message": {"content": ""}}, "empty"),
        (dict(response(), done_reason="length"), "truncated"),
        (dict(response(), done=False), "truncated"),
    ],
)
def test_response_outcomes_are_distinguished(result, expected):
    assert parse_response(result)["generation_status"] == expected


def test_resume_skips_saved_refusals_and_preserves_real_parent_ids(frozen):
    run_dir, settings = frozen
    client = FakeClient(
        [
            response("", status="refused", reason="Unable to perform this request."),
            response("A pair of cats sleep on a sofa that is blue."),
            response("On the kitchen's wooden table rests a bowl containing apples."),
        ]
    )
    first = generate(run_dir, settings, client=client, limit=1)
    assert first["generation_counts"] == {"refused": 1, "pending": 2}
    assert len(client.calls) == 1
    assert first["structural_checks_passed"] == 0
    finished = generate(run_dir, settings, client=client)
    assert len(client.calls) == 3  # No new call for the completed refusal.
    assert finished["remaining"] == 0
    assert finished["human_validated"] == finished["meaning_verified"] == 0
    with (run_dir / "candidates.csv").open() as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 2
    assert set(STANDARD_COLUMNS).issubset(exported[0])
    assert {row["parent_prompt_id"] for row in exported} == {"SMOKE_2", "SMOKE_3"}
    assert all(row["human_validated"] == "False" and row["validated_level"] == "" for row in exported)
    _, _, outcomes = load_run(run_dir)
    assert outcomes["SMOKE_1"]["request_attempts"][0]["response"]["done"] is True
    assert all(payload["stream"] is False for _, payload in client.calls)
    assert (
        json.loads(client.calls[0][1]["messages"][1]["content"])["description"]
        == _smoke_parents()[0]["original_prompt"]
    )


def test_only_transient_technical_errors_get_one_retry(frozen):
    run_dir, settings = frozen
    client = FakeClient(
        [
            OllamaError("temporary", retryable=True),
            response(),
            OllamaError("invalid request", status=400),
            response("", status="refused", reason="Declined"),
        ]
    )
    summary = generate(run_dir, settings, client=client)
    assert len(client.calls) == 4
    assert summary["generation_counts"] == {"produced": 1, "technical_error": 1, "refused": 1}
    assert len(load_run(run_dir)[2]["SMOKE_1"]["request_attempts"]) == 2


@pytest.mark.parametrize("change", ["digest", "options"])
def test_resume_rejects_different_model_or_settings(frozen, change):
    run_dir, settings = frozen
    client = FakeClient([response()])
    generate(run_dir, settings, client=client, limit=1)
    settings = deepcopy(settings)
    if change == "digest":
        client.digest = "another-digest"
    else:
        settings["options"]["temperature"] = 0.9
    with pytest.raises(ValueError, match="settings changed"):
        generate(run_dir, settings, client=client)
    assert len(client.calls) == 1


def test_screen_rejects_identical_originals_and_all_duplicate_candidates(frozen):
    run_dir, settings = frozen
    client = FakeClient(
        [
            response(_smoke_parents()[0]["original_prompt"]),
            response("The same sentence."),
            response("The same sentence."),
        ]
    )
    result = generate(run_dir, settings, client=client)
    assert result["structural_checks_passed"] == 0
    with (run_dir / "pairs.csv").open() as handle:
        pairs = list(csv.DictReader(handle))
    assert "matches_original_prompt" in pairs[0]["screening_flags"]
    assert all("duplicate_candidate" in row["screening_flags"] for row in pairs[1:])


def test_long_input_is_recorded_without_shortening_or_calling_model(frozen):
    run_dir, settings = frozen
    settings["max_parent_chars"] = 1
    client = FakeClient([])
    result = generate(run_dir, settings, client=client)
    assert result["generation_counts"] == {"input_too_long": 3}
    assert client.calls == []


def test_report_works_offline_before_any_generation(frozen):
    run_dir, _ = frozen
    summary = write_report(run_dir)
    assert summary["generation_counts"] == {"pending": 3}
    assert summary["completed_outcomes"] == 0
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "candidates.csv").read_text().startswith("prompt_id,")
    assert main(["report", "--run-dir", str(run_dir)]) == 0


def test_client_rejects_missing_download_and_non_local_endpoints(tmp_path):
    settings = load_settings()
    client = OllamaClient(settings)
    client.request = lambda endpoint, **kwargs: {"version": "1"} if endpoint == "/api/version" else {"models": []}
    with pytest.raises(OllamaError, match="not ready"):
        client.identity()
    settings["base_url"] = "https://example.com"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="local Ollama"):
        load_settings(config)


def test_request_sends_json_to_local_ollama_without_a_proxy():
    settings = load_settings()
    client = OllamaClient(settings)
    calls = []

    class FakeOpener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return io.BytesIO(json.dumps(response()).encode())

    client.opener = FakeOpener()
    assert client.request("/api/chat", {"model": settings["model"], "stream": False})["done"]
    request, timeout = calls[0]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert json.loads(request.data)["stream"] is False
    assert timeout == settings["timeout_seconds"]
