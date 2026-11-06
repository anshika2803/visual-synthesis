from types import SimpleNamespace

import pytest

from visual_synthesis.backends.image import DiffusersBackend, batch_groups, with_defaults
from visual_synthesis.config import MODEL_DEFAULTS

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")


class RecordingPipe:
    """Stands in for a diffusers pipeline: records each call, returns one image per prompt."""

    def __init__(self, fail_batches=False):
        self.calls, self.fail_batches = [], fail_batches

    def __call__(self, **call):
        self.calls.append(call)
        prompts = call["prompt"] if isinstance(call["prompt"], list) else [call["prompt"]]
        if self.fail_batches and len(prompts) > 1:
            raise RuntimeError("no list prompts here")
        return SimpleNamespace(images=[PIL.new("RGB", (8, 8)) for _ in prompts])


def backend(pipe, hooks=None):
    model = {**MODEL_DEFAULTS, "id": "stub", "repo_id": "stub/stub"}
    instance = DiffusersBackend(model)
    instance.pipe, instance.hooks = pipe, hooks
    return instance


def requests(n, **extra):
    return [{"prompt_id": f"p{i}", "prompt": f"prompt {i}", "seed": 100 + i, **extra} for i in range(n)]


def test_compatible_requests_share_a_call_and_keep_their_own_seeds():
    pipe = RecordingPipe()
    images = backend(pipe).generate(requests(3))

    assert len(pipe.calls) == 1 and pipe.calls[0]["prompt"] == ["prompt 0", "prompt 1", "prompt 2"]
    seeds = [generator.initial_seed() for generator in pipe.calls[0]["generator"]]
    assert seeds == [100, 101, 102]
    assert [(image["prompt_id"], image["seed"], image["batch_size"]) for image in images] == [
        ("p0", 100, 3), ("p1", 101, 3), ("p2", 102, 3)]


def test_hooks_and_multi_image_requests_get_one_call_each():
    model = {**MODEL_DEFAULTS, "id": "stub", "repo_id": "stub/stub"}
    filled = [with_defaults(r, model) for r in requests(3)]
    assert [len(group) for group in batch_groups(filled, hooks=True)] == [1, 1, 1]
    assert [len(group) for group in batch_groups(filled)] == [3]
    mixed = filled[:1] + [{**filled[1], "steps": 50}] + filled[2:]
    assert [len(group) for group in batch_groups(mixed)] == [1, 1, 1]  # different settings never share a call
    several = [{**request, "num_images": 2} for request in filled]
    assert [len(group) for group in batch_groups(several)] == [1, 1, 1]


def test_a_failed_batch_is_generated_one_at_a_time():
    pipe = RecordingPipe(fail_batches=True)
    images = backend(pipe).generate(requests(2))
    assert [len(call["prompt"]) if isinstance(call["prompt"], list) else 1 for call in pipe.calls] == [2, 1, 1]
    assert [image["batch_size"] for image in images] == [1, 1]
