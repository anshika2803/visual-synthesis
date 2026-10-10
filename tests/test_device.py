import pytest

torch = pytest.importorskip("torch")

from visual_synthesis.backends.device import pick_dtype  # noqa: E402


def test_float16_is_honoured_on_metal_but_not_by_default():
    assert pick_dtype("float16", "mps") is torch.float16
    assert pick_dtype(None, "mps") is torch.float32
    assert pick_dtype("bfloat16", "mps") is torch.float32


def test_cpu_stays_float32_and_cuda_defaults_to_half():
    assert pick_dtype("float16", "cpu") is torch.float32
    assert pick_dtype(None, "cuda:0") is torch.float16
