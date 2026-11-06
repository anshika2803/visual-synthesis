import json

import numpy as np
import pytest

from visual_synthesis.activations import ActivationHooks, ablate, load_intervention, steer


def captured_hooks(conditional, unconditional=()):
    # Exercise capture finalization without a model or GPU forward pass.
    hooks = ActivationHooks.__new__(ActivationHooks)
    hooks.buffers = {"layer": {"cond": list(conditional), "uncond": list(unconditional)}}
    hooks.intervention = None
    hooks.stats = {"edited": 0, "change": 0.0}
    hooks.step = len(conditional) - 1
    hooks.outputs = {}
    return hooks


def test_large_captures_round_trip_without_fp16_overflow(tmp_path):
    vectors = np.array([[65536, -100000, 1.25], [131072, -200000, 2.5]], dtype=np.float32)
    hooks = captured_hooks(vectors, vectors / 2)
    with np.errstate(over="raise", invalid="raise"):
        result = hooks.end()
    capture = result["capture"]["layer"]
    assert capture["steps"].dtype == np.float32
    np.testing.assert_array_equal(capture["steps"], vectors)
    np.testing.assert_array_equal(capture["mean"], vectors.mean(axis=0))
    np.testing.assert_array_equal(result["uncond"]["layer"], vectors / 2)
    path = tmp_path / "activation.npz"
    np.savez(path, **{f"layer/{kind}": values for kind, values in capture.items()})
    with np.load(path) as saved:
        assert np.isfinite(saved["layer/steps"]).all()
        np.testing.assert_array_equal(saved["layer/steps"], vectors)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", ["conditional", "unconditional"])
def test_non_finite_captures_fail_validation(value, kind):
    good = np.array([[1, 2, 3]], dtype=np.float32)
    bad = np.array([[1, value, 3]], dtype=np.float32)
    hooks = captured_hooks(bad if kind == "conditional" else good, bad if kind == "unconditional" else ())
    with pytest.raises(RuntimeError, match=f"non-finite {kind} capture"):
        hooks.end()


def test_ablation_sets_the_projection_to_the_benign_mean_and_keeps_the_rest():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(8)
    d /= np.linalg.norm(d)
    mu, h = rng.standard_normal(8), rng.standard_normal((5, 8))

    edited = ablate(h, mu, d, 1.0)
    np.testing.assert_allclose(edited @ d, np.full(5, mu @ d))
    np.testing.assert_allclose(edited - (edited @ d)[:, None] * d, h - (h @ d)[:, None] * d)  # orthogonal part kept
    np.testing.assert_array_equal(ablate(h, mu, d, 0.0), h)
    # Half strength moves the projection half way.
    np.testing.assert_allclose(ablate(h, mu, d, 0.5) @ d, (h @ d + mu @ d) / 2)


def test_steering_moves_the_harmful_mean_onto_the_benign_mean():
    mu_benign, mu_harmful = np.array([1.0, 0, 0]), np.array([4.0, 4, 0])
    d = (mu_harmful - mu_benign) / np.linalg.norm(mu_harmful - mu_benign)
    np.testing.assert_allclose(steer(mu_harmful, d, np.linalg.norm(mu_harmful - mu_benign), 1.0), mu_benign)
    np.testing.assert_array_equal(steer(mu_harmful, d, 5.0, 0.0), mu_harmful)


def e4_dir(tmp_path):
    layer = "transformer.transformer_blocks.3"
    arrays = {f"{layer}/mu_benign": np.zeros(4), f"{layer}/mu_harmful": np.array([3.0, 4.0, 0, 0]),
              f"{layer}/mean_diff": np.array([0.6, 0.8, 0, 0]), f"{layer}/probe": np.array([1.0, 0, 0, 0]),
              f"{layer}/random_0": np.array([0, 0, 1.0, 0])}
    np.savez(tmp_path / "directions.npz", **arrays)
    (tmp_path / "statistics.json").write_text(json.dumps({"layer": layer}))
    return layer


@pytest.mark.parametrize(("vector", "scale"), [("safety", 5.0), ("probe", 3.0), ("random:0", 5.0)])
def test_intervention_reads_the_chosen_layer_and_scales_steering(tmp_path, vector, scale):
    layer = e4_dir(tmp_path)
    settings = {"e4_dir": str(tmp_path), "layer": None, "vector": vector, "mode": "steering", "alpha": 1,
                "passes": "all", "steps": [0, 1]}
    loaded = load_intervention(settings)
    assert loaded["layer"] == layer
    # safety: |delta|; probe: the harmful mean's projection; random: the same norm as safety.
    assert loaded["scale"] == pytest.approx(scale)


def test_an_intervention_cannot_use_a_layer_without_a_direction(tmp_path):
    e4_dir(tmp_path)
    settings = {"e4_dir": str(tmp_path), "layer": "unet.mid_block.attentions.0", "vector": "safety",
                "mode": "ablation", "alpha": 1, "passes": "all", "steps": [0, 1]}
    with pytest.raises(KeyError, match="layers"):
        load_intervention(settings)
