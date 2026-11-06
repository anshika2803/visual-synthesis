"""Each model's hook adapter on a tiny, randomly initialised copy of its real diffusers denoiser.

No weights are downloaded: the block classes, output tuples, CFG batch order and
joint text+image sequences are the real ones, only smaller. For every adapter:

    capture       the captured vector is the mean image-token vector of the conditional pass
    alpha = 0     the denoiser output is bit-identical to the unhooked one, and stays so after remove()
    cond only     editing the conditional pass leaves the unconditional pass and the text tokens untouched
    misconfigured a hook pointed at the text stream raises instead of editing it
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from visual_synthesis.activations import ActivationHooks  # noqa: E402

TEXT = 6  # text tokens
SIZE = 64  # pixels: 8 x 8 latents, so 16 image tokens with 16 pixels per token


def unet():
    return diffusers.UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=2, sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32, attention_head_dim=(2, 4), use_linear_projection=True, addition_embed_type="text_time",
        addition_time_embed_dim=8, transformer_layers_per_block=(1, 2), projection_class_embeddings_input_dim=80,
        norm_num_groups=8)


def unet_call(batch):
    return (torch.randn(batch, 4, 8, 8), torch.tensor(10)), {
        "encoder_hidden_states": torch.randn(batch, TEXT, 32), "return_dict": False,
        "added_cond_kwargs": {"text_embeds": torch.randn(batch, 32), "time_ids": torch.randn(batch, 6)}}


def flux_ids():
    ids = torch.zeros(16, 3)
    ids[:, 1], ids[:, 2] = torch.arange(16) // 4, torch.arange(16) % 4
    return ids


CASES = {
    "sd-xl": {
        "order": "uncond_first", "root": "unet", "build": unet, "call": unet_call,
        "hook": {"layers": ["unet.down_blocks.1.attentions.1", "unet.mid_block.attentions.0",
                            "unet.up_blocks.0.attentions.2"], "output_index": 0, "cfg": "batch_uncond_first"},
        "expected": lambda out: out[0][1].mean(dim=(1, 2)),
    },
    "sd-3.5": {
        "order": "uncond_first", "root": "transformer",
        "build": lambda: diffusers.SD3Transformer2DModel(
            sample_size=8, patch_size=2, in_channels=4, num_layers=2, attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=16, caption_projection_dim=16, pooled_projection_dim=8, out_channels=4,
            pos_embed_max_size=8),
        "call": lambda batch: ((), {"hidden_states": torch.randn(batch, 4, 8, 8),
                                    "encoder_hidden_states": torch.randn(batch, TEXT, 16),
                                    "pooled_projections": torch.randn(batch, 8),
                                    "timestep": torch.full((batch,), 10), "return_dict": False}),
        "hook": {"layers": ["transformer.transformer_blocks.0", "transformer.transformer_blocks.1"], "output_index": 1,
                 "cfg": "batch_uncond_first", "pixels_per_token": 16},
        "expected": lambda out: out[1][1].mean(dim=0), "wrong": {"output_index": 0},
    },
    "pixart-sigma": {
        "order": "uncond_first", "root": "transformer",
        "build": lambda: diffusers.PixArtTransformer2DModel(
            num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8, num_layers=2,
            cross_attention_dim=16, sample_size=8, patch_size=2, caption_channels=12, norm_num_groups=8,
            use_additional_conditions=False),
        "call": lambda batch: ((torch.randn(batch, 4, 8, 8),), {
            "encoder_hidden_states": torch.randn(batch, TEXT, 12), "timestep": torch.full((batch,), 10),
            "added_cond_kwargs": {"resolution": None, "aspect_ratio": None}, "return_dict": False}),
        "hook": {"layers": ["transformer.transformer_blocks.0", "transformer.transformer_blocks.1"],
                 "cfg": "batch_uncond_first", "pixels_per_token": 16},
        "expected": lambda out: out[1].mean(dim=0), "wrong": {"pixels_per_token": 8},
    },
    "flux-dev": {
        "order": "sequential", "root": "transformer",
        "build": lambda: diffusers.FluxTransformer2DModel(
            patch_size=1, in_channels=16, num_layers=1, num_single_layers=2, attention_head_dim=8,
            num_attention_heads=2, joint_attention_dim=16, pooled_projection_dim=8, axes_dims_rope=(2, 2, 4)),
        "call": lambda batch: ((), {"hidden_states": torch.randn(batch, 16, 16),
                                    "encoder_hidden_states": torch.randn(batch, TEXT, 16),
                                    "pooled_projections": torch.randn(batch, 8), "timestep": torch.tensor([0.5]),
                                    "img_ids": flux_ids(), "txt_ids": torch.zeros(TEXT, 3), "return_dict": False}),
        "hook": {"layers": ["transformer.transformer_blocks.0", "transformer.single_transformer_blocks.1"],
                 "output_index": 1, "cfg": "sequential", "pixels_per_token": 16},
        "expected": lambda out: out[1][0].mean(dim=0), "wrong": {"output_index": 0},
    },
    "qwen-image": {
        "order": "sequential", "root": "transformer",
        "build": lambda: diffusers.QwenImageTransformer2DModel(
            patch_size=2, in_channels=16, out_channels=4, num_layers=2, attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=16, axes_dims_rope=(2, 2, 4)),
        "call": lambda batch: ((), {"hidden_states": torch.randn(batch, 16, 16),
                                    "encoder_hidden_states": torch.randn(batch, TEXT, 16),
                                    "encoder_hidden_states_mask": torch.ones(batch, TEXT),
                                    "timestep": torch.tensor([0.5]), "img_shapes": [[(1, 4, 4)]] * batch,
                                    "return_dict": False}),
        "hook": {"layers": ["transformer.transformer_blocks.0", "transformer.transformer_blocks.1"],
                 "output_index": 1, "cfg": "sequential", "pixels_per_token": 16},
        "expected": lambda out: out[1][0].mean(dim=0), "wrong": {"output_index": 0},
    },
    "z-image-turbo": {
        "order": "cond_first", "root": "transformer",
        "build": lambda: diffusers.ZImageTransformer2DModel(
            in_channels=4, dim=32, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=16,
            axes_dims=[4, 6, 6], axes_lens=[64, 32, 32]),
        "call": lambda batch: (([torch.randn(4, 1, 8, 8) for _ in range(batch)], torch.full((batch,), 0.5),
                                [torch.randn(TEXT, 16) for _ in range(batch)]), {"return_dict": False}),
        # [image (16, padded to 32), caption (padded to 32)]: only the first 16 positions are the image.
        "hook": {"layers": ["transformer.layers.0", "transformer.layers.1"], "cfg": "batch_cond_first",
                 "pixels_per_token": 16, "joint": "image_first"},
        "expected": lambda out: out[0, :16].mean(dim=0), "text": lambda out: out[:, 16:],
        "wrong": {"joint": None},
    },
    "glm-image": {
        "order": "sequential", "root": "transformer",
        "build": lambda: diffusers.GlmImageTransformer2DModel(
            patch_size=2, in_channels=4, out_channels=4, num_layers=2, attention_head_dim=8, num_attention_heads=2,
            text_embed_dim=12, time_embed_dim=16, condition_dim=8, prior_vq_quantizer_codebook_size=32),
        "call": lambda batch: ((), {"hidden_states": torch.randn(batch, 4, 8, 8),
                                    "encoder_hidden_states": torch.randn(batch, TEXT, 12),
                                    "prior_token_id": torch.randint(0, 32, (batch, 16)),
                                    "prior_token_drop": torch.zeros(batch, 16, dtype=torch.bool),
                                    "timestep": torch.tensor([10.0]), "target_size": torch.tensor([[64.0, 64.0]]),
                                    "crop_coords": torch.zeros(batch, 2), "return_dict": False}),
        "hook": {"layers": ["transformer.transformer_blocks.0", "transformer.transformer_blocks.1"],
                 "output_index": 0, "cfg": "sequential", "pixels_per_token": 16},
        "expected": lambda out: out[0][0].mean(dim=0), "wrong": {"output_index": 1},
    },
}
HOOK_DEFAULTS = {"output_index": None, "pixels_per_token": None, "joint": None}
# "order" is how the pipeline really lays out the passes; the hook config must agree with it.


def setup(name):
    """``(case, root, model, calls)``: one denoising step's calls, cond first, as the pipeline makes them."""
    case = CASES[name]
    torch.manual_seed(0)
    model = case["build"]().eval()
    if case["order"] == "sequential":
        calls = [case["call"](1), case["call"](1)]  # cond call, then uncond call, same timestep
    else:
        calls = [case["call"](2)]  # one call holding both passes
    return case, SimpleNamespace(**{case["root"]: model}), model, calls


def run(model, calls):
    """The denoiser outputs of one step, as one tensor per call."""
    with torch.inference_mode():
        outputs = [model(*args, **kwargs)[0] for args, kwargs in calls]
    return [torch.stack(out) if isinstance(out, list) else out for out in outputs]


def hooks_for(case, root, layers=None, intervention=None, **hook):
    hooks = ActivationHooks(root, {**HOOK_DEFAULTS, **case["hook"], **hook},
                            layers if layers is not None else case["hook"]["layers"], intervention)
    hooks.begin(SIZE, SIZE, 1)
    return hooks


def steering(layer, dim, alpha=1.0, passes="all"):
    direction = np.ones(dim, np.float32) / np.sqrt(dim)
    return {"layer": layer, "mode": "steering", "vector": "random:0", "alpha": alpha, "direction": direction,
            "mu": np.zeros(dim, np.float32), "scale": 5.0, "passes": passes, "steps": (0.0, 1.0)}


def passes(case, outputs):
    """``(cond, uncond)`` parts of a step's denoiser outputs, by the pipeline's real layout."""
    if case["order"] == "sequential":
        return outputs[0], outputs[1]
    first, second = outputs[0][0], outputs[0][1]
    return (first, second) if case["order"] == "cond_first" else (second, first)


@pytest.mark.parametrize("name", CASES)
def test_capture_is_the_conditional_image_stream(name):
    case, root, model, calls = setup(name)
    layer = case["hook"]["layers"][0]
    raw = []
    handle = model.get_submodule(layer.partition(".")[2]).register_forward_hook(
        lambda module, args, output: raw.append(output) and None)
    hooks = hooks_for(case, root)
    run(model, calls)
    result = hooks.end()
    handle.remove()
    hooks.remove()

    captured = result["capture"][layer]
    assert captured["steps"].shape[0] == 1 and result["stats"]["steps"] == 1
    np.testing.assert_allclose(captured["mean"], case["expected"](raw[0]).float().numpy(), rtol=1e-5, atol=1e-6)
    assert layer in result["uncond"]  # every case runs with CFG, so the uncond pass is seen too


@pytest.mark.parametrize("name", CASES)
def test_alpha_zero_and_removal_leave_the_output_unchanged(name):
    case, root, model, calls = setup(name)
    baseline = run(model, calls)
    layer = case["hook"]["layers"][-1]
    dim = len(hooks_capture(case, root, model, calls, layer))
    for alpha in (0.0, 1.0):
        hooks = hooks_for(case, root, [], steering(layer, dim, alpha))
        outputs = run(model, calls)
        assert hooks.end()["stats"]["edited_calls"] == len(calls)  # alpha = 0 still runs the edit path
        hooks.remove()
        assert all(torch.equal(a, b) for a, b in zip(baseline, outputs, strict=True)) == (alpha == 0)
    for before, after in zip(baseline, run(model, calls), strict=True):
        assert torch.equal(before, after)


def hooks_capture(case, root, model, calls, layer):
    hooks = hooks_for(case, root, [layer])
    run(model, calls)
    mean = hooks.end()["capture"][layer]["mean"]
    hooks.remove()
    return mean


@pytest.mark.parametrize("name", CASES)
def test_cond_only_edit_spares_the_uncond_pass_and_the_text_tokens(name):
    case, root, model, calls = setup(name)
    layer = case["hook"]["layers"][0]
    baseline = run(model, calls)
    before = []
    handle = model.get_submodule(layer.partition(".")[2]).register_forward_hook(
        lambda module, args, output: before.append(output) and None)
    run(model, calls)
    handle.remove()

    dim = len(hooks_capture(case, root, model, calls, layer))
    hooks = hooks_for(case, root, [], steering(layer, dim, passes="cond"))
    after = []  # registered after the edit hook, so it sees the edited output
    handle = model.get_submodule(layer.partition(".")[2]).register_forward_hook(
        lambda module, args, output: after.append(output) and None)
    edited = run(model, calls)
    handle.remove()
    hooks.remove()

    (cond_before, uncond_before), (cond_after, uncond_after) = passes(case, baseline), passes(case, edited)
    assert torch.equal(uncond_before, uncond_after)
    assert not torch.equal(cond_before, cond_after)
    index = case["hook"].get("output_index")
    if isinstance(before[0], tuple) and len(before[0]) > 1:  # dual stream: the text element is untouched
        other = 1 - index
        assert before[0][other] is None or torch.equal(before[0][other], after[0][other])
    if "text" in case:  # joint sequence: the text positions are untouched
        assert torch.equal(case["text"](before[0]), case["text"](after[0]))


@pytest.mark.parametrize("name", [name for name, case in CASES.items() if "wrong" in case])
def test_a_hook_on_the_wrong_stream_raises(name):
    case, root, model, calls = setup(name)
    hooks = hooks_for(case, root, **case["wrong"])
    with pytest.raises(RuntimeError, match="tokens"):
        run(model, calls)
    hooks.remove()


def test_batch_layouts_count_every_call_as_a_step():
    """SD 1.5's PNDM scheduler repeats a timestep (958, 925, 925, ...): two calls are two steps."""
    case, root, model, calls = setup("sd-xl")
    hooks = hooks_for(case, root)
    run(model, calls + calls)  # the same timestep twice
    result = hooks.end()
    hooks.remove()
    assert result["stats"]["steps"] == 2
    assert all(len(values["steps"]) == 2 for values in result["capture"].values())


def test_sequential_cfg_on_a_batched_pipeline_raises():
    case, root, model, _ = setup("qwen-image")
    hooks = hooks_for(case, root)
    with pytest.raises(RuntimeError, match="batches CFG"):
        run(model, [case["call"](2)])
    hooks.remove()
