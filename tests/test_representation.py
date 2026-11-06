import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from visual_synthesis import representation  # noqa: E402

LAYERS = ["blocks.0", "blocks.1"]


def capture_run(tmp_path, n=40, dim=12, steps=3, signal=(0.0, 3.0)):
    """A fake capture run: harmful prompts are shifted along axis 0, by ``signal[i]`` in layer i."""
    rng = np.random.default_rng(0)
    rows = []
    for split in ("discovery", "validation"):
        for group in ("benign", "harmful", "near_boundary"):
            for i in range(n):
                arrays = {}
                for layer, shift in zip(LAYERS, signal, strict=True):
                    vectors = rng.standard_normal((steps, dim)).astype(np.float32)
                    vectors[:, 0] += shift * (group == "harmful") + 0.5 * shift * (group == "near_boundary")
                    arrays[f"{layer}/steps"], arrays[f"{layer}/mean"] = vectors.astype(np.float16), vectors.mean(0)
                path = tmp_path / f"{split}_{group}_{i}.npz"
                np.savez(path, **arrays)
                rows.append({"prompt_id": path.stem, "subset_group": group, "subset_split": split,
                             "generation_status": "success", "activation_path": str(path)})
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "generations.parquet")
    return frame


def test_the_separating_layer_is_chosen_and_its_direction_recovered(tmp_path):
    table, arrays, projections, steps, statistics = representation.analyze(capture_run(tmp_path), LAYERS)

    assert statistics["layer"] == "blocks.1"
    by_layer = table.set_index("layer")
    assert by_layer.loc["blocks.0", "cv_auc"] < 0.75 < 0.95 < by_layer.loc["blocks.1", "cv_auc"]
    assert by_layer.loc["blocks.1", "split_half_cosine"] > 0.8
    assert abs(arrays["blocks.1/mean_diff"][0]) > 0.9  # the planted axis
    for seed in representation.RANDOM_SEEDS:
        assert np.linalg.norm(arrays[f"blocks.1/random_{seed}"]) == pytest.approx(1.0, abs=1e-6)
    # Validation projections in units of |delta|: benign ~0, near-boundary ~0.5, harmful ~1.
    groups = projections[projections["layer"] == "blocks.1"].set_index("group")["mean"]
    assert groups["benign"] < groups["near_boundary"] < groups["harmful"]
    assert set(steps["step"]) == {0, 1, 2}
    assert statistics["n"]["discovery"] == {"benign": 40, "harmful": 40}


def test_ties_go_to_the_earliest_layer():
    table = pd.DataFrame({"layer": ["a", "b", "c"], "cv_auc": [0.990, 0.995, 0.97]})
    assert representation.choose_layer(table) == "a"
    assert representation.choose_layer(table, tie=0.001) == "b"


def test_the_command_writes_the_artifacts_that_generate_reads(tmp_path):
    (tmp_path / "run").mkdir()
    capture_run(tmp_path / "run", n=15)
    output = tmp_path / "e4"
    assert representation.main(["--capture-run", str(tmp_path / "run"), "--output", str(output)]) == 0
    statistics = json.loads((output / "statistics.json").read_text())
    with np.load(output / "directions.npz") as arrays:
        assert f"{statistics['layer']}/mean_diff" in arrays.files
    assert {"layer_auc.csv", "group_projection.csv", "step_auc.csv"} <= {p.name for p in output.iterdir()}
