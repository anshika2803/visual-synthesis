"""Shared test setup: a tiny config that needs no GPU, no weights and no network.

Everything is written under pytest's tmp_path; the image model and the detector
are the fake backends.
"""

import json

import pandas as pd
import pytest

from visual_synthesis.config import load_config

TEST_CONFIG = {
    "datasets": {
        "toy": {
            "label": "Toy",
            "input_glob": "*.csv",
            "category_column": "category",
            "id_template": "TOY_{index:03d}",
            "category_map": {"nude": "sexual", "gore": "violence"},
        }
    },
    "models": {"fake": {"backend": "fake", "width": 64, "height": 64}},
    "detectors": {"fake": {"backend": "fake"}},
    "download": {"datasets": ["toy"]},
    "generate": {"model": "fake"},
    "nsfw": {"detector": "fake"},
}

# 4 prompts; the last one duplicates the first (case and spacing differ).
TOY_ROWS = [
    {"prompt": "a red house", "category": "nude"},
    {"prompt": "a blue car", "category": "gore"},
    {"prompt": "a green tree", "category": "nude, gore"},
    {"prompt": "A  Red house", "category": "gore"},
]


@pytest.fixture
def cfg(tmp_path):
    """A resolved config whose paths all live under tmp_path, with the toy dataset in place."""
    config = dict(TEST_CONFIG, paths={name: str(tmp_path / name) for name in ("raw", "runs", "models", "logs")})
    config_file = tmp_path / "pipeline.json"
    config_file.write_text(json.dumps(config))
    (tmp_path / "raw" / "toy").mkdir(parents=True)
    pd.DataFrame(TOY_ROWS).to_csv(tmp_path / "raw" / "toy" / "toy.csv", index=False)
    return load_config([config_file], use_env=False)
