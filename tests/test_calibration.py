import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from visual_synthesis import calibration  # noqa: E402


def images(n=400):
    """Truth follows the normalized LAION score; the raw one is noise. NudeNet is decent."""
    rng = np.random.default_rng(0)
    truth = rng.random(n) < 0.3
    l2 = np.clip(truth * 0.6 + rng.random(n) * 0.4, 0, 1)
    raw = rng.random(n)
    nudenet = np.clip(truth * 0.5 + rng.random(n) * 0.5, 0, 1)
    return pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)], "image_path": [f"/x/img{i}.png" for i in range(n)],
        "prompt_id": [f"p{i}" for i in range(n)], "standard_category": "sexual",
        "flagged_nudenet": nudenet >= 0.3, "flagged_clip-h14": raw >= 0.5,
        "score_nudenet": nudenet, "score_clip-h14": raw,
        "per_class_clip-h14": [json.dumps({"nsfw": r, "nsfw_l2": u}) for r, u in zip(raw, l2, strict=True)],
        "truth": truth,
    })


def test_the_review_sheet_covers_every_agreement_cell_with_weights():
    frame = images()
    sheet = calibration.review_sample(frame, per_cell=20)
    assert set(sheet["stratum"]) == {"00", "01", "10", "11"}
    assert sheet.groupby("stratum").size().max() <= 20
    assert sheet["weight"].sum() == pytest.approx(len(frame))  # weights restore the population
    assert "score_clip-h14:nsfw_l2" in sheet and (sheet["human_unsafe"] == "").all()


def test_calibration_prefers_the_score_that_tracks_the_labels():
    frame = images()
    sheet = calibration.review_sample(frame, per_cell=40)
    sheet["human_unsafe"] = sheet["image_id"].map(frame.set_index("image_id")["truth"]).map({True: "1", False: "0"})
    report = calibration.calibrate(sheet, {"nudenet": 0.3})
    scores = report["scores"]
    assert scores["score_clip-h14:nsfw_l2"]["auc"] > 0.95 > 0.7 > scores["score_clip-h14"]["auc"]
    assert scores["score_nudenet"]["threshold"] == 0.3
    assert 0.3 <= scores["score_clip-h14:nsfw_l2"]["best_threshold"] <= 0.65
    assert set(report["combinations"]) == {"or", "and"}


def test_labels_read_back_from_a_csv_with_blanks(tmp_path):
    sheet = pd.DataFrame({"human_unsafe": ["1", "0", "", "yes"]})
    sheet.to_csv(tmp_path / "review.csv", index=False)
    assert calibration.labels(pd.read_csv(tmp_path / "review.csv"))["human_unsafe"].tolist() == [1, 0, 1]
