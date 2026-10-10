from visual_synthesis.pipeline import run_stage
from visual_synthesis.stages import s01_download


def test_hand_placed_dataset_is_verified(cfg):
    cfg["datasets"]["toy"]["files"] = ["toy.csv"]
    result = run_stage(cfg, s01_download)
    assert result["status"] == "success"


def test_dataset_is_downloaded_from_url(cfg, tmp_path):
    source = tmp_path / "remote.csv"
    source.write_text("prompt,category\na cat,nude\n")
    cfg["datasets"]["toy"].update(url=source.as_uri(), dir="from_url")

    result = run_stage(cfg, s01_download)

    assert result["downloaded"] == ["toy"]
    assert (tmp_path / "raw" / "from_url" / "remote.csv").read_text() == source.read_text()
