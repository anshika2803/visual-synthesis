import sys
from types import SimpleNamespace

from visual_synthesis.download import download_hf
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


def test_hf_download_repairs_a_populated_partial_cache(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        (tmp_path / "model.safetensors").write_bytes(b"checkpoint")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    status, path, _ = download_hf("example/model", tmp_path)
    assert status == "downloaded" and path == tmp_path
    assert len(calls) == 1 and calls[0]["force_download"] is False
    assert (tmp_path / "model.safetensors").is_file()
