"""Fetching datasets, model weights and detector weights.

Everything the pipeline needs from the network lives here. Downloads are
idempotent: something already on disk is skipped unless ``force`` is set.

Every function returns one result dict::

    {"kind": "dataset", "id": "i2p", "status": "downloaded", "path": ..., "detail": ...}

status is one of: downloaded, skipped (already there), present (hand-placed
files verified), missing (hand-placed files not there), failed.
"""

import logging
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from visual_synthesis.config import dataset_dir
from visual_synthesis.io import file_sha256, safe_name

log = logging.getLogger(__name__)

USER_AGENT = "visual-synthesis"
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
FAILED_STATUSES = ("failed", "missing")


def result(kind, item_id, status, path, detail=""):
    level = logging.ERROR if status in FAILED_STATUSES else logging.INFO
    log.log(level, "%s %s: %s (%s) %s", kind, item_id, status, path, detail)
    return {"kind": kind, "id": item_id, "status": status, "path": str(path), "detail": detail}


def is_populated(directory):
    directory = Path(directory)
    return directory.is_dir() and any(directory.iterdir())


# ------------------------------------------------------------------ sources


def download_url(url, target_dir, sha256=None, force=False, timeout=600):
    """Download one URL into ``target_dir``. Returns ``(status, path, detail)``.

    The file is written to ``<name>.part`` first and only renamed once complete
    (and once its checksum matches, when ``sha256`` is given). Archives are
    unpacked next to the downloaded file.
    """
    filename = Path(urllib.parse.urlparse(url).path).name or "download.bin"
    destination = Path(target_dir) / filename
    if destination.exists() and not force:
        return "skipped", destination, "already downloaded"

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    log.info("downloading %s", url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        return "failed", destination, f"{url}: {exc}"

    if sha256 and file_sha256(partial) != sha256:
        partial.unlink()
        return "failed", destination, f"sha256 mismatch for {url}"
    partial.replace(destination)

    if destination.name.endswith(ARCHIVE_SUFFIXES):
        is_tar = not destination.name.endswith(".zip")
        shutil.unpack_archive(destination, destination.parent, **({"filter": "data"} if is_tar else {}))
        log.info("unpacked %s", destination)
    return "downloaded", destination, url


def download_hf(repo_id, target_dir, repo_type="model", revision=None, allow_patterns=None, force=False):
    """Download a HuggingFace repo snapshot. Returns ``(status, path, detail)``."""
    target_dir = Path(target_dir)
    if is_populated(target_dir) and not force:
        return "skipped", target_dir, "already downloaded"

    from huggingface_hub import snapshot_download

    log.info("downloading %s %s from HuggingFace", repo_type, repo_id)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(target_dir),
            allow_patterns=allow_patterns or None,
            force_download=force,
        )
    except Exception as exc:  # network and auth errors come in many types
        return "failed", target_dir, f"{repo_id}: {exc}"
    return "downloaded", target_dir, repo_id


# ----------------------------------------------------------------- per kind


def download_dataset(cfg, dataset_id):
    """Put a dataset's raw files in ``<paths.raw>/<dataset dir>/``.

    Uses the dataset's ``url`` (one or several) or ``hf_repo``. With neither,
    the files are expected to be placed by hand, and are only checked.
    """
    dataset = cfg["datasets"][dataset_id]
    target = dataset_dir(cfg, dataset_id)
    force = cfg["download"]["force"]
    if not dataset["enabled"]:
        return result("dataset", dataset_id, "skipped", target, "enabled=false")

    expected = [target / name for name in dataset["files"]]
    have_all = bool(expected) and all(path.exists() for path in expected)
    if have_all and not force:
        return result("dataset", dataset_id, "skipped", target, "files already present")

    if dataset["url"]:
        urls = [dataset["url"]] if isinstance(dataset["url"], str) else list(dataset["url"])
        outcomes = [
            download_url(url, target, dataset["sha256"], force, cfg["download"]["timeout_seconds"])
            for url in urls
        ]
        failures = [detail for status, _, detail in outcomes if status == "failed"]
        if failures:
            return result("dataset", dataset_id, "failed", target, "; ".join(failures))
        downloaded = any(status == "downloaded" for status, _, _ in outcomes)
        return result("dataset", dataset_id, "downloaded" if downloaded else "skipped", target, ", ".join(urls))

    if dataset["hf_repo"]:
        status, path, detail = download_hf(
            dataset["hf_repo"], target, "dataset", dataset["hf_revision"], dataset["files"], force
        )
        return result("dataset", dataset_id, status, path, detail)

    # No source configured: the files are placed by hand.
    missing = [path.name for path in expected if not path.exists()]
    if not target.is_dir() or missing or not is_populated(target):
        return result("dataset", dataset_id, "missing", target, f"place {missing or 'the files'} in {target}")
    return result("dataset", dataset_id, "present", target, "hand-placed files verified")


def download_model(cfg, model_id):
    """Fetch a model's weights into ``<paths.models>/<local_dir or id>/``."""
    model = cfg["models"][model_id]
    target = Path(cfg["paths"]["models"]) / safe_name(model["local_dir"] or model_id)
    if model["backend"] != "diffusers":
        return result("model", model_id, "skipped", target, f"backend={model['backend']} needs no weights")
    status, path, detail = download_hf(
        model["repo_id"], target, "model", model["revision"], model["allow_patterns"], cfg["download"]["force"]
    )
    return result("model", model_id, status, path, detail)


def download_detector(cfg, detector_id):
    """Fetch a detector's weights, when its backend needs any."""
    detector = cfg["detectors"][detector_id]
    target = Path(cfg["paths"]["models"]) / safe_name(detector["local_dir"] or detector_id)
    if not detector.get("repo_id") and not detector.get("url"):
        # NudeNet, for example, ships its weights inside the Python package.
        return result("detector", detector_id, "skipped", target, "weights ship with the package")

    outcomes = []
    force = cfg["download"]["force"]
    if detector.get("url"):
        urls = [detector["url"]] if isinstance(detector["url"], str) else list(detector["url"])
        for u in urls:
            status, path, detail = download_url(
                u, target, force=force, timeout=cfg["download"].get("timeout_seconds", 600)
            )
            outcomes.append((status, path, detail))

    if detector.get("repo_id"):
        status, path, detail = download_hf(
            detector["repo_id"], target, allow_patterns=detector.get("allow_patterns"), force=force
        )
        outcomes.append((status, path, detail))

    failures = [detail for status, _, detail in outcomes if status == "failed"]
    if failures:
        return result("detector", detector_id, "failed", target, "; ".join(failures))
    downloaded = any(status == "downloaded" for status, _, _ in outcomes)
    detail = ", ".join(str(d) for _, _, d in outcomes)
    return result("detector", detector_id, "downloaded" if downloaded else "skipped", target, detail)
