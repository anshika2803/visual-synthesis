"""Stage 1 -- download datasets, model weights and detector weights.

Which ones is decided by the ``download`` section of the config; the URLs and
repos come from each dataset/model/detector entry.
"""

import logging

from visual_synthesis.download import FAILED_STATUSES, download_dataset, download_detector, download_model

log = logging.getLogger(__name__)

NAME = "download"
SLUG = "s01_download"
HELP = "Download datasets, model weights and detector weights."
NEEDS_GPU = False


def run(cfg, run, source=None):
    wanted = cfg["download"]
    plan = (
        [("dataset", item, download_dataset) for item in wanted["datasets"]]
        + [("model", item, download_model) for item in wanted["models"]]
        + [("detector", item, download_detector) for item in wanted["detectors"]]
    )
    if not plan:
        log.info("nothing to download: list ids under 'download' in the config, or pass --dataset/--model")
        return {"rows_in": 0, "rows_out": 0}

    if cfg["run"]["dry_run"]:
        for kind, item, _ in plan:
            log.info("dry run: would download %s %s", kind, item)
        return {"rows_in": len(plan), "rows_out": 0, "planned": len(plan)}

    results = [fetch(cfg, item) for _, item, fetch in plan]
    failed = [r["id"] for r in results if r["status"] in FAILED_STATUSES]
    return {
        "rows_in": len(results),
        "rows_out": len(results) - len(failed),
        "downloaded": [r["id"] for r in results if r["status"] == "downloaded"],
        "failed": failed,
        "results": results,
    }
