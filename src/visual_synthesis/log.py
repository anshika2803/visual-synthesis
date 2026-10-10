"""Logging: readable lines on the console, JSON lines on disk.

Every module logs through the standard library::

    log = logging.getLogger(__name__)
    log.info("converted %s: %d rows", name, rows)

Each run gets its own folder, ``logs/<run_id>/``:

    run.jsonl      every record of every stage in the run, as JSON
    <stage>.log    the same records for one stage, human-readable
    errors.log     warnings and errors only

Every JSON record carries timestamp, level, run_id, stage, module, function
and line number, so a log line can always be traced to its source.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(stage)s] %(message)s  (%(filename)s:%(lineno)d)"

# Set by setup_logging and stamped onto every record.
context = {"run_id": "-", "stage": "-"}


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.run_id = context["run_id"]
        record.stage = context["stage"]
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            "run_id": record.run_id,
            "stage": record.stage,
            "module": record.name,
            "filename": record.filename,
            "func_name": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level="info", log_dir=None, run_id="-", stage="-", console_json=False):
    """(Re)configure logging. With ``log_dir`` set, also write the per-run files."""
    context.update(run_id=run_id, stage=stage)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.DEBUG)

    def add(handler, formatter, handler_level):
        handler.setFormatter(formatter)
        handler.setLevel(handler_level)
        handler.addFilter(ContextFilter())
        root.addHandler(handler)

    text = logging.Formatter(TEXT_FORMAT, datefmt="%H:%M:%S")
    console_level = getattr(logging, str(level).upper(), logging.INFO)
    add(logging.StreamHandler(sys.stderr), JsonFormatter() if console_json else text, console_level)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        add(logging.FileHandler(log_dir / "run.jsonl", encoding="utf-8"), JsonFormatter(), logging.DEBUG)
        add(logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8"), text, logging.DEBUG)
        add(logging.FileHandler(log_dir / "errors.log", encoding="utf-8"), text, logging.WARNING)

    # Third-party libraries are chatty at DEBUG; keep their warnings only.
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "filelock", "fsspec", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
