"""Progress reporting for the long stages (generate, nsfw).

On a terminal it is a tqdm bar. Anywhere else (Slurm, CI, a redirected log)
tqdm switches itself off and a log line is written every ``log_every`` items
instead, so batch logs stay readable.

    with Progress(len(pending), "generate", "img", log_every=10) as progress:
        ...
        progress.advance(2, failed=1)       # safe to call from worker threads
"""

import logging
import secrets
import string
import time
from contextlib import ExitStack
from threading import Lock

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger(__name__)


class Progress:
    def __init__(self, total, description, unit="item", log_every=0, model_id=None):
        self.total = total
        self.description = description
        self.unit = unit
        self.log_every = log_every
        self.model_id = model_id
        # Auto-generate a 4-letter unique tag; user cannot supply it.
        self.tag = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(4))
        self.done = 0
        self.fields = {}
        self._logged = 0
        self._started = time.perf_counter()
        self._lock = Lock()
        self._stack = ExitStack()
        self.bar = None

    def __enter__(self):
        # Route log records through tqdm so they print above the bar instead of through it.
        self._stack.enter_context(logging_redirect_tqdm())
        desc = self.description
        parts = []
        if self.model_id:
            parts.append(self.model_id)
        if self.tag:
            parts.append(self.tag)
        if parts:
            desc = f"{self.description} [{', '.join(parts)}]"
        self.bar = self._stack.enter_context(
            tqdm(total=self.total, desc=desc, unit=self.unit, disable=None, dynamic_ncols=True)
        )
        return self

    def __exit__(self, *exc_info):
        return self._stack.__exit__(*exc_info)

    def advance(self, count=1, **fields):
        """Record ``count`` finished items. Numeric ``fields`` (e.g. failed=1) are added to the bar's totals."""
        with self._lock:
            self.done += count
            for key, value in fields.items():
                self.fields[key] = self.fields.get(key, 0) + value
            self.bar.update(count)
            shown = {key: value for key, value in self.fields.items() if value}
            if shown:
                self.bar.set_postfix(shown, refresh=False)
            # Off a terminal the bar is disabled; fall back to periodic log lines.
            if self.bar.disable and self.log_every and self.done - self._logged >= self.log_every:
                self._logged = self.done
                rate = self.done / max(time.perf_counter() - self._started, 1e-9)
                extra = "".join(f", {key}={value}" for key, value in shown.items())
                suffix = ""
                parts = []
                if self.model_id:
                    parts.append(self.model_id)
                if self.tag:
                    parts.append(self.tag)
                if parts:
                    suffix = f" [{', '.join(parts)}]"
                log.info("%s %d/%d%s (%.2f/s%s)", self.description, self.done, self.total, suffix, rate, extra)
