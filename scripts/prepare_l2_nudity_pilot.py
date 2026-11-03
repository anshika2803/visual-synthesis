#!/usr/bin/env python3
"""Prepare source-linked nudity-boundary inputs for the normal project CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_synthesis.l2_nudity_pilot import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
