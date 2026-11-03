#!/usr/bin/env python3
"""Exercise generation plus semantic classification, or classify an existing run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_synthesis.l2_pair_review import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
