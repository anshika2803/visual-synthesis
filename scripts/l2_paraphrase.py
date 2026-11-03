#!/usr/bin/env python3
"""Run the local L2 workflow without installing additional Python packages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_synthesis.l2_paraphrase import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
