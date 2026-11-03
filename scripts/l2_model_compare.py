#!/usr/bin/env python3
"""Run the frozen, paired text-generator comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_synthesis.l2_model_comparison import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
