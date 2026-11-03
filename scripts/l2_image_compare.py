#!/usr/bin/env python3
"""Compare original, current and revised paraphrases through real image generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_synthesis.l2_image_comparison import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
