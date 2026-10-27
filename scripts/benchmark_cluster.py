#!/usr/bin/env python3
"""Run the cluster input, dependency, GPU and cache checks from a checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_synthesis.benchmark_cluster import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
