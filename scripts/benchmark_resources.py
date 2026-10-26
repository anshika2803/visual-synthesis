#!/usr/bin/env python3
"""Convenience entry point: python scripts/benchmark_resources.py --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_synthesis.cli import cli  # noqa: E402

if __name__ == "__main__":
    cli(["benchmark", *sys.argv[1:]])
