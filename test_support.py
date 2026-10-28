"""Command-line smoke test helper required by the lab structure.

Run from this folder with: python test_support.py
For the full suite run: python -m pytest -q tests
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_smoke_suite() -> int:
    root = Path(__file__).resolve().parent
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=root, text=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(run_smoke_suite())
