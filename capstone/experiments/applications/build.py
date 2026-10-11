#!/usr/bin/env python3
"""Build a delegated port with the application-memory study observers."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.argv.append("--instrument")
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                      "ports/common/application/build.py"), run_name="__main__")
