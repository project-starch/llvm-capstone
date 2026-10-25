#!/usr/bin/env python3
"""Build a delegated port with the application-memory study observers."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.argv.append("--instrument")
    # The study's observers exist on the physical profile only (level0's
    # statistics); build.py refuses them on virtual, its default.
    if not any(a == "--profile" or a.startswith("--profile=") for a in sys.argv[1:]):
        sys.argv += ["--profile", "physical"]
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                      "ports/common/application/build.py"), run_name="__main__")
