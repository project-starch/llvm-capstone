#!/usr/bin/env python3
"""Run a CTest replay mode with retained logs and an exact event comparison."""

import argparse
import pathlib
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recording", type=pathlib.Path)
parser.add_argument("results", type=pathlib.Path)
parser.add_argument("mode", choices=("spatial", "backing", "sublet"))
args = parser.parse_args()
port = pathlib.Path(__file__).resolve().parents[2]
for name in ("commands.bin", "recorded.bin"):
    if not (args.recording / name).is_file():
        raise SystemExit(
            "Run the native build and CTest first, or set FFPOOL_RECORDING_DIR."
        )
args.results.mkdir(parents=True, exist_ok=True)
run = pathlib.Path(tempfile.mkdtemp(prefix=f"qemu-{args.mode}-", dir=args.results))
print(f"QEMU test artifacts: {run}", flush=True)
subprocess.run(
    [
        sys.executable,
        port / "host/run-qemu.py",
        args.recording / "commands.bin",
        run,
        "--protection",
        args.mode,
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        port / "host/memory/analyze.py",
        "compare",
        args.recording / "recorded.bin",
        run / "capstone.bin",
    ],
    check=True,
)
