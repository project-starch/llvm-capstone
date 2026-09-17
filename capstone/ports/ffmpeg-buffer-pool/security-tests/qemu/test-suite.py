#!/usr/bin/env python3
"""Give each CTest security run its own retained result directory."""

import argparse
import pathlib
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=pathlib.Path)
args = parser.parse_args()
args.results.mkdir(parents=True, exist_ok=True)
run = pathlib.Path(tempfile.mkdtemp(prefix="security-", dir=args.results))
print(f"Security test artifacts: {run}", flush=True)
subprocess.run(["bash", pathlib.Path(__file__).with_name("run.sh"), run], check=True)
