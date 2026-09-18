#!/usr/bin/env python3
"""Prepare a build-owned FFmpeg source tree from the verified release archive."""

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=pathlib.Path)
parser.add_argument("sha256")
parser.add_argument("version")
parser.add_argument("destination", type=pathlib.Path)
parser.add_argument("variant", choices=("stock", "traced", "ported"))
args = parser.parse_args()
port = pathlib.Path(__file__).resolve().parent.parent

# Check again at build time, including when an offline cache was replaced
# after CMake configuration. Never patch a caller's original source checkout.
with args.archive.open("rb") as archive:
    if hashlib.file_digest(archive, "sha256").hexdigest() != args.sha256:
        raise SystemExit(f"FFmpeg archive SHA256 mismatch: {args.archive}")

args.destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=args.destination.parent) as temporary:
    with tarfile.open(args.archive) as archive:
        archive.extractall(temporary, filter="data")
    source = pathlib.Path(temporary) / f"ffmpeg-{args.version}"
    if args.variant != "stock":
        subprocess.run(
            [sys.executable, port / "host/instrument-pools.py", source],
            check=True,
        )
    if args.variant == "ported":
        subprocess.run(
            [sys.executable, port / "patches/apply-pool-port.py", source], check=True
        )
    # Failed extraction or patch application leaves the last completed tree intact.
    if args.destination.exists():
        shutil.rmtree(args.destination)
    source.rename(args.destination)
    (args.destination / "prepared.stamp").touch()
