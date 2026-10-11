#!/usr/bin/env python3
"""Extract a pinned archive (APR, or apr-util with --name) and apply its ordered, versioned patches."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("archive", type=Path)
p.add_argument("sha256")
p.add_argument("version")
p.add_argument("destination", type=Path)
p.add_argument("--patch-tool", default="patch")
p.add_argument("--name", default="apr", help="archive top-level name: apr or apr-util")
p.add_argument("--patches", type=Path, default=None, help="patch directory; default patches/ beside cmake/")
a = p.parse_args()
with a.archive.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != a.sha256:
        p.error(f"{a.name} archive SHA256 mismatch")
a.destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=a.destination.parent) as temporary:
    with tarfile.open(a.archive) as archive:
        archive.extractall(temporary, filter="data")
    source = Path(temporary) / f"{a.name}-{a.version}"
    patch_dir = a.patches or Path(__file__).resolve().parent.parent / "patches"
    patches = sorted(patch_dir.glob("*.patch"))
    if not patches:
        p.error("missing allocator patch series")
    for patch in patches:
        subprocess.run(
            [a.patch_tool, "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
            cwd=source,
            check=True,
        )
    if a.destination.exists():
        shutil.rmtree(a.destination)
    source.rename(a.destination)
    (a.destination / "prepared.stamp").touch()
