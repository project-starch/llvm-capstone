#!/usr/bin/env python3
"""Extract the pinned allocator source and apply the ordered, versioned patches."""

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
# ported: the patches that make pymalloc run as a standalone, capability-correct
# allocator (0001, 0002). protected: also 0003, the Sublet protection with the
# two-operand CREVOKE of the capstone-sublet bitstream. protected-virtual: also
# 0004, which ends a block with the one-operand CREVOKE of the virtual profile.
# reference: 0001 only, the native comparison build.
p.add_argument("--variant", choices=("ported", "protected", "protected-virtual", "reference"), default="ported")
a = p.parse_args()
with a.archive.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != a.sha256:
        p.error("CPython archive SHA256 mismatch")
a.destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=a.destination.parent) as temporary:
    with tarfile.open(a.archive) as archive:
        archive.extractall(temporary, filter="data")
    source = Path(temporary) / f"Python-{a.version}"
    patches = sorted(
        (Path(__file__).resolve().parent.parent / "patches").glob("*.patch")
    )
    if not patches:
        p.error("missing allocator patch series")
    for patch in patches:
        if a.variant == "reference" and "-0001-" not in patch.name:
            continue
        if a.variant == "ported" and ("-0003-" in patch.name or "-0004-" in patch.name):
            continue
        if a.variant == "protected" and "-0004-" in patch.name:
            continue
        subprocess.run(
            [a.patch_tool, "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
            cwd=source,
            check=True,
        )
    if a.destination.exists():
        shutil.rmtree(a.destination)
    source.rename(a.destination)
    (a.destination / "prepared.stamp").touch()
