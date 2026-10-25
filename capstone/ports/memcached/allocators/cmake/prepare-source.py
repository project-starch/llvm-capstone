#!/usr/bin/env python3
"""Extract the pinned memcached archive and apply the ordered, versioned patches."""

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
# ported: the shims (0001) and the hooks into the port's ledger (0002), for the native,
# CheriBSD and PoisonCap arms. protected: the shims and the application's Sublet patch
# (../app/patches/*-0006-*), the two lifetime instructions in slabs.c and cache.c, with
# no hooks and no ledger: the Sublet arm on the capstone-application target.
p.add_argument("--variant", choices=("ported", "protected"), default="ported")
a = p.parse_args()
with a.archive.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != a.sha256:
        p.error("memcached archive SHA256 mismatch")
a.destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=a.destination.parent) as temporary:
    with tarfile.open(a.archive) as archive:
        archive.extractall(temporary, filter="data")
    source = Path(temporary) / f"memcached-{a.version}"
    port = Path(__file__).resolve().parent.parent
    patches = sorted((port / "patches").glob("*.patch"))
    if not patches:
        p.error("missing allocator patch series")
    if a.variant == "protected":
        sublet = sorted((port.parent / "app" / "patches").glob("*-0006-*.patch"))
        if len(sublet) != 1:
            p.error("expected exactly one application patch 0006, found %d" % len(sublet))
        patches = [patch for patch in patches if "-0001-" in patch.name] + sublet
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
