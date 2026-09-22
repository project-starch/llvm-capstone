"""Verify the pinned release, extract only the wmem subtree, apply the strict patch series."""

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
p.add_argument("destination", type=Path)
p.add_argument("--variant", choices=("reference", "ported"), default="ported")
a = p.parse_args()
with a.archive.open("rb") as f:
    if hashlib.file_digest(f, "sha256").hexdigest() != a.sha256:
        p.error("archive checksum mismatch")
root = a.archive.name.removesuffix(".tar.xz")
subtrees = (f"{root}/wsutil/wmem/", f"{root}/epan/wmem_scopes.")
a.destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=a.destination.parent) as temporary:
    with tarfile.open(a.archive) as archive:
        members = [m for m in archive if m.isfile() and m.name.startswith(subtrees)]
        if not members:
            p.error("archive lacks the wmem subtree")
        archive.extractall(temporary, members=members, filter="data")
    source = Path(temporary) / root
    patches = sorted(
        (Path(__file__).resolve().parent.parent / "patches").glob("*.patch")
    )
    if not patches:
        p.error("missing patch series")
    if a.variant == "ported":
        for patch in patches:
            subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
                cwd=source,
                check=True,
            )
    if a.destination.exists():
        shutil.rmtree(a.destination)
    source.rename(a.destination)
    (a.destination / "prepared.stamp").touch()
