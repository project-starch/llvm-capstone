#!/usr/bin/env python3
"""Check patch ordering, duplicate rejection, and preservation after preparation failure."""

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

archive, prepared = map(Path, sys.argv[1:])
port = Path(__file__).resolve().parents[2]
patches = sorted((port / "patches").glob("*.patch"))
FILES = ("slabs.c", "slabs.h", "cache.c", "cache.h", "queue.h")
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    with tarfile.open(archive) as source:
        for name in FILES:
            source.extract(f"memcached-1.6.45/{name}", root, filter="data")
    tree = root / "memcached-1.6.45"
    # The second patch must not apply before the first.
    assert (
        subprocess.run(
            ["git", "apply", "--check", patches[1]], cwd=tree, capture_output=True
        ).returncode
        != 0
    )
    for patch in patches:
        subprocess.run(["git", "apply", patch], cwd=tree, check=True)
    # Nor may either apply twice.
    for patch in patches:
        assert (
            subprocess.run(
                ["git", "apply", "--check", patch], cwd=tree, capture_output=True
            ).returncode
            != 0
        )
    for name in FILES:
        assert (tree / name).read_bytes() == (prepared / name).read_bytes(), name
    marker = tree / "preserve-me"
    marker.write_text("previous complete source")
    result = subprocess.run(
        [sys.executable, port / "cmake/prepare-source.py", archive, "0" * 64, "1.6.45", tree],
        capture_output=True,
    )
    assert result.returncode != 0 and marker.read_text() == "previous complete source"
