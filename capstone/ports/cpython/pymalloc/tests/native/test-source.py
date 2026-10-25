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
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    with tarfile.open(archive) as source:
        for name in ("Objects/obmalloc.c", "Include/internal/pycore_obmalloc.h"):
            source.extract(f"Python-3.13.7/{name}", root, filter="data")
    tree = root / "Python-3.13.7"
    assert (
        subprocess.run(
            ["git", "apply", "--check", patches[2]], cwd=tree, capture_output=True
        ).returncode
        != 0
    )
    # The native build prepares the `ported` variant: 0001 and 0002. 0003, the
    # Sublet protection, must still apply on top of them.
    for patch in patches[:2]:
        subprocess.run(["git", "apply", patch], cwd=tree, check=True)
    assert (
        subprocess.run(
            ["git", "apply", "--check", patches[0]], cwd=tree, capture_output=True
        ).returncode
        != 0
    )
    for name in ("Objects/obmalloc.c", "Include/internal/pycore_obmalloc.h"):
        assert (tree / name).read_bytes() == (prepared / name).read_bytes()
    subprocess.run(["git", "apply", "--check", patches[2]], cwd=tree, check=True)
    marker = tree / "preserve-me"
    marker.write_text("previous complete source")
    result = subprocess.run(
        [
            sys.executable,
            port / "cmake/prepare-source.py",
            archive,
            "0" * 64,
            "3.13.7",
            tree,
        ],
        capture_output=True,
    )
    assert result.returncode != 0 and marker.read_text() == "previous complete source"
