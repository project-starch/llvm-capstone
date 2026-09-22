#!/usr/bin/env python3
"""Check the apr-util patch series: ordering, duplicate rejection, and that the prepared file is what the patches produce."""

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

archive, prepared = map(Path, sys.argv[1:])
port = Path(__file__).resolve().parents[2]
patches = sorted((port / "patches/apr-util").glob("*.patch"))
FILE = "buckets/apr_buckets_alloc.c"
assert len(patches) == 2, patches
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    with tarfile.open(archive) as source:
        source.extract(f"apr-util-1.6.3/{FILE}", root, filter="data")
    tree = root / "apr-util-1.6.3"
    # The second patch must not apply before the first.
    assert subprocess.run(["git", "apply", "--check", patches[1]], cwd=tree, capture_output=True).returncode != 0
    for patch in patches:
        subprocess.run(["git", "apply", patch], cwd=tree, check=True)
    # Nor may either apply twice.
    for patch in patches:
        assert subprocess.run(["git", "apply", "--check", patch], cwd=tree, capture_output=True).returncode != 0
    assert (tree / FILE).read_bytes() == (prepared / FILE).read_bytes(), FILE
print("apr-util source preparation: ordered, non-repeatable, byte-identical")
