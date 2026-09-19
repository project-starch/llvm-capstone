"""Negative controls for the pinned source and ordered patch interface."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

archive = Path(sys.argv[1])
prepared = Path(sys.argv[2])
port = Path(__file__).resolve().parents[2]
patches = sorted((port / "patches").glob("*.patch"))
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    with tarfile.open(archive) as tar:
        member = tar.getmember("whisper.cpp-1.9.4/ggml/src/ggml.c")
        tar.extract(member, root, filter="data")
    source = root / "whisper.cpp-1.9.4"
    # The recorder patch needs extraction/lifetime context and cannot go first.
    assert (
        subprocess.run(
            ["git", "apply", "--check", str(patches[2])],
            cwd=source,
            capture_output=True,
        ).returncode
        != 0
    )
    for patch in patches[:2]:
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        assert (
            subprocess.run(
                ["git", "apply", "--check", str(patch)], cwd=source, capture_output=True
            ).returncode
            != 0
        )
    assert (source / "ggml/src/ggml.c").read_bytes() == (
        prepared / "ggml/src/ggml.c"
    ).read_bytes()
    (source / "sentinel").write_text("preserve")
    assert (
        subprocess.run(
            [
                sys.executable,
                str(port / "cmake/prepare-source.py"),
                str(archive),
                "0" * 64,
                str(source),
            ],
            capture_output=True,
        ).returncode
        != 0
    )
    assert (source / "sentinel").read_text() == "preserve"
print("PASS strict patch series, duplicate/order and checksum controls")
