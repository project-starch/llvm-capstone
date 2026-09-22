"""Negative controls for the pinned source and ordered patch interface."""

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

archive = Path(sys.argv[1])
prepared = Path(sys.argv[2])
port = Path(__file__).resolve().parents[2]
patches = sorted((port / "patches").glob("*.patch"))
units = (
    "wmem_core.c",
    "wmem_user_cb.c",
    "wmem_allocator_simple.c",
    "wmem_allocator_block.c",
    "wmem_allocator_block_fast.c",
    "wmem_allocator_strict.c",
)
root = archive.name.removesuffix(".tar.xz")
with tempfile.TemporaryDirectory() as temporary:
    with tarfile.open(archive) as tar:
        members = [
            m for m in tar if m.isfile() and m.name.startswith(f"{root}/wsutil/wmem/")
        ]
        tar.extractall(temporary, members=members, filter="data")
    source = Path(temporary) / root
    assert patches, "missing patch series"
    for patch in patches:
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        assert (
            subprocess.run(
                ["git", "apply", "--check", str(patch)], cwd=source, capture_output=True
            ).returncode
            != 0
        )
    for unit in units:
        assert (source / "wsutil/wmem" / unit).read_bytes() == (
            prepared / "wsutil/wmem" / unit
        ).read_bytes(), unit
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
print("PASS strict patch series, duplicate and checksum controls")
