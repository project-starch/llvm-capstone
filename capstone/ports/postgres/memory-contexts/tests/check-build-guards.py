#!/usr/bin/env python3
"""Exercise the source integrity guards."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

port, build, archive, version, checksum, cc, make = sys.argv[1:]
port, build, archive = map(Path, (port, build, archive))


def rejects(command, message):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 or message not in result.stdout + result.stderr:
        raise SystemExit(f"Guard did not reject {message!r}: {result}")


with tempfile.TemporaryDirectory(dir=build) as temporary:
    work = Path(temporary)
    source = work / f"postgresql-{version}"
    prepare = [
        sys.executable,
        str(port / "cmake/prepare-source.py"),
        "--archive",
        str(archive),
        "--source",
        str(source),
        "--variants",
        str(work / "variants"),
        "--patches",
        str(port / "patches"),
        "--version",
        version,
        "--cc",
        cc,
        "--make",
        make,
        "--sha256",
    ]
    rejects([*prepare, "0" * 64], "archive checksum mismatch")
    upstream = build / "source" / f"postgresql-{version}"
    for original in upstream.glob("src/backend/utils/mmgr/*.c"):
        target = source / original.relative_to(upstream)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    with (source / "src/backend/utils/mmgr/aset.c").open("a") as modified:
        modified.write("\n/* altered source negative control */\n")
    rejects([*prepare, checksum], "Modified upstream source")


print("Source integrity negative controls passed")
