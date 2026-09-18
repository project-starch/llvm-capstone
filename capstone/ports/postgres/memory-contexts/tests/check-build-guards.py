#!/usr/bin/env python3
"""Exercise source integrity, image-size and cross-ABI configuration guards."""

from pathlib import Path
import shutil
import struct
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

    # A valid ELF64 header and PT_LOAD with a loadable span just over 2 MiB.
    ident = b"\x7fELF\x02\x01\x01".ljust(16, b"\0")
    header = struct.pack(
        "<16sHHIQQQIHHHHHH", ident, 2, 243, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0
    )
    segment = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, 0, 2 * 1024 * 1024 + 1, 16)
    image = work / "oversized.dom"
    image.write_bytes(header + segment)
    rejects(
        [sys.executable, str(port / "cmake/check-domain.py"), str(image)],
        "loader permits at most 2 MiB",
    )

    domain, linux = work / "domain", work / "linux"
    domain.mkdir()
    linux.mkdir()
    (domain / "regions.json").write_text('{"arena": 1}')
    (linux / "regions.json").write_text('{"arena": 2}')
    rejects(
        [
            sys.executable,
            str(port / "host/run-qemu.py"),
            str(work / "unused.a11"),
            str(work / "results"),
            "--domain-build",
            str(domain),
            "--linux-build",
            str(linux),
        ],
        "region configurations differ",
    )
print("Source integrity, image-size and region-mismatch negative controls passed")
