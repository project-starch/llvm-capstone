#!/usr/bin/env python3
"""Verify the release and prepare host headers plus independent patched variants."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
for name in ("archive", "source", "variants", "patches"):
    parser.add_argument("--" + name, type=Path, required=True)
for name in ("version", "sha256", "cc", "make"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
args.archive.parent.mkdir(parents=True, exist_ok=True)
with Path(str(args.archive) + ".lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not args.archive.exists():
        url = f"https://ftp.postgresql.org/pub/source/v{args.version}/postgresql-{args.version}.tar.bz2"
        with tempfile.NamedTemporaryFile(
            dir=args.archive.parent, delete=False
        ) as download:
            temporary = Path(download.name)
        try:
            urllib.request.urlretrieve(url, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != args.sha256:
                raise SystemExit("PostgreSQL download checksum mismatch")
            temporary.replace(args.archive)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != args.sha256:
        raise SystemExit("Cached PostgreSQL archive checksum mismatch")
args.source.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.archive) as archive:
    if not args.source.exists():
        archive.extractall(args.source.parent, filter="data")
    # Generated files may change; the manager and patched header must remain upstream.
    originals = [
        *(
            args.source / f"src/backend/utils/mmgr/{name}.c"
            for name in (
                "aset",
                "mcxt",
                "generation",
                "slab",
                "bump",
                "alignedalloc",
                "memdebug",
            )
        ),
        args.source / "src/include/utils/memutils_memorychunk.h",
    ]
    for original in originals:
        member = f"postgresql-{args.version}/{original.relative_to(args.source)}"
        if original.read_bytes() != archive.extractfile(member).read():
            raise SystemExit(f"Modified upstream source: {original}")
config = args.source / "src/include/pg_config.h"
stamp = args.source / ".headers-ready"
with (args.source.parent / "prepare.log").open("a") as log:
    if not stamp.exists():
        env = dict(os.environ, CC=args.cc)
        # Host configuration generates headers only. Domain variants override
        # ABI alignment below; no PostgreSQL server is cross-compiled.
        subprocess.run(
            [
                "./configure",
                "--without-icu",
                "--without-zlib",
                "--without-readline",
                "--without-libxml",
                "--quiet",
            ],
            cwd=args.source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [args.make, "-C", "src", "submake-generated-headers"],
            cwd=args.source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        stamp.touch()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def patched(original, patches):
    data = original.read_bytes()
    for name in patches:
        with tempfile.TemporaryDirectory(dir=args.variants) as temporary:
            target = Path(temporary) / original.name
            target.write_bytes(data)
            with (args.patches / name).open("rb") as patch:
                subprocess.run(
                    ["patch", "--batch", "-s", "-F0", "-p0", str(target)],
                    stdin=patch,
                    check=True,
                )
            data = target.read_bytes()
    return data


args.variants.mkdir(parents=True, exist_ok=True)
for mode in ("spatial", "sublet"):
    root = args.variants / mode
    suffixes = ["capstone"] + (["sublet"] if mode == "sublet" else [])
    put(
        root / "aset.c",
        patched(
            args.source / "src/backend/utils/mmgr/aset.c",
            [f"aset-{s}.patch" for s in suffixes],
        ),
    )
    put(
        root / "include/utils/memutils_memorychunk.h",
        patched(
            args.source / "src/include/utils/memutils_memorychunk.h",
            [f"memorychunk-{s}.patch" for s in suffixes],
        ),
    )
    text = config.read_text()
    if text.count("#define MAXIMUM_ALIGNOF 8\n") != 1:
        raise SystemExit(
            "Expected an eight-byte host alignment before the Capstone ABI override"
        )
    put(
        root / "include/pg_config.h",
        text.replace(
            "#define MAXIMUM_ALIGNOF 8\n", "#define MAXIMUM_ALIGNOF 16\n"
        ).encode(),
    )
print(
    f"PostgreSQL {args.version}: verified source, generated headers, spatial/Sublet variants"
)
