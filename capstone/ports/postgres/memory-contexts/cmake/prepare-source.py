#!/usr/bin/env python3
"""Verify the release and prepare host headers plus independent patched variants."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--patch-tool", default="patch")
for name in ("archive", "source", "variants", "patches"):
    parser.add_argument("--" + name, type=Path, required=True)
for name in ("version", "sha256", "cc", "make"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
# CMake downloads through the shared cache; recheck before extracting or patching.
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
            patch_path = args.patches / f"postgresql-{args.version}-{name}.patch"
            with patch_path.open("rb") as patch:
                subprocess.run(
                    [
                        args.patch_tool,
                        "--batch",
                        "--forward",
                        "--fuzz=0",
                        "-s",
                        "-p1",
                        str(target),
                    ],
                    stdin=patch,
                    check=True,
                )
            data = target.read_bytes()
    return data


args.variants.mkdir(parents=True, exist_ok=True)
for mode in ("spatial", "sublet"):
    root = args.variants / mode
    aset_patches = ["0001-allocset-capstone-size-classes"]
    chunk_patches = ["0002-memorychunk-capstone-alignment"]
    if mode == "sublet":
        chunk_patches.append("0003-memorychunk-sublet-metadata-indices")
        aset_patches.append("0004-allocset-sublet-context-revocation")
    put(
        root / "aset.c",
        patched(
            args.source / "src/backend/utils/mmgr/aset.c",
            aset_patches,
        ),
    )
    put(
        root / "include/utils/memutils_memorychunk.h",
        patched(
            args.source / "src/include/utils/memutils_memorychunk.h",
            chunk_patches,
        ),
    )
    if mode == "sublet":
        for manager, patch in (
            ("slab", "0005-slab-sublet-lifetimes"),
            ("generation", "0006-generation-sublet-lifetimes"),
            ("bump", "0007-bump-sublet-lifetimes"),
        ):
            put(
                root / f"{manager}.c",
                patched(args.source / f"src/backend/utils/mmgr/{manager}.c", [patch]),
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
