#!/usr/bin/env python3
"""Verify the release and prepare host headers plus independent patched variants."""

import argparse
import hashlib
import os
import shutil
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


# The files a variant may patch, relative to the source root. Each variant is
# a mirror of them with its ordered patches applied, so a patch may span
# several files and add new ones.
MIRRORED = [
    *(f"src/backend/utils/mmgr/{name}" for name in (
        "aset.c", "mcxt.c", "generation.c", "slab.c", "bump.c", "Makefile", "meson.build")),
    "src/include/utils/memutils_memorychunk.h",
]
# What the replay compiles from a variant; everything else comes from the source.
VARIANT_SOURCES = {
    "spatial": ["aset.c"],
    "sublet": ["aset.c", "mcxt.c", "generation.c", "slab.c", "bump.c", "sublet.c"],
}
VARIANT_HEADERS = {
    "spatial": ["memutils_memorychunk.h"],
    "sublet": ["memutils_memorychunk.h", "memutils_sublet.h"],
}
VARIANT_PATCHES = {
    "spatial": ["0001-allocset-capstone-size-classes", "0002-memorychunk-capstone-alignment"],
    "sublet": [
        "0001-allocset-capstone-size-classes",
        "0002-memorychunk-capstone-alignment",
        "0003-memory-contexts-sublet-lifetimes",
    ],
}


def variant_tree(patches):
    tree = Path(tempfile.mkdtemp(dir=args.variants))
    for relative in MIRRORED:
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((args.source / relative).read_bytes())
    for name in patches:
        patch_path = args.patches / f"postgresql-{args.version}-{name}.patch"
        with patch_path.open("rb") as patch:
            subprocess.run(
                [args.patch_tool, "--batch", "--forward", "--fuzz=0", "-s", "-p1", "-d", str(tree)],
                stdin=patch,
                check=True,
            )
    return tree


args.variants.mkdir(parents=True, exist_ok=True)
for mode in ("spatial", "sublet"):
    root = args.variants / mode
    tree = variant_tree(VARIANT_PATCHES[mode])
    try:
        for name in VARIANT_SOURCES[mode]:
            put(root / name, (tree / "src/backend/utils/mmgr" / name).read_bytes())
        for name in VARIANT_HEADERS[mode]:
            put(root / "include/utils" / name, (tree / "src/include/utils" / name).read_bytes())
    finally:
        shutil.rmtree(tree)
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
