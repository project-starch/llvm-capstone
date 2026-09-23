#!/usr/bin/env python3
"""Pack CPython's pure-Python standard library as lib/python313.zip for a domain.

Run it with a native CPython of the SAME version (3.13), whose bytecode is the
one the domain's interpreter reads: the archive holds compiled .pyc files at the
legacy location (PyZipFile.writepy), which zipimport loads without the source.

Why a zip: import reads it with open, read, lseek and fstat, all of which the
hostcall file service serves. A stdlib DIRECTORY would need a listing of it
(os.listdir, getdents64), which the service does not have yet. The entries are
STORED, not deflated, so zipimport needs no zlib.

Left out: the test suite, GUI and packaging trees, and each package's own tests.

Usage:  make-stdlib-zip.py <Lib-dir> <out.zip>
"""

import pathlib
import sys
import zipfile

SKIP = {"test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3", "pydoc_data",
        "site-packages", "__pycache__", "venv", "turtle.py", "__phello__"}
SKIP_PARTS = {"tests", "test", "idle_test"}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    lib, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    if sys.version_info[:2] != (3, 13):
        print(f"ERROR: this is Python {sys.version.split()[0]}; the domain's interpreter is 3.13 "
              f"and reads only 3.13 bytecode", file=sys.stderr)
        return 2
    if not (lib / "os.py").is_file() or not (lib / "encodings" / "__init__.py").is_file():
        print(f"ERROR: {lib} is not a CPython Lib directory", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.PyZipFile(out, "w", compression=zipfile.ZIP_STORED, optimize=0) as z:
        for p in sorted(lib.iterdir()):
            if p.name in SKIP or p.name.startswith("."):
                continue
            if p.is_dir() and (p / "__init__.py").is_file():
                z.writepy(str(p), filterfunc=lambda f: not SKIP_PARTS & set(pathlib.Path(f).parts))
            elif p.suffix == ".py":
                z.writepy(str(p))
        names = set(z.namelist())
    for needed in ("os.pyc", "encodings/__init__.pyc", "encodings/utf_8.pyc", "codecs.pyc"):
        if needed not in names:
            print(f"ERROR: {out} has no {needed}", file=sys.stderr)
            return 2
    print(f"{out}: {len(names)} entries, {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
