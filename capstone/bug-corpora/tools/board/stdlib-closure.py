#!/usr/bin/env python3
"""stdlib-closure.py --python NATIVE --lib LIB --out FILE SCRIPT...: the CPython stdlib files a set of scripts can
import, for a board boot whose payload cannot carry the whole Lib (the vp2 Tier 3 cap leaves ~8.5 MB; the 3.13 Lib
is 86 MB gzipped).

modulefinder runs under NATIVE, the same CPython release as the image, over every SCRIPT, and over the interpreter's
startup modules. The result is the set of LIB files they reach, written relative to LIB, one per line. It is an
over-approximation of static imports and misses dynamic ones, so a run on a trimmed stdlib is valid only after a QEMU
A/B shows every reading equal to the full stdlib's.

DYNAMIC packages are added whole. The A/B caught the first one (2026-10-19): codecs are imported by NAME at lookup
time (codecs.lookup -> encodings.search_function -> __import__("encodings." + name)), which modulefinder cannot see,
so allocator-boundary case 17's trigger raised `LookupError: unknown encoding: hz` on the trimmed Lib and never reached
its defect. A missing DYNAMIC package is an error, not an empty addition.
"""
import argparse
import subprocess
import sys
from pathlib import Path

STARTUP = ["site", "encodings", "encodings.utf_8", "encodings.aliases", "encodings.latin_1", "encodings.ascii",
           "codecs", "io", "abc", "os", "stat", "posixpath", "genericpath", "_collections_abc", "_sitebuiltins",
           "runpy", "importlib", "importlib.util", "importlib.machinery", "sysconfig", "traceback", "linecache"]

DYNAMIC = ["encodings"]

FINDER = r'''
import sys, os, modulefinder
lib, out, startup, dynamic, scripts = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(","), sys.argv[5:]
need, errors = set(), 0
def add(mf):
    for m in mf.modules.values():
        if m.__file__ and m.__file__.startswith(lib + os.sep):
            need.add(os.path.relpath(m.__file__, lib))
for f in scripts:
    mf = modulefinder.ModuleFinder(path=[os.path.dirname(f), lib])
    try:
        mf.run_script(f)
    except Exception as e:
        errors += 1
        print(f"ERROR {f}: {type(e).__name__}: {e}", file=sys.stderr)
    add(mf)
for mod in startup:
    mf = modulefinder.ModuleFinder(path=[lib])
    try:
        mf.import_hook(mod)
    except ImportError:
        pass
    add(mf)
static = len(need)
for pkg in dynamic:
    root = os.path.join(lib, pkg)
    if not os.path.isdir(root):
        errors += 1
        print(f"ERROR dynamic package {pkg} not found under {lib}", file=sys.stderr)
        continue
    for dirpath, _, files in os.walk(root):
        need.update(os.path.relpath(os.path.join(dirpath, f), lib) for f in files if f.endswith(".py"))
open(out, "w").write("\n".join(sorted(need)) + "\n")
print(f"{len(scripts)} scripts, {errors} errors, {len(need)} files ({static} static, +{len(need) - static} from dynamic packages {dynamic})")
sys.exit(2 if errors or not need else 0)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python", type=Path, required=True, help="the native CPython of the image's release")
    p.add_argument("--lib", type=Path, required=True, help="that release's Lib")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("scripts", nargs="+", type=Path)
    a = p.parse_args()
    lib = a.lib.resolve()
    r = subprocess.run([str(a.python), "-c", FINDER, str(lib), str(a.out.resolve()), ",".join(STARTUP), ",".join(DYNAMIC),
                        *[str(s.resolve()) for s in a.scripts]], cwd=str(a.python.parent),
                       env={"PYTHONPATH": str(lib), "PATH": "/usr/bin:/bin"})
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
