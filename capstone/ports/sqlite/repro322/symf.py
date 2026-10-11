#!/usr/bin/env python3
"""Name the function a domain fault landed in.

llvm-symbolizer on these images returns a local label (`.Lpcrel_hi155`), which names
nothing; the preceding FUNCTION symbol does. Usage: symf.py <dom> <file-offset-hex>.
"""
import os, subprocess, sys

nm = os.path.join(os.environ.get("CAPSTONE_LLVM_BIN", ""), "llvm-nm")
dom, off = sys.argv[1], int(sys.argv[2], 16)
out = subprocess.check_output([nm, "-C", "--defined-only", "-n", "--format=posix", dom]).decode()
best = None
for line in out.splitlines():
    f = line.split()
    if len(f) < 3 or f[1] not in ("T", "t", "W", "w") or f[0].startswith(".L"):
        continue
    try:
        a = int(f[2], 16)
    except ValueError:
        continue
    if a <= off and (best is None or a > best[0]):
        best = (a, f[0])
print("%s+0x%x" % (best[1], off - best[0]) if best else hex(off))
