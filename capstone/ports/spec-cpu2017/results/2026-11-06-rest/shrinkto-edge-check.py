#!/usr/bin/env python3
"""shrinkto-edge-check.py <elf>...: for an image built with -capstone-shrinkto-end-adjust=1, show that no data object
SHRINKTO narrows can push cursor + size + 1 past the end of the launcher's EXACT image window (vexec built
-DCAPSTONE_VEXEC_EXACT_IMAGE grants the page-rounded end of the last PT_LOAD, not a power of two). Exit 0 if every
image is safe, 1 if any would trap, 2 if no image could be read."""
import subprocess, sys, os
R = os.environ.get("READELF", "llvm-readelf"); bad = seen = 0
for e in sys.argv[1:]:
    try:
        L = subprocess.run([R, "-lW", e], capture_output=True, text=True, check=True).stdout
        S = subprocess.run([R, "-sW", e], capture_output=True, text=True, check=True).stdout
    except Exception as x:
        print(f"{e}: cannot read ({x})"); continue
    loads = [l.split() for l in L.splitlines() if l.strip().startswith("LOAD")]
    win = (max(int(x[2], 16) + int(x[5], 16) for x in loads) + 4095) & ~4095
    objs = [(int(p[1], 16) + int(p[2]), p[7], int(p[2])) for p in (l.split() for l in S.splitlines())
            if len(p) >= 8 and p[3] == "OBJECT" and 0 < int(p[2]) and int(p[2]) + 1 <= 2047]
    if not objs: print(f"{e}: no SHRINKTO-sized object"); continue
    end, name, size = max(objs); seen += 1; ok = end + 1 <= win; bad |= not ok
    print(f"{e}: last object {name} ({size} B) ends 0x{end:x}; window ends 0x{win:x}; slack {win - end} B: {'SAFE' if ok else 'TRAPS'}")
sys.exit(2 if not seen else 1 if bad else 0)
