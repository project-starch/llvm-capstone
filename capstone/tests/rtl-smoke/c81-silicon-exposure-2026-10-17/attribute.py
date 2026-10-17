#!/usr/bin/env python3
"""Attribute scan-memcpy-expansion.py's A-only runs to real (non-.L) functions, with their addresses.
Usage: attribute.py [-v] PLAIN.dom TWIN.dom (-v prints each run's instructions). Needs C81_SCANNER_DIR (the compiler
lane's C-81 folder, for scan-memcpy-expansion.py), C81_OBJDUMP and C81_NM. A run is keyed by (enclosing real function, bytes, min unit, insns); the multiset
difference plain - twin is what the libcall lowering diverted."""
import os, sys, subprocess, collections, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("s", os.path.join(os.environ["C81_SCANNER_DIR"], "scan-memcpy-expansion.py"))
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
OD = os.environ["C81_OBJDUMP"]; NM = os.environ["C81_NM"]

def real_syms(img):
    out = []
    for l in subprocess.run([NM, "-n", img], capture_output=True, text=True, check=True).stdout.splitlines():
        p = l.split()
        if len(p) == 3 and p[1] in "tTwW" and not p[2].startswith(".L"):
            out.append((int(p[0], 16), p[2]))
    if not out:
        raise SystemExit(f"ERROR: no symbols in {img}")
    return sorted(out)

def runs_with_addr(img, thr=16):
    syms = real_syms(img); addrs = [a for a, _ in syms]
    import bisect
    res = []; txt = {}
    for fn, ins in s.disassemble(Path(img), OD).items():
        cur = []
        def flush():
            if cur:
                moved = sum(w for k, w, a in cur if k == "s")
                if moved >= thr and max(w for _, w, _ in cur) < 16:
                    a0 = cur[0][2]; i = bisect.bisect_right(addrs, a0) - 1
                    res.append(((syms[i][1], moved, min(w for _, w, _ in cur), len(cur)), a0 - syms[i][0],
                                "; ".join(txt[a] for _, _, a in cur)))
        for a, m, r in ins:
            if m in s.LOADS: cur.append(("l", s.LOADS[m], a)); txt[a] = f"{m} {r}"
            elif m in s.STORES: cur.append(("s", s.STORES[m], a)); txt[a] = f"{m} {r}"
            elif s.BREAKERS.match(m): flush(); cur = []
        flush()
    return res

VERBOSE = "-v" in sys.argv; sys.argv = [a for a in sys.argv if a != "-v"]
A, B = runs_with_addr(sys.argv[1]), runs_with_addr(sys.argv[2])
cb = collections.Counter(k for k, _, _ in B)
seen = collections.Counter()
for k, off, t in sorted(A, key=lambda x: (x[0][0], x[1])):
    seen[k] += 1
    if seen[k] > cb[k]:
        fn, moved, unit, n = k
        print(f"  {fn}+{off:#x}  {moved}B unit={unit} ({n} insns)")
        if VERBOSE: print(f"      {t}")
