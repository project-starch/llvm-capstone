#!/usr/bin/env python3
"""scan-memcpy-expansion.py's run detector with a narrower break set: only control flow and ldc/stc end a run, so
capability address arithmetic (cincoffset, cincoffsetimm, movc, ...) between the chunks of a scalar copy no longer
splits it below the threshold. Diff plain vs twin by (function, bytes, min unit, insns); print the plain-only runs
with their instructions. Usage: rescan-nosplit.py PLAIN TWIN [threshold]"""
import os, re, sys, collections, importlib.util, subprocess, bisect
from pathlib import Path
spec = importlib.util.spec_from_file_location("s", os.path.join(os.environ["C81_SCANNER_DIR"], "scan-memcpy-expansion.py"))
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
OD = os.environ["C81_OBJDUMP"]; NM = os.environ["C81_NM"]
BRK = re.compile(r"^(j|jr|jal|jalr|b[a-z]+|ret|ecall|ebreak|ldc|stc|tail|call|mret|sret|wfi|fence)\b")
def real_syms(img):
    out = []
    for l in subprocess.run([NM, "-n", img], capture_output=True, text=True, check=True).stdout.splitlines():
        p = l.split()
        if len(p) == 3 and p[1] in "tTwW" and not p[2].startswith(".L"): out.append((int(p[0], 16), p[2]))
    return sorted(out)
def runs(img, thr):
    syms = real_syms(img); addrs = [a for a, _ in syms]; res = []
    for fn, ins in s.disassemble(Path(img), OD).items():
        cur = []
        def flush():
            if cur:
                moved = sum(w for k, w, a, t in cur if k == "s")
                if moved >= thr and any(k == "l" for k, *_ in cur):
                    a0 = cur[0][2]; i = bisect.bisect_right(addrs, a0) - 1
                    res.append(((syms[i][1], moved, min(w for _, w, _, _ in cur), len(cur)), a0 - syms[i][0], "; ".join(t for *_, t in cur)))
        for a, m, r in ins:
            if m in s.LOADS: cur.append(("l", s.LOADS[m], a, f"{m} {r}"))
            elif m in s.STORES: cur.append(("s", s.STORES[m], a, f"{m} {r}"))
            elif BRK.match(m): flush(); cur = []
        flush()
    return res
thr = int(sys.argv[3]) if len(sys.argv) > 3 else 16
A, B = runs(sys.argv[1], thr), runs(sys.argv[2], thr)
cb = collections.Counter(k for k, _, _ in B); seen = collections.Counter(); n = 0
print(f"plain runs {len(A)} twin runs {len(B)} (load-carrying, >= {thr} B stored, ldc/stc and control flow break)")
for k, off, t in sorted(A, key=lambda x: (x[0][0], x[1])):
    seen[k] += 1
    if seen[k] > cb[k]:
        n += 1; print(f"  {k[0]}+{off:#x}  {k[1]}B unit={k[2]} ({k[3]} insns)\n      {t[:300]}")
print(f"plain-only load-carrying runs: {n}")
