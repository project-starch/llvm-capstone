#!/usr/bin/env python3
"""compare-sv.py <name> <uart.txt> [--tree A|B]: one S-29 bare arm's SR/SV readings against the RTL lane's simulated
vector for that image (capstone/tests/rtl-smoke/satpstale-bare-2026-10-19/results/rtl-sim-vectors.txt, dev
8e1df1e05a3c), the tree's section. Values are compared as 64-bit integers, as compare-vp2.py does. Exit 0 only on the
same count and values; 1 on a difference; 2 if the arm, the table or the readings are missing (no data is never a pass)."""
import re, sys, pathlib
TABLE = pathlib.Path("/tmp/capstone/land-board/capstone/tests/rtl-smoke/satpstale-bare-2026-10-19/results/rtl-sim-vectors.txt")
def table(tree="A"):
    t, sec = {}, None
    for l in TABLE.read_text().splitlines():
        m = re.match(r"^== tree ([AB]) ", l)
        if m: sec = m.group(1); continue
        m = re.match(r"^VEC (\S+)\s+\((0x[0-9a-f]+)\): (\d+) \| (.*)$", l)
        if m and sec == tree: t[m.group(1)] = (int(m.group(3)), [int(v, 0) & (2**64 - 1) for v in m.group(4).split()])
    return t
if __name__ == "__main__":
    a = sys.argv[1:]; tree = "A"
    if "--tree" in a: i = a.index("--tree"); tree = a[i + 1]; del a[i:i + 2]
    T = table(tree); n = a[0]
    if len(T) != 7: print(f"ERROR: the tree {tree} table gave {len(T)} arms, not 7"); sys.exit(2)
    if n not in T: print(f"ERROR: {n} not in the tree {tree} table"); sys.exit(2)
    cnt, exp = T[n]
    if len(exp) != cnt: print(f"ERROR: {n}: table count {cnt} but {len(exp)} values"); sys.exit(2)
    p = pathlib.Path(a[1]); out = p.read_text(errors="replace") if p.exists() else ""
    sr = re.findall(r"SR ([0-9A-Fa-f]{16})", out); sv = [int(x, 16) for x in re.findall(r"SV ([0-9A-Fa-f]{16})", out)]
    if not sr and not sv: print(f"NO-READING {n}: no SR/SV lines in {a[1]}"); sys.exit(2)
    ok = sv == exp and (not sr or int(sr[0], 16) == cnt)
    print(f"   {n}: read count {int(sr[0], 16) if sr else '?'} values {[hex(v) for v in sv]}; want {cnt} {[hex(v) for v in exp]}")
    print(f"VERDICT {n}: {'AS PRE-REGISTERED for tree ' + tree if ok else 'NOT tree ' + tree}")
    sys.exit(0 if ok else 1)
