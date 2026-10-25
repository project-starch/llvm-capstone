#!/usr/bin/env python3
"""Count mem* call targets per function in two images (plain, libcall twin); the twin's extra references are the
inline expansions the plain image carries. Fix over the first version: an `addi rd, rs, imm` whose rs holds an auipc
value counts whatever rd is (the call sites use `auipc a2; addi a3, a2, X; jalr a3`). Also counts `jal` to a target."""
import re, subprocess, sys, collections, bisect, os
OD = os.environ["C81_OBJDUMP"]
NM = os.environ["C81_NM"]
def count(img):
    syms = {}; real = []
    for l in subprocess.run([NM, "-n", img], capture_output=True, text=True, check=True).stdout.splitlines():
        p = l.split()
        if len(p) == 3 and p[1] in "tTwW":
            syms.setdefault(int(p[0], 16), p[2])
            if not p[2].startswith(".L"): real.append((int(p[0], 16), p[2]))
    real.sort(); ra = [a for a, _ in real]
    targets = {a: n for a, n in syms.items() if n in ("memcpy", "memset", "memmove")}
    if "memcpy" not in targets.values(): raise SystemExit(f"ERROR: no memcpy symbol in {img}: {targets}")
    out = subprocess.run([OD, "-d", "--no-show-raw-insn", img], capture_output=True, text=True, check=True).stdout
    auipc = {}; c = collections.Counter(); sites = collections.defaultdict(list)
    for l in out.splitlines():
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", l)
        if not m: continue
        a = int(m.group(1), 16); mn = m.group(2); ops = [x.strip() for x in m.group(3).split(",")]
        t = None
        if mn == "auipc":
            auipc[ops[0]] = (a, int(ops[1], 16)); continue
        if mn == "addi" and len(ops) == 3 and ops[1] in auipc:
            base, imm = auipc[ops[1]]; imm = imm - (1 << 20) if imm >= (1 << 19) else imm
            t = base + (imm << 12) + int(ops[2], 0)
        elif mn in ("jal", "call", "tail") and ops:
            mm = re.match(r"([0-9a-f]+)", ops[-1]); t = int(mm.group(1), 16) if mm else None
        if ops and ops[0] in auipc and mn not in ("auipc",) and not mn.startswith(("s", "b", "j")):
            auipc.pop(ops[0], None)   # rd overwritten: its auipc value is no longer live (stores/branches have no rd)
        if t in targets:
            i = bisect.bisect_right(ra, a) - 1; f = real[i][1]; c[(f, targets[t])] += 1; sites[(f, targets[t])].append(a - real[i][0])
    return c, sites
A, sa = count(sys.argv[1]); B, sb = count(sys.argv[2])
ta = collections.Counter(); tb = collections.Counter()
for (f, n), k in A.items(): ta[n] += k
for (f, n), k in B.items(): tb[n] += k
print("plain totals", dict(ta), " twin totals", dict(tb))
more = B - A
print(f"twin-only references: {sum(more.values())} in {len({f for f, _ in more})} functions")
for (f, n), k in sorted(more.items()):
    print(f"  {f} {n} x{k} (twin offsets {[hex(o) for o in sb[(f, n)]][:6]})")
less = A - B
if less: print("references only in plain:", sorted(less.items())[:10])
