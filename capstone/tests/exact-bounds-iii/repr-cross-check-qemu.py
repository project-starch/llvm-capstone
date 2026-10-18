#!/usr/bin/env python3
"""Cross-check capstone/runtime/include/capstone/representable.h against capstone-qemu's COMPACT encoder.

repr-cross-check.py checks the header against the RTL encoder. The QEMU prediction kits run the compact profile,
where helper_csderive refuses (cause 29) any child that does not survive cap_compress/cap_uncompress; the model of
that round trip is qemu_compact_repr.py (a 64-bit port of capstone-qemu target/riscv/cap_compress.c at
b290f5656600). Q-15's placement repairs size and place children with capstone_repr_round/capstone_repr_ok, so the
header must agree with THIS encoder too, and capstone_repr_round must pad only at the tail.

Checks, all from the compiled header:
  1. capstone_repr_ok(base, len) == the compact round trip, over boundary and random (base, len) pairs;
  2. positive control: sublet-lifetimes case 27's shape (base+1, 4096) is refused by both;
  3. negative control: a base-ignoring predicate must disagree with the encoder;
  4. capstone_repr_round(len) = r: r >= len, r is accepted at every r-granule-aligned base tried, the pad r - len is
     below the granule of r, and no length in [len, r) is accepted at that base (minimality).
Exits non-zero on any failure.
"""
import os, pathlib, random, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qemu_compact_repr import representable
INC = next((p / "capstone/runtime/include" for p in HERE.parents
            if (p / "capstone/runtime/include/capstone/representable.h").is_file()), None)
if INC is None: sys.exit("ERROR: representable.h not found")
CC = os.environ.get("CAPSTONE_CLANG") or os.environ.get("CC") or "clang"

def header(cases, broken=False):
    mutant = '#define capstone_repr_ok(b, l) (((l) % capstone_repr_granule(l)) == 0)\n' if broken else ''
    drv = ('#include <stdio.h>\n#include <capstone/representable.h>\n' + mutant +
           'int main(void){unsigned long b,l;while(scanf("%lu %lu",&b,&l)==2)'
           'printf("%d %lu %lu\\n",capstone_repr_ok(b,l),capstone_repr_round(l),capstone_repr_granule(l));return 0;}\n')
    with tempfile.TemporaryDirectory() as t:
        c, x = pathlib.Path(t, "d.c"), pathlib.Path(t, "d")
        c.write_text(drv)
        subprocess.run([CC, "-O1", "-I", str(INC), str(c), "-o", str(x)], check=True)
        out = subprocess.run([str(x)], input="".join(f"{b} {l}\n" for b, l in cases),
                             capture_output=True, text=True, check=True).stdout.split("\n")
    return [tuple(int(v) for v in line.split()) for line in out if line.strip()]

rng = random.Random(15)
cases = []
for k in range(12, 31):                                   # boundaries around every power of two from 4 KiB
    for l in (1 << k) - 1, 1 << k, (1 << k) + 1, (1 << k) + (1 << (k - 9)), (1 << k) + (1 << (k - 9)) - 1:
        g = 1 << max(0, k - 9)
        for b in (0x80000000, 0x80000000 + g, 0x80000000 + g // 2 if g > 1 else 0x80000001, 0x80000001, 0x3fa4fe8000):
            cases.append((b, l))
for l in (1, 15, 16, 4095):
    cases += [(0x80000001, l), (0x80000000, l)]
for _ in range(20000):                                    # random: log-uniform length, base with random alignment
    l = int(2 ** rng.uniform(0, 30)) or 1
    b = (rng.getrandbits(46) >> rng.randrange(0, 24)) << rng.randrange(0, 24)
    cases.append((b & ((1 << 47) - 1), l))
ans = header(cases)
bad = [(b, l, a[0]) for (b, l), a in zip(cases, ans) if bool(a[0]) != representable(b, l)[0]]
acc = sum(a[0] for a in ans)
print(f"1. repr_ok vs the compact round trip: {len(cases)} pairs, header accepts {acc}, disagreements {len(bad)}")
for b, l, s in bad[:5]: print(f"   base={b:#x} len={l} header={s} encoder={representable(b, l)[0]}")
c27 = header([(0x80001001, 4096)])[0][0]
print(f"2. case 27 shape (0x80001001, 4096): header={c27} encoder={int(representable(0x80001001, 4096)[0])} (both must be 0)")
mut = sum(bool(a[0]) != representable(b, l)[0] for (b, l), a in zip(cases, header(cases, broken=True)))
print(f"3. base-ignoring mutant disagreements: {mut} (must be > 0)")
pad_fail, pad_max, rounded = [], 0, 0
lens = sorted({l for _, l in cases} | {int(2 ** rng.uniform(12, 30)) for _ in range(3000)})
rs = [a[1] for a in header([(0, l) for l in lens])]
gs = [a[2] for a in header([(0, r) for r in rs])]       # the granule of each rounded length, one compile
for l, r, g in zip(lens, rs, gs):
    ok = r >= l and r - l < g and all(representable(base, r)[0] for base in (0x80000000, 0x80000000 + g, 0x3fa4000000 + 7 * g))
    probe = range(l, r) if r - l <= 4096 else sorted({l, r - 1} | {rng.randrange(l, r) for _ in range(256)})
    ok = ok and not any(representable(0x80000000 + g, L)[0] for L in probe)   # exhaustive up to a 4 KiB pad
    if not ok: pad_fail.append((l, r, g))
    rounded += r != l; pad_max = max(pad_max, r - l)
print(f"4. repr_round tail pad: {len(lens)} lengths, {rounded} rounded, max pad {pad_max}, failures {len(pad_fail)}")
for f in pad_fail[:5]: print("   len=%d round=%d granule=%d" % f)
ok = not bad and c27 == 0 and not representable(0x80001001, 4096)[0] and mut > 0 and not pad_fail
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
