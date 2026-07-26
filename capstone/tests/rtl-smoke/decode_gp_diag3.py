#!/usr/bin/env python3
"""Decode a gp_diag3 DEBUG line captured with LADDER_DBG_SLOTS=45.

Slot map (dbg_i == res[3+i]):
  dbg0..8    the 9 probes A..I
  dbg9..28   res[12..32) -- unused, expected 0
  dbg29..36  W1 = res[32..40)  seeded STRAIGHT-LINE, read by probes B/C/D/G/H
  dbg37..44  W2 = res[40..48)  seeded by probe F's LOOP, read straight-line by F

The point of the run: the host reads these windows AFTER the domain returned.
If W1 comes back as the seeds while probe C (a straight-line read of that same
window, inside the domain) summed to garbage, the fault is load-side, confirmed
for the whole window rather than just W1[0..3].
"""
import re
import sys

SEEDS = [256 << i for i in range(8)]
PROBES = [
    ("A", "loop over GLOBAL array (gp cap-table)", 65280),
    ("B", "loop over res[] (SHARED REGION)", 65280),
    ("C", "STRAIGHT-LINE read of res[] (no loop)", 65280),
    ("D", "loop over res[] at a CONSTANT index", 2048),
    ("E", "loop over a LOCAL STACK array", 65280),
    ("F", "loop STORING into res[], straight-line readback", 65280),
    ("G", "loop over res[] via a WALKING POINTER", 65280),
    ("H", "NESTED loop over res[] (v2 fold shape)", 255),
    ("I", "canary", 0xC0FFEE),
]
RUN1 = [65280, 2308734352, 2174516624, 2048, 134282752,
        65280, 2308734352, 134218537, 12648430]


def classify(v, exp):
    t = []
    if 0x80000000 <= v < 0x100000000:
        t.append(f"DRAM-address-like (0x80000000+0x{v - 0x80000000:x})")
    if v and (v & (v - 1)) == 0:
        t.append(f"power of two = 2^{v.bit_length() - 1}")
    if v == 0:
        t.append("ZERO -- store never landed?")
    d = v - exp
    if d and abs(d) in SEEDS:
        t.append(f"differs from expected by exactly seed {abs(d)}")
    return ("  <-- " + "; ".join(t)) if t else ""


text = sys.stdin.read() if len(sys.argv) < 2 else " ".join(sys.argv[1:])
v = {int(k): int(x) for k, x in re.findall(r"dbg(\d+)=(\d+)", text)}
if not v:
    sys.exit("no dbgN=... found in input")
print(f"parsed {len(v)} slots\n")

print("=== probes (dbg0..8) ===")
for i, (nm, desc, exp) in enumerate(PROBES):
    got = v.get(i)
    if got is None:
        continue
    ok = got == exp
    rep = "" if got == RUN1[i] else f"  [!] DIFFERS from run 1 ({RUN1[i]})"
    print(f"  {nm} dbg{i:<2} {'OK ' if ok else 'BAD'} got={got:<12} "
          f"exp={exp:<8} {desc}{rep}")

print("\n=== windows read back by the HOST, after return ===")
verdict = {}
for name, base, note in (("W1 res[32..40)", 29, "straight-line stores"),
                         ("W2 res[40..48)", 37, "probe F's LOOP stores")):
    got = [v.get(base + k) for k in range(8)]
    if any(g is None for g in got):
        print(f"  {name}: INCOMPLETE (need slots {base}..{base + 7})")
        continue
    good = got == SEEDS
    verdict[name] = good
    print(f"  {name} [{note}]: {'ALL SEEDS PRESENT' if good else 'MISMATCH'}")
    for k, g in enumerate(got):
        if g != SEEDS[k]:
            print(f"      [{k}] got {g} (0x{g:x}) expected "
                  f"{SEEDS[k]}{classify(g, SEEDS[k])}")

unused = [k for k in range(9, 29) if v.get(k)]
if unused:
    print(f"\n  [!] nonzero in the unused gap res[12..32): "
          f"{ {k: v[k] for k in unused} }")

print("\n=== verdict ===")
w1, w2 = verdict.get("W1 res[32..40)"), verdict.get("W2 res[40..48)")
if w1 is True:
    print("  W1 intact in memory, yet probes B/C/G/H summed it wrong INSIDE the")
    print("  domain -> LOAD-SIDE fault CONFIRMED across the whole window, not")
    print("  just W1[0..3]. Store side is exonerated.")
elif w1 is False:
    print("  W1 is CORRUPT in memory -> a store-side component after all;")
    print("  the load-side conclusion from the 33-slot run was premature.")
if w2 is True:
    print("  W2 intact -> probe F's LOOP stores land correctly too.")
elif w2 is False:
    print("  W2 corrupt -> loop stores do NOT land; F's correct sum was luck.")
