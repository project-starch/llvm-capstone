#!/usr/bin/env python3
"""Negated-cursor census: every CINCOFFSET (opcode 0x5b, funct3 1, funct7 0x0c) whose offset register (rs2) was last
written, earlier in the same function and in program order, by a negation `sub rd, x0, rs` (0x40000033 | rs<<20 | rd<<7).
That is the hoisted `cap - base` shape. Raw words, FUNC symbols for function boundaries, parse count printed, exit 2 on
no parse. Usage: negcursor-census.py <llvm-bin> <image>..."""
import re, subprocess, sys
LB = sys.argv[1]; rc = 0
for img in sys.argv[2:]:
    syms = sorted((int(p[1], 16), int(p[2]), p[7]) for p in (l.split() for l in subprocess.run(
        [f'{LB}/llvm-readelf', '-sW', img], capture_output=True, text=True).stdout.splitlines())
        if len(p) >= 8 and p[3] == 'FUNC' and p[1] != '0')
    starts = {a: n for a, s, n in syms}
    def fn(pc):
        best = '?'
        for a, s, n in syms:
            if a <= pc < a + max(s, 1): best = n
        return best
    out = subprocess.run([f'{LB}/llvm-objdump', '-d', img], capture_output=True, text=True).stdout
    n = cinc = 0; neg = {}; hits = []; own = []; copy = {}   # copy[rd] = rs when rd := rs (the cursor's integer)
    for l in out.splitlines():
        m = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){2,4})', l)
        if not m: continue
        pc = int(m.group(1), 16); b = bytes.fromhex(m.group(2).replace(' ', '')); n += 1
        if pc in starts: neg.clear(); copy.clear()
        if len(b) != 4: continue
        w = int.from_bytes(b, 'little'); op = w & 0x7f; rd = (w >> 7) & 31; f3 = (w >> 12) & 7
        rs1 = (w >> 15) & 31; rs2 = (w >> 20) & 31; f7 = w >> 25
        if op == 0x5b and f3 == 1 and f7 == 0x0c:
            cinc += 1
            if rs2 in neg:
                hits.append((pc, fn(pc), neg[rs2][0]))
                if neg[rs2][1] is not None and neg[rs2][1] == rs1:   # cap + (-its own cursor): cursor -> ~0
                    own.append((pc, fn(pc), neg[rs2][0]))
        if op == 0x33 and f3 == 0 and f7 == 0x20 and rs1 == 0:      # sub rd, x0, rs  (neg)
            neg[rd] = (pc, copy.get(rs2))
            copy.pop(rd, None)
            continue
        if op == 0x13 and f3 == 0 and (w >> 20) == 0 and rs1 != 0:   # mv rd, rs (addi rd, rs, 0)
            copy[rd] = copy.get(rs1, rs1); neg.pop(rd, None); continue
        if op in (0x03, 0x13, 0x17, 0x1b, 0x33, 0x37, 0x3b, 0x5b, 0x67, 0x6f):
            neg.pop(rd, None); copy.pop(rd, None)
            for r in [r for r, v in copy.items() if v == rd]: copy.pop(r)   # the source capability was overwritten
    print(f"== {img.split('/')[-1]}: {n} instructions parsed, {cinc} CINCOFFSET, {len(hits)} by a negated register, "
          f"{len(own)} by the NEGATED OWN CURSOR (cap - its cursor)")
    for pc, f, at in own: print(f"   OWN {pc:#x} in {f} (neg at {at:#x})")
    if n == 0: rc = 2
sys.exit(rc)
