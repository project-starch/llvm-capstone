#!/usr/bin/env python3
"""Scan for the R-20 shape across ALL capability ops, not just `stc`.

The compiler workaround only keeps x10 out of an `stc` base. But check_cap_op
(ariane_pkg.sv:902-912) also covers LDC, MOVC, CINCOFFSET, SCC, SHRINK, ... so in principle
ANY capability op with rs1 == x10 and rd != x10 loses the same clobber claim, and the shape is:

    <cap op with rs1 == x10, rd != x10>  ->  an instruction that WRITES x10  ->  a reader of x10

Raw-encoding decode; capability ops are opcode 0x5B (and 0x7B for the CAP* family).
STC is the S-type case (funct3 == 4) where the decoder sets rd = rs2.
"""
import struct, sys

def rd_of(w):
    op = w & 0x7f
    f3 = (w >> 12) & 7
    if op == 0x5b and f3 == 4:          # STC: rd = rs2 per decoder.sv
        return (w >> 20) & 0x1f
    return (w >> 7) & 0x1f

def writes_x10(w):
    op = w & 0x7f
    # branches (0x63) and stores (0x23) have no rd
    if op in (0x63, 0x23):
        return False
    if op == 0x5b and ((w >> 12) & 7) == 4:   # stc writes no GPR
        return False
    return ((w >> 7) & 0x1f) == 10

def reads_x10(w):
    return ((w >> 15) & 0x1f) == 10 or ((w >> 20) & 0x1f) == 10

def scan(path, window=4):
    b = open(path, 'rb').read()
    ws = []
    off = 0x1000
    while off <= len(b) - 4:
        ws.append(struct.unpack_from('<I', b, off)[0]); off += 4
    per_op = {}
    total = 0
    for i, w in enumerate(ws):
        op = w & 0x7f
        if op not in (0x5b, 0x7b):
            continue
        if ((w >> 15) & 0x1f) != 10:      # rs1 must be x10
            continue
        if rd_of(w) == 10:                # rd == x10 restores the claim
            continue
        for j in range(i + 1, min(i + 1 + window, len(ws))):
            if not writes_x10(ws[j]):
                continue
            for k in range(j + 1, min(j + 1 + window, len(ws))):
                if reads_x10(ws[k]):
                    key = (op, (w >> 12) & 7)
                    per_op[key] = per_op.get(key, 0) + 1
                    total += 1
                    break
            break
    return total, per_op

for p in sys.argv[1:]:
    t, per = scan(p)
    print(f'  {p.split("/")[-1]:24s} WIDE vulnerable shape = {t}')
    for (op, f3), n in sorted(per.items(), key=lambda x: -x[1])[:6]:
        print(f'      opcode {op:#04x} funct3 {f3}: {n}')
