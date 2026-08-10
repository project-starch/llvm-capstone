#!/usr/bin/env python3
"""Residual R-20 exposure, restricted to ops that can actually deliver the wrong VALUE.

Line 568 drops x10's clobber claim for ANY non-CAPENTER capability op with rs1 == x10 and
rd != x10 -- but the wrong value is supplied by the rs1-cursor forwarding mux, which is gated
by check_fwd_rs1 (ariane_pkg.sv:929-935) = {SPLIT, MOVC, CJALR, CCSRRW, STC}. LDC and
CINCOFFSET are NOT in that set, so they can lose the stall without corrupting a value.

Encodings (asm_insn.h):  opcode 0x5B
  SPLIT  funct3 1 funct7 0x6      MOVC   funct3 1 funct7 0xa
  CJALR  funct3 5 (I-type)        CCSRRW funct3 7 (I-type)      STC funct3 4 (S-type)
"""
import struct, sys

def kind(w):
    if (w & 0x7f) != 0x5b: return None
    f3 = (w >> 12) & 7
    f7 = (w >> 25) & 0x7f
    if f3 == 1 and f7 == 0x6:  return 'SPLIT'
    if f3 == 1 and f7 == 0xa:  return 'MOVC'
    if f3 == 5:                return 'CJALR'
    if f3 == 7:                return 'CCSRRW'
    if f3 == 4:                return 'STC'
    return None

def rd_of(w):
    return ((w >> 20) & 0x1f) if ((w & 0x7f) == 0x5b and ((w >> 12) & 7) == 4) else ((w >> 7) & 0x1f)

def writes_x10(w):
    op = w & 0x7f
    if op in (0x63, 0x23): return False
    if op == 0x5b and ((w >> 12) & 7) == 4: return False
    return ((w >> 7) & 0x1f) == 10

def reads_x10(w):
    return ((w >> 15) & 0x1f) == 10 or ((w >> 20) & 0x1f) == 10

def scan(path, window=4):
    b = open(path, 'rb').read(); ws = []; off = 0x1000
    while off <= len(b) - 4:
        ws.append(struct.unpack_from('<I', b, off)[0]); off += 4
    hits = {}
    sites = []
    for i, w in enumerate(ws):
        k = kind(w)
        if not k: continue
        if ((w >> 15) & 0x1f) != 10 or rd_of(w) == 10: continue
        for j in range(i + 1, min(i + 1 + window, len(ws))):
            if not writes_x10(ws[j]): continue
            for m in range(j + 1, min(j + 1 + window, len(ws))):
                if reads_x10(ws[m]):
                    hits[k] = hits.get(k, 0) + 1
                    sites.append((i * 4 + 0x10000, k))
                    break
            break
    return hits, sites

for p in sys.argv[1:]:
    h, s = scan(p)
    tot = sum(h.values())
    print(f'  {p.split("/")[-1]:22s} value-corrupting shape = {tot}   {h if h else ""}')
    for va, k in s[:6]:
        print(f'      {k:7s} @ {va:#x}')
