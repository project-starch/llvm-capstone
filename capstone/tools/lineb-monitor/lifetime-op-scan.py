#!/usr/bin/env python3
"""Scan binaries for Capstone lifetime-op encodings (opcode 0x5b, funct3 1) at EVERY 2-byte alignment, by funct7 and
the operand fields each op fixes. usage: lifetime-op-scan.py <file> [<file>...]. Prints per file and op the count and
the first offsets. One decoder for every binary."""
import sys, struct
OPS = {0x00: ('REVOKE', 'rd0rs2_0'), 0x08: ('MREV', 'rs2_0'), 0x0b: ('DROP', 'rd0rs2_0'),
       0x51: ('CDERIVE', ''), 0x52: ('CREVOKE', 'rd0'), 0x54: ('CSNODE', ''), 0x09: ('INIT', '')}
def ok(w, rule):
    rd, rs2 = (w >> 7) & 31, (w >> 20) & 31
    return {'rd0rs2_0': rd == 0 and rs2 == 0, 'rs2_0': rs2 == 0, 'rd0': rd == 0, '': True}[rule]
for p in sys.argv[1:]:
    b = open(p, 'rb').read(); hits = {}
    for i in range(0, len(b) - 3, 2):
        w = struct.unpack_from('<I', b, i)[0]
        if w & 0x7f == 0x5b and (w >> 12) & 7 == 1 and (w >> 25) in OPS:
            name, rule = OPS[w >> 25]
            if ok(w, rule): hits.setdefault(name, []).append(i)
    print(f"== {p} ({len(b)} bytes)")
    for name in ('REVOKE', 'MREV', 'DROP', 'CDERIVE', 'CREVOKE', 'CSNODE', 'INIT'):
        v = hits.get(name, [])
        print(f"   {name:8s} {len(v):6d}  {[hex(x) for x in v[:5]]}")
