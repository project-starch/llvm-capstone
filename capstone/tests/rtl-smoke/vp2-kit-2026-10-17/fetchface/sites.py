#!/usr/bin/env python3
"""Per-function node-invalidating and derive sites by raw word (opcode 0x5b, funct3 1, funct7 0x00 REVOKE, 0x0b DROP,
0x52 CREVOKE, 0x53 CSRETIRE, 0x51 CDERIVE), function from the disassembler's <symbol>: headers. Exit 2 on no parse."""
import re, subprocess, sys, collections
OBJ, imgs = sys.argv[1], sys.argv[2:]
NAMES = {0x00: 'REVOKE', 0x0b: 'DROP', 0x52: 'CREVOKE', 0x53: 'CSRETIRE', 0x51: 'CDERIVE'}
rc = 0
RE = OBJ.replace('llvm-objdump', 'llvm-readelf')
for img in imgs:
    syms = sorted((int(q[1], 16), int(q[2]), q[7]) for q in (l.split() for l in subprocess.run(
        [RE, '-sW', img], capture_output=True, text=True).stdout.splitlines())
        if len(q) >= 8 and q[3] == 'FUNC' and q[1] != '0')
    def func(a):
        best = '?'
        for b, z, nm in syms:
            if b <= a < b + max(z, 1): best = nm
        return best
    n = 0; c = collections.Counter()
    for line in subprocess.run([OBJ, '-d', img], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})', line)
        if not m: continue
        n += 1; w = int.from_bytes(bytes.fromhex(m.group(2).replace(' ', '')), 'little')
        if w & 0x7f == 0x5b and (w >> 12) & 7 == 1 and (w >> 25) in NAMES:
            c[(NAMES[w >> 25], func(int(m.group(1), 16)))] += 1
    print(f"== {img}: {n} instructions parsed")
    if not n: rc = 2
    for (op, f), k in sorted(c.items()): print(f"   {op:<9} {k:>3}  {f}")
sys.exit(rc)
