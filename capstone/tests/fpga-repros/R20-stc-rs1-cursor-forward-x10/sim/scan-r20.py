#!/usr/bin/env python3
"""Count the R-20 vulnerable shape properly.

The shape is:  stc <v>, off(a0)   ->   an instruction that WRITES a0   ->   a reader of a0.
An earlier version of this scan only checked whether the NEXT instruction read a0, which missed
the common `stc a1,off(a0); ldc a0,0(a1); <reader>` form and reported zero for an image that
had plenty. Uses llvm-objdump TEXT (the disassembler decodes stc-with-base-a0 fine), and the
count is cross-checked against a raw-encoding count of `stc` with rs1==x10.
"""
import re, subprocess, sys, struct

OD = 'llvm/cmake-build-debug/bin/llvm-objdump'

def raw_stc_a0(path):
    b = open(path, 'rb').read(); n = 0; off = 0x1000
    while off <= len(b) - 4:
        w = struct.unpack_from('<I', b, off)[0]
        if (w & 0x7f) == 0x5b and ((w >> 12) & 7) == 4 and ((w >> 15) & 0x1f) == 10:
            n += 1
        off += 4
    return n

def scan(path, window=6):
    d = subprocess.run([OD, '-d', '--triple=capstone64-unknown-elf', path],
                       capture_output=True, text=True).stdout
    ins = []
    for l in d.split('\n'):
        m = re.match(r'^\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*\t(.*)$', l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2).strip()))
    stc_a0 = vuln = 0
    sites = []
    for i, (a, t) in enumerate(ins):
        m = re.match(r'stc\s+(\S+),\s*(\S+)\(a0\)\s*$', t)
        if not m:
            continue
        stc_a0 += 1
        # walk forward for an instruction that WRITES a0, then one that READS a0
        for j in range(i + 1, min(i + 1 + window, len(ins))):
            tj = ins[j][1]
            writes_a0 = re.match(r'\S+\s+a0\s*,', tj) is not None
            if not writes_a0:
                continue
            for k in range(j + 1, min(j + 1 + window, len(ins))):
                if re.search(r'\ba0\b', ins[k][1]):
                    vuln += 1
                    sites.append(a)
                    break
            break
    return stc_a0, vuln, sites, raw_stc_a0(path)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        s, v, sites, raw = scan(p)
        flag = '' if s == raw else f'   <-- TEXT/RAW MISMATCH ({s} vs {raw}); instrument suspect'
        print(f'  {p.split("/")[-1]:28s} stc-base-a0={s:5d} (raw {raw:5d}){flag}   VULNERABLE={v:5d}')
        if sites[:5]:
            print(f'      first sites: {" ".join(hex(x) for x in sites[:5])}')
