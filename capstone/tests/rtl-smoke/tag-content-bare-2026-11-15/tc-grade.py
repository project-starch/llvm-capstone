#!/usr/bin/env python3
"""Grade one tag-content-bare-2026-11-15 boot against its PREREG (llvm-capstone 8c905539b551).
usage: tc-grade.py <uart.txt> <tag-content.bin as booted>   exit 0 PASS, 1 FAIL, 2 VOID, 3 P3-UNGRADED, 4 NO DATA"""
import sys, re, hashlib
IMG = '273e77e9da57f4d5c96d0fde4fbc762af58b493ee1e10a80d55edc55232d192a'
EXP = [0x400, 0x0, 0xffff, 0x0, 0x101, 0x4, 0x0]
MUT = [0x200, 0x200, 0x0, 0x1, 0x101, 0x4, 0x0]
ARM = lambda g: 'P0' if g < 256 else 'P1' if g < 512 else 'P2' if g < 768 else 'P3'
uart, binp = sys.argv[1:3]
t = open(uart, errors='replace').read()
sr = [int(x, 16) for x in re.findall(r'^SR ([0-9A-Fa-f]{16})\s*$', t, re.M)]
sv = [int(x, 16) for x in re.findall(r'^SV ([0-9A-Fa-f]{16})\s*$', t, re.M)]
h = hashlib.sha256(open(binp, 'rb').read()).hexdigest()
print('image', h[:16], 'match' if h == IMG else 'MISMATCH', '| SR', sr, '| SV', [hex(v) for v in sv])
if not sr and not sv: print('NO DATA: no SR/SV lines in', uart); sys.exit(4)
why = []
if h != IMG: why.append('image hash differs')
if sr != [7] or len(sv) != 7: why.append('SR count not 7')
if len(sv) == 7 and sv[6] != 0: why.append('traps %d' % sv[6])
if len(sv) == 7 and sv[4] != 0x101: why.append('ones read %#x != 0x101' % sv[4])
if why: print('VOID:', '; '.join(why)); sys.exit(2)
if sv[1] > 0:
    print('FAIL: %d wrong tag bytes; first granule %d (%s) byte %#x%s' % (sv[1], sv[2], ARM(sv[2]), sv[3],
          ' (= the mutant vector)' if sv == MUT else '')); sys.exit(1)
if sv[5] != 4:
    print('P3 UNGRADED: page fills %d != 4; readings 1-3:' % sv[5], [hex(v) for v in sv[:3]]); sys.exit(3)
if sv[:4] == EXP[:4]: print('PASS: readings 1-4 = expected; fills 4; traps 0'); sys.exit(0)
print('FAIL: readings 1-4', [hex(v) for v in sv[:4]], '!= expected'); sys.exit(1)
