#!/usr/bin/env python3
"""C-50 gate (docs/ref/ISSUES.md). Find the capstone64 codegen signature: an INTEGER add off sp/s0 whose result is then used
as a memory-access base within the next few instructions (addi rX, sp|s0, k ... sd/ld .., (rX)).
A capability frame pointer needs cincoffset; addi yields an untagged integer that faults as
a base. usage: scan-addi-sp.py <objdump -d output>"""
import re, sys
insn = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
func = re.compile(r'^[0-9a-f]+ <([^>]+)>:')
lines, cur = [], '?'
for l in open(sys.argv[1]):
    m = func.match(l)
    if m: cur = m.group(1); continue
    m = insn.match(l)
    if m: lines.append((cur, m.group(1), m.group(2), m.group(3)))
hits = []
for i, (fn, addr, op, ops) in enumerate(lines):
    m = re.match(r'(\w+),\s*(sp|s0|fp),\s*(-?(?:0x)?[0-9a-f]+)$', ops) if op == 'addi' else None
    if not m: continue
    rd = m.group(1)
    for fn2, a2, op2, ops2 in lines[i + 1:i + 8]:
        if fn2 != fn: break
        if re.search(r'\(%s\)$' % re.escape(rd), ops2) and op2 in ('sd','sw','sh','sb','ld','lw','lwu','lh','lhu','lb','lbu','stc','ldc'):
            hits.append((fn, addr, f'addi {ops}', a2, f'{op2} {ops2}')); break
        if re.match(r'%s\b' % re.escape(rd), ops2.split(',')[0] if ops2 else ''):
            break   # rd overwritten before any use as a base
for h in hits: print('HIT %-40s %s %-24s -> %s %s' % h)
if not lines:
    sys.exit(f'ERROR: no instructions parsed from {sys.argv[1]}')
print(f'{len(hits)} hit(s) in {len(lines)} instructions')
sys.exit(1 if hits else 0)
