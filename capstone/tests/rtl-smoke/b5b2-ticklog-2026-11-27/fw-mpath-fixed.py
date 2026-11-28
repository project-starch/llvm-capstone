#!/usr/bin/env python3
"""fw-mpath-fixed.py FW_ELF: the M-path instructions of a machine-timer interrupt from U that the firmware wrapper's
two minstret reads do not see (runtime lane, 2026-11-27, Amendment 1). Walks the disassembly:
- _trap_handler from its first instruction, the from-M branch not taken, through the jal to sbi_trap_handler, then from
  the instruction after the jal to its mret, every "beq t0,t1" (the restore's register-unchanged test) taken;
- sbi_trap_handler from its entry to its first "csrr ...,minstret" (excluded), and from its second minstret read to
  its ret along the machine-timer branch (the beq on the mcause compare taken), the second read itself included.
Prints the three counts and their sum. A path that leaves the walked function, or a missing landmark, is an error."""
import re, subprocess, sys

elf = sys.argv[1]


def dis(fn):
    out = subprocess.run(['riscv64-unknown-elf-objdump', '-d', '--no-show-raw-insn', f'--disassemble={fn}', elf],
                         capture_output=True, text=True).stdout
    ins = []
    for l in out.splitlines():
        m = re.match(r'^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$', l)
        if m:
            ins.append((int(m[1], 16), m[2], m[3]))
    if not ins:
        sys.exit(f'fw-mpath-fixed: no {fn} in {elf}')
    return ins


def target(ops):
    m = re.search(r'\b([0-9a-f]{8})\b', ops)
    return int(m[1], 16) if m else None


def walk(ins, start, stop, taken):
    at = {a: i for i, (a, _, _) in enumerate(ins)}
    i, n = at[start], 0
    while True:
        a, op, ops = ins[i]
        n += 1
        if stop(a, op, ops):
            return n
        if op in ('j',) or (op.startswith('b') and taken(a, op, ops)):
            t = target(ops)
            if t not in at:
                sys.exit(f'fw-mpath-fixed: branch at {a:#x} leaves the walk')
            i = at[t]
            continue
        i += 1


th = dis('_trap_handler')
jal = [a for a, op, ops in th if op == 'jal' and 'sbi_trap_handler' in ops]
if len(jal) != 1:
    sys.exit(f'fw-mpath-fixed: {len(jal)} jal to sbi_trap_handler in _trap_handler')
ent = walk(th, th[0][0], lambda a, op, ops: a == jal[0], lambda a, op, ops: False)
after = [a for a, _, _ in th if a > jal[0]][0]
ext = walk(th, after, lambda a, op, ops: op == 'mret', lambda a, op, ops: ops.startswith('t0,t1'))
w = dis('sbi_trap_handler')
reads = [a for a, op, ops in w if op == 'csrr' and ops.endswith('minstret')]
if len(reads) != 2:
    sys.exit(f'fw-mpath-fixed: {len(reads)} minstret reads in sbi_trap_handler')
pro = walk(w, w[0][0], lambda a, op, ops: a == reads[0], lambda a, op, ops: False) - 1
epi = walk(w, reads[1], lambda a, op, ops: op == 'ret', lambda a, op, ops: op == 'beq')
print(f'asm entry {ent} + asm exit {ext} + wrapper before its first read {pro} + wrapper from its second read {epi} '
      f'= {ent + ext + pro + epi}')
