#!/usr/bin/env python3
"""C-50 gate (docs/ref/ISSUES.md): find an INTEGER address formed from a capability stack
pointer and then used as a memory-access base.

    usage: scan-addi-sp.py <llvm-objdump -d --no-show-raw-insn output>

A capability stack/frame pointer must be offset with cincoffset[imm]. `addi rX, sp, k` or
`add rX, sp, rY` yields an untagged integer, and a load/store through rX faults (cause 24).
That is what the compiler emitted for C-50.

HOW IT TRACKS, so the gate cannot be dodged by instruction layout (three layouts that
dodged the first version, found by audit on 2026-09-23, are the negative tests):
  * Capability frame registers are sp always, and s0/fp only after the function makes it
    one (`movc s0, sp` or `cincoffsetimm s0, ...`). At -O1, s0 is often an ordinary
    integer register, and integer adds off it are then legitimate.
  * TAINT: `addi rX, <capframe>, k` or `add rX, <capframe>, rY` (either order) taints rX.
    Taint propagates through `mv`/`addi`/`add` whose source is tainted.
  * A HIT is any load/store (including ldc/stc) whose base register is tainted.
  * A register loses its taint only when an instruction WRITES it. Stores write nothing:
    their first operand is a source. Only real function symbols end a function; `.L`
    local labels do not.
Exit status: 1 on any hit, and 1 (with ERROR) when no instructions were parsed.
"""
import re
import sys

INSN = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
FUNC = re.compile(r'^[0-9a-f]+ <([^>]+)>:')
MEM = re.compile(r'(-?(?:0x)?[0-9a-fA-F]+)?\((\w+)\)$')
LOADS = {'ld', 'lw', 'lwu', 'lh', 'lhu', 'lb', 'lbu', 'ldc', 'flw', 'fld', 'lr.w', 'lr.d'}
STORES = {'sd', 'sw', 'sh', 'sb', 'stc', 'fsw', 'fsd'}


def regs(ops):
    return [o.strip() for o in ops.split(',')] if ops else []


def scan(path):
    hits, n = [], 0
    fn, capframe, taint = '?', {'sp'}, {}
    for line in open(path, errors='replace'):
        m = FUNC.match(line)
        if m:
            if not m.group(1).startswith('.L') and not m.group(1).startswith('$'):
                fn, capframe, taint = m.group(1), {'sp'}, {}
            continue
        m = INSN.match(line)
        if not m:
            continue
        n += 1
        addr, op, ops = m.group(1), m.group(2), m.group(3)
        r = regs(ops)
        if op in LOADS or op in STORES:
            mm = MEM.search(ops)
            if mm and mm.group(2) in taint:
                hits.append((fn, taint[mm.group(2)], f'{addr}: {op} {ops}'))
            if op in STORES:
                continue                      # a store writes no register
        if not r:
            continue
        rd, src = r[0], r[1:]
        if op == 'movc' and rd in ('s0', 'fp') and src[:1] == ['sp']:
            capframe.add(rd); taint.pop(rd, None); continue
        if op == 'cincoffsetimm' and rd in ('s0', 'fp') and src[:1] and src[0] in capframe:
            capframe.add(rd); taint.pop(rd, None); continue
        new = None
        if op in ('addi', 'add') and any(s in capframe for s in src[:2]):
            new = f'{addr}: {op} {ops}'
        elif op in ('mv', 'addi', 'add') and any(s in taint for s in src[:2]):
            new = next(taint[s] for s in src[:2] if s in taint)
        if rd in capframe and rd != 'sp':
            capframe.discard(rd)              # s0 overwritten: no longer a capability frame
        if new:
            taint[rd] = new
        else:
            taint.pop(rd, None)
    return hits, n


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    hits, n = scan(sys.argv[1])
    if n == 0:
        sys.exit(f'ERROR: no instructions parsed from {sys.argv[1]}')
    for fn, origin, use in hits:
        print(f'HIT {fn:<40} {origin:<32} -> {use}')
    print(f'{len(hits)} hit(s) in {n} instructions')
    sys.exit(1 if hits else 0)


if __name__ == '__main__':
    main()
