#!/usr/bin/env python3
"""Scan a Capstone disassembly for the C-32 shape.

THE SHAPE, calibrated against the committed reproducer
llvm/test/CodeGen/Capstone/c32-movc-untagged-live.ll:

    <int-producing insn> rs, ...      # rs now holds an UNTAGGED, integer-bridged value
    ...
    movc  rd, rs                      # RTL nulls rs (it is not a NONLIN capability)
    ...
    <any read of rs> before rs is redefined   # reads 0 on silicon, correct under QEMU

The DEF SIDE is the discriminator, and it is what separates this from the looser
"movc rd, rs with rs read again before redefinition" shape. That looser shape ALSO
matches real_cap_copy -- the reproducer's own CONTROL -- where rs was defined by a
movc and so holds a genuine tagged capability that survives. Classifying the def is
what tells the defect from the control.

Registers are the merged file (ctN is xN), so a name is matched as written.

LIMITS, stated because they bound every number this prints:
  * Linear scan. Control flow is not followed; a branch between the movc and the
    read is not modelled, so a hit is a CANDIDATE and a miss is not a proof.
  * `unknown-def` means the definer was not found within the window (an incoming
    argument, or a def further back). Those are NOT counted as candidates.
"""
import re, sys, collections

# Destination is a genuine capability (tagged).
CAP_DEF = {'movc', 'ldc', 'cincoffset', 'cincoffsetimm', 'scc', 'cmove', 'auipcc',
           'cspecialrw', 'csetbounds', 'csetboundsimm', 'cgetpcc', 'candperm',
           'cseal', 'ccleartag', 'cbuildcap', 'csetaddr'}
# Destination is a plain integer (untagged by construction).
INT_DEF = {'mv', 'addi', 'li', 'lui', 'auipc', 'ld', 'lw', 'lwu', 'lb', 'lbu', 'lh', 'lhu',
           'slli', 'srli', 'srai', 'sll', 'srl', 'sra', 'and', 'andi', 'or', 'ori',
           'xor', 'xori', 'add', 'addw', 'addiw', 'sub', 'subw', 'sext.w', 'zext.w',
           'mul', 'mulw', 'div', 'divw', 'divu', 'divuw', 'rem', 'remw', 'remu', 'remuw',
           'neg', 'negw', 'not', 'seqz', 'snez', 'sltu', 'slt', 'slti', 'sltiu',
           'lcc', 'cgettag', 'cgetbase', 'cgetlen', 'cgetoffset', 'cgetaddr',
           'cgetperm', 'cgettype', 'slliw', 'srliw', 'sraiw', 'mulh', 'mulhu', 'rdcycle'}
# No destination register at all.
NO_DEF = {'sb', 'sh', 'sw', 'sd', 'stc', 'beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu',
          'beqz', 'bnez', 'blez', 'bgez', 'bltz', 'bgtz', 'ble', 'bgt', 'j', 'jr',
          'ret', 'ecall', 'ebreak', 'fence', 'fence.i', 'unimp', 'nop', 'cjr'}

INSN = re.compile(r'^\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*\t(\S+)(?:\s+(.*?))?\s*$')
SYM = re.compile(r'^([0-9a-f]+) <(.+)>:')
REG = re.compile(r'\b(zero|ra|sp|gp|tp|fp|t[0-6]|s1[01]|s[0-9]|a[0-7]|x[0-9]+|c[0-9]+)\b')


def strip_comment(s):
    return s.split('#')[0].strip() if s else ''


def regs(operands):
    return REG.findall(operands or '')


def parse(path):
    """-> list of (addr, mnem, operands, func)"""
    out, func = [], '?'
    for line in open(path, errors='ignore'):
        m = SYM.match(line)
        if m:
            # .L* are local labels, NOT function boundaries -- objdump splits on them.
            if not m.group(2).startswith('.L'):
                func = m.group(2)
            continue
        m = INSN.match(line)
        if m:
            out.append((int(m.group(1), 16), m.group(2), strip_comment(m.group(3)), func))
    return out


def defreg(mnem, ops):
    if mnem in NO_DEF:
        return None
    r = regs(ops)
    return r[0] if r else None


def usereg(mnem, ops):
    r = regs(ops)
    return r if mnem in NO_DEF else r[1:]


def scan(insns, window=400):
    hits = []
    for i, (addr, mnem, ops, func) in enumerate(insns):
        if mnem != 'movc':
            continue
        r = regs(ops)
        if len(r) < 2:
            continue
        rd, rs = r[0], r[1]
        if rd == rs:
            continue
        # --- def side: how was rs last defined?
        srcdef = None
        for j in range(i - 1, max(-1, i - window), -1):
            a2, m2, o2, f2 = insns[j]
            if f2 != func:
                break
            if defreg(m2, o2) == rs:
                srcdef = m2
                break
        if srcdef is None:
            cls = 'unknown-def'
        elif srcdef in INT_DEF:
            cls = 'INT-DEF'
        elif srcdef in CAP_DEF:
            cls = 'cap-def'
        else:
            cls = 'unclassified:' + srcdef
        # --- live side: is rs READ after the movc, before being redefined?
        live = None
        for j in range(i + 1, min(len(insns), i + window)):
            a2, m2, o2, f2 = insns[j]
            if f2 != func:
                break
            if rs in usereg(m2, o2):
                live = (a2, m2, o2)
                break
            if defreg(m2, o2) == rs:
                break
        if live:
            hits.append((addr, func, rd, rs, srcdef, cls, live))
    return hits


if __name__ == '__main__':
    verbose = '--quiet' not in sys.argv
    for path in [a for a in sys.argv[1:] if not a.startswith('--')]:
        insns = parse(path)
        hits = scan(insns)
        by = collections.Counter(h[5] for h in hits)
        cands = [h for h in hits if h[5] == 'INT-DEF']
        print(f"file: {path}")
        print(f"  instructions parsed : {len(insns)}")
        print(f"  movc total          : {sum(1 for i in insns if i[1] == 'movc')}")
        print(f"  movc w/ live source : {len(hits)}   <- the LOOSE shape (over-reports)")
        for k, v in by.most_common():
            print(f"      {v:6d}  {k}")
        print(f"  C-32 CANDIDATES (INT-DEF source, still live): {len(cands)}")
        if verbose:
            for h in cands[:40]:
                print(f"      {h[0]:#x} {h[1]}: movc {h[2]}, {h[3]}  (src def by '{h[4]}'), "
                      f"later read {h[6][0]:#x} '{h[6][1]} {h[6][2]}'")
            if len(cands) > 40:
                print(f"      ... {len(cands)-40} more")
        print()
