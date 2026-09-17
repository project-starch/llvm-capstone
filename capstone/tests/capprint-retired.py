#!/usr/bin/env python3
# capprint-retired.py LOG.iss [...] -- CAPPRINT readings aligned to RETIREMENTS, never de-duplicated by heuristic.
#
# The RVFI trace (LOG, same stem as LOG.iss) records every retired instruction; a CAPPRINT is opcode 0x7B /
# funct3 0 / funct7 0x9 with the printed register in rs1, and a trap's handler ends in an mret (0x30200073).
# The .iss holds the $display output, which fires at EXECUTE on the FLU, so it also carries SPECULATIVE prints:
# instructions younger than a faulting DYN op that executed before the flush and re-executed after it (a
# window can hold several prints), and wrong-path prints after a branch mispredict.
#
# Alignment by EPOCH. Every retired print has an epoch = number of mrets retired before it; every raw print
# has an epoch = number of `Exception:` lines logged before its cycle. A real print's two epochs agree (an
# older, unflushed print retires before the trap; a re-executed print prints after it); a speculative print's
# raw epoch is SMALLER than its retirement's. So each retired print, in order, takes the earliest unmatched
# raw print of its register in its epoch at a cycle after the previous match. Leftovers are the speculative
# prints, each reported with what flushed it. When raw and retired counts already agree and the registers
# match in order there is nothing to resolve and the epochs are not needed (fixtures whose traps are ecalls).
# The tool REFUSES (exit 2) rather than print a plausible table when trace traps and logged exceptions
# disagree or any retired print finds no raw print. Replaced the "same register within 200 cycles" rule,
# which merged DISTINCT consecutive prints of one register (17 of 29 kept on r12-recl-stale, 2026-09-16), and
# the "same register next with an exception between" rule, which cannot see a two-print flush window.
import re, sys
ABI = {0:'x0',1:'ra',2:'sp',3:'gp',4:'tp',5:'t0',6:'t1',7:'t2',8:'s0',9:'s1',10:'a0',11:'a1',12:'a2',13:'a3',
       14:'a4',15:'a5',16:'a6',17:'a7',18:'s2',19:'s3',20:'s4',21:'s5',22:'s6',23:'s7',24:'s8',25:'s9',
       26:'s10',27:'s11',28:'t3',29:'t4',30:'t5',31:'t6'}
rc = 0
def fmt(k, v):
    if k == 'cap': return 'revnode_id=%d (gen %d, index %d)' % (v, v >> 16, v & 0xffff)
    if k == 'torn': return 'capability, value UNREADABLE (line torn by harness output)'
    return str(v)
for iss in sys.argv[1:]:
    log = iss[:-4] if iss.endswith('.iss') else iss + '.log'
    retired, traps = [], 0                                   # (pc, reg, epoch)
    for ln in open(log, errors='replace'):
        m = re.match(r'\s*\d+\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]{8})\)', ln)
        if not m: continue
        w = int(m.group(2), 16)
        if w == 0x30200073: traps += 1; continue
        if (w & 0x7f) == 0x7b and ((w >> 12) & 7) == 0 and (w >> 25) == 0x9:
            retired.append((int(m.group(1), 16), (w >> 15) & 31, traps))
    raw, excs = [], []                                       # (cycle, reg, kind, value); exception cycles
    for ln in open(iss, errors='replace'):
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Reg\[\s*(\d+)\]:\s*([0-9a-fA-F]+)\s*$', ln)
        if m: raw.append((int(m.group(1)), int(m.group(2)), 'int', int(m.group(3), 16))); continue
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Reg\[\s*(\d+)\]:\s*Cursor:.*Revnode_id\s*:\s*(\d+)', ln)
        if m: raw.append((int(m.group(1)), int(m.group(2)), 'cap', int(m.group(3)))); continue
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Reg\[\s*(\d+)\]:\s*Cursor:', ln)   # torn by harness stdout
        if m: raw.append((int(m.group(1)), int(m.group(2)), 'torn', None)); continue
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Exception:\s*(\S+)', ln)
        if m: excs.append((int(m.group(1)), m.group(2)))
    exc_cycles = [c for c, _ in excs]
    def raw_epoch(c): return sum(1 for e in exc_cycles if e < c)
    print('== %s  retired CAPPRINTs %d, raw prints %d, exceptions %d, trace traps %d' % (iss.split('/')[-1], len(retired), len(raw), len(excs), traps))
    out, spec = [], []
    if len(raw) == len(retired) and all(r[1] == p[1] for r, p in zip(raw, retired)):
        out = [(p[0], r[0], r[1], r[2], r[3]) for r, p in zip(raw, retired)]   # nothing to resolve
    elif traps != len(excs):
        print('  REFUSED: %d traps in the trace but %d exceptions logged -- epochs cannot be aligned' % (traps, len(excs))); rc = 2
    else:
        used = [False] * len(raw); last_c = -1
        for pc, r, ep in retired:
            j = next((k for k, (c, pr, kd, v) in enumerate(raw) if not used[k] and pr == r and c > last_c and raw_epoch(c) == ep), None)
            if j is None:
                print('  ALIGNMENT FAILED: retired print at pc %#x (%s, epoch %d) has no raw print' % (pc, ABI[r], ep)); rc = 2; break
            used[j] = True; last_c = raw[j][0]; out.append((pc, raw[j][0], r, raw[j][2], raw[j][3]))
        for k, (c, pr, kd, v) in enumerate(raw):
            if used[k]: continue
            e = next((ec for ec in exc_cycles if ec > c), None)
            re_c = next((o[1] for o in out if o[2] == pr and o[1] > c), None)
            why = ('flushed by the exception at %d, re-executed at %d' % (e, re_c)) if (e and re_c and e < re_c) else 'wrong-path (branch mispredict) or unmatched'
            spec.append((c, pr, kd, v, why))
    if rc != 2 and len(out) != len(retired): print('  ALIGNMENT INCOMPLETE'); rc = 2
    for pc, c, r, k, v in out: print('  %#010x  cyc %6d  %-3s  %s' % (pc, c, ABI[r], fmt(k, v)))
    for c, r, k, v, why in sorted(spec): print('  [speculative %s at cycle %d = %s: %s]' % (ABI[r], c, fmt(k, v), why))
    print('  exceptions at cycles: %s' % (', '.join('%d %s' % (c, n) for c, n in excs) or 'none'))
sys.exit(rc)
