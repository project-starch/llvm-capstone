#!/usr/bin/env python3
# capprint-retired.py LOG.iss -- (capstone/tests; used by the R-12 reclaimer fixtures) CAPPRINT readings aligned to RETIREMENTS, not de-duplicated by heuristic.
# The RVFI trace (LOG, same stem) records every retired instruction; a CAPPRINT is opcode 0x7B / funct3 0 /
# funct7 0x9 with the printed register in rs1. The .iss holds the $display output, which fires at EXECUTE
# on the FLU, so it also carries SPECULATIVE prints: (a) a print re-executed after a late exception flush
# -- same register twice with an Exception line between; (b) a wrong-path print after a branch
# mispredict -- a register the retired sequence does not expect there. Both are skipped by aligning the
# raw sequence to the retired one; the tool REFUSES (exit 2) if the alignment does not consume every raw
# print, rather than printing a plausible table. capprint-last.py's "same register within 200 cycles"
# rule merged DISTINCT consecutive prints of one register (r12-recl-stale prints a3 four times running).
import re, sys
ABI = {0:'x0',1:'ra',2:'sp',3:'gp',4:'tp',5:'t0',6:'t1',7:'t2',8:'s0',9:'s1',10:'a0',11:'a1',12:'a2',13:'a3',
       14:'a4',15:'a5',16:'a6',17:'a7',18:'s2',19:'s3',20:'s4',21:'s5',22:'s6',23:'s7',24:'s8',25:'s9',
       26:'s10',27:'s11',28:'t3',29:'t4',30:'t5',31:'t6'}
rc = 0
for iss in sys.argv[1:]:
    log = iss[:-4] if iss.endswith('.iss') else iss + '.log'
    retired = []  # (pc, reg)
    for ln in open(log, errors='replace'):
        m = re.match(r'\s*\d+\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]{8})\)', ln)
        if not m: continue
        w = int(m.group(2), 16)
        if (w & 0x7f) == 0x7b and ((w >> 12) & 7) == 0 and (w >> 25) == 0x9:
            retired.append((int(m.group(1), 16), (w >> 15) & 31))
    raw = []      # (cycle, reg, kind, value) and exceptions as (cycle, None, 'exc', name)
    for ln in open(iss, errors='replace'):
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Reg\[\s*(\d+)\]:\s*([0-9a-fA-F]+)\s*$', ln)
        if m: raw.append((int(m.group(1)), int(m.group(2)), 'int', int(m.group(3), 16))); continue
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Reg\[\s*(\d+)\]:\s*Cursor:.*Revnode_id\s*:\s*(\d+)', ln)
        if m: raw.append((int(m.group(1)), int(m.group(2)), 'cap', int(m.group(3)))); continue
        # a capability print whose line the harness's own stdout tore through (seen: the final
        # '*** SUCCESS ***' message spliced into a Reg[11] line at cycle 19845): the print happened, its
        # value is unreadable. Keep it in the alignment as UNREADABLE rather than losing the retirement.
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Reg\[\s*(\d+)\]:\s*Cursor:', ln)
        if m: raw.append((int(m.group(1)), int(m.group(2)), 'torn', None)); continue
        m = re.search(r'\[Cycle\s*(\d+)\]\s*Exception:\s*(\S+)', ln)
        if m: raw.append((int(m.group(1)), None, 'exc', m.group(2)))
    prints = [x for x in raw if x[2] != 'exc']
    excs = [x[0] for x in raw if x[2] == 'exc']
    print('== %s  retired CAPPRINTs %d, raw prints %d, exceptions %d' % (iss.split('/')[-1], len(retired), len(prints), len(excs)))
    out, skipped, j = [], [], 0
    for i, (pc, r) in enumerate(retired):
        while j < len(prints):
            c, pr, k, v = prints[j]
            if pr != r:
                skipped.append((c, pr, 'wrong-path (retired sequence expects %s)' % ABI[r])); j += 1; continue
            nxt = prints[j + 1] if j + 1 < len(prints) else None
            if nxt and nxt[1] == r and any(c < e < nxt[0] for e in excs):
                skipped.append((c, pr, 'flushed by the exception at %d, re-executed at %d' % (next(e for e in excs if c < e < nxt[0]), nxt[0]))); j += 1; continue
            out.append((pc, c, r, k, v)); j += 1; break
        else:
            print('  ALIGNMENT FAILED: retired print #%d (pc %#x, %s) has no raw print left' % (i + 1, pc, ABI[r])); rc = 2; break
    leftover = prints[j:]
    for pc, c, r, k, v in out:
        print('  %#010x  cyc %6d  %-3s  %s' % (pc, c, ABI[r], ('revnode_id=%d (gen %d, index %d)' % (v, v >> 16, v & 0xffff)) if k == 'cap' else ('capability, value UNREADABLE (line torn by harness output)' if k == 'torn' else str(v))))
    for c, r, why in skipped: print('  [skipped speculative print of %s at cycle %d: %s]' % (ABI[r], c, why))
    for c, r, k, v in leftover: print('  UNCONSUMED raw print of %s at cycle %d = %s -- alignment is NOT trusted' % (ABI[r], c, v)); rc = 2
    if len(out) != len(retired): rc = 2
    print('  exceptions at cycles: %s' % (', '.join(map(str, excs)) or 'none'))
sys.exit(rc)
