#!/usr/bin/env python3
"""Static detector for the R-18 trigger.

  usage: scan-r18-trigger.py <llvm-objdump> <triple> <dom>...
  e.g.   python3 scan-r18-trigger.py llvm/cmake-build-debug/bin/llvm-objdump \
                 capstone64-unknown-elf /tmp/capstone/ladder-fpga/*.dom

WHAT IT ANSWERS: which binaries can hit R-18 at all, without running anything. Measured
2026-08-08 across 30 ladder rungs: most have ZERO sites; beebs_crc32big has 5, the bl*/al
family 4 each. Both DOCUMENTED silicon miscompiles -- matmult_int (R-1) and beebs_recursion
-- have ZERO, which is why R-18 does not explain them and why the "the workaround will fix
R-1" hypothesis was dropped without spending a board boot on it.

READ THE ZERO CORRECTLY, because it was first read wrongly. matmult_int has NINE
`movc rd, zero` instructions; an early claim that it "contains no movc at all" came from a
`grep -c` that silently returned 0 -- the ugrep trap this project documents elsewhere. The
zero here does NOT mean "no capability ops". It means none of those shadowed registers is
ever used as the DATA OPERAND OF A STORE, which is what the trigger requires, and that was
re-verified by an independent hand-written check before the R-1 hypothesis was dropped.

LIMITS, so a zero is not over-read: this is a per-function linear last-writer scan with NO
control-flow analysis, so a capability-written register that reaches a store along a branch
edge is missed. A non-zero count is solid; a zero means "no straight-line site", not "safe".: an ordinary integer store whose DATA register was
last written by a capability-producing op, so it carries a non-zero capability metadata
shadow and the dcache misclassifies the store (wt_dcache_mem.sv:138).

Per the RTL, ANY GPR write updates the metadata shadow (issue_read_operands.sv:1663-1665,
under the integer write-enable), and a non-FLU writeback carries cap_result='0'
(scoreboard.sv:246) -- so an integer op SCRUBS the shadow. Only a capability-producing op
leaves it non-zero. That makes this a simple last-writer question per register."""
import re, subprocess, sys

CAP_OPS = {'movc','cincoffset','cincoffsetimm','scc','ldc','lcc','split','splitlo',
           'delin','capcreate','seal','unseal','shrink','shrinkto','tighten','mrev',
           'revoke','drop','init','sccsr','ccsrrw'}
INT_STORES = {'sb','sh','sw','sd'}

def scan(dom, objdump, triple):
    out = subprocess.run([objdump,'-d','--triple='+triple,dom],
                         capture_output=True,text=True).stdout
    shadow = {}          # reg -> instruction that last wrote it (cap op) or None
    hits, fn = [], '?'
    for line in out.splitlines():
        m = re.match(r'^[0-9a-f]+ <(.+)>:', line.strip())
        if m: fn = m.group(1); shadow.clear(); continue
        m = re.match(r'^\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(\S+)\s*(.*)$', line)
        if not m: continue
        addr, mnem, rest = m.group(1), m.group(2), m.group(3)
        ops = [o.strip() for o in rest.split(',')]
        if mnem in INT_STORES and ops:
            src = ops[0]
            if shadow.get(src):
                hits.append((fn, addr, mnem, rest.strip(), src, shadow[src]))
            continue
        if not ops: continue
        rd = ops[0]
        shadow[rd] = ('%s %s' % (mnem, rest.strip())) if mnem in CAP_OPS else None
    return hits

if __name__ == '__main__':
    objdump, triple = sys.argv[1], sys.argv[2]
    for dom in sys.argv[3:]:
        h = scan(dom, objdump, triple)
        name = dom.split('/')[-1]
        print("%-22s trigger sites: %d" % (name, len(h)))
        for fn,addr,mnem,rest,src,producer in h[:6]:
            print("    %-18s %s: %-4s %-24s  <- %s written by `%s`" % (fn,addr,mnem,rest,src,producer))
