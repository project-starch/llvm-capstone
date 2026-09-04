#!/usr/bin/env python3
"""Find capability uses whose operand had its tag stripped by a scalar move.

`mv rd, rs` is ADDI rd, rs, 0. On this machine that clears rd's tag -- which is
exactly what an integer copy wants, and a disaster if rd is then used where a
TAGGED capability is required. (It used to be spelled PseudoSCALAR_COPY_I128 and
PseudoTRUNC_CAP; those are gone with the i128 capability carrier, but copyPhysReg
still emits a bare ADDI for every integer copy, so the shape is unchanged.) The model asserts on
it (helper_csdelin's `rd_v->tag`) and silicon faults.

This is the failure class that lit CANNOT see: every .ll test passed while every
capability passed through inline asm came out untagged. It is found by reading
the binary, so this reads the binary.

    llvm-objdump -d prog.dom > d.txt && scan-tag-stripped-caps.py d.txt

Exit status is 1 when anything is flagged, so it can gate a build.

CONTROLS, because a scanner that never fires proves nothing. Verified on the
CoreMark domain built while the bitcast selection was wrong: 8 hits, each a
`mv rX, rX` immediately before `delin rX`. Zero on the same domain once the
bitcast became a plain COPY. Re-run both arms after any change to copyPhysReg or
to how a capability is selected -- a clean scan means nothing until this is done.

ponytail: the scan is per basic block and per register, and it gives up on a
register as soon as anything redefines it. That misses a strip whose use is in
another block. Widening it needs real dataflow; this catches the shape that has
actually occurred.
"""
import re, sys
# A `mv rd, rs` (addi rd, rs, 0) clears the tag. Flag it when rd is then used
# where a TAGGED capability is required, before rd is redefined.
CAP_USE = {                     # opcode -> operand indices that must be tagged
 'delin':[0], 'scc':[1], 'cincoffset':[1], 'cincoffsetimm':[1], 'movc':[1],
 # SHRINK is 0, NOT 1. It is `Constraints = "$rd = $cap_in"` and prints as
 # `shrink $rd, $rs1, $rs2`, so the capability is the FIRST printed operand and
 # $rs1/$rs2 are plain integer bounds. Checking 1 meant checking a register that
 # can never legitimately be tagged: eight `mv` into a bounds register were
 # reported as tag strips, and the real operand was never checked at all.
 'shrink':[0],
 'tighten':[1], 'seal':[1], 'init':[1], 'lcc':[1], 'mrev':[1],
 'revoke':[0], 'ldc':[1], 'stc':[1], 'cjalr':[1],
 # EVERY memory access on this target goes through a capability base, not just
 # ldc/stc. Leaving these out is what made the scan come back clean while the
 # domain faulted with "Cap mem access requires capability" on a plain `ld`.
 'ld':[1], 'lw':[1], 'lwu':[1], 'lh':[1], 'lhu':[1], 'lb':[1], 'lbu':[1],
 'sd':[1], 'sw':[1], 'sh':[1], 'sb':[1],
 'fld':[1], 'flw':[1], 'fsd':[1], 'fsw':[1],
}
def ops(text):
    text = re.sub(r'#.*$', '', text)
    return [o.strip() for o in text.split(',')] if text else []
hits = 0
for path in sys.argv[1:]:
    cur = None
    pend = {}                   # reg -> (line no, text) awaiting a capability use
    for n, l in enumerate(open(path, errors='replace')):
        m = re.match(r'^[0-9a-f]+ <([^>]+)>:', l)
        if m: cur = m.group(1); pend.clear(); continue
        m2 = re.match(r'^\s*[0-9a-f]+:(?:\s+[0-9a-f]{2})+\s+(.*)$', l)
        if not m2: continue
        ins = m2.group(1).strip()
        p = re.split(r'\s+', ins, maxsplit=1)
        if not p: continue
        op, rest = p[0], (p[1] if len(p) > 1 else '')
        a = ops(rest)
        if op == 'mv' and len(a) == 2:
            pend[a[0]] = (n + 1, ins); continue
        if op in CAP_USE:
            for i in CAP_USE[op]:
                # memory operands look like `0x10(a3)`
                if i < len(a):
                    r = re.sub(r'.*\(|\).*', '', a[i])
                    if r in pend:
                        ln, mvtext = pend[r]
                        print(f"{path}:{ln}  {cur}: `{mvtext}` then line {n+1} `{ins}`")
                        hits += 1
        # A CALL redefines every caller-saved register, so nothing pending
        # survives it. Without this the scanner reads an ARGUMENT set up before a
        # call and a RETURN VALUE used after it as the same register -- one false
        # positive in CoreMark, `mv a0, a3` before a cjalr paired with the
        # `movc a1, a0` that copies the result out. A tool that cries wolf gets
        # ignored, which costs the same as one that never fires. The call's own
        # target operand is checked above, before this clears.
        if op in ('cjalr', 'jalr', 'jal'):
            pend.clear(); continue
        for r in list(pend):    # any redefinition clears the pending state
            if a and a[0] == r and op != 'mv': del pend[r]
print(f"-- {hits} tag-stripped capability uses")
sys.exit(1 if hits else 0)
