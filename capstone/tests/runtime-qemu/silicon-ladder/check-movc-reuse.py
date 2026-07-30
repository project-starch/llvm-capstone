#!/usr/bin/env python3
"""Find the C-14 codegen pattern: a capability register reused as a scalar.

BOARD-ISOLATED 2026-07-30, one variable, one session:

    beebs_primer1  control                                   PASS
    gpw16b         64 B, 16 elements, no reuse               PASS (583391941, exact)
    gpw16          64 B, 16 elements, WITH reuse             FAIL

Same size, same element count, same access shape. The only difference is that gpw16's
store loop reuses the address register as the loop counter:

    addiw a6, a4, 1
    slli/srli a4              ; index*4
    cincoffset a4, a3, a4     ; a4 is now a CAPABILITY (&g[i])
    sw    a6, 0(a4)
    movc  a4, a6              ; a SCALAR written over that same register
    bne   a6, a5, back        ; next iteration consumes a4 as an integer

gpsz keeps the index in its own integer register and passes at 64 elements. The compiler
emits the reuse form when the stored value is derived from the index (g[i] = i+1) and
not otherwise (g[i] = i*3+7).

Why it is silicon-only: on CVA6 the capability metadata lives in a SEPARATE shadow
register file, so a register can carry capability state that a scalar write does not
clear. QEMU keeps one unified value per register, so this pattern is invisible there --
the same structural blind spot as the DELIN and bounds-compression divergences.

This scans a linked domain for the pattern so a codegen fix can be verified, and so a
new benchmark can be checked before it costs a board session.

Exit 1 if any occurrence is found.
"""
import re
import subprocess
import sys

CAP_DEF = re.compile(r'^\s*[0-9a-f]+:\s+(cincoffset|cincoffsetimm|ldc|scc|split|movc)\s+(\w+)')
MOVC = re.compile(r'^\s*[0-9a-f]+:\s+movc\s+(\w+),\s*(\w+)')
INT_USE = re.compile(r'^\s*[0-9a-f]+:\s+'
                     r'(slli|srli|srai|addi|addiw|add|addw|sub|subw|xor|or|and|mul|'
                     r'sll|srl|beq|bne|blt|bge|bltu|bgeu)\w*\s+(.*)$')
INSN = re.compile(r'^\s*[0-9a-f]+:\s+(\S+)\s*(.*)$')


def scan(path, window=8):
    try:
        out = subprocess.run(['llvm-objdump', '-d', '--no-show-raw-insn', path],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print('%-28s ERROR %s' % (path.split('/')[-1], exc))
        return []

    lines = out.splitlines()
    cap_regs = {}          # reg -> index of the instruction that made it a capability
    hits = []
    for i, ln in enumerate(lines):
        m = CAP_DEF.match(ln)
        if m and m.group(1) != 'movc':
            cap_regs[m.group(2).rstrip(',')] = i

        mv = MOVC.match(ln)
        if mv:
            rd, rs = mv.group(1).rstrip(','), mv.group(2)
            # Two qualifiers, both learned from the gpw16-vs-gpsz pair:
            #  - `movc rd, zero` is loop INITIALISATION, not reuse of a live capability.
            #    gpsz does exactly that (`movc t1, zero`) and PASSES on the board.
            #  - the capability definition must be RECENT. The failing shape has
            #    `cincoffset a4, a3, a4` two instructions before `movc a4, a6`, inside
            #    one loop body; a definition far above is almost certainly a different
            #    basic block, and this scanner does not track control flow.
            recent = rd in cap_regs and (i - cap_regs[rd]) <= window
            if rs != 'zero' and recent:
                # rd held a capability and is now written by movc. Does anything in the
                # next few instructions consume it as an INTEGER before it is redefined
                # as a capability again?
                for j in range(i + 1, min(i + 1 + window, len(lines))):
                    nxt = lines[j]
                    cm = CAP_DEF.match(nxt)
                    if cm and cm.group(2).rstrip(',') == rd and cm.group(1) != 'movc':
                        break                      # redefined as a capability: fine
                    iu = INT_USE.match(nxt)
                    if iu and re.search(r'\b%s\b' % re.escape(rd), iu.group(2)):
                        hits.append((ln.strip(), nxt.strip()))
                        break
            if rd in cap_regs:
                del cap_regs[rd]
            continue

        # any other definition of a register clears its capability status
        gm = INSN.match(ln)
        if gm and gm.group(2):
            first = gm.group(2).split(',')[0].strip()
            if first in cap_regs and not CAP_DEF.match(ln):
                del cap_regs[first]
    return hits


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: check-movc-reuse.py <domain.elf> [...]')
    bad = 0
    for p in sys.argv[1:]:
        hits = scan(p)
        print('%-28s C-14 capability-register reuse: %d' % (p.split('/')[-1], len(hits)))
        for a, b in hits[:6]:
            print('    %-40s -> %s' % (a, b))
        if hits:
            bad += 1
    sys.exit(1 if bad else 0)

# ACCURACY AGAINST KNOWN BOARD OUTCOMES (2026-07-30) -- 5 of 8. Read before trusting it:
#
#   gpw16          FAIL  detected 1   correct
#   gpw16b         PASS  detected 0   correct
#   gpsz           PASS  detected 0   correct
#   gpcp           PASS  detected 0   correct
#   beebs_primer1  PASS  detected 0   correct
#   gpn2           FAIL  detected 0   MISS
#   gpn4           FAIL  detected 0   MISS
#   gpw2           ran   detected 1   FALSE POSITIVE
#
# So this pattern is NOT the whole story. The gpw16-vs-gpw16b pair is a genuine
# one-variable, same-session result and the pattern is real, but it does not predict
# gpn2/gpn4 (which fail without it) or gpw2 (which has it and did not wedge). Treat this
# as a lead and a fix-verification aid, not as the C-14 root cause. The scanner also does
# not track control flow, so "recent" is approximated by an instruction window.
