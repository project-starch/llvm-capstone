#!/usr/bin/env python3
"""Find the C-14 codegen pattern: one register used as BOTH a capability and a scalar
inside a single loop body.

BOARD EVIDENCE (2026-07-30, every run with a control in the same session). Sorting each
measured rung by presence of this pattern reproduces every outcome:

    beebs_primer1  no pattern   PASS  582955588 (exact)
    gpsz           no pattern   PASS  607423941 (exact)
    gpcp           no pattern   PASS   23404485 (exact)
    gpw16b         no pattern   PASS  583391941 (exact)
    gpw2           PATTERN      WRONG DATA (3950255460, expected 3983810698)
    gpw4/8/16      PATTERN      wedge
    gpn1use0       PATTERN      wedge
    gpn2/gpn4      PATTERN      wedge
    gpn2use0/1     PATTERN      wedge

The failing shape, from gpn2's domain_main:

    203ac: addiw      a6, a4, 1
    203b0: slli       a4, a4, 0x20
    203b4: srli       a4, a4, 0x1e     ; a4 = index*4  (INTEGER use)
    203b8: cincoffset a4, a3, a4       ; a4 := capability &g[i]   (CAPABILITY def)
    203bc: sw         a6, 0(a4)
    203c0: movc       a4, a6           ; a live SCALAR written over a4
    203c4: bne        a6, a5, 203ac    ; back-edge

gpw16b/gpsz compile the same algorithm with the index in its own integer register (the
capability lands in a7, the counter stays in a4) and pass. The compiler only produces the
aliased form when the stored value is derived from the loop index.

WHY LOOP-SCOPED. An earlier version of this scanner walked a forward instruction window
and scored only 5/8 -- it missed gpn2 and gpn4 because the INTEGER use sits *above* the
movc and is reached only through the back-edge. Scoping to the loop body fixes that. That
miss is worth remembering: it briefly looked like counter-evidence against a hypothesis
that was in fact correct.

Note `cincoffset rd, cap, rd` alone is NOT the tell -- gpsz has exactly that on a7 and
passes. The tell is that the SAME register is also written with a live scalar in the same
loop and then consumed as an integer.

Silicon-only by construction: CVA6 keeps capability metadata in a separate shadow
register file, whereas QEMU keeps one unified value per register, so every one of these
rungs is QEMU-green.

Exit 1 if any occurrence is found.
"""
import re
import subprocess
import sys

CAP_DEF = re.compile(r'^\s*([0-9a-f]+):\s+(cincoffset|cincoffsetimm|ldc|scc|split)\s+(\w+)')
MOVC = re.compile(r'^\s*([0-9a-f]+):\s+movc\s+(\w+),\s*(\w+)')
BRANCH = re.compile(r'^\s*([0-9a-f]+):\s+(?:beq|bne|blt|bge|bltu|bgeu|j)\w*\s+.*?0x([0-9a-f]+)')
INT_USE = re.compile(r'^\s*([0-9a-f]+):\s+'
                     r'(?:slli|srli|srai|addi|addiw|add|addw|sub|subw|xor|xori|or|ori|'
                     r'and|andi|mul|mulw|sll|srl|beq|bne|blt|bge|bltu|bgeu)\s+(.*)$')
ADDR = re.compile(r'^\s*([0-9a-f]+):')


def scan(path):
    try:
        out = subprocess.run(['llvm-objdump', '-d', '--no-show-raw-insn', path],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print('%-28s ERROR %s' % (path.split('/')[-1], exc))
        return []

    lines = [l for l in out.splitlines() if ADDR.match(l)]
    addr_of = [int(ADDR.match(l).group(1), 16) for l in lines]
    index_of = {a: i for i, a in enumerate(addr_of)}

    hits, seen = [], set()
    for i, ln in enumerate(lines):
        b = BRANCH.match(ln)
        if not b:
            continue
        target = int(b.group(2), 16)
        if target >= addr_of[i] or target not in index_of:
            continue                                   # not a backward branch
        lo, hi = index_of[target], i                   # loop body, inclusive

        cap_defs, movc_defs, int_uses = {}, {}, set()
        for j in range(lo, hi + 1):
            body = lines[j]
            cd = CAP_DEF.match(body)
            if cd:
                cap_defs[cd.group(3).rstrip(',')] = body.strip()
            mv = MOVC.match(body)
            if mv and mv.group(3).rstrip(',') != 'zero':
                movc_defs[mv.group(2).rstrip(',')] = body.strip()
            iu = INT_USE.match(body)
            if iu:
                int_uses.update(re.findall(r'\b([a-z]\w*)\b', iu.group(2)))

        for reg in sorted(set(cap_defs) & set(movc_defs) & int_uses):
            key = (lo, reg)
            if key in seen:
                continue
            seen.add(key)
            hits.append((hex(target), reg, cap_defs[reg], movc_defs[reg]))
    return hits


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: check-movc-reuse.py <domain.elf> [...]')
    bad = 0
    for p in sys.argv[1:]:
        hits = scan(p)
        print('%-28s C-14 capability/scalar register aliasing: %d' %
              (p.split('/')[-1], len(hits)))
        for target, reg, cd, mv in hits[:6]:
            print('    loop@%-9s %-4s  %-34s | %s' % (target, reg, cd, mv))
        if hits:
            bad += 1
    sys.exit(1 if bad else 0)
