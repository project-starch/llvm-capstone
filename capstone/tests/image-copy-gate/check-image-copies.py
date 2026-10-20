#!/usr/bin/env python3
"""Image-link gate for #246 (C-81): every memcpy/memmove a linked image carries must copy capabilities.

Since #246, a copy of 16 bytes or more whose 16-byte alignment the compiler cannot prove calls memcpy/memmove
instead of being inlined as scalar loads and stores. That is only a fix if the definition the image links copies
whole granules as capabilities, as musl-capstone's string_bounds_safe.c does when the two cursors agree mod 16. A
freestanding image that brings its own byte-loop memcpy now routes those copies through it and strips tags exactly
as the inline code did, silently. A static link pulls a definition in only when something references it, so a
definition present in the image is a definition reached.

Per image: llvm-objdump --disassemble-symbols=memcpy,memmove; a function passes when its body has `ldc rD, ..(rB)`
and a later `stc rD, ..(rS)` with the same rD and neither base sp (the prologue's spill of s0 to the stack is
`stc s0, 0(sp)` and must not count). Verdicts: COPIES-CAPS, NO-CAP-COPY (fail), ABSENT (not linked, so not reached).

Exit 0 every image passes; 1 any NO-CAP-COPY; 2 cannot check (no image given, unreadable, no symbol table) --
never a pass. --expect-fail inverts the verdict for planted controls: exit 0 only if every image FAILS.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

INSN = re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2} )+\s*(\S+)\s+(.*)$')
MEM = re.compile(r'^(\w+),\s*(-?(?:0x)?[0-9a-f]*)\((\w+)\)$')


def functions(objdump, image):
    out = subprocess.run([objdump, '-d', '--disassemble-symbols=memcpy,memmove', str(image)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f'objdump rc {out.returncode}')
    bodies, cur = {}, None
    for line in out.stdout.splitlines():
        m = re.match(r'^[0-9a-f]+ <(memcpy|memmove)>:$', line)
        if m:
            cur = bodies.setdefault(m.group(1), [])
            continue
        if cur is not None:
            i = INSN.match(line)
            if i:
                cur.append((i.group(1), i.group(2).strip()))
    return bodies


def copies_caps(body):
    loaded = set()
    for op, args in body:
        m = MEM.match(args)
        if not m:
            continue
        reg, _, base = m.groups()
        if base in ('sp', 'x2'):
            continue
        if op == 'ldc':
            loaded.add(reg)
        elif op == 'stc' and reg in loaded:
            return True
    return False


def has_symtab(readelf, image):
    out = subprocess.run([readelf, '-S', '-W', str(image)], capture_output=True, text=True)
    return out.returncode == 0 and ' .symtab ' in out.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('images', nargs='*', type=Path)
    p.add_argument('--llvm-bin', type=Path, required=True, help='directory with llvm-objdump and llvm-readelf')
    p.add_argument('--expect-fail', action='store_true', help='planted controls: every image must FAIL')
    a = p.parse_args()
    if not a.images:
        print('check-image-copies: no images given -- nothing checked', file=sys.stderr)
        return 2
    objdump, readelf = a.llvm_bin/'llvm-objdump', a.llvm_bin/'llvm-readelf'
    failed = unchecked = 0
    for image in a.images:
        if not image.is_file() or not has_symtab(readelf, image):
            print(f'CANNOT-CHECK {image}: missing, not ELF, or no symbol table')
            unchecked += 1
            continue
        try:
            bodies = functions(objdump, image)
        except RuntimeError as e:
            print(f'CANNOT-CHECK {image}: {e}')
            unchecked += 1
            continue
        verdicts = {}
        for fn in ('memcpy', 'memmove'):
            if fn not in bodies:
                verdicts[fn] = 'ABSENT'
            elif not bodies[fn]:
                verdicts[fn] = 'CANNOT-CHECK'
                unchecked += 1
            else:
                verdicts[fn] = 'COPIES-CAPS' if copies_caps(bodies[fn]) else 'NO-CAP-COPY'
        bad = [fn for fn, v in verdicts.items() if v == 'NO-CAP-COPY']
        failed += bool(bad)
        print(f"{'FAIL' if bad else 'ok  '} {image} memcpy={verdicts['memcpy']} memmove={verdicts['memmove']}")
    n = len(a.images)
    print(f'check-image-copies: {n} images, {failed} with a non-capability copy, {unchecked} cannot-check')
    if unchecked:
        return 2
    if a.expect_fail:
        return 0 if failed == n else 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
