#!/usr/bin/env python3
"""Image-link gate for #246 (C-81): every memcpy/memmove a linked image carries must copy capabilities.

Since #246, a copy of 16 bytes or more whose 16-byte alignment the compiler cannot prove calls memcpy/memmove
instead of being inlined as scalar loads and stores. That is only a fix if the definition the image links copies
whole granules as capabilities, as musl-capstone's string_bounds_safe.c does when the two cursors agree mod 16. A
freestanding image that brings its own byte-loop memcpy now routes those copies through it and strips tags exactly
as the inline code did, silently. A static link pulls a definition in only when something references it, so a
definition present in the image is a definition reached.

Per image: each defined memcpy/memmove FUNC symbol is disassembled by ADDRESS RANGE, its value to value+size, never
by --disassemble-symbols: that option stops at the next symbol, and a local label such as `.Lpcrel_hi0` inside the
function (an `auipc` before its loop) cut a body short. A body whose decoded bytes fall short of the symbol size is
CANNOT-CHECK, never a verdict. A function passes when its body has `ldc rD, ..(rB)`
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


def symbols(readelf, image):
    """Defined FUNC symbols named memcpy/memmove: [(name, value, size)]."""
    out = subprocess.run([readelf, '-s', '-W', str(image)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f'readelf rc {out.returncode}')
    found = []
    for line in out.stdout.splitlines():
        w = line.split()
        if len(w) >= 8 and w[3] == 'FUNC' and w[6] != 'UND' and w[7] in ('memcpy', 'memmove'):
            found.append((w[7], int(w[1], 16), int(w[2])))
    return found


def body(objdump, image, value, size):
    """(instructions, decoded bytes) for [value, value+size)."""
    out = subprocess.run([objdump, '-d', f'--start-address={value:#x}', f'--stop-address={value + size:#x}',
                          str(image)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f'objdump rc {out.returncode}')
    insns, nbytes = [], 0
    for line in out.stdout.splitlines():
        i = INSN.match(line)
        if i:
            insns.append((i.group(1), i.group(2).strip()))
            nbytes += len(re.match(r'^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)', line).group(1).split())
    return insns, nbytes


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
        verdicts = {'memcpy': 'ABSENT', 'memmove': 'ABSENT'}
        try:
            for fn, value, size in symbols(readelf, image):
                insns, nbytes = body(objdump, image, value, size)
                if size == 0 or nbytes != size:
                    v = f'CANNOT-CHECK({nbytes}/{size} bytes decoded)'
                    unchecked += 1
                else:
                    v = 'COPIES-CAPS' if copies_caps(insns) else 'NO-CAP-COPY'
                # Several definitions of one name: the worst verdict stands.
                if verdicts[fn] == 'ABSENT' or verdicts[fn] == 'COPIES-CAPS':
                    verdicts[fn] = v
        except RuntimeError as e:
            print(f'CANNOT-CHECK {image}: {e}')
            unchecked += 1
            continue
        bad = [fn for fn, v in verdicts.items() if v == 'NO-CAP-COPY']
        failed += bool(bad)
        tag = 'FAIL' if bad else 'CANT' if any(v.startswith('CANNOT') for v in verdicts.values()) else 'ok  '
        print(f"{tag} {image} memcpy={verdicts['memcpy']} memmove={verdicts['memmove']}")
    n = len(a.images)
    print(f'check-image-copies: {n} images, {failed} with a non-capability copy, {unchecked} cannot-check')
    if unchecked:
        return 2
    if a.expect_fail:
        return 0 if failed == n else 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
