#!/usr/bin/env python3
"""The layout gate between the two arms' builds (build.py output directories, capstone mode).

  check-arms.py <T1 build dir> <T2 build dir> [--tsv gate.tsv]

1. Objects: every benchmark and harness object is byte-identical across the arms (the SDK enters at
   the link only). Any difference fails.
2. Per image pair (same image name, so the same K): the section headers (`llvm-readelf -S`) must be
   identical, and in the linker maps every input section and every symbol must sit at the same
   address with the same size -- except heap.c's own (`libapplication-runtime.a(heap.c.obj)`),
   which necessarily differ. Paths are compared with each arm's SDK root and build directory
   replaced by placeholders. A pair that differs anywhere else FAILS and the program is blocked.

--negative-control shows the gate can fire: T1 against a T2 build linked with the SDK's default
link.ld (heap.c first, so everything after it moves) must FAIL for every pair. --k-check requires
every K>0 variant to equal K0 outside heap.c.
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

READELF = os.path.join(os.environ.get('CAPSTONE_LLVM_BIN', ''), 'llvm-readelf')
HEAP = 'libapplication-runtime.a(heap.c.obj)'
SYMBOL_TABLES = {'.symtab', '.strtab', '.shstrtab'}


def section_headers(image):
    """The ALLOCATED sections' headers (name, type, address, offset, size, flags, alignment). The
    symbol and string tables carry no address and legitimately differ: heap.c's own locals."""
    out = subprocess.run([READELF, '-S', '--wide', str(image)], capture_output=True, text=True, check=True).stdout
    rows = []
    for l in out.splitlines():
        if not l.strip().startswith('[') or ']' not in l or l.strip().startswith('[Nr]'):
            continue
        f = l.split(']', 1)[1].split()
        flags = f[6] if len(f) == 10 else ''
        if 'A' in flags:
            rows.append(f)
    return rows


def read_map(path, subst):
    """{(level, out, in, name): [(vma, size)...]} with heap.c's entries in a separate dict."""
    text = path.read_text()
    for old, new in subst:
        text = text.replace(old, new)
    entries, heap = defaultdict(list), defaultdict(list)
    out_sec = in_sec = ''
    for line in text.splitlines()[1:]:
        m = re.match(r'\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+(\d+) (.*)$', line)
        if not m:
            continue
        vma, size, rest = int(m.group(1), 16), int(m.group(3), 16), m.group(5)
        indent = len(rest) - len(rest.lstrip(' '))
        name = rest.strip()
        if indent == 0:
            out_sec, in_sec = name, ''
            if name.startswith('BENCH_HEAP_SHIFT'):
                continue  # the K shift itself; equal for a pair of the same K
            key = (0, out_sec, '', name)
        elif indent == 8:
            in_sec = name
            key = (1, out_sec, in_sec, '')
        else:
            key = (2, out_sec, in_sec, name)
        # heap.c's input sections and symbols, and the layout script's own bookkeeping of its region
        # (the K shift, the region's end markers), are the allocator's; everything else is compared
        if out_sec in SYMBOL_TABLES:
            continue  # unallocated; T2's heap.c has one local more than T1's
        in_heap = HEAP in in_sec or 'bench_heap' in name.lower()
        (heap if in_heap else entries)[key].append((vma, size))
    return entries, heap


def compare(img1, img2, subst1, subst2):
    problems = []
    if section_headers(img1) != section_headers(img2):
        h1, h2 = section_headers(img1), section_headers(img2)
        diff = [f'{a} | {b}' for a, b in zip(h1, h2) if a != b][:3]
        problems.append('section headers differ: ' + '; '.join(' '.join(x) if isinstance(x, list) else x for x in diff))
    e1, heap1 = read_map(img1.with_suffix('.map'), subst1)
    e2, heap2 = read_map(img2.with_suffix('.map'), subst2)
    keys = set(e1) | set(e2)
    bad = [k for k in keys if sorted(e1.get(k, [])) != sorted(e2.get(k, []))]
    if bad:
        bad.sort()
        problems.append(f'{len(bad)} map entries differ outside heap.c, first: {bad[0]} '
                        f'{e1.get(bad[0])} | {e2.get(bad[0])}')
    inputs = sum(1 for k in keys if k[0] == 1)
    heap_inputs = sum(1 for k in set(heap1) | set(heap2) if k[0] == 1)
    return problems, inputs, heap_inputs


def subst_for(build):
    """Placeholders for the arm's own paths in its maps: the SDK root and the build directory."""
    build = build.resolve()
    first = (build / 'images.tsv').read_text().splitlines()[1].split('\t')[0]
    mm = re.search(r'(\S+)/sdk/libapplication-runtime\.a\(', (build / f'{first}.map').read_text())
    return [(mm.group(1), '<SDKROOT>'), (str(build), '<OUT>')]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('t1', type=Path)
    p.add_argument('t2', type=Path)
    p.add_argument('--tsv', type=Path)
    p.add_argument('--k-check', action='store_true',
                   help='also require, within T1, that every K variant equals K0 outside heap.c')
    p.add_argument('--negative-control', type=Path,
                   help='a T2 build linked with the SDK default link.ld (build.py --layout default): '
                        'every pair against T1 must FAIL, or the gate cannot fire')
    a = p.parse_args()
    bad_objects = []
    for o in sorted((a.t1 / 'obj').rglob('*.o')):
        rel = o.relative_to(a.t1 / 'obj')
        other = a.t2 / 'obj' / rel
        if not other.exists() or other.read_bytes() != o.read_bytes():
            bad_objects.append(str(rel))
    s1, s2 = subst_for(a.t1), subst_for(a.t2)
    rows = []
    images = [l.split('\t') for l in (a.t1 / 'images.tsv').read_text().splitlines()[1:] if l.strip()]
    for image, program, k, *_ in images:
        i1, i2 = a.t1 / f'{image}.dom', a.t2 / f'{image}.dom'
        if not i2.exists():
            rows.append((image, 'FAIL', 0, 0, 'no T2 image'))
            continue
        problems, inputs, heap_inputs = compare(i1, i2, s1, s2)
        rows.append((image, 'FAIL' if problems else 'PASS', inputs, heap_inputs, '; '.join(problems)))
    print(f'objects: {len(bad_objects)} differ between the arms' + (': ' + ', '.join(bad_objects) if bad_objects else ''))
    passed = sum(r[1] == 'PASS' for r in rows)
    print(f'layout gate: {passed} of {len(rows)} image pairs PASS')
    for r in rows:
        if r[1] != 'PASS':
            print(f'  FAIL {r[0]}: {r[4]}')
    selftest_ok = True
    if a.k_check:
        kimgs = [(i, prog, k) for i, prog, k, *_ in images if k != '0']
        kbad = []
        for i, prog, k in kimgs:
            problems, _, _ = compare(a.t1 / f'{prog}.K0.dom', a.t1 / f'{i}.dom', s1, s1)
            if problems:
                kbad.append(f'{i}: {problems[0][:160]}')
        print(f'K check: {len(kimgs) - len(kbad)} of {len(kimgs)} K>0 images equal K0 outside heap.c')
        for b in kbad:
            print('  FAIL', b)
        selftest_ok = selftest_ok and not kbad
    if a.negative_control:
        sc = subst_for(a.negative_control)
        fired = total = 0
        for image, program, k, *_ in images:
            other = a.negative_control / f'{image}.dom'
            if not other.exists():
                continue
            total += 1
            problems, _, _ = compare(a.t1 / f'{image}.dom', other, s1, sc)
            fired += bool(problems)
        print(f'negative control (default link.ld): the gate FAILED {fired} of {total} pairs (must be all)')
        selftest_ok = selftest_ok and total > 0 and fired == total
    if a.tsv:
        with a.tsv.open('w') as f:
            f.write('image\tgate\tinput_sections_compared\theap_input_sections\tproblems\n')
            for r in rows:
                f.write('\t'.join(map(str, r)) + '\n')
    return 0 if not bad_objects and passed == len(rows) and selftest_ok else 1


if __name__ == '__main__':
    sys.exit(main())
