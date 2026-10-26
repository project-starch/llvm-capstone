#!/usr/bin/env python3
"""Re-derive sizes.tsv and kvariants.txt from a sizing boot (run-qemu.py --icount results).

  resize.py --results results.json --label T1 --count-label T1c [--target 1.2e7] [--check-only]

Sizing: under -icount shift=0 one guest ns is one retired instruction, so an iteration's ns is its
instruction count (kernel work during the bracket included). The bracket's own cost is ctl/empty's
median; a program's new size is size x target / (median ns - empty), rounded to two significant
digits. A row of sizes.tsv with a third field ("pin: reason") is never changed.

K variants: a program goes into kvariants.txt when its counting build made any allocator call
(malloc, free, realloc, calloc) inside a timed iteration.

Non-reentrant programs (a check that changes between iterations, or a failed iteration) go into
nonreentrant.txt; run-qemu.py --nonreentrant runs them "0 1".

--check-only changes nothing and lists every program whose median is outside [0.6, 1.6] x target;
it exits 1 if any unpinned program is.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def round2(x):
    k = 10 ** max(0, int(math.log10(max(x, 1))) - 1)
    return max(1, round(x / k) * k)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, action='append', required=True)
    p.add_argument('--label', default='T1')
    p.add_argument('--count-label', default='T1c')
    p.add_argument('--target', type=float, default=1.2e7)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    res = {}
    for f in a.results:
        d = json.loads(f.read_text())
        if not d['meta']['icount']:
            sys.exit(f'{f}: not an --icount run')
        for label, entries in d['results'].items():
            res.setdefault(label, {}).update(entries)
    runs, counts = res[a.label], res.get(a.count_label, {})
    med = {}
    for image, e in runs.items():
        timed = [x['ns'] for x in e['bench'] if not x['it'].startswith('w')]
        if timed and image.endswith('.K0'):
            med[image[:-3]] = statistics.median(timed)
    empty = med.get('ctl-empty')
    if empty is None:
        sys.exit('no ctl-empty result: the bracket cost is unknown')
    lines = (HERE / 'sizes.tsv').read_text().splitlines()
    out, bad, changed = [], [], 0
    for line in lines:
        if line.startswith('#') or not line.strip():
            out.append(line)
            continue
        f = line.split('\t')
        name, size, pin = f[0], int(f[1]), (f[2] if len(f) > 2 else '')
        m = med.get(name)
        if m is not None and not (0.6 * a.target <= m <= 1.6 * a.target):
            bad.append((name, size, int(m), pin))
        if a.check_only or pin or m is None:
            out.append(line)
            continue
        new = round2(size * a.target / max(m - empty, 1))
        changed += new != size
        out.append(f'{name}\t{new}')
    if a.check_only:
        for name, size, m, pin in bad:
            print(f'{name:28s} size={size:<8} median={m:>12} {"(pinned: " + pin + ")" if pin else "OUT OF RANGE"}')
        unpinned = [b for b in bad if not b[3]]
        print(f'{len(med)} programs measured; {len(unpinned)} unpinned outside [0.6, 1.6] x {a.target:g}; '
              f'bracket (ctl-empty) = {int(empty)} guest instructions')
        return 1 if unpinned else 0
    (HERE / 'sizes.tsv').write_text('\n'.join(out) + '\n')
    alloc = []
    for image, e in counts.items():
        per = [x for x in e['alloc'] if x['it'] != 'total' and not x['it'].startswith('w')]
        if per and any(x[k] for x in per for k in ('malloc', 'free', 'realloc', 'calloc')):
            alloc.append(image[:-3] if image.endswith('.K0') else image)
    if not counts:
        sys.exit(f'no {a.count_label} results: kvariants.txt not written')
    (HERE / 'kvariants.txt').write_text(
        '# programs that allocate inside the timed region (counting build, resize.py): built at K = 0..3\n' +
        '\n'.join(sorted(alloc)) + '\n')
    nonre = []
    for image, e in runs.items():
        b = e['bench']
        if image.endswith('.K0') and b and (len({x['check'] for x in b}) > 1 or not all(x['ok'] for x in b)):
            nonre.append(image[:-3])
    (HERE / 'nonreentrant.txt').write_text(
        '# programs whose iterations do not repeat (a check that changes, or verification lost after the\n'
        '# first iteration) in the sizing run (resize.py): run one timed iteration per process, "0 1"\n' +
        '\n'.join(sorted(nonre)) + '\n')
    print(f'sizes: {changed} changed; kvariants: {len(alloc)} programs; non-reentrant: {len(nonre)}; '
          f'bracket {int(empty)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
