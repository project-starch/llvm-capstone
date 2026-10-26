#!/usr/bin/env python3
"""The QEMU functional table, one row per program, from run-qemu.py results.

  functional-table.py --results results.json [--results ...] --t1 T1 --t2 T2 [--native native]
                      [--images <T1 build dir>/images.tsv] --out functional.tsv

Columns: program, images (the K variants run per arm), T1_ok, T2_ok, native_check_equal, failure.
An arm is ok when every image of the program exited 0 with every iteration verified. The native
column compares the per-iteration check values of the K0 image of each arm with the native build
(n/a where there is no native result). A failure carries the image, its exit status, how many
iterations completed, and the [CAPSTONE] and capstone-exec fault lines from the console.
"""
import argparse
import json
import sys
from pathlib import Path


def load(paths):
    merged = {}
    for p in paths:
        for label, entries in json.loads(Path(p).read_text())['results'].items():
            merged.setdefault(label, {}).update(entries)
    return merged


def verdict(e):
    if not e:
        return False, 'not run'
    b = e['bench']
    total = (e['info']['warmup'] + e['info']['iterations']) if e.get('info') else '?'
    ok = e['rc'] == 0 and b and all(x['ok'] for x in b) and len(b) == total
    if ok:
        return True, ''
    faults = [l for l in e['other'] if l.startswith('[CAPSTONE]') or 'domain fault' in l or 'collect skipped' in l]
    failed = sum(not x['ok'] for x in b)
    return False, (f'rc={e["rc"]}{" timeout" if e.get("timeout") else ""} iterations={len(b)}/{total} '
                   f'verify-failed={failed} ' + ' | '.join(f[:220] for f in faults[:3]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, action='append', required=True)
    p.add_argument('--t1', default='T1')
    p.add_argument('--t2', default='T2')
    p.add_argument('--native', default='native')
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = load(a.results)
    by_prog = {}
    for line in a.images.read_text().splitlines()[1:]:
        if line.strip():
            image, program = line.split('\t')[:2]
            by_prog.setdefault(program, []).append(image)
    rows = ['\t'.join(['program', 'images', 'T1_ok', 'T2_ok', 'native_check_equal', 'failure'])]
    counts = {'T1': 0, 'T2': 0, 'neq': 0}
    for program, images in by_prog.items():
        fails, oks = [], {}
        for arm in (a.t1, a.t2):
            allok = True
            for image in images:
                ok, why = verdict(r.get(arm, {}).get(image))
                if not ok:
                    allok = False
                    fails.append(f'{arm} {image}: {why}')
            oks[arm] = allok
        n = r.get(a.native, {}).get(program)
        nchk = [x['check'] for x in n['bench']] if n and n['bench'] else None
        eq = []
        for arm in (a.t1, a.t2):
            e = r.get(arm, {}).get(f'{program}.K0')
            if e and e['bench'] and nchk:
                eq.append([x['check'] for x in e['bench']] == nchk)
        neq = 'n/a' if not eq else ('yes' if all(eq) else 'NO')
        if n and n['rc'] != 0:
            fails.append(f'native: rc={n["rc"]}')
        counts['T1'] += oks[a.t1]
        counts['T2'] += oks[a.t2]
        counts['neq'] += neq == 'yes'
        rows.append('\t'.join([program, str(len(images)), 'yes' if oks[a.t1] else 'NO',
                               'yes' if oks[a.t2] else 'NO', neq, '; '.join(fails)]))
    a.out.write_text('\n'.join(rows) + '\n')
    print(f'{len(by_prog)} programs: T1 ok {counts["T1"]}, T2 ok {counts["T2"]}, native check equal {counts["neq"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
