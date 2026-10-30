#!/usr/bin/env python3
"""Plain addendum 1, rule 1: drop the iterations in which another task ran inside a Plain timed region.

  plain-exclude.py <runs.tsv> <runs.tsv.kern.tsv> --out <runs.tsv> --out-kern <kern.tsv> --set-aside <runs.tsv>
                   [--report <file>]

One boot per invocation (the rule is per kit). User-mode rdcycle/rdinstret count the whole hart, so a Plain iteration
in which another task ran carries that task's cycles; the Capstone arms count inside their own context and do not.
A Plain timed iteration is EXCLUDED when its `timer` count exceeds its program's median Plain timer count in the same
kit by more than 10 % plus 2 ticks (timer > 1.1 * median + 2). The same (proc, it) of that program is dropped from
every other arm of the kit, so every ratio compares the same iterations. Warm-up rows (it = w*) are kept as they are:
perf-analyze.py drops them.

A program with more than 2 of its timed iterations excluded in any one Plain process is SET ASIDE: all its rows go
to --set-aside, to be reported per program and kept out of every suite geomean. The controls ctl-empty and ctl-loop
are never set aside: they are in no geomean, and they calibrate plain-correct.py, so they lose only the excluded rows.

Exit 1 on no Plain rows, or a Plain timed row with no BENCH-KERN counterpart: "no data" is an error, never a pass.
"""
import argparse
import csv
import statistics as st
import sys
from collections import defaultdict


def die(msg):
    print(f'plain-exclude: {msg}', file=sys.stderr)
    sys.exit(1)


def write(path, fields, rows):
    with open(path, 'w') as f:
        f.write('\t'.join(fields) + '\n')
        for r in rows:
            f.write('\t'.join(r[x] for x in fields) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs')
    ap.add_argument('kern')
    ap.add_argument('--out', required=True)
    ap.add_argument('--out-kern', required=True)
    ap.add_argument('--set-aside', required=True)
    ap.add_argument('--report')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.runs), delimiter='\t'))
    krows = list(csv.DictReader(open(a.kern), delimiter='\t'))
    if not rows:
        die(f'no rows in {a.runs}')
    if len({(r['board'], r['boot']) for r in rows}) != 1:
        die('one boot per invocation: the rule is per kit')
    fields, kfields = list(rows[0].keys()), list(krows[0].keys()) if krows else []
    kern = {(k['step'], k['it']): k for k in krows}
    timed = lambda r: not r['it'].startswith('w')
    plain = [r for r in rows if r['arm'] == 'Plain' and timed(r)]
    if not plain:
        die('no Plain timed rows')
    timer = defaultdict(list)
    for r in plain:
        k = kern.get((r['step'], r['it']))
        if k is None:
            die(f'Plain {r["program"]} proc {r["proc"]} it {r["it"]} has no BENCH-KERN row')
        timer[r['program']].append(int(k['timer']))
    med = {p: st.median(v) for p, v in timer.items()}
    drop, per_proc = set(), defaultdict(int)
    for r in plain:
        t = int(kern[(r['step'], r['it'])]['timer'])
        if t > 1.1 * med[r['program']] + 2:
            drop.add((r['program'], r['proc'], r['it']))
            per_proc[(r['program'], r['proc'])] += 1
    # The controls are in no suite geomean and are plain-correct.py's calibration: they lose only their excluded rows.
    aside = {p for (p, _), n in per_proc.items() if n > 2 and p not in ('ctl-empty', 'ctl-loop')}
    keep, side, gone = [], [], defaultdict(int)
    for r in rows:
        if (r['program'], r['proc'], r['it']) in drop:
            gone[(r['program'], r['arm'])] += 1
            continue
        (side if r['program'] in aside else keep).append(r)
    kept_steps = {(r['step'], r['it']) for r in keep}
    write(a.out, fields, keep)
    write(a.out_kern, kfields, [k for k in krows if (k['step'], k['it']) in kept_steps])
    write(a.set_aside, fields, side)
    boot = rows[0]['boot']
    rep = [f'{boot}: {len(drop)} of {len(plain)} Plain timed iterations excluded (timer > 1.1 x median + 2), '
           f'{sum(gone.values())} rows dropped across arms; {len(aside)} program(s) set aside: {sorted(aside)}']
    for p in sorted({p for p, _, _ in drop}):
        n = sum(1 for q, _, _ in drop if q == p)
        rep.append(f'  {p}: median timer {med[p]:g}; excluded {n} '
                   f'({", ".join(f"proc {pr} it {it}" for q, pr, it in sorted(drop) if q == p)})'
                   + ('  SET ASIDE' if p in aside else ''))
    text = '\n'.join(rep) + '\n'
    print(text, end='')
    if a.report:
        open(a.report, 'w').write(text)


if __name__ == '__main__':
    main()
