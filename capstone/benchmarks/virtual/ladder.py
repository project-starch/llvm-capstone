#!/usr/bin/env python3
"""The microbenchmark ladder Plain -> Caps -> ObjBounds -> MapLife -> ObjLife, from four sources of board rows.

  ladder.py --plain RUNS --caps RUNS... --objb RUNS... --rec RUNS... --out DIR [--plain-raw RUNS] [--capsv RUNS...]
            [--boot B] [--seed S]

Sources (each a parse-board.py runs table, .gz allowed; one row per timed iteration):
  plain  the Plain kits (board 1, bitstream 3), arms Plain, T1 (= MapLife), T3 (= ObjLife), after plain-exclude.py and
         plain-correct.py (the Plain section's corrected reading). --plain-raw adds the uncorrected reading beside it.
  caps   the Caps kits (bitstream 3), arms Caps, MapLife, ObjLife.
  objb   the ObjBounds kits (the variant bitstream), arms T1 and T3, relabelled HERE: T1 -> ObjBounds, T3 -> ObjLife-v.
         The bitstream is what makes T1 ObjBounds (PREREG-t1-t2-sampling.md, "ObjBounds", Analysis).
  rec    the bitstream-3 campaign records (results/2026-10-27-t3all-bitstream3, results/2026-10-28-t3heap-bitstream3),
         arms T1 (= MapLife) and T3 (= ObjLife); T2 rows are set aside.
  capsv  optional: the Caps kits (unchanged) on the ObjBounds bitstream, relabelled HERE: Caps -> Caps-v (capabilities
         without the per-access lifetime check), MapLife -> ObjBounds (a repeat of the objb source's arm), ObjLife ->
         ObjLife-v (PREREG-t1-t2-sampling.md, "Caps-v"). It is loaded last, so the other sources' bootstrap draws, and
         every table row that does not involve it, are those of a run without it.

Estimator, as perf-analyze.py report: per (source, program, arm), the mean of each process's mean log cycles over its
timed iterations (it = w* dropped). A direct ratio B/A is exp(mean B - mean A) with Welch's 95 % interval on the
process means; this holds for in-boot pairs and for the pre-registered cross-bitstream pairs alike (the latter are
labelled "cross" and carry that caveat). A chained ratio (a ladder bar through MapLife) takes its interval from a
two-level bootstrap (processes, then iterations within a process; B replicates per arm; independent across sources).
Suite geomeans take the bootstrap too, over the programs that have every term.

Groups: beebs (the 93, -libc variants included, as the figure), beebs-up (80), beebs-libc (13: beebs programs whose
name ends in -libc), rv8 (7), coremark (1). Controls (ctl-empty, ctl-loop, renamed per boot by plain-correct.py) enter
no group. A program missing from a source is absent from every ratio that needs it (Plain's set-aside programs).

Outputs in DIR: per-program.tsv (direct ratios), ladder.tsv (the five bars per program, normalized to Plain; six with
--capsv), per-group.tsv (geomeans); with --capsv also capsv-grade.txt, the Caps-v predictions C3-C6 graded against
their pre-registered bands (C1, C2 are the bare kit's and the gate's). Exit 1 on a failed process, a program seen twice in one source, or a source with no rows:
"no data" is an error, never an empty table.
"""
import argparse
import csv
import gzip
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = {
    'plain': {'Plain': 'Plain', 'T1': 'MapLife', 'T3': 'ObjLife'},
    'caps': {'Caps': 'Caps', 'MapLife': 'MapLife', 'ObjLife': 'ObjLife'},
    'objb': {'T1': 'ObjBounds', 'T3': 'ObjLife-v'},
    'rec': {'T1': 'MapLife', 'T3': 'ObjLife'},
    'capsv': {'Caps': 'Caps-v', 'MapLife': 'ObjBounds', 'ObjLife': 'ObjLife-v'},
}
SKIP = {'plain': set(), 'caps': set(), 'objb': set(), 'rec': {'T2'}, 'capsv': set()}
SUITES = ('beebs-up', 'beebs-libc', 'rv8', 'coremark')


def die(msg):
    print(f'ladder: {msg}', file=sys.stderr)
    sys.exit(1)


def rows_of(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def load(src, paths):
    """{program: {arm: {(boot, proc): [(cycles, instret)]}}}, {program: suite}"""
    data, suite, where = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), {}, defaultdict(set)
    n = 0
    for path in paths:
        for r in rows_of(path):
            if r['arm'] in SKIP[src]:
                continue
            arm = ARMS[src].get(r['arm'])
            if arm is None:
                die(f'{src} {path}: unknown arm {r["arm"]!r}')
            if r['rc'] != '0':
                die(f'{src} {path}: {r["program"]} {r["arm"]} proc {r["proc"]} rc={r["rc"]}')
            if r['it'].startswith('w'):
                continue
            p = r['program']
            suite[p] = r['suite']
            where[p].add((path, r['board'], r['boot']))
            data[p][arm][(r['boot'], r['proc'])].append((int(r['cycles']), int(r['instret'])))
            n += 1
    if not n:
        die(f'{src}: no timed rows in {paths}')
    for p, w in where.items():
        if len(w) != 1 and not p.startswith('ctl-'):
            die(f'{src}: {p} seen in more than one boot: {sorted(w)}')
    return data, suite


def group_of(p, s):
    if s == 'ctl' or p.startswith('ctl-empty') or p.startswith('ctl-loop'):
        return []
    if s == 'beebs':
        return ['beebs', 'beebs-libc' if p.endswith('-libc') else 'beebs-up']
    return [s]


class Arm:
    def __init__(self, procs, rng, B):
        self.g = [np.log(np.array([c for c, _ in v], dtype=float)) for v in procs.values()]
        self.m = np.array([g.mean() for g in self.g])
        self.mean = float(self.m.mean())
        self.ins = float(np.median([i for v in procs.values() for _, i in v]))
        k = len(self.g)
        self.boot = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, k, k)
            self.boot[b] = np.mean([rng.choice(self.g[j], len(self.g[j])).mean() for j in pick])


def welch(a, b):
    va, vb = a.m.var(ddof=1) / len(a.m), b.m.var(ddof=1) / len(b.m)
    d = b.mean - a.mean
    se = math.sqrt(va + vb)
    if se == 0:
        return d, d, d
    dof = (va + vb) ** 2 / (va ** 2 / (len(a.m) - 1) + vb ** 2 / (len(b.m) - 1))
    t = stats.t.ppf(0.975, dof)
    return d, d - t * se, d + t * se


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plain', required=True)
    ap.add_argument('--plain-raw')
    ap.add_argument('--caps', nargs='+', required=True)
    ap.add_argument('--objb', nargs='+', required=True)
    ap.add_argument('--rec', nargs='+', required=True)
    ap.add_argument('--capsv', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    srcs = {'plain': [a.plain], 'caps': a.caps, 'objb': a.objb, 'rec': a.rec}
    if a.plain_raw:
        srcs['plainraw'] = [a.plain_raw]
        ARMS['plainraw'], SKIP['plainraw'] = ARMS['plain'], SKIP['plain']
    if a.capsv:
        srcs['capsv'] = a.capsv
    D, suite = {}, {}
    for s, paths in srcs.items():
        d, su = load(s, paths)
        suite.update(su)
        D[s] = {p: {arm: Arm(procs, rng, a.boot) for arm, procs in arms.items()} for p, arms in d.items()}

    def arm(s, p, x):
        return D.get(s, {}).get(p, {}).get(x)

    # Direct ratios: (name, source of A, arm A, source of B, arm B, kind)
    DIRECT = [
        ('ObjLife/MapLife rec', 'rec', 'MapLife', 'rec', 'ObjLife', 'in-boot'),
        ('ObjLife/MapLife plainkits', 'plain', 'MapLife', 'plain', 'ObjLife', 'in-boot'),
        ('ObjLife/MapLife capskits', 'caps', 'MapLife', 'caps', 'ObjLife', 'in-boot'),
        ('MapLife/Caps', 'caps', 'Caps', 'caps', 'MapLife', 'in-boot'),
        ('ObjLife/Caps', 'caps', 'Caps', 'caps', 'ObjLife', 'in-boot'),
        ('MapLife/Plain', 'plain', 'Plain', 'plain', 'MapLife', 'in-boot'),
        ('ObjLife/Plain', 'plain', 'Plain', 'plain', 'ObjLife', 'in-boot'),
        ('MapLife/Plain raw', 'plainraw', 'Plain', 'plainraw', 'MapLife', 'in-boot'),
        ('ObjLife/Plain raw', 'plainraw', 'Plain', 'plainraw', 'ObjLife', 'in-boot'),
        ('ObjLife-v/ObjBounds', 'objb', 'ObjBounds', 'objb', 'ObjLife-v', 'in-boot'),
        ('MapLife/ObjBounds', 'objb', 'ObjBounds', 'rec', 'MapLife', 'cross'),
        ('ObjLife/ObjBounds', 'objb', 'ObjBounds', 'rec', 'ObjLife', 'cross'),
        ('Caps-v/ObjBounds capsvkits', 'capsv', 'ObjBounds', 'capsv', 'Caps-v', 'in-boot'),
        ('ObjBounds repeat', 'objb', 'ObjBounds', 'capsv', 'ObjBounds', 'cross-day'),
        ('ObjLife-v repeat', 'objb', 'ObjLife-v', 'capsv', 'ObjLife-v', 'cross-day'),
        ('Caps-v/Caps', 'caps', 'Caps', 'capsv', 'Caps-v', 'cross'),
    ]
    # Ladder bars normalized to Plain: a sum of (source, arm B, arm A) log differences, each term from one source.
    LADDER = {
        'Caps': [('caps', 'Caps', 'MapLife'), ('plain', 'MapLife', 'Plain')],
        'Caps-v': [('capsv', 'Caps-v', 'ObjBounds'), (('objb', 'ObjBounds'), ('rec', 'MapLife')),
                   ('plain', 'MapLife', 'Plain')],
        'ObjBounds': [(('objb', 'ObjBounds'), ('rec', 'MapLife')), ('plain', 'MapLife', 'Plain')],
        'MapLife': [('plain', 'MapLife', 'Plain')],
        'ObjLife': [('plain', 'ObjLife', 'Plain')],
    }

    def term(t, p):
        """(log difference, bootstrap replicates) of one ladder term, or None"""
        if isinstance(t[0], tuple):
            (sb, xb), (sa, xa) = t
        else:
            sb, xb, xa = t
            sa = sb
        b, c = arm(sb, p, xb), arm(sa, p, xa)
        if b is None or c is None:
            return None
        return b.mean - c.mean, b.boot - c.boot

    progs = sorted(p for p in suite if group_of(p, suite[p]))
    a.out.mkdir(parents=True, exist_ok=True)
    per, logs = [], defaultdict(dict)
    for name, sa, xa, sb, xb, kind in DIRECT:
        for p in progs:
            A, B = arm(sa, p, xa), arm(sb, p, xb)
            if A is None or B is None:
                continue
            d, lo, hi = welch(A, B)
            per.append((name, kind, suite[p], p, f'{len(A.m)}/{len(B.m)}', math.exp(d), math.exp(lo), math.exp(hi),
                        B.ins / A.ins))
            logs[name][p] = B.boot - A.boot
    with (a.out / 'per-program.tsv').open('w') as f:
        f.write('ratio\tkind\tsuite\tprogram\tprocs\tpoint\tlo\thi\tins_ratio\n')
        for r in per:
            f.write('\t'.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in r) + '\n')
    lad = {}
    with (a.out / 'ladder.tsv').open('w') as f:
        f.write('suite\tprogram\tbar\tpoint\tlo\thi\n')
        for p in progs:
            for bar, terms in LADDER.items():
                ts = [term(t, p) for t in terms]
                if any(t is None for t in ts):
                    continue
                d = sum(t[0] for t in ts)
                bt = sum(t[1] for t in ts)
                lad.setdefault(bar, {})[p] = bt
                f.write(f'{suite[p]}\t{p}\t{bar}\t{math.exp(d):.4f}\t{math.exp(np.quantile(bt, 0.025)):.4f}\t'
                        f'{math.exp(np.quantile(bt, 0.975)):.4f}\n')
    point = {name: {p: next(r[5] for r in per if r[0] == name and r[3] == p) for p in v} for name, v in logs.items()}
    with (a.out / 'per-group.tsv').open('w') as f:
        f.write('ratio\tgroup\tprograms\tgeomean\tlo\thi\n')
        everything = [(n, logs[n], point[n]) for n, *_ in DIRECT if n in logs]
        everything += [(f'ladder {b}/Plain', lad[b], None) for b in LADDER if b in lad]
        for name, bt, pt in everything:
            for g in ('beebs', 'beebs-up', 'beebs-libc', 'rv8', 'coremark'):
                ps = [p for p in bt if g in group_of(p, suite[p])]
                if not ps:
                    continue
                reps = np.mean([bt[p] for p in ps], axis=0)
                if pt is not None:
                    gm = math.exp(np.mean([math.log(pt[p]) for p in ps]))
                else:
                    gm = math.exp(np.mean([sum(t[0] for t in (term(x, p) for x in LADDER[name.split()[1].split('/')[0]]))
                                           for p in ps]))
                f.write(f'{name}\t{g}\t{len(ps)}\t{gm:.4f}\t{math.exp(np.quantile(reps, 0.025)):.4f}\t'
                        f'{math.exp(np.quantile(reps, 0.975)):.4f}\n')
    print((a.out / 'per-group.tsv').read_text(), end='')
    if a.capsv:
        grade(a.out)


def grade(out):
    """capsv-grade.txt: C3-C6 of PREREG-t1-t2-sampling.md "Caps-v", read from the tables just written."""
    G = {(r['ratio'], r['group']): r for r in rows_of(out / 'per-group.tsv')}
    P = defaultdict(dict)
    for r in rows_of(out / 'per-program.tsv'):
        P[r['ratio']][r['program']] = r
    lines, verdicts = [], []

    def gm(ratio, g):
        r = G.get((ratio, g))
        if r is None:
            die(f'grade: no {ratio!r} geomean for {g}: no data is an error')
        return float(r['geomean'])

    def check(name, ok, text):
        verdicts.append(ok)
        lines.append(f'{name} {"PASS" if ok else "MISS"}  {text}')

    for g in SUITES:
        v = gm('ObjBounds repeat', g)
        check('C3', abs(v - 1) <= 0.003, f'{g}: ObjBounds repeat / ObjBounds {v:.4f}, band 1.000 +- 0.3 %')
    for g in SUITES:
        v, want = gm('Caps-v/ObjBounds capsvkits', g), 1 / gm('MapLife/Caps', g)
        check('C4', abs(v - want) <= 0.005, f'{g}: Caps-v / ObjBounds in-boot {v:.4f}, predicted {want:.4f} +- 0.005')
    pp = P.get('Caps-v/ObjBounds capsvkits')
    if not pp:
        die('grade: no per-program Caps-v/ObjBounds ratios')
    out_of = sorted(p for p, r in pp.items() if not 0.98 <= float(r['point']) <= 1.02)
    check('C4', not out_of, f'per program: {len(pp) - len(out_of)} of {len(pp)} within 0.98-1.02'
          + (f'; outside: {", ".join(f"{p} {float(pp[p]["point"]):.4f}" for p in out_of)}' if out_of else ''))
    pi = P.get('Caps-v/Caps')
    if not pi:
        die('grade: no per-program Caps-v/Caps ratios')
    off = sorted(p for p, r in pi.items() if abs(float(r['ins_ratio']) - 1) > 0.0002)
    worst = max(pi, key=lambda p: abs(float(pi[p]['ins_ratio']) - 1))
    check('C5', not off, f'instructions Caps-v / Caps(bitstream 3): {len(pi) - len(off)} of {len(pi)} within 0.02 %, '
          f'largest {worst} {float(pi[worst]["ins_ratio"]):.6f}' + (f'; outside: {", ".join(off)}' if off else ''))
    for g in SUITES:
        v, ref = gm('ladder Caps-v/Plain', g), gm('ladder ObjBounds/Plain', g)
        check('C6', abs(v / ref - 1) <= 0.01, f'{g}: Caps-v / Plain {v:.4f}, ObjBounds / Plain {ref:.4f}, band +- 1 %')
    lines.append(f'{sum(verdicts)} of {len(verdicts)} checks PASS')
    (out / 'capsv-grade.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
