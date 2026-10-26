#!/usr/bin/env python3
"""T1 (Sublet spatial) against T2 (spatial + temporal) from the board's per-iteration records.

Input: a TSV with a header and one row per timed iteration (warm-up rows have it = w0, w1, ... and are dropped):
    board  boot  suite  program  arm  proc  it  cycles  instret  ns  rc
  arm   T1 or T2;  proc  the process index within (program, arm);  rc  the process's exit status;
  cycles, instret  the iteration's counts inside CSRUNV windows (perf module's bench_{cycles,instret} marks).

  pilot   variance components of log cycles per program and arm (one-way random effects: between processes,
          within a process), the instret check, and the main run's dimensions from the pre-registered rule
          (PREREG-t1-t2-sampling.md).
  report  per program: T2/T1 = exp(mean of process means of log cycles, T2 - T1) with Welch's 95 % interval on the
          process means, a two-level bootstrap interval as a check, the instret ratio; per suite: the geometric
          mean of the ratios with a two-level bootstrap interval; box plots per suite.

Exit 1 on no data, a failed process, a program whose arms ran on different boards or boots, or fewer than
--min-procs processes in an arm: "no data" is an error, never an empty table.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def die(msg):
    print(f'perf-analyze: {msg}', file=sys.stderr)
    sys.exit(1)


def load(path, min_procs):
    rows = list(csv.DictReader(open(path), delimiter='\t'))
    if not rows:
        die(f'no rows in {path}')
    need = {'board', 'boot', 'suite', 'program', 'arm', 'proc', 'it', 'cycles', 'instret', 'rc'}
    if not need <= set(rows[0]):
        die(f'missing columns {sorted(need - set(rows[0]))}')
    data = defaultdict(lambda: {'T1': defaultdict(list), 'T2': defaultdict(list)})
    suite, where, problems = {}, defaultdict(set), []
    for r in rows:
        p = r['program']
        suite[p] = r['suite']
        where[p].add((r['board'], r['boot']))
        if r['arm'] not in ('T1', 'T2'):
            problems.append(f'{p}: arm {r["arm"]!r}')
            continue
        if r['rc'] != '0':
            problems.append(f'{p} {r["arm"]} proc {r["proc"]}: rc={r["rc"]}')
            continue
        if r['it'].startswith('w'):
            continue
        if not r['cycles'] or not r['instret'] or int(r['cycles']) <= 0:
            problems.append(f'{p} {r["arm"]} proc {r["proc"]} it {r["it"]}: no counters ({r["cycles"]!r})')
            continue
        data[p][r['arm']][r['proc']].append((int(r['cycles']), int(r['instret'])))
    for p, arms in data.items():
        if len(where[p]) != 1:
            problems.append(f'{p}: arms ran in more than one board/boot {sorted(where[p])}')
        for arm in ('T1', 'T2'):
            if len(arms[arm]) < min_procs:
                problems.append(f'{p} {arm}: {len(arms[arm])} processes, fewer than {min_procs}')
    if problems:
        die('refusing to report:\n  ' + '\n  '.join(sorted(set(problems))))
    return data, suite


def logc(procs):
    """list of arrays of log cycles, one per process"""
    return [np.log(np.array([c for c, _ in v], dtype=float)) for v in procs.values()]


def components(groups):
    """one-way random effects (unbalanced allowed): (sigma_p, sigma_i) of log cycles, by the ANOVA estimator"""
    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    N = n.sum()
    means = np.array([g.mean() for g in groups])
    grand = np.concatenate(groups).mean()
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ssb = (n * (means - grand) ** 2).sum()
    msw = ssw / (N - k) if N > k else 0.0
    msb = ssb / (k - 1) if k > 1 else 0.0
    n0 = (N - (n ** 2).sum() / N) / (k - 1) if k > 1 else n.mean()
    var_p = max((msb - msw) / n0, 0.0)
    return math.sqrt(var_p), math.sqrt(msw), dict(k=k, N=int(N), msb=msb, msw=msw, n0=n0)


def upper_sigma(sigma, dof, conf=0.80):
    """one-sided upper confidence bound of a standard deviation from a chi-square with dof degrees of freedom"""
    if dof <= 0:
        return float('inf')
    return sigma * math.sqrt(dof / stats.chi2.ppf(1 - conf, dof))


def welch(m1, m2):
    """95 % interval of mean(m2) - mean(m1) on process means (log scale)"""
    a, b = np.asarray(m1), np.asarray(m2)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = b.mean() - a.mean()
    se = math.sqrt(va + vb)
    if se == 0:
        return d, d, d
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, dof)
    return d, d - t * se, d + t * se


def two_level(rng, groups, B):
    """B bootstrap replicates of the arm's mean of process means: resample processes, then iterations within"""
    k = len(groups)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, k, k)
        out[b] = np.mean([rng.choice(groups[j], len(groups[j])).mean() for j in pick])
    return out


def cmd_pilot(a):
    data, suite = load(a.runs, 2)
    print('program\tarm\tprocs\titers\tsigma_p\tsigma_i\tsigma_p_up80\tsigma_i_up80\tinstret_constant')
    worst_p = worst_i = 0.0
    for p in sorted(data):
        for arm in ('T1', 'T2'):
            g = logc(data[p][arm])
            sp, si, d = components(g)
            sp_up = upper_sigma(sp, d['k'] - 1) if sp > 0 else upper_sigma(math.sqrt(max(d['msb'], 0) / d['n0']), d['k'] - 1)
            si_up = upper_sigma(si, d['N'] - d['k'])
            ins = {i for v in data[p][arm].values() for _, i in v}
            # The rule dimensions on the pilot PROGRAMS (PREREG, 'The pilot'), not on the controls: ctl-empty's
            # region is ~2e4 instructions, so its relative noise says nothing about a benchmark's.
            if not p.startswith('ctl-'):
                worst_p, worst_i = max(worst_p, sp_up), max(worst_i, si_up)
            print(f'{p}\t{arm}\t{d["k"]}\t{d["N"]}\t{sp:.5f}\t{si:.5f}\t{sp_up:.5f}\t{si_up:.5f}\t'
                  f'{"yes" if len(ins) == 1 else f"no ({len(ins)} values)"}')
    # The pre-registered rule.
    h = math.log(1 + a.h)
    i_star = math.sqrt(a.c_proc * worst_i ** 2 / (a.c_iter * max(worst_p, 1e-12) ** 2))
    I = int(min(max(round(i_star), 3), 20))
    # Student's t, not 1.96: the report's interval is Welch's on P process means per arm, whose degrees of
    # freedom are about 2(P - 1). At P = 3 the z rule under-sizes the interval by about 1.4x (audit 2026-10-26).
    def half(P):
        return stats.t.ppf(0.975, 2 * (P - 1)) * math.sqrt(2 * (worst_p ** 2 / P + worst_i ** 2 / (P * I)))
    P = 3
    while half(P) > h and P < 1000:
        P += 1
    per_prog_s = 2 * P * (a.c_proc + (I + a.warmup) * a.c_iter)
    print(f'\nworst sigma_p (up80) {worst_p:.5f}, worst sigma_i (up80) {worst_i:.5f}')
    print(f'rule: I* = {i_star:.2f} -> I = {I}; P = {P} processes per arm for h = {a.h:.2%}')
    print(f'cost: {per_prog_s:.0f} s per program; {a.programs} programs over {a.boards} boards = '
          f'{per_prog_s * a.programs / a.boards / 3600:.2f} h of runs per board (cap {a.cap_h} h)')
    if per_prog_s * a.programs / a.boards / 3600 > a.cap_h:
        # Relax h to what the cap allows at the same I.
        P = max(3, int(a.cap_h * 3600 * a.boards / a.programs // (2 * (a.c_proc + (I + a.warmup) * a.c_iter))))
        hh = half(P)
        print(f'CAP BINDS: P = {P}, achievable half-width {math.exp(hh) - 1:.2%}')


def cmd_report(a):
    data, suite = load(a.runs, a.min_procs)
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    per, boots = {}, {}
    for p in sorted(data):
        g1, g2 = logc(data[p]['T1']), logc(data[p]['T2'])
        m1, m2 = [g.mean() for g in g1], [g.mean() for g in g2]
        d, lo, hi = welch(m1, m2)
        b = two_level(rng, g2, a.boot) - two_level(rng, g1, a.boot)
        boots[p] = b
        i1 = np.array([i for v in data[p]['T1'].values() for _, i in v], dtype=float)
        i2 = np.array([i for v in data[p]['T2'].values() for _, i in v], dtype=float)
        per[p] = dict(suite=suite[p], procs=f'{len(g1)}/{len(g2)}', ratio=math.exp(d), lo=math.exp(lo), hi=math.exp(hi),
                      blo=math.exp(np.quantile(b, 0.025)), bhi=math.exp(np.quantile(b, 0.975)),
                      ins_ratio=float(np.median(i2) / np.median(i1)),
                      ins_const='yes' if len(set(i1)) == 1 and len(set(i2)) == 1 else 'no',
                      verdict='no measurable difference' if lo <= 0 <= hi else ('T2 slower' if lo > 0 else 'T2 faster'))
    cols = ['suite', 'program', 'procs', 'ratio', 'lo', 'hi', 'blo', 'bhi', 'ins_ratio', 'ins_const', 'verdict']
    with (a.out / 'per-program.tsv').open('w') as f:
        f.write('\t'.join(cols) + '\n')
        for p, d in sorted(per.items(), key=lambda kv: (kv[1]['suite'], kv[0])):
            v = {**d, 'program': p}
            f.write('\t'.join(f'{v[c]:.4f}' if isinstance(v[c], float) else str(v[c]) for c in cols) + '\n')
    with (a.out / 'per-suite.tsv').open('w') as f:
        f.write('suite\tprograms\tgeomean\tlo\thi\tT2_slower\tT2_faster\tno_difference\n')
        for s in sorted(set(suite.values())):
            ps = [p for p in per if per[p]['suite'] == s]
            gm = math.exp(np.mean([math.log(per[p]['ratio']) for p in ps]))
            gb = np.exp(np.mean([boots[p] for p in ps], axis=0))
            v = [per[p]['verdict'] for p in ps]
            f.write(f'{s}\t{len(ps)}\t{gm:.4f}\t{np.quantile(gb, 0.025):.4f}\t{np.quantile(gb, 0.975):.4f}\t'
                    f'{v.count("T2 slower")}\t{v.count("T2 faster")}\t{v.count("no measurable difference")}\n')
            boxes(s, sorted(ps), data, a.out / f'box-{s}.svg')
    print((a.out / 'per-suite.tsv').read_text(), end='')


def boxes(s, ps, data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(ps) + 2), 4.5))
    pos, vals, colors = [], [], []
    for i, p in enumerate(ps):
        t1 = np.array([c for v in data[p]['T1'].values() for c, _ in v], dtype=float)
        t2 = np.array([c for v in data[p]['T2'].values() for c, _ in v], dtype=float)
        med = np.median(t1)
        vals += [t1 / med, t2 / med]
        pos += [3 * i, 3 * i + 1]
        colors += ['#9AA6B4', '#E39A4B']
    bp = ax.boxplot(vals, positions=pos, widths=0.8, whis=1.5, patch_artist=True,
                    medianprops=dict(color='black'), flierprops=dict(marker='o', markersize=3))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.axhline(1.0, color='#444', linewidth=0.8)
    ax.set_xticks([3 * i + 0.5 for i in range(len(ps))])
    ax.set_xticklabels(ps, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel("cycles / the program's T1 median")
    ax.set_title(f'{s}: T1 (grey) and T2 (orange), every timed iteration')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pilot')
    p.add_argument('runs', type=Path)
    p.add_argument('--h', type=float, default=0.01, help='target 95 %% half-width of T2/T1 (fraction)')
    p.add_argument('--c-proc', type=float, default=7.0, help='seconds per process launch on the board')
    p.add_argument('--c-iter', type=float, default=0.4, help='seconds per timed iteration on the board')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--programs', type=int, default=102)
    p.add_argument('--boards', type=int, default=2)
    p.add_argument('--cap-h', type=float, default=3.0)
    r = sub.add_parser('report')
    r.add_argument('runs', type=Path)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--min-procs', type=int, default=3)
    r.add_argument('--boot', type=int, default=10000)
    r.add_argument('--seed', type=int, default=20261026)
    a = ap.parse_args()
    (cmd_pilot if a.cmd == 'pilot' else cmd_report)(a)


if __name__ == '__main__':
    main()
