#!/usr/bin/env python3
"""Collect the --size 100 result runs of the reuse-distance comparison, check each, compare the repetitions,
and plot one figure per system.

    collect.py MIRROR OUT [--plot]

MIRROR holds copies of the hosts' run directories, one per host: MIRROR/<host>/speedtest1/reuse-size100/
set<N>-poisoncap-{off,on}/result.json (speedtest1/run.py) and MIRROR/<host>/reuse-capstone-runs/
set<N>-sublet-{off,on}/results.jsonl (run-sublet.py, row <arm>-size100: spatial is off,
block-bounds on). A run directory without its record is still running and is skipped.

OUT gets runs.json (each run's result lines only: hashes, exit status, the test times, the wrapper's two
report lines; home paths scrubbed), checks.txt (every check, every run) and, with --plot, plots/ from
plot.py: per system, the lowest set whose two runs are admissible and use qualified binaries, with
the x axis shared by both systems. A run is admissible when it exits 0, completes the 20 tests, and its
exit report has installed=1, error=0, 32 bins, the four identities, >= 1,000 issues and no SQL step or
reset error. Repetitions of one arm must agree in every counter and bin; checks.txt says whether they do.
Exit 0 only if every collected run is admissible and every repetition agrees.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'platform'))
sys.path.insert(0, str(HERE))
import records  # noqa: E402
import reuse_report  # noqa: E402

TESTS = [100, 110, 120, 130, 140, 142, 145, 160, 190, 210, 240, 250, 260, 280, 300, 320, 400, 410, 980, 990]
# The binaries each arm was qualified with (qualification.txt); others are plotted only as preliminary.
QUALIFIED = {
    ('poisoncap', 'off'): {'aad6a979'}, ('poisoncap', 'on'): {'97b02659'},
    ('sublet', 'off'): {'9836684d'}, ('sublet', 'on'): {'6bfe4d1b'},
}
SUBLET_ROW = {'off': 'spatial-size100', 'on': 'block-bounds-size100'}


def discover(mirror):
    runs = []
    for f in sorted(mirror.glob('*/speedtest1/reuse-size100/set*-poisoncap-*/result.json')):
        m = re.fullmatch(r'set(\d+)-poisoncap-(off|on)', f.parent.name)
        r = json.loads(f.read_text())
        runs.append({'system': 'poisoncap', 'set': int(m[1]), 'arm': m[2], 'host': f.parts[len(mirror.parts)],
                     'binary_sha256': r.get('binary_sha256'), 'kernel_sha256': r.get('kernel_sha256'),
                     'invocation': records.scrub(r.get('invocation')), 'exit': r.get('exit'), 'seconds': r.get('seconds'),
                     'stdout': r.get('stdout') or '', 'stderr': r.get('stderr') or ''})
    for f in sorted(mirror.glob('*/reuse-capstone-runs/set*-sublet-*/results.jsonl')):
        m = re.fullmatch(r'set(\d+)-sublet-(off|on)', f.parent.name)
        rows = {json.loads(l)['name']: json.loads(l) for l in f.read_text().splitlines() if l.strip()}
        r = rows.get(SUBLET_ROW[m[2]], {})
        runs.append({'system': 'sublet', 'set': int(m[1]), 'arm': m[2], 'host': f.parts[len(mirror.parts)],
                     'binary_sha256': r.get('image_sha256'), 'module_args': r.get('module_args'),
                     'exit': r.get('rc'), 'ran': r.get('ran', False), 'fault': r.get('fault'),
                     'stdout': r.get('stdout') or '', 'stderr': r.get('stderr') or ''})
    return runs


def check(run):
    """Fill run['problems'] (empty: admissible) and run['report'] (the exit report and SQL-CHECK lines)."""
    p = []
    tests = [(int(i), float(s)) for i, s in reuse_report.TEST.findall(run['stdout'])]
    run['tests'] = [[i, s] for i, s in tests]
    tot = reuse_report.TOTAL.findall(run['stdout'])
    run['total_seconds'] = float(tot[-1]) if tot else None
    run['report'] = [reuse_report.report_line(run['stderr']), reuse_report.sql_line(run['stderr'])]
    if run['exit'] != 0:
        p.append(f"exit {run['exit']}")
    if [i for i, _ in tests] != TESTS:
        p.append(f'tests completed: {len(tests)} of 20')
    gap, sql = run['report']
    if not gap:
        p.append('no REUSE-GAP report at exit')
    else:
        d = reuse_report.fields(gap)
        n, bins = reuse_report.counters(d), d['bins']
        run['counters'], run['bins'] = n, bins
        if d['installed'] != '1' or d['error'] != '0':
            p.append(f"installed={d['installed']} error={d['error']} first_error={d.get('first_error')}")
        if len(bins) != 32:
            p.append(f'{len(bins)} bins')
        p += [name.replace('==', '!=') for name in reuse_report.failed_identities(d)]
        if n['issues'] < 1000:
            p.append(f"issues {n['issues']} < 1000")
    if not sql:
        p.append('no SQL-CHECK line')
    else:
        s = reuse_report.fields(sql)
        if s.get('step_other') != '0' or s.get('reset_err') != '0':
            p.append(f'SQL errors: {sql}')
    run['problems'] = p
    run['qualified'] = (run['binary_sha256'] or '')[:8] in QUALIFIED[(run['system'], run['arm'])]


def verdict_lines(runs):
    """ADMISSIBLE or INADMISSIBLE per run."""
    lines = []
    for r in runs:
        tag = f"{r['system']} set{r['set']} {r['arm']:3} ({r['host']}, {(r['binary_sha256'] or '?')[:8]}"
        tag += ', qualified binary)' if r['qualified'] else ', NOT a qualified binary)'
        if r['problems']:
            lines.append(f"INADMISSIBLE {tag}: {'; '.join(r['problems'])}")
        else:
            c = r['counters']
            lines.append(f"ADMISSIBLE {tag}: issues={c['issues']} reuses={c['reuses']} failed={c['failed']} "
                         f"total={r['total_seconds']}s")
    return lines


def repetition_lines(runs):
    """(lines, all agree): the admissible repetitions of each arm must agree exactly."""
    lines, agree = [], True
    for key in sorted({(r['system'], r['arm']) for r in runs}):
        good = [r for r in runs if (r['system'], r['arm']) == key and not r['problems']]
        if len(good) > 1:
            ref = good[0]
            same = all(r['counters'] == ref['counters'] and r['bins'] == ref['bins'] for r in good[1:])
            agree &= same
            lines.append(f"{'IDENTICAL' if same else 'DIFFERENT'} repetitions {key[0]} {key[1]}: sets "
                         f"{[r['set'] for r in good]}" + ('' if same else ' ' + str(
                             [(r['set'], r['counters'], r['bins'][:16]) for r in good])))
    return lines, agree


def write_records(out, runs):
    keep = ('system', 'set', 'arm', 'host', 'binary_sha256', 'kernel_sha256', 'module_args', 'invocation', 'exit',
            'seconds', 'total_seconds', 'tests', 'report', 'qualified', 'problems')
    rows = [{k: r[k] for k in keep if k in r} for r in runs]
    (out / 'runs.json').write_text(json.dumps(rows, indent=1) + '\n')
    # plot.py reads RESULTS_JSONL#ROW with the report lines as stderr.
    (out / 'reports.jsonl').write_text(''.join(json.dumps(
        {'name': f"{r['system']}-set{r['set']}-{r['arm']}", 'stderr': '\n'.join(x for x in r['report'] if x)}) + '\n'
        for r in runs))


def plot_systems(out, runs):
    """(lines, all plotted): per system, the lowest set whose two runs are admissible, preferring qualified
    binaries, on an x axis shared by both systems."""
    lines, ok = [], True
    good = [r for r in runs if not r['problems']]
    last = max([b for r in good for b in range(32) if r['bins'][b]] + [0]) + 2
    xmax = 2 ** last - 1
    for system, name in (('poisoncap', 'PoisonCap'), ('sublet', 'Sublet')):
        pairs = {}
        for r in good:
            if r['system'] == system:
                pairs.setdefault(r['set'], {})[r['arm']] = r
        full = sorted(s for s, d in pairs.items() if set(d) == {'off', 'on'})
        qual = [s for s in full if pairs[s]['off']['qualified'] and pairs[s]['on']['qualified']]
        pick = (qual or full or [None])[0]
        if pick is None:
            lines.append(f'PLOT {name}: no admissible off/on pair yet')
            continue
        stem = out / 'plots' / f'{system}-size100'
        cmd = [sys.executable, str(HERE / 'plot.py'), str(stem), '--xmax', str(xmax),
               '--control', f"Temporal off={out / 'reports.jsonl'}#{system}-set{pick}-off",
               '--protected', f"Temporal on={out / 'reports.jsonl'}#{system}-set{pick}-on"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        ok &= res.returncode == 0
        said = (res.stdout + res.stderr).strip().replace(f'{out}/', '')   # paths relative to OUT
        lines.append(f"PLOT {name}: set {pick}{'' if pick in qual else ' (PRELIMINARY: not the qualified binaries)'}"
                     f", xmax {xmax:,}: {records.scrub(said)}")
    return lines, ok


def main():
    mirror, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs = discover(mirror)
    if not runs:
        sys.exit(f'no finished run under {mirror}')
    for r in runs:
        check(r)
    lines = verdict_lines(runs)
    repetitions, agree = repetition_lines(runs)
    lines += repetitions
    ok = agree and not any(r['problems'] for r in runs)
    write_records(out, runs)
    if '--plot' in sys.argv[3:]:
        plotted, plots_ok = plot_systems(out, runs)
        lines += plotted
        ok &= plots_ok
    (out / 'checks.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
