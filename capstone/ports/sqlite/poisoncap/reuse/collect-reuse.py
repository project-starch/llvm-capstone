#!/usr/bin/env python3
"""Collect the --size 100 result runs of the reuse-distance comparison, check each, compare the repetitions,
and plot one figure per system.

    collect-reuse.py MIRROR OUT [--plot]

MIRROR holds copies of the hosts' run directories, one per host: MIRROR/<host>/speedtest1/reuse-size100/
set<N>-poisoncap-{off,on}/result.json (speedtest1/run.py) and MIRROR/<host>/reuse-capstone-runs/
set<N>-sublet-{off,on}/results.jsonl (run-reuse-capstone.py, row <arm>-size100: spatial is off,
block-bounds on). A run directory without its record is still running and is skipped.

OUT gets runs.json (each run's result lines only: hashes, exit status, the test times, the wrapper's two
report lines; home paths scrubbed), checks.txt (every check, every run) and, with --plot, plots/ from
plot-reuse.py: per system, the lowest set whose two runs are admissible and use qualified binaries, with
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
import records  # noqa: E402

TESTS = [100, 110, 120, 130, 140, 142, 145, 160, 190, 210, 240, 250, 260, 280, 300, 320, 400, 410, 980, 990]
TEST = re.compile(r'^\s*(\d{3}) - .*?\.{3,}\s*([0-9.]+)s\s*$', re.M)
TOTAL = re.compile(r'^\s*TOTAL\.+\s*([0-9.]+)s\s*$', re.M)
# The binaries each arm was qualified with (qualification.txt); others are plotted only as preliminary.
QUALIFIED = {
    ('poisoncap', 'off'): {'aad6a979'}, ('poisoncap', 'on'): {'97b02659'},
    ('sublet', 'off'): {'9836684d'}, ('sublet', 'on'): {'6bfe4d1b'},
}
SUBLET_ROW = {'off': 'spatial-size100', 'on': 'block-bounds-size100'}
FIELDS = ('attempts', 'issues', 'releases', 'reuses', 'distinct', 'failed', 'moved', 'inplace', 'live')


def report_lines(err):
    gap = re.findall(r'^REUSE-GAP memsys5 at=exit .*$', err, re.M)
    sql = re.findall(r'^SQL-CHECK .*$', err, re.M)
    return [gap[-1] if gap else None, sql[-1] if sql else None]


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
    """Fill run['problems'] (empty: admissible) and run['report'] (parsed exit report)."""
    p = []
    tests = [(int(i), float(s)) for i, s in TEST.findall(run['stdout'])]
    run['tests'] = [[i, s] for i, s in tests]
    tot = TOTAL.findall(run['stdout'])
    run['total_seconds'] = float(tot[-1]) if tot else None
    run['report'] = report_lines(run['stderr'])
    if run['exit'] != 0:
        p.append(f"exit {run['exit']}")
    if [i for i, _ in tests] != TESTS:
        p.append(f'tests completed: {len(tests)} of 20')
    gap, sql = run['report']
    if not gap:
        p.append('no REUSE-GAP report at exit')
    else:
        d = dict(kv.split('=', 1) for kv in gap.split()[3:])
        bins = [int(x) for x in d['bins'].split(',')]
        n = {k: int(d[k]) for k in FIELDS}
        run['counters'], run['bins'] = n, bins
        if d['installed'] != '1' or d['error'] != '0':
            p.append(f"installed={d['installed']} error={d['error']} first_error={d.get('first_error')}")
        if len(bins) != 32:
            p.append(f'{len(bins)} bins')
        if n['attempts'] != n['issues'] + n['failed']:
            p.append('attempts != issues + failed')
        if n['issues'] != n['distinct'] + n['reuses']:
            p.append('issues != distinct + reuses')
        if sum(bins) != n['reuses']:
            p.append('sum(bins) != reuses')
        if n['live'] != n['issues'] - n['releases']:
            p.append('live != issues - releases')
        if n['issues'] < 1000:
            p.append(f"issues {n['issues']} < 1000")
    if not sql:
        p.append('no SQL-CHECK line')
    else:
        s = dict(kv.split('=', 1) for kv in sql.split()[1:])
        if s.get('step_other') != '0' or s.get('reset_err') != '0':
            p.append(f'SQL errors: {sql}')
    run['problems'] = p
    run['qualified'] = (run['binary_sha256'] or '')[:8] in QUALIFIED[(run['system'], run['arm'])]


def main():
    mirror, out = Path(sys.argv[1]), Path(sys.argv[2])
    plot = '--plot' in sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    runs = discover(mirror)
    if not runs:
        sys.exit(f'no finished run under {mirror}')
    lines, ok = [], True
    for r in runs:
        check(r)
        tag = f"{r['system']} set{r['set']} {r['arm']:3} ({r['host']}, {(r['binary_sha256'] or '?')[:8]}"
        tag += ', qualified binary)' if r['qualified'] else ', NOT a qualified binary)'
        if r['problems']:
            ok = False
            lines.append(f"INADMISSIBLE {tag}: {'; '.join(r['problems'])}")
        else:
            c = r['counters']
            lines.append(f"ADMISSIBLE {tag}: issues={c['issues']} reuses={c['reuses']} failed={c['failed']} "
                         f"total={r['total_seconds']}s")
    # Repetitions of one arm must agree exactly.
    for key in sorted({(r['system'], r['arm']) for r in runs}):
        good = [r for r in runs if (r['system'], r['arm']) == key and not r['problems']]
        if len(good) > 1:
            ref = good[0]
            same = all(r['counters'] == ref['counters'] and r['bins'] == ref['bins'] for r in good[1:])
            ok &= same
            lines.append(f"{'IDENTICAL' if same else 'DIFFERENT'} repetitions {key[0]} {key[1]}: sets "
                         f"{[r['set'] for r in good]}" + ('' if same else ' ' + str(
                             [(r['set'], r['counters'], r['bins'][:16]) for r in good])))
    keep = ('system', 'set', 'arm', 'host', 'binary_sha256', 'kernel_sha256', 'module_args', 'invocation', 'exit',
            'seconds', 'total_seconds', 'tests', 'report', 'qualified', 'problems')
    rows = [{k: r[k] for k in keep if k in r} for r in runs]
    (out / 'runs.json').write_text(json.dumps(rows, indent=1) + '\n')
    # plot-reuse.py reads RESULTS_JSONL#ROW with the report lines as stderr.
    (out / 'reports.jsonl').write_text(''.join(json.dumps(
        {'name': f"{r['system']}-set{r['set']}-{r['arm']}", 'stderr': '\n'.join(x for x in r['report'] if x)}) + '\n'
        for r in runs))
    if plot:
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
            cmd = [sys.executable, str(HERE / 'plot-reuse.py'), str(stem), '--xmax', str(xmax),
                   '--control', f"Temporal off={out / 'reports.jsonl'}#{system}-set{pick}-off",
                   '--protected', f"Temporal on={out / 'reports.jsonl'}#{system}-set{pick}-on"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            ok &= res.returncode == 0
            said = (res.stdout + res.stderr).strip().replace(f'{out}/', '')   # paths relative to OUT
            lines.append(f"PLOT {name}: set {pick}{'' if pick in qual else ' (PRELIMINARY: not the qualified binaries)'}"
                         f", xmax {xmax:,}: {records.scrub(said)}")
    (out / 'checks.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
