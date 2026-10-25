#!/usr/bin/env python3
"""Qualify the reuse-distance instrument on a platform before any speedtest1 number counts.

    qualify-reuse.py poisoncap QUAL_DIR          (speedtest1-guest.py runs: QUAL_DIR/<name>/result.json)
    qualify-reuse.py capstone RESULTS_JSONL ARM  (run-reuse-capstone.py rows named <arm>-<name>)

Names: test-wrapped, test-traced, test-plain, st-wrapped-size1 (or st-size1), st-plain-size1.
  1. check-reuse-wrap.py on the three test programs (the independent oracle, the identities, the SQL
     positive control, and identical output with and without the wrapper);
  2. the speedtest1 pilot: wrapped and unwrapped exit 0 and complete the same tests; the wrapper reports
     installed=1, error=0 and its identities; SQL-CHECK reports no step or reset error.
Prints PASS/FAIL per check; exit 0 only if all pass.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TEST = re.compile(r'^\s*(\d{3}) - .*?\.{3,}\s*[0-9.]+s\s*$', re.M)


def load(kind, where, arm):
    runs = {}
    if kind == 'poisoncap':
        for d in Path(where).iterdir():
            f = d / 'result.json'
            if f.exists():
                r = json.loads(f.read_text())
                runs[d.name] = {'rc': r.get('exit'), 'stdout': r.get('stdout') or '', 'stderr': r.get('stderr') or ''}
    else:
        for line in Path(where).read_text().splitlines():
            r = json.loads(line)
            if r['name'].startswith(arm + '-'):
                runs[r['name'][len(arm) + 1:]] = {'rc': r.get('rc'), 'stdout': r.get('stdout') or '',
                                                  'stderr': r.get('stderr') or ''}
    if 'st-size1' in runs:
        runs['st-wrapped-size1'] = runs.pop('st-size1')
    return runs


def report(err):
    m = re.findall(r'^REUSE-GAP memsys5 at=exit (.*)$', err, re.M)
    if not m:
        return None
    d = dict(kv.split('=', 1) for kv in m[-1].split())
    d['bins'] = [int(x) for x in d['bins'].split(',')]
    return d


def main():
    kind, where = sys.argv[1], sys.argv[2]
    arm = sys.argv[3] if kind == 'capstone' else 'poisoncap'
    runs = load(kind, where, arm)
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'} [{arm}] {name}{(': ' + str(detail)) if detail and not cond else ''}")
        ok &= bool(cond)

    need = ['test-wrapped', 'test-traced', 'test-plain', 'st-wrapped-size1', 'st-plain-size1']
    check('all five processes present', all(n in runs for n in need), sorted(runs))
    if not all(n in runs for n in need):
        return 1
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / 'w.out').write_text(runs['test-wrapped']['stdout'])
        (t / 'w.err').write_text(runs['test-wrapped']['stderr'])
        (t / 'p.out').write_text(runs['test-plain']['stdout'])
        (t / 't.err').write_text(runs['test-traced']['stderr'])
        r = subprocess.run([sys.executable, str(HERE / 'check-reuse-wrap.py'), t / 'w.out', t / 'w.err',
                            t / 'p.out', t / 't.err'], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith('FAIL'):
                print(f'  {line}')
        check('test programs: check-reuse-wrap.py ALL PASS', r.returncode == 0, r.stdout.strip().splitlines()[-1:])
    w, p = runs['st-wrapped-size1'], runs['st-plain-size1']
    check('pilot: both exit 0', w['rc'] == 0 and p['rc'] == 0, (w['rc'], p['rc']))
    tw, tp = TEST.findall(w['stdout']), TEST.findall(p['stdout'])
    check('pilot: same tests completed', tw == tp and len(tw) > 0, (tw, tp))
    rep = report(w['stderr'])
    check('pilot: wrapper report at exit', rep is not None)
    if rep:
        n = {k: int(rep[k]) for k in ('attempts', 'issues', 'releases', 'reuses', 'distinct', 'failed', 'live')}
        check('pilot: installed=1 error=0', rep['installed'] == '1' and rep['error'] == '0', rep.get('first_error'))
        check('pilot: attempts == issues + failed', n['attempts'] == n['issues'] + n['failed'])
        check('pilot: issues == distinct + reuses', n['issues'] == n['distinct'] + n['reuses'])
        check('pilot: sum(bins) == reuses', sum(rep['bins']) == n['reuses'])
        check('pilot: live == issues - releases', n['live'] == n['issues'] - n['releases'])
        print(f"  [{arm}] pilot: attempts={n['attempts']} reuses={n['reuses']} distinct={n['distinct']} "
              f"failed={n['failed']} live={n['live']} moved={rep['moved']} bins={rep['bins'][:16]}")
    sql = re.findall(r'^SQL-CHECK (.*)$', w['stderr'], re.M)
    s = dict(kv.split('=', 1) for kv in sql[-1].split()) if sql else {}
    check('pilot: no SQL step or reset error', s.get('step_other') == '0' and s.get('reset_err') == '0', s)
    print('ALL PASS' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
