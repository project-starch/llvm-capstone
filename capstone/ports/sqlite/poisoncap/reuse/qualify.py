#!/usr/bin/env python3
"""Qualify the reuse-distance instrument on a platform before any speedtest1 number counts.

    qualify.py poisoncap QUAL_DIR          (speedtest1/run.py runs: QUAL_DIR/<name>/result.json)
    qualify.py capstone RESULTS_JSONL ARM  (run-sublet.py rows named <arm>-<name>)

Names: test-wrapped, test-traced, test-plain, st-wrapped-size1 (or st-size1), st-plain-size1.
  1. check-wrapper.py on the three test programs (the independent oracle, the identities, the SQL
     positive control, and identical output with and without the wrapper);
  2. the speedtest1 pilot: wrapped and unwrapped exit 0 and complete the same tests; the wrapper reports
     installed=1, error=0 and its identities; SQL-CHECK reports no step or reset error.
Prints PASS/FAIL per check; exit 0 only if all pass.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reuse_report  # noqa: E402

PROCESSES = ['test-wrapped', 'test-traced', 'test-plain', 'st-wrapped-size1', 'st-plain-size1']
# The counters the pilot line shows and the workload check reports.
PILOT_COUNTERS = ('attempts', 'issues', 'releases', 'reuses', 'distinct', 'failed', 'live')


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


class Checks:
    """Prints PASS or FAIL per check, tagged with the arm; ok stays True only while every check holds."""

    def __init__(self, arm):
        self.arm, self.ok = arm, True

    def __call__(self, name, cond, detail=''):
        print(f"{'PASS' if cond else 'FAIL'} [{self.arm}] {name}{(': ' + str(detail)) if detail and not cond else ''}")
        self.ok &= bool(cond)


def check_test_programs(check, runs):
    """The three test programs exit 0, and check-wrapper.py passes on their output."""
    for n in ('test-wrapped', 'test-traced', 'test-plain'):
        check(f'{n} exit 0', runs[n]['rc'] == 0, runs[n]['rc'])
    check('test-traced stdout equals test-wrapped stdout', runs['test-traced']['stdout'] == runs['test-wrapped']['stdout'])
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / 'w.out').write_text(runs['test-wrapped']['stdout'])
        (t / 'w.err').write_text(runs['test-wrapped']['stderr'])
        (t / 'p.out').write_text(runs['test-plain']['stdout'])
        (t / 't.err').write_text(runs['test-traced']['stderr'])
        r = subprocess.run([sys.executable, str(HERE / 'check-wrapper.py'), t / 'w.out', t / 'w.err',
                            t / 'p.out', t / 't.err'], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith('FAIL'):
                print(f'  {line}')
        check('test programs: check-wrapper.py ALL PASS', r.returncode == 0, r.stdout.strip().splitlines()[-1:])


def check_pilot(check, w, p, arm):
    """speedtest1 at --size 1, wrapped (W) and unwrapped (P): the same 20 tests, an admissible report
    that saw the workload, and no SQL error."""
    check('pilot: both exit 0', w['rc'] == 0 and p['rc'] == 0, (w['rc'], p['rc']))
    tw = [i for i, _ in reuse_report.TEST.findall(w['stdout'])]
    tp = [i for i, _ in reuse_report.TEST.findall(p['stdout'])]
    check('pilot: same 20 tests completed', tw == tp and len(tw) == 20, (len(tw), len(tp)))
    rep = reuse_report.snapshot(w['stderr'])
    check('pilot: wrapper report at exit', rep is not None)
    if rep:
        n = {k: int(rep[k]) for k in PILOT_COUNTERS}
        check('pilot: installed=1 error=0', rep['installed'] == '1' and rep['error'] == '0', rep.get('first_error'))
        quarantine = os.environ.get('REUSE_EXPECT_QUARANTINE') == '1'
        check('pilot: the observer saw the workload (>= 1000 allocations issued, none failed, '
              + ('a quarantining allocator: no reuse required)' if quarantine else '>= 100 reuses)'),
              n['issues'] >= 1000 and n['failed'] == 0 and (quarantine or n['reuses'] >= 100), n)
        counts = reuse_report.counters(rep)
        for name, holds in reuse_report.IDENTITIES:
            check(f'pilot: {name}', holds(counts, rep['bins']))
        print(f"  [{arm}] pilot: attempts={n['attempts']} reuses={n['reuses']} distinct={n['distinct']} "
              f"failed={n['failed']} live={n['live']} moved={rep['moved']} bins={rep['bins'][:16]}")
    line = reuse_report.sql_line(w['stderr'])
    s = reuse_report.fields(line) if line else {}
    check('pilot: no SQL step or reset error', s.get('step_other') == '0' and s.get('reset_err') == '0', s)
    check('pilot: SQL calls were counted (>= 1000 steps)', int(s.get('step_row', 0)) + int(s.get('step_done', 0)) >= 1000, s)


def main():
    kind, where = sys.argv[1], sys.argv[2]
    arm = sys.argv[3] if kind == 'capstone' else 'poisoncap'
    runs = load(kind, where, arm)
    check = Checks(arm)
    check('all five processes present', all(n in runs for n in PROCESSES), sorted(runs))
    if not all(n in runs for n in PROCESSES):
        return 1
    check_test_programs(check, runs)
    check_pilot(check, runs['st-wrapped-size1'], runs['st-plain-size1'], arm)
    print('ALL PASS' if check.ok else 'FAILED')
    return 0 if check.ok else 1


if __name__ == '__main__':
    sys.exit(main())
