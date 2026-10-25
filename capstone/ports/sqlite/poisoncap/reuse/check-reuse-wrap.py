#!/usr/bin/env python3
"""Check reuse-wrap.c against test-reuse-wrap.c's own record of its calls.

    check-reuse-wrap.py WRAPPED_STDOUT WRAPPED_STDERR UNWRAPPED_STDOUT TRACED_STDERR

The oracle reads the call record (M/F/R lines up to ALLOC-DONE) and recomputes, by its own code, what
the wrapper must report at the "alloc-done" snapshot: attempts, issues, releases, reuses, distinct,
failed, moved, in place and the 32 bins. sqlite3_realloc returning the same address is no event
(SQLite may not even call xRealloc then). It also requires the "start" snapshot (after
sqlite3_initialize) to be empty, the reconciliation identities at exit, the SQL positive control
(steps 101 x4 then 19; resets 0 x4 then 19; SQL-CHECK step_other=1 first=19 reset_err=1 first=19,
step_done 4 or, with lld wrapping sqlite3_exec's own steps, more) and, given the unwrapped run's stdout, that the program's output, every returned address
included, is identical with and without the wrapper. sqlite3_initialize itself allocates and frees
through memsys5 before the program's first call; the oracle is seeded with those events, read from a
build with -DREUSE_WRAP_TRACE (its "EV" lines before the "start" snapshot), and that build's reports must
equal the untraced build's. Exit 0 only if every check holds.
"""
import os
import re
import sys

FIELDS = ('attempts', 'issues', 'releases', 'reuses', 'distinct', 'failed', 'moved', 'inplace')


def snapshot(err, label):
    m = re.search(rf'^REUSE-GAP memsys5 at={label} (.*)$', err, re.M)
    if not m:
        return None
    d = dict(kv.split('=', 1) for kv in m.group(1).split())
    d['bins'] = [int(x) for x in d['bins'].split(',')]
    return d


def oracle(out, pre):
    attempts = issues = releases = reuses = failed = moved = inplace = 0
    state = {}  # address -> 'live' or attempts at release
    bins = [0] * 32
    distinct = 0
    for kind, a, at in pre:  # sqlite3_initialize's own events, before the program's first call
        attempts = max(attempts, at)
        if kind == 'I':
            state[a] = 'live'
            issues += 1
            distinct += 1
        else:
            state[a] = at
            releases += 1

    def issue(a):
        nonlocal issues, reuses, distinct
        s = state.get(a)
        assert s != 'live', f'oracle: issue of live {a:x}'
        if s is None:
            distinct += 1
        else:
            gap = attempts - s
            b = 0
            while b < 31 and gap > (1 << (b + 1)) - 1:
                b += 1
            bins[b] += 1
            reuses += 1
        state[a] = 'live'
        issues += 1

    def release(a):
        nonlocal releases
        assert state.get(a) == 'live', f'oracle: release of {a:x} not live'
        state[a] = attempts
        releases += 1

    for line in out.split('ALLOC-DONE')[0].splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == 'M':
            attempts += 1
            if f[2] != 'null':
                issue(int(f[2], 16))
            else:
                failed += 1
        elif f[0] == 'F':
            release(int(f[1], 16))
        elif f[0] == 'R':
            old, new = f[1], f[3]
            if new == old:
                inplace += 1  # counted by the wrapper only if SQLite called xRealloc: not compared
                continue
            attempts += 1
            old = int(old, 16)
            if new == 'null':
                failed += 1
                continue
            moved += 1
            issue(int(new, 16))
            release(old)
    return dict(attempts=attempts, issues=issues, releases=releases, reuses=reuses, distinct=distinct,
                failed=failed, moved=moved, inplace=inplace, bins=bins)


def main():
    out, err = open(sys.argv[1]).read(), open(sys.argv[2]).read()
    plain, traced = open(sys.argv[3]).read(), open(sys.argv[4]).read()
    pre = [(k, int(a, 16), int(at)) for k, a, at in
           re.findall(r'^EV ([IF]) ([0-9a-f]+) (\d+)$', traced.split('at=start')[0], re.M)]
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'} {name}{(': ' + detail) if detail and not cond else ''}")
        ok &= bool(cond)

    start, done, end = snapshot(err, 'start'), snapshot(err, 'alloc-done'), snapshot(err, 'exit')
    check('three snapshots present', start and done and end)
    if not (start and done and end):
        return 1
    check('installed', done['installed'] == '1')
    # zip() and sum() over a short list would pass a report that dropped bins: require all 32.
    for label, d in (('start', start), ('alloc-done', done), ('exit', end)):
        check(f'{label} reports 32 bins', len(d['bins']) == 32, str(len(d['bins'])))
    check('start snapshot holds exactly the seeded events', int(start['attempts']) == max([0] + [x[2] for x in pre])
          and int(start['issues']) == sum(1 for x in pre if x[0] == 'I'), f"start attempts={start['attempts']}")
    exp = oracle(out, pre)
    print(f'oracle seeded with {len(pre)} events of sqlite3_initialize')
    # Only the wrapper's own lines: a platform's launcher may add its own (with per-process addresses).
    strip = lambda e: [l for l in e.splitlines() if l.startswith(('REUSE-GAP', 'SQL-CHECK'))]
    check('traced build reports exactly what the untraced build reports', strip(traced) == strip(err))
    for k in FIELDS:
        if k == 'inplace':
            continue
        check(f'alloc-done {k}', int(done[k]) == exp[k], f'wrapper {done[k]} oracle {exp[k]}')
    check('alloc-done bins', done['bins'] == exp['bins'], f"wrapper {done['bins']} oracle {exp['bins']}")
    if os.environ.get('REUSE_EXPECT_QUARANTINE') == '1':
        # An allocator that withholds freed blocks (PoisonCap's quarantine of 4,096 entries) returns none
        # within a short sequence: the oracle then covers issues, releases and moves, not the bins (the
        # bin code is the same on every platform and is checked where reuse happens).
        check('quarantining allocator: no reuse in the sequence, moves exercised',
              exp['reuses'] == 0 and exp['moved'] > 10, f"reuses={exp['reuses']} moved={exp['moved']}")
    else:
        check('oracle exercised reuse and moves', exp['reuses'] > 100 and exp['moved'] > 10,
              f"reuses={exp['reuses']} moved={exp['moved']}")
    for label, d in (('alloc-done', done), ('exit', end)):
        n = {k: int(d[k]) for k in FIELDS}
        check(f'{label} error=0', d['error'] == '0', d.get('first_error', ''))
        check(f'{label} attempts == issues + failed', n['attempts'] == n['issues'] + n['failed'])
        check(f'{label} issues == distinct + reuses', n['issues'] == n['distinct'] + n['reuses'])
        check(f'{label} sum(bins) == reuses', sum(d['bins']) == n['reuses'])
        check(f'{label} live == issues - releases', int(d['live']) == n['issues'] - n['releases'])
    # The exit report has no oracle of its own (the SQL part's allocations are SQLite's), but it must
    # extend the alloc-done snapshot: every counter and every bin at least as large, and the SQL part
    # must have allocated (a wrapper that zeroed or rewrote its counters at exit fails here).
    for k in FIELDS:
        check(f'exit {k} >= alloc-done {k}', int(end[k]) >= int(done[k]), f'{end[k]} < {done[k]}')
    check('exit attempts > alloc-done attempts (the SQL part allocates)', int(end['attempts']) > int(done['attempts']))
    check('exit bins >= alloc-done bins, bin by bin', all(e >= d for e, d in zip(end['bins'], done['bins'])))
    steps = [int(x) for x in re.findall(r'^STEP (-?\d+)$', out, re.M)]
    resets = [int(x) for x in re.findall(r'^RESET (-?\d+)$', out, re.M)]
    check('SQL steps 101 x4 then 19', steps == [101] * 4 + [19], str(steps))
    check('SQL resets 0 x4 then 19', resets == [0] * 4 + [19], str(resets))
    found = re.findall(r'^SQL-CHECK (.*)$', err, re.M)  # one per snapshot: the last is at exit
    sql = dict(kv.split('=', 1) for kv in found[-1].split()) if found else {}
    # step_done is 4 with GNU ld (the program's own calls only) and more with lld, which also wraps
    # sqlite3_exec's internal steps; the error itself must be counted exactly once either way.
    check('SQL-CHECK catches the one error',
          int(sql.get('step_done', 0)) >= 4 and sql.get('step_other') == '1' and sql.get('first_step_other') == '19'
          and sql.get('reset_err') == '1' and sql.get('first_reset_err') == '19', str(sql))
    check('output identical without the wrapper (every address)', plain == out,
          f'{len(plain.splitlines())} vs {len(out.splitlines())} lines')
    print('ALL PASS' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
