#!/usr/bin/env python3
"""check.py RESULTS_JSON: each step of run.sh against the plan's table; exit 1 on any miss.

    scan    old: LEAK at slab-chunk offsets 0 and 16 and at cache-object offset 0, exit 1
            new: no LEAK, exit 0
    stale   both: SIGSEGV with cause 25 at the announced access stale-chunk
    bounds  both: chunk-last-byte reads, then SIGSEGV with cause 28 at one-past-chunk
    move    both: the other page's chunk, the earlier page's chunk and the taker's chunk read back
            what was written, the taker sits in the moved page, then SIGSEGV with cause 25 at
            moved-page-chunk
    move-prealloc (new): as move, without the taker
"""
import json
import re
import sys


def last_access(text):
    begun = re.findall(r'^ACCESS-BEGIN (\S+)', text, re.M)
    return begun[-1] if begun else None


def done(text, name, value):
    return re.search(rf'^ACCESS-DONE {re.escape(name)} value={value}$', text, re.M) is not None


def fault_at(r, cause, access):
    misses = []
    if r['kind'] != 'signal' or r['value'] != 11:
        misses.append(f"ended {r['kind']} {r['value']}, not SIGSEGV")
    if not r['fault'] or f'cause={cause} ' not in r['fault'] + ' ':
        misses.append(f"fault line {r['fault']!r}, not cause {cause}")
    if last_access(r['output']) != access:
        misses.append(f"last announced access {last_access(r['output'])}, not {access}")
    return misses


def check(name, r):
    arm, _, mode = name.partition('-')
    text = r['output']
    leaks = set(re.findall(r'^LEAK (\S+) offset=(-?\d+)', text, re.M))
    if mode == 'scan' and arm == 'old':
        misses = [f'no LEAK {o} offset={n}' for o, n in
                  (('slab-chunk', '0'), ('slab-chunk', '16'), ('cache-object', '0')) if (o, n) not in leaks]
        if r['kind'] != 'exit' or r['value'] != 1:
            misses.append(f"ended {r['kind']} {r['value']}, not exit 1")
        return misses
    elif mode == 'scan' and arm == 'new':
        misses = [f'LEAK {o} offset={n}' for o, n in sorted(leaks)]
        if r['kind'] != 'exit' or r['value'] != 0:
            misses.append(f"ended {r['kind']} {r['value']}, not exit 0")
        return misses
    elif mode == 'stale':
        return fault_at(r, 25, 'stale-chunk')
    elif mode == 'bounds':
        misses = [] if re.search(r'^ACCESS-DONE chunk-last-byte ', text, re.M) else ['chunk-last-byte did not read']
        return misses + fault_at(r, 28, 'one-past-chunk')
    elif mode in ('move', 'move-prealloc'):
        misses = [f'{n} did not read back {v}' for n, v in
                  (('other-page-chunk', '0xb1'), ('earlier-page-chunk', '0xe1')) if not done(text, n, v)]
        if mode == 'move':
            if not done(text, 'taker-chunk', '0xc1'):
                misses.append('taker-chunk did not read back 0xc1')
            if not re.search(r'^REUSED .* in_moved_page=1$', text, re.M):
                misses.append('the taker is not in the moved page')
        return misses + fault_at(r, 25, 'moved-page-chunk')
    else:
        return [f'unknown step {name}']


def main():
    results = json.load(open(sys.argv[1]))
    failed = 0
    for r in results:
        misses = check(r['name'], r)
        print(f"{'PASS' if not misses else 'FAIL'} {r['name']}: {r['kind']} {r['value']}"
              + (f" | {r['fault']}" if r['fault'] else '') + ''.join(f'\n    {m}' for m in misses))
        failed += bool(misses)
    expected = {f'old-{m}' for m in ('scan', 'stale', 'bounds', 'move')} | \
               {f'new-{m}' for m in ('scan', 'stale', 'bounds', 'move', 'move-prealloc')}
    absent = sorted(expected - {r['name'] for r in results})
    for name in absent:
        print(f'FAIL {name}: no result')
    print(f'sublet-layout: {len(results) - failed} of {len(expected)} steps as predicted')
    return 1 if failed or absent else 0


if __name__ == '__main__':
    sys.exit(main())
