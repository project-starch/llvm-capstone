#!/usr/bin/env python3
"""check.py BOOT_LOG: the verdicts of the layout-fix unit test (run.sh).

header  A live caller loads the words in front of its chunk. The unfixed image (the patch of
        5060011844e) finds the block's tagged reference in the header and exits 23. The fixed
        image faults with cause 28 at the first load, so the header lies outside the chunk's
        bounds, and the domain dies by SIGSEGV (status 139).
stale   A load through a pointer after pfree faults with cause 25 in both images.
table   Every header reader through the block table succeeds in both images.
The fixed image also passes the lifetime-failures modes: CDERIVE fault injection, large sizes and
the failure-atomic large realloc.
"""
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text().replace('\r', '')


def segment(arm, mode):
    m = re.search(rf'^[^\n]*TRAPTEST: begin {arm} {mode}\n(.*?)^[^\n]*TRAPTEST: end {arm} {mode} status (\d+)$',
                  log, re.M | re.S)
    assert m, f'{arm} {mode}: no reading'
    return m.group(1), int(m.group(2))


body, status = segment('unfixed', 'header')
leak = re.search(r'TRAPTEST: postgres header reference at -(\d+) covers (\d+) bytes', body)
assert leak and status == 23, f'unfixed header: {status}\n{body}'
print(f'unfixed header: tagged block reference at -{leak.group(1)}, bounds {leak.group(2)} bytes '
      '(negative control holds)')

body, status = segment('fixed', 'header')
first = body.split('TRAPTEST: postgres header load -16 armed\n', 1)
assert len(first) == 2 and 'sublet: fault cause=28 ' in first[1], f'fixed header: no cause 28\n{body}'
assert 'header reference' not in body and 'load -32 armed' not in body and status == 139, \
    f'fixed header: {status}\n{body}'
print('fixed header: the first load in front of the chunk faults with cause 28')

for arm in ('fixed', 'unfixed'):
    body, status = segment(arm, 'stale')
    after = body.split('TRAPTEST: postgres stale armed\n', 1)
    assert len(after) == 2 and 'sublet: fault cause=25 ' in after[1], f'{arm} stale\n{body}'
    print(f'{arm} stale: cause 25')
    body, status = segment(arm, 'table')
    assert 'TRAPTEST: postgres table PASS' in body and status == 0, f'{arm} table: {status}\n{body}'
    print(f'{arm} table: PASS')

for mode, line in (('injection', 'TRAPTEST: postgres injection PASS'),
                   ('large', 'TRAPTEST: postgres large success'),
                   ('external', None)):
    body, status = segment('fixed', mode)
    assert status == 0 and (line is None or line in body), f'fixed {mode}: {status}\n{body}'
    print(f'fixed {mode}: PASS')
assert 'TRAPTEST: done' in log
print('layout-fix unit test PASS')
