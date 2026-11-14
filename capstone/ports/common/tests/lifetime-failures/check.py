#!/usr/bin/env python3
"""Check unit verdicts and associate each stale probe with its kernel fault."""
import sys
from pathlib import Path
s = Path(sys.argv[1]).read_text().replace('\r', '')
for port in ['postgres', 'mruby', 'perl']:
    assert f'TRAPTEST: {port} injection PASS' in s, port
    assert f'TRAPTEST: {port} fixed status 0' in s, port
    control = 11 if port == 'postgres' else 139
    assert f'TRAPTEST: {port} unfixed status {control}' in s, port
    segment = s.split(f'TRAPTEST: {port} stale armed\n', 1)[1]
    segment = segment.split(f'TRAPTEST: {port} stale status', 1)[0]
    assert 'sublet: fault cause=25 ' in segment, port
    print(port, 'injection, negative control, stale cause 25 PASS')
assert 'TRAPTEST: postgres large success' in s, 'postgres large'
assert 'TRAPTEST: postgres large status 0' in s, 'postgres large'
assert 'TRAPTEST: postgres unfixed large status 0' not in s, 'postgres large control'
assert 'TRAPTEST: postgres external status 0' in s, 'postgres external'
assert 'TRAPTEST: postgres unfixed external status 0' not in s, 'postgres external control'
print('postgres large sizes, failure-atomic large realloc and its control PASS')
assert 'TRAPTEST: done' in s
