#!/usr/bin/env python3
"""Prepare commands-only replay inputs or compare independently observed events."""
import argparse
import collections
import csv
import json
from pathlib import Path
import struct

MAGIC = 0x4650465452433031
HEADER = struct.Struct('<4Q')
EVENT = struct.Struct('<10Q')
FIELDS = ('op', 'pool', 'lease', 'backing', 'size', 'aux', 'allocations', 'gap',
          'live_bytes', 'retained_bytes')

def read(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f'{path}: short header')
    magic, count, status, arena = HEADER.unpack_from(data)
    if magic != MAGIC or len(data) != HEADER.size + count * EVENT.size:
        raise ValueError(f'{path}: invalid format or truncated file')
    rows = list(EVENT.iter_unpack(data[HEADER.size:]))
    if status or not rows or rows[-1][0] != 5:
        raise ValueError(f'{path}: incomplete run, status={status}, events={count}')
    return rows, arena

def prepare(source, out):
    rows, _ = read(source)
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    for row in rows:
        row = list(row)
        # Erase every recorded outcome; the replayer cannot copy one by mistake.
        for i in (3, 6, 7, 8, 9):
            row[i] = 0
        commands.append(row)
    data = HEADER.pack(MAGIC, len(rows), 0, 0)
    data += b''.join(EVENT.pack(*r) for r in commands)
    (out / 'commands.bin').write_bytes(data)
    bad = bytearray(data)
    for i, row in enumerate(commands):
        if row[0] == 3:
            struct.pack_into('<Q', bad, HEADER.size + i * EVENT.size + 16, 2**63)
            break
    else:
        raise ValueError('no RETURN available for negative control')
    (out / 'invalid-return.bin').write_bytes(bad)
    print(f'prepared {len(rows)} commands; all recorded outcomes removed')

def stats(rows):
    gets = [r for r in rows if r[0] == 2]
    gaps = collections.Counter(r[7] for r in gets if r[7])
    return dict(events=len(rows), pools=sum(r[0] == 1 for r in rows),
                allocations=len(gets), backing_allocations=sum(r[7] == 0 for r in gets),
                reuses=sum(gaps.values()), reuse_fraction=sum(gaps.values()) / len(gets),
                max_gap=max(gaps, default=0), gap_histogram=dict(sorted(gaps.items())),
                peak_live_bytes=max(r[8] for r in rows),
                peak_retained_bytes=max(r[9] for r in rows),
                final_live_bytes=rows[-1][8], final_retained_bytes=rows[-1][9])

def compare(original, native, capstone, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, _ = read(original)
    result = {'scope': 'AVBufferPool payload bytes and address reuse; unprotected serial replay',
              'original': stats(rows), 'comparisons': {}}
    for label, path in [('native', native), ('capstone_qemu', capstone)]:
        other, arena = read(path)
        mismatches = sum(a != b for a, b in zip(rows, other)) + abs(len(rows) - len(other))
        result['comparisons'][label] = dict(mismatching_events=mismatches,
            allocator_arena_high_water_bytes=arena, **stats(other))
        if mismatches:
            for i, (a, b) in enumerate(zip(rows, other)):
                if a != b:
                    print(f'{label}: first disagreement at event {i}: {dict(zip(FIELDS,a))} vs {dict(zip(FIELDS,b))}')
                    break
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'original-events.csv').open('w') as f:
        w = csv.writer(f); w.writerow(('event',) + FIELDS)
        w.writerows((i,) + row for i, row in enumerate(rows))
    print(json.dumps(result, indent=2))
    if any(c['mismatching_events'] for c in result['comparisons'].values()):
        raise SystemExit('replay differs: comparison saved, do not claim agreement')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('source', type=Path); prep.add_argument('out', type=Path)
    cmp = sub.add_parser('compare')
    for name in ('original', 'native', 'capstone', 'out'): cmp.add_argument(name, type=Path)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.source, a.out)
    else: compare(a.original, a.native, a.capstone, a.out)
