#!/usr/bin/env python3
"""Recheck the archived complete-interpreter oracle and gap histograms."""
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'applications'))
from reuse_gap_metrics import parse_reuse_gap


def read(archive, name):
    return archive.extractfile('./' + name).read()


def check(arm_set, archive_name):
    with tarfile.open(HERE / archive_name) as archive:
        rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
        assert len(rows) == 3 * len(arm_set)
        assert {(row['point']['arm'], row['repetition']) for row in rows} == {
            (arm, repetition) for arm in arm_set for repetition in range(3)}
        for row in rows:
            assert row['status'] == 'pass'
            directory = row['point']['id'] + '-' + str(row['repetition'])
            assert read(archive, directory + '/stdout') == b'EXP-OK cpython 552\n'
            stderr = read(archive, directory + '/stderr').decode()
            gap = parse_reuse_gap(stderr, 'PYM_REUSE_GAP')
            assert gap == row['reuse_gap']
            assert gap['issues'] == (80192 if archive_name.startswith('cheribsd') else 77005)
            assert gap['error'] == 0 and sum(gap['bins']) == gap['reuses']
    return len(rows)


if __name__ == '__main__':
    total = check({'pymalloc-spatial', 'pymalloc-sublet'}, 'capstone-raw.tar.gz')
    total += check({'poisoncap-spatial'}, 'cheribsd-raw.tar.gz')
    print(f'validated {total}/9 complete CPython processes')
