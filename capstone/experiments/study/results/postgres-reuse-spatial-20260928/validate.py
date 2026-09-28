#!/usr/bin/env python3
"""Recheck the archived SQL oracle and inner allocator histogram."""
import hashlib
import importlib.machinery
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'applications'))
from reuse_gap_metrics import parse_reuse_gap

runner = importlib.machinery.SourceFileLoader(
    'cheribsd_runner', str(HERE.parents[2] / 'applications/cheribsd-run.py')).load_module()


def read(archive, name):
    return archive.extractfile('./' + name).read()


with tarfile.open(HERE / 'cheribsd-raw.tar.gz') as archive:
    manifest = json.loads(read(archive, 'manifest.json'))
    build = json.loads((HERE / 'build-manifest.json').read_text())
    rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
    assert len(rows) == 3
    assert sorted(row['repetition'] for row in rows) == [0, 1, 2]
    assert build['binary_sha256'] == manifest['files']['/tmp/postgres-study']['sha256']
    assert build['reuse_gap_observer'] is True
    points = json.loads(read(archive, 'points.json'))
    assert len(points) == 1
    point = points[0]
    assert point['expected_pg_rows_sha256'] == '1969a1709ec3bb734a1a8b1aecd73409dc3837b873588eb2866565f30ed08f8c'
    for row in rows:
        assert row['status'] == 'pass' and row['point'] == point
        directory = point['id'] + '-' + str(row['repetition'])
        stdout = read(archive, directory + '/stdout').decode()
        stderr = read(archive, directory + '/stderr').decode()
        sql = runner.pg_rows(stdout)
        assert len(sql) == 22 and 'count = "1500"' in sql[-2]
        assert hashlib.sha256(('\n'.join(sql) + '\n').encode()).hexdigest() == point['expected_pg_rows_sha256']
        assert 'EXP-GUEST-EXIT 0' in stderr
        gap = parse_reuse_gap(stdout, 'PG_REUSE_GAP')
        assert gap == row['reuse_gap']
        assert (gap['issues'], gap['releases'], gap['reuses'], gap['distinct'], gap['error']) == (
            54004, 51149, 44974, 9030, 0)
        assert 'PG_POISONCAP mode=0 sweeps=0 ' in stdout
print('validated 3/3 complete PostgreSQL processes')
