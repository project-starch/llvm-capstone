#!/usr/bin/env python3
"""Recheck complete PostgreSQL Sublet runs and their inner gap reports."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'applications'))
from reuse_gap_metrics import parse_reuse_gap


def read(archive, name):
    return archive.extractfile('./' + name).read()


with tarfile.open(HERE / 'capstone-raw.tar.gz') as archive:
    manifest = json.loads(read(archive, 'manifest.json'))
    build = json.loads((HERE / 'build-manifest.json').read_text())
    points = json.loads(read(archive, 'points.json'))
    rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
    assert len(points) == 1 and len(rows) == 3
    point = points[0]
    assert point['reuse_gap'] == 'PG_REUSE_GAP'
    assert point['fresh_tree'] == {'source': 'pgstudy/pgdata16-final',
                                   'destination': 'pgstudy/pgdata-current'}
    assert sorted(row['repetition'] for row in rows) == [0, 1, 2]
    assert build['reuse_gap'] is True
    assert build['image_sha256'] == manifest['images'][point['image']]
    assert len({row['boot_id'] for row in rows}) == 1
    for row in rows:
        assert row['status'] == 'pass' and row['point'] == point
        assert row['before']['node_capacity'] == 262144
        directory = point['id'] + '-' + str(row['repetition'])
        stdout = read(archive, directory + '/stdout')
        stderr = read(archive, directory + '/stderr').decode()
        assert hashlib.sha256(stdout).hexdigest() == point['expected_stdout_sha256']
        assert len(stdout) == point['expected_stdout_bytes']
        phases = [line.split('phase=', 1)[1].split()[0]
                  for line in stderr.splitlines() if line.startswith('EXP-MEM ')]
        assert phases == ['startup', 'exit']
        gap = parse_reuse_gap(stderr, 'PG_REUSE_GAP')
        assert gap == row['reuse_gap']
        assert (gap['issues'], gap['releases'], gap['reuses'], gap['distinct'], gap['error']) == (
            54032, 51177, 43705, 10327, 0)
print('validated 3/3 complete PostgreSQL Sublet processes')
