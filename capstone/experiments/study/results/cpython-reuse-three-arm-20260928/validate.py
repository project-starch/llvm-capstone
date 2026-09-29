#!/usr/bin/env python3
"""Recheck the archived complete-interpreter oracle and gap histograms."""
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'applications'))
from reuse_gap_metrics import parse_reuse_gap
PHASES = ['startup', 'baseline', 'live-0', 'released-0', 'live-1',
          'released-1', 'live-2', 'released-2', 'exit']


def read(archive, name):
    return archive.extractfile('./' + name).read()


def check(arm_set, archive_name):
    with tarfile.open(HERE / archive_name) as archive:
        manifest = json.loads(read(archive, 'manifest.json'))
        flavor = 'cheribsd' if archive_name.startswith('cheribsd') else 'capstone'
        build = json.loads((HERE / (flavor + '-build-manifest.json')).read_text())
        rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
        assert len(rows) == 3 * len(arm_set)
        assert {(row['point']['arm'], row['repetition']) for row in rows} == {
            (arm, repetition) for arm in arm_set for repetition in range(3)}
        for row in rows:
            assert row['status'] == 'pass'
            directory = row['point']['id'] + '-' + str(row['repetition'])
            assert read(archive, directory + '/stdout') == b'EXP-OK cpython 552\n'
            stderr = read(archive, directory + '/stderr').decode()
            prefix = 'EXP-CHERI ' if archive_name.startswith('cheribsd') else 'EXP-MEM '
            phases = [line.split('phase=', 1)[1].split()[0]
                      for line in stderr.splitlines() if line.startswith(prefix)]
            assert phases == PHASES
            gap = parse_reuse_gap(stderr, 'PYM_REUSE_GAP')
            assert gap == row['reuse_gap']
            assert gap['issues'] == (80192 if archive_name.startswith('cheribsd') else 77005)
            assert gap['error'] == 0 and sum(gap['bins']) == gap['reuses']
            if archive_name.startswith('cheribsd'):
                assert manifest['files']['/tmp/python-study']['sha256'] == build['binary_sha256']
                assert 'PYM_POISONCAP mode=0 sweeps=0 ' in stderr
            else:
                assert row['image_sha256'] == build['image_sha256']
                assert row['before']['node_capacity'] == 262144
    return len(rows)


if __name__ == '__main__':
    total = check({'pymalloc-spatial', 'pymalloc-sublet'}, 'capstone-raw.tar.gz')
    total += check({'poisoncap-spatial'}, 'cheribsd-raw.tar.gz')
    print(f'validated {total}/9 complete CPython processes')
