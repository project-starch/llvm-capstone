#!/usr/bin/env python3
"""Recompute the four PostgreSQL inner-reuse arms from their raw processes."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'applications'))
from reuse_gap_metrics import parse_reuse_gap
spec = importlib.util.spec_from_file_location('cheribsd_runner', HERE.parents[2] / 'applications/cheribsd-run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ORACLE = '1969a1709ec3bb734a1a8b1aecd73409dc3837b873588eb2866565f30ed08f8c'


def read(archive, name):
    return archive.extractfile('./' + name).read()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    runs, hashes = [], {}
    build = json.loads((HERE / 'build-manifest.json').read_text())
    assert build['reused_source_root'] is False
    assert build['reuse_gap_observer'] is True
    assert build['chunk_metadata_capacity'] == 65536
    assert build['block_metadata_capacity'] == 8192
    assert build['poisoncap_policy'] == 'published-sqlite-thresholds-transferred-full-queue-corrected'
    archive_path = HERE / 'cheribsd-raw.tar.gz'
    hashes[archive_path.name] = sha(archive_path)
    with tarfile.open(archive_path) as archive:
        manifest = json.loads(read(archive, 'manifest.json'))
        points = json.loads(read(archive, 'points.json'))
        rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
        assert len(points) == 2 and len(rows) == 6
        assert {(r['point']['mode'], r['repetition']) for r in rows} == {(m, n) for m in (0, 1) for n in range(3)}
        assert manifest['guest_default_revocation'] == '0'
        assert 'runtime_revocation_async: 1' in manifest['platform']
        assert 'runtime_revocation_every_free_default: 0' in manifest['platform']
        assert manifest['files']['/tmp/postgres-study']['sha256'] == build['binary_sha256']
        assert manifest['files']['/tmp/policy-lib/libc.so.7']['sha256'] == 'a22e5b4a61f854c7006ec3abf1a5ba25ce043eaaa1c3fcd9f1a74e3eb95cc690'
        assert manifest['runner_sha256'] == hashlib.sha256(read(archive, 'runner.py')).hexdigest()
        for row in rows:
            point = row['point']
            assert point in points and row['status'] == 'pass'
            assert point['expected_pg_rows_sha256'] == ORACLE
            assert point['nested_policy'] == 'published-sqlite-thresholds-corrected-v1'
            assert point['revocation'] == 1
            assert manifest['process_environments'][point['id']]['_RUNTIME_REVOCATION_ENABLE'] == '1'
            directory = point['id'] + '-' + str(row['repetition'])
            stdout = read(archive, directory + '/stdout').decode()
            stderr = read(archive, directory + '/stderr').decode()
            assert runner.verdict(point, 0, stdout, stderr) == 'pass'
            gap = parse_reuse_gap(stdout, 'PG_REUSE_GAP')
            assert gap == row['reuse_gap']
            ledgers = {}
            for line in stdout.splitlines():
                line = re.sub(r'^(backend> )+', '', line)
                if line.startswith(('PG_POISONCAP ', 'PG_POISONCAP_POLICY ', 'PG_POISONCAP_METADATA ', 'PG_POISONCAP_QUEUE ')):
                    ledgers[line.split()[0]] = {k: int(v) for k, v in (part.split('=') for part in line.split()[1:])}
            assert ledgers['PG_POISONCAP']['hands'] == gap['issues']
            queue = ledgers['PG_POISONCAP_QUEUE']
            assert queue['pending'] + queue['blocks'] <= 4096
            runs.append(dict(arm=point['arm'], rep=row['repetition'], **gap, ledgers=ledgers))
    for directory, arm in [('postgres-reuse-capstone-spatial-20260928', 'capstone-spatial'),
                           ('postgres-reuse-sublet-20260928', 'capstone-sublet')]:
        source = HERE.parent / directory
        subprocess.run([sys.executable, str(source / 'validate.py')], check=True)
        path = source / 'capstone-raw.tar.gz'
        hashes['../' + directory + '/capstone-raw.tar.gz'] = sha(path)
        with tarfile.open(path) as archive:
            rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
            for row in rows:
                runs.append(dict(arm=arm, rep=row['repetition'], **row['reuse_gap']))
    for arm in {r['arm'] for r in runs}:
        samples = [r for r in runs if r['arm'] == arm]
        assert len(samples) == 3
        assert all(s['error'] == 0 and sum(s['bins']) == s['reuses'] for s in samples)
    summary = dict(schema=1, workload='postgres-work', label='PostgreSQL · SQL',
                   runs=runs, raw_sha256=hashes,
                   workload_kind='Complete backend qualification SQL; not pgbench',
                   baseline_kind='Capstone original layout; CheriBSD matched adapter layout',
                   outer_policy='Application libc enabled; guest setup services disabled',
                   oracle_sha256=ORACLE)
    target = HERE / 'reuse-summary.json'
    encoded = json.dumps(summary, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == encoded, 'saved summary differs from recomputed raw evidence'
    else:
        target.write_text(encoded)
    print('validated 12/12 complete PostgreSQL processes, four arms, three repetitions')
    return summary


if __name__ == '__main__':
    validate()
