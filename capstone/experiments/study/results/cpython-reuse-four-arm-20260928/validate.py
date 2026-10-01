#!/usr/bin/env python3
"""Recompute all four complete-interpreter reuse histograms from raw evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'applications'))
from reuse_gap_metrics import parse_reuse_gap


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(archive, name):
    return archive.extractfile('./' + name).read()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    runner = module(HERE.parents[2] / 'applications/cheribsd-run.py', 'cheribsd_runner')
    original = HERE.parent / 'cpython-reuse-three-arm-20260928'
    capstone = module(original / 'validate.py', 'capstone_archive')
    assert capstone.check({'pymalloc-spatial', 'pymalloc-sublet'}, 'capstone-raw.tar.gz') == 6
    build = json.loads((HERE / 'build-manifest.json').read_text())
    assert build['reuse_gap_observer'] and build['pointer_bytes'] == 16
    assert build['application_cflags'] == '-O1 -Wno-error'
    runs = []
    path = HERE / 'cheribsd-raw.tar.gz'
    with tarfile.open(path) as archive:
        manifest = json.loads(read(archive, 'manifest.json'))
        points = json.loads(read(archive, 'points.json'))
        rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
        assert len(points) == 2 and len(rows) == 6
        assert {(r['point']['mode'], r['repetition']) for r in rows} == {(m, i) for m in (0, 1) for i in range(3)}
        assert manifest['guest_default_revocation'] == '0'
        assert 'runtime_revocation_async: 1' in manifest['platform']
        assert 'runtime_revocation_every_free_default: 0' in manifest['platform']
        assert manifest['files']['/tmp/python-study']['sha256'] == build['binary_sha256']
        assert manifest['files']['/tmp/pyhome/lib/python313.zip']['sha256'] == build['stdlib_zip_sha256']
        workload_sha = manifest['files']['/tmp/objects.py']['sha256']
        assert manifest['files']['/tmp/libc-fixed.so.7']['sha256'] == 'a22e5b4a61f854c7006ec3abf1a5ba25ce043eaaa1c3fcd9f1a74e3eb95cc690'
        assert manifest['runner_sha256'] == hashlib.sha256(read(archive, 'runner.py')).hexdigest()
        for row in rows:
            point = row['point']
            assert row['status'] == 'pass' and point in points
            assert point['argv'][4:] == ['8', '3', '0']
            assert point['nested_policy'] == 'published-sqlite-thresholds-corrected-v1'
            assert point['revocation'] == 1 and point['reuse_gap'] == 'PYM_REUSE_GAP'
            assert manifest['process_environments'][point['id']]['_RUNTIME_REVOCATION_ENABLE'] == '1'
            directory = point['id'] + '-' + str(row['repetition'])
            stdout = read(archive, directory + '/stdout').decode()
            stderr = read(archive, directory + '/stderr').decode()
            assert stdout == point['expected_stdout'] == 'EXP-OK cpython 552\n'
            assert runner.verdict(point, 0, stdout, stderr) == 'pass'
            gap = parse_reuse_gap(stderr, 'PYM_REUSE_GAP')
            assert gap == row['reuse_gap']
            reports = [line for line in stderr.splitlines() if line.startswith('PYM_POISONCAP ')]
            assert len(reports) == 1
            ledger = {k: int(v) for k, v in (word.split('=') for word in reports[0].split()[1:])}
            runs.append(dict(arm=point['arm'], rep=row['repetition'], **gap, ledger=ledger))
    with tarfile.open(original / 'capstone-raw.tar.gz') as archive:
        manifest = json.loads(read(archive, 'manifest.json'))
        assert manifest['workload_inputs']['/mnt/host/experiments/objects.py'] == workload_sha
        rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
        for row in rows:
            arm = {'pymalloc-spatial': 'capstone-spatial', 'pymalloc-sublet': 'capstone-sublet'}[row['point']['arm']]
            runs.append(dict(arm=arm, rep=row['repetition'], **row['reuse_gap']))
    assert len(runs) == 12 and len({r['arm'] for r in runs}) == 4
    for row in runs:
        assert row['error'] == 0 and sum(row['bins']) == row['reuses']
    summary = dict(schema=1, workload='cpython-objects8', label='CPython · JSON/GC', runs=runs,
                   raw_sha256={'cheribsd-raw.tar.gz': sha(path),
                               '../cpython-reuse-three-arm-20260928/capstone-raw.tar.gz': sha(original / 'capstone-raw.tar.gz')},
                   workload_kind='Complete interpreter JSON/GC qualification, objects.py 8 3 0; not pyperformance',
                   outer_policy='Application libc enabled; guest setup services disabled',
                   metric='Observed same-start reuse, indexed by successful new lifetimes; three repetitions per arm')
    encoded = json.dumps(summary, indent=2, sort_keys=True) + '\n'
    target = HERE / 'reuse-summary.json'
    if target.exists():
        assert target.read_text() == encoded, 'summary differs from raw evidence'
    else:
        target.write_text(encoded)
    print('validated 12/12 complete CPython processes, four arms, three repetitions')
    return summary


if __name__ == '__main__':
    validate()
