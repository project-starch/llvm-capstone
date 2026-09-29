#!/usr/bin/env python3
"""Recompute all four Perl SV-head reuse histograms from the raw archives."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
APPLICATIONS = HERE.parents[2] / 'applications'
sys.path.insert(0, str(APPLICATIONS))
from reuse_gap_metrics import parse_reuse_gap

PHASES = ['startup', 'baseline', 'live-0', 'released-0', 'live-1', 'released-1',
          'live-2', 'released-2', 'exit']
ORACLE = 'EXP-OK perl 2357760\n'
PLATFORM = {  # the CPython four-arm campaign's Capstone platform, same files
    'qemu': 'f9795d413ad1910a954dc918fcf469a2a13725a50d48d249dbe8a70cd741f85d',
    'kernel': 'b3193f0a7eccafabcd3ba89b8246afa29c391aa0d38cf54362f55379f84a09dd',
    'firmware': 'fc3f4795efd771dbec751429ade16b333b9ace474ec6160dfef6ded8ae253b43',
    'rootfs': 'fb55450e273e42e1a592cb0e8329632e1694da2323e41a68ddd587c41c94326f',
    'launcher': '921921f3dd529ce85638d9c6ba63d43173a071a39b931a089304458348da1d53',
    'job_helper': '5be7879d33c2483570de238bc9b16a730f22cd20d8d2f531547a0bdbbb803cef',
}
FIXED_LIBC = 'a22e5b4a61f854c7006ec3abf1a5ba25ce043eaaa1c3fcd9f1a74e3eb95cc690'
ARMS = {'sv-heads-spatial': 'capstone-spatial', 'sv-heads-sublet': 'capstone-sublet',
        'poisoncap-spatial': 'poisoncap-spatial', 'poisoncap-temporal': 'poisoncap-temporal'}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(archive, name):
    return archive.extractfile('./' + name).read()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ledger(stderr):
    lines = [line for line in stderr.splitlines() if line.startswith('PERL_SV_HEADS ')]
    assert len(lines) == 1, 'exactly one closing SV-head ledger'
    words = [word.split('=', 1) for word in lines[0].split()[1:]]
    fields = dict(words)
    assert len(fields) == len(words), 'duplicate ledger field'
    platform = fields.pop('platform')
    return platform, {k: int(v) for k, v in fields.items()}


def common_ledger(row, gap, mode):
    assert row['mode'] == mode and row['head_bytes'] == 48 and row['per_page'] == 84
    assert row['issues'] == gap['issues'] == gap['attempts']
    assert row['releases'] == gap['releases'] and row['issues'] == row['releases'] + row['live']
    assert row['retired'] == 0 and row['pages'] == row['peak_pages'] <= row['max_pages']
    # A page is added only when no head is issuable, so every earlier page's
    # heads were all issued: the last page alone may be partly unused.
    assert (row['pages'] - 1) * row['per_page'] < gap['distinct'] <= row['pages'] * row['per_page']
    assert gap['error'] == 0 and sum(gap['bins']) == gap['reuses']


def capstone_runs(build):
    runner = module(APPLICATIONS / 'run.py', 'capstone_runner')
    runs = []
    with tarfile.open(HERE / 'capstone-raw.tar.gz') as archive:
        manifest = json.loads(read(archive, 'manifest.json'))
        points = json.loads(read(archive, 'points.json'))
        rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
        assert manifest['runner_sha256'] == sha(read(archive, 'runner.py'))
        identity = manifest['platform']
        assert {k: v['sha256'] for k, v in identity['files'].items()} == PLATFORM
        assert identity['environment'] == {'CAPSTONE_REV_NODES': '262144', 'CAPSTONE_GP_NONLIN': '1'}
        assert set(manifest['images'].values()) == {build['image_sha256']}
        assert len(points) == 2 and len(rows) == 6
        assert {(r['point']['arm'], r['repetition']) for r in rows} == \
            {(a, i) for a in ('sv-heads-spatial', 'sv-heads-sublet') for i in range(3)}
        inputs = manifest['workload_inputs']
        for row in rows:
            point = row['point']
            assert row['status'] == 'pass' and point in points and row['boot_id'] == manifest['boot_id']
            assert point['argv'] == ['/mnt/host/experiments/records.pl', '512', '3', '0']
            mode = {'sv-heads-spatial': 0, 'sv-heads-sublet': 1}[point['arm']]
            assert point['environment']['PERL_SUBLET_MODE'] == str(mode)
            assert point['expected_node_capacity'] == row['before']['node_capacity'] == 262144
            assert not (row['after']['live_domains'] or row['after']['live_regions'] or row['after']['live_bytes'])
            directory = point['id'] + '-' + str(row['repetition'])
            stdout_raw = read(archive, directory + '/stdout')
            stdout, stderr = stdout_raw.decode(), read(archive, directory + '/stderr').decode()
            assert stdout == point['expected_stdout'] == ORACLE
            assert runner.verdict(point, 0, False, stdout, stderr, row['exit'], stdout_raw) == 'pass'
            gap = parse_reuse_gap(stderr, 'PERL_REUSE_GAP')
            assert gap == row['reuse_gap']
            platform, heads = ledger(stderr)
            assert platform == 'capstone'
            common_ledger(heads, gap, mode)
            carved = heads['pages'] * heads['per_page']
            assert heads['split'] == heads['delin'] - heads['revoke'] == carved
            assert heads['revoke'] == (heads['releases'] if mode else 0) and heads['init'] == 0
            assert heads['mrev'] == carved + heads['revoke']
            runs.append(dict(arm=ARMS[point['arm']], rep=row['repetition'], **gap,
                             ledger=heads))
    return runs, inputs


def cheribsd_runs(build, inputs):
    runner = module(APPLICATIONS / 'cheribsd-run.py', 'cheribsd_runner')
    runs = []
    with tarfile.open(HERE / 'cheribsd-raw.tar.gz') as archive:
        manifest = json.loads(read(archive, 'manifest.json'))
        points = json.loads(read(archive, 'points.json'))
        rows = [json.loads(line) for line in read(archive, 'runs.jsonl').splitlines()]
        assert manifest['runner_sha256'] == sha(read(archive, 'runner.py'))
        assert manifest['guest_default_revocation'] == '0'
        assert 'runtime_revocation_async: 1' in manifest['platform']
        files = {guest: spec['sha256'] for guest, spec in manifest['files'].items()}
        assert files['/tmp/perl-study'] == build['binary_sha256']
        assert files['/tmp/libc-fixed.so.7'] == FIXED_LIBC
        for guest, share in (('/tmp/records.pl', 'records.pl'),
                             ('/tmp/perl-lib/strict.pm', 'perl-lib/strict.pm'),
                             ('/tmp/perl-lib/warnings.pm', 'perl-lib/warnings.pm')):
            assert files[guest] == inputs['/mnt/host/experiments/' + share], guest
        assert len(points) == 2 and len(rows) == 6
        assert {(r['point']['mode'], r['repetition']) for r in rows} == {(m, i) for m in (0, 1) for i in range(3)}
        for row in rows:
            point = row['point']
            mode = point['mode']
            assert row['status'] == 'pass' and point in points
            assert point['argv'][2:] == ['512', '3', '0']
            assert point['nested_policy'] == 'published-sqlite-thresholds-corrected-v1'
            assert point['revocation'] == 1 and point['reuse_gap'] == 'PERL_REUSE_GAP'
            environment = manifest['process_environments'][point['id']]
            assert environment['_RUNTIME_REVOCATION_ENABLE'] == '1'
            assert environment['PERL_POISONCAP_MODE'] == str(mode)
            directory = point['id'] + '-' + str(row['repetition'])
            stdout_raw = read(archive, directory + '/stdout')
            stdout, stderr = stdout_raw.decode(), read(archive, directory + '/stderr').decode()
            assert stdout == point['expected_stdout'] == ORACLE
            assert runner.verdict(point, 0, stdout, stderr, stdout_raw) == 'pass'
            assert [s['phase'] for s in runner.samples(stderr)] == PHASES
            gap = parse_reuse_gap(stderr, 'PERL_REUSE_GAP')
            assert gap == row['reuse_gap']
            platform, heads = ledger(stderr)
            assert platform == 'cheribsd-poisoncap'
            common_ledger(heads, gap, mode)
            assert heads['queued'] == 0
            assert heads['sweeps'] == heads['full_drains'] + heads['threshold_drains'] + heads['teardown_drains']
            assert heads['poison_bytes'] == heads['clear_bytes'] == heads['zero_bytes']
            assert heads['poison_bytes'] == (heads['releases'] * heads['head_bytes'] if mode else 0)
            runs.append(dict(arm=ARMS[point['arm']], rep=row['repetition'], **gap,
                             ledger=heads))
    return runs


def validate():
    capstone_build = json.loads((HERE / 'capstone-build-manifest.json').read_text())
    cheribsd_build = json.loads((HERE / 'cheribsd-build-manifest.json').read_text())
    assert capstone_build['application'] == 'perl' and capstone_build['nested'] == 'perl'
    assert capstone_build['heap'] == 'level0'
    assert cheribsd_build['application_cflags'] == '-O1' and cheribsd_build['pointer_bytes'] == 16
    assert cheribsd_build['nested_allocator'].startswith('perl-sv-heads')
    runs, inputs = capstone_runs(capstone_build)
    runs += cheribsd_runs(cheribsd_build, inputs)
    assert len(runs) == 12 and len({r['arm'] for r in runs}) == 4
    for arm in ARMS.values():
        same = [r for r in runs if r['arm'] == arm]
        assert len({r['issues'] for r in same}) == 1, arm + ': issue count varies'
    summary = dict(
        schema=1, workload='perl-records512', label='Perl · records',
        runs=sorted(runs, key=lambda r: (r['arm'], r['rep'])),
        raw_sha256={name: sha((HERE / name).read_bytes())
                    for name in ('capstone-raw.tar.gz', 'cheribsd-raw.tar.gz')},
        workload_kind='Complete Perl 5.36.3 interpreter, records.pl 512 3 0; not a standard benchmark score',
        boundary='SV heads only; SV bodies, hash entries and OP slabs keep their upstream allocators',
        outer_policy='CheriBSD process revocation enabled; guest default disabled',
        metric='Observed same-start reuse, indexed by successful new lifetimes; three repetitions per arm')
    encoded = json.dumps(summary, indent=2, sort_keys=True) + '\n'
    target = HERE / 'reuse-summary.json'
    if target.exists():
        assert target.read_text() == encoded, 'summary differs from raw evidence'
    else:
        target.write_text(encoded)
    print('validated 12/12 complete Perl processes, four arms, three repetitions')
    return summary


if __name__ == '__main__':
    validate()
