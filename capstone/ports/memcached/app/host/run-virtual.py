#!/usr/bin/env python3
"""Compare the virtual four-worker server to native memcached; judge worker faults."""
import argparse
import base64
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[3]
VIRTUAL = REPO / 'capstone/runtime/virtual'
spec = importlib.util.spec_from_file_location('judge', APP.parents[1] / 'common/application/check-safety.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker_counts(data):
    counts = {}
    for value in re.findall(rb'^MC-WORKER (\d+) conn$', data, re.M):
        n = int(value)
        counts[n] = counts.get(n, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'adapter', 'qemu', 'images', 'cross-cc', 'pthread', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, default=21299)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--native-reference', type=Path,
                   help='Reuse native transcripts from a passing gate with identical native binaries and harness source')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    harness_source = APP / 'host/mc-harness/mc-harness.c'
    host_harness = a.work / 'mc-harness-host'
    subprocess.run(['gcc', '-O1', '-Wall', str(harness_source), '-lpthread', '-o', str(host_harness)], check=True)
    subprocess.run([str(a.cross_cc), '-O1', '-Wall', str(harness_source), '-lpthread', '-o', str(stage/'mc-harness')], check=True)
    inputs = dict(launcher=a.adapter/'capstone-vexec', module=a.adapter/'module/capstone_vm.ko',
                  job=a.adapter/'capstone-job', pthread=a.pthread,
                  memcached=a.build/'domain/memcached.dom', marker=a.build/'marker/memcached.dom',
                  safety=a.build/'safety/memcached-safety-virtual.dom',
                  native=a.build/'native/bin/memcached', native_marker=a.build/'marker/memcached-native',
                  qemu=a.qemu, harness_source=harness_source)
    inputs.update({name: a.images/name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    for name, filename in [('launcher', 'capstone-vexec'), ('module', 'capstone_vm.ko'),
                           ('job', 'capstone-job'), ('pthread', 'pthread.dom'),
                           ('memcached', 'memcached.dom'), ('marker', 'marker.dom'), ('safety', 'safety.dom')]:
        shutil.copy2(inputs[name], stage/filename)
    flags = ['-l', '127.0.0.1', '-p', str(a.port), '-U', '0', '-m', '64', '-t', '4']
    native = {}
    reference_record = None
    if a.native_reference:
        reference_record = a.native_reference/'result.json'
        recorded = json.loads(reference_record.read_text())
        if recorded.get('status') != 'PASS' or any(
                recorded.get('input_sha256', {}).get(name) != sha(inputs[name])
                for name in ('native', 'native_marker', 'harness_source')):
            p.error('native reference does not match this passing harness/binary combination')
    for label, stop, perturb, marked in [('null1', 'TERM', 'none', False), ('null2', 'TERM', 'none', False),
                                         ('value', 'TERM', 'value', False), ('cas', 'TERM', 'cas', False),
                                         ('usr1', 'USR1', 'none', False), ('marker', 'TERM', 'none', True)]:
        out = a.work / ('native-' + label)
        out.mkdir()
        if a.native_reference:
            for file in (a.native_reference/('native-'+label)).iterdir():
                if file.is_file(): shutil.copy2(file, out/file.name)
        else:
            with (out/'harness.log').open('w') as log:
                subprocess.run([str(host_harness), '--out', str(out), '--port', str(a.port),
                            '--stop', stop, '--perturb', perturb, '--',
                            str(inputs['native_marker' if marked else 'native']), *flags],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        native[label] = {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()}
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        print(('PASS ' if passed else 'FAIL ') + name, flush=True)

    reference = native['null1']['transcript.norm']
    check('native_null', reference == native['null2']['transcript.norm'])
    for label in ('value', 'cas'):
        check('oracle_detects_' + label, reference != native[label]['transcript.norm'])
    script = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1',
              'insmod capstone_vm.ko || exit 1', 'chmod 666 /dev/capstone-vm',
              'export CAPSTONE_EXEC_DIAGNOSTICS=1',
              'echo MC_BEGIN:pthread', './capstone-vexec ./pthread.dom', 'echo MC_END:pthread:$?']
    predictions = {}
    for line in (APP/'host/safety-expect.txt').read_text().splitlines():
        words = line.split('#')[0].split()
        if len(words) == 4 and words[0] == 'sublet':
            predictions.setdefault(int(words[1]), []).append(words[2:])
    if not predictions:
        raise ValueError('no Sublet predictions for the virtual outer heap')
    cases = [('normal1', 'memcached', 'TERM', 0), ('normal2', 'memcached', 'TERM', 0),
             ('usr1', 'memcached', 'USR1', 0), ('marker', 'marker', 'TERM', 0)]
    cases += [(f'fx{n}', 'safety', 'TERM', n) for n in sorted(predictions)]
    for label, image, stop, fixture in cases:
        out = '/tmp/mc-' + label
        options = f'--fixture {fixture}' if fixture else f'--stop {stop}'
        # capstone-job forwards TERM, but not USR1; signal its launcher child.
        if not fixture and stop == 'USR1':
            options += ' --signal-child'
        script += [f'echo MC_BEGIN:{label}', f'mkdir -p {out}; chmod 777 {out}',
                   f'CAPSTONE_FAULT_RECORD={out}/fault.txt ./mc-harness --out {out} --port {a.port} {options} -- '
                   f'./capstone-job {out}/job.json --user 65534:65534 -- ./capstone-vexec ./{image}.dom ' + ' '.join(flags),
                   'mc_status=$?',
                   f'[ ! -f {out}/transcript.norm ] || gzip -c {out}/transcript.norm > {out}/transcript.norm.gz',
                   f'for name in transcript.norm.gz status.txt identity.txt job.json server.out server.err fault.txt; do '
                   f'f={out}/$name; [ -f "$f" ] || continue; echo MC_FILE:{label}:$name; base64 "$f"; echo MC_FILE_END; done',
                   f'echo MC_END:{label}:$mc_status']
    script += ['rmmod capstone_vm', 'echo MC_CLEANUP:$?', 'echo VIRTUAL_STAGED_DONE']
    gate = '\n'.join(script) + '\n'
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([sys.executable, str(VIRTUAL/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work/'guest'), '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    parts = {label: (text, int(status)) for label, text, status in
             re.findall(r'^MC_BEGIN:([^\n]+)\n(.*?)^MC_END:\1:(\d+)\n', log, re.M | re.S)}
    pthread_text, pthread_status = parts.get('pthread', ('', -1))
    for mark in ('unsupported_clone_refused', 'tls mutex cond tagged_join', 'timed_wait', 'independent_blocking_io', 'private_epoll_events', 'shared_tls_transport', '64_joined_lifetimes'):
        check('pthread_' + mark.replace(' ', '_'), pthread_status == 0 and 'VIRTUAL_PTHREAD_OK ' + mark in pthread_text)
    fixtures = []
    for label, image, stop, number in cases:
        text, status = parts.get(label, ('', -1))
        files = {name: base64.b64decode(data) for name, data in
                 re.findall(r'^MC_FILE:' + label + r':([^\n]+)\n(.*?)^MC_FILE_END\n', text, re.M | re.S)}
        if 'transcript.norm.gz' in files:
            files['transcript.norm'] = gzip.decompress(files.pop('transcript.norm.gz'))
        saved = a.work / label
        saved.mkdir()
        for name, data in files.items():
            (saved/name).write_bytes(data)
        check(label + '_harness', status == 0)
        if not number:
            n = native['marker' if label == 'marker' else 'usr1' if stop == 'USR1' else 'null1']
            check(label + '_protocol', files.get('transcript.norm') == n['transcript.norm'])
            check(label + '_identity', files.get('identity.txt', b'').strip() == b'STAT pointer_size 128' and
                  n['identity.txt'].strip() == b'STAT pointer_size 64')
            job = json.loads(files.get('job.json', b'{}'))
            expected_exit = re.search(rb' exit=(\d+)', n['status.txt'])
            check(label + '_exit', expected_exit and job == dict(version=1, kind='exit', value=int(expected_exit[1])))
            if label == 'marker':
                nc, vc = worker_counts(n['server.err']), worker_counts(files.get('server.err', b''))
                check('four_workers', nc == vc and set(vc) == set(range(4)))
                check('worker_gate_control', nc != worker_counts(b''))
                stderr = re.sub(rb'^MC-WORKER \d+ conn\n', b'', files.get('server.err', b''), flags=re.M)
            else:
                stderr = files.get('server.err', b'')
            check(label + '_stderr', stderr == re.sub(rb'^MC-WORKER \d+ conn\n', b'', n['server.err'], flags=re.M) and not files.get('fault.txt'))
        else:
            job = json.loads(files.get('job.json', b'{}'))
            result = dict(kind=job.get('kind'), value=job.get('value'))
            if files.get('fault.txt'):
                result['fault'] = files['fault.txt'].decode().strip()
            stdout = files.get('server.out', b'').decode(errors='replace').replace('MCAPP-', 'FFAPP-')
            diagnostics = text + files.get('server.err', b'').decode(errors='replace')
            try:
                (got, detail, _), info = judge.classify(stdout, diagnostics, result, number)
                passed = all(judge.matches(got, detail, info, *want, number) for want in predictions[number])
            except ValueError as error:
                got, detail, passed = 'ERROR', str(error), False
            fixtures.append(dict(fixture=number, passed=passed, outcome=got, detail=detail,
                                 expected=predictions[number]))
            check(label + '_prediction', passed)
    check('adapter_cleanup', run.returncode == 0 and '\nMC_CLEANUP:0\n' in log)
    record = dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
                  checks=checks, fixtures=fixtures, workers=4, harts=1, virtual_vm_abi=5,
                  native_reference_sha256=sha(reference_record) if reference_record else None,
                  native_transcript_sha256={label: hashlib.sha256(files['transcript.norm']).hexdigest()
                                            for label, files in native.items()},
                  safety_prediction_arm='sublet',
                  known_gaps=['nested slab/cache lifetimes', 'bipbuffer logical reuse'],
                  input_sha256={name: sha(path) for name, path in inputs.items()},
                  runner_sha256=sha(Path(__file__)), gate_sha256=hashlib.sha256(gate.encode()).hexdigest(),
                  predictions_sha256=sha(APP/'host/safety-expect.txt'))
    (a.work/'result.json').write_text(json.dumps(record, indent=2) + '\n')
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
