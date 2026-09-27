#!/usr/bin/env python3
"""Run application points in one CheriBSD guest over loopback SSH.

The common ports/host CheriBSD Guest owns boot and snapshot lifetime. This
runner owns only its application processes; no guest restart or case retry.
"""
import argparse
import fcntl
import os
import sys
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from allocation_metrics import allocation_samples, valid_allocations


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples(stderr):
    return [dict((k, v if k == 'phase' else int(v))
                 for k, v in (word.split('=', 1) for word in line.split()[1:]))
            for line in stderr.splitlines() if line.startswith('EXP-CHERI ')]


def verdict(point, rc, stdout, stderr):
    if rc: return 'transport-error'
    exits = re.findall(r'^EXP-GUEST-EXIT (\d+)$', stderr, re.M)
    if len(exits) != 1: return 'missing-exit-evidence'
    if int(exits[0]) != 0: return 'guest-error'
    if stdout != point['expected_stdout']: return 'oracle-mismatch'
    try:
        memory = samples(stderr)
        if [s['phase'] for s in memory] != point['expected_phases']: return 'missing-phases'
        if not memory or any(s['heap_error'] or s['shadow_error'] or
                             s['revocation'] != point['revocation'] or
                             any(v < 0 for k, v in s.items() if k != 'phase') or
                             not (0 <= s['allocated'] <= s['active'] <= s['resident'])
                             for s in memory): return 'bad-metrics'
    except (ValueError, KeyError): return 'bad-metrics'
    if point.get('allocations') and not valid_allocations(stderr, point['expected_phases']):
        return 'bad-allocation-metrics'
    return 'pass'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', type=Path, help='Attach to an owned existing guest')
    p.add_argument('--sdk', type=Path)
    p.add_argument('--rootfs', type=Path)
    p.add_argument('--disk', type=Path)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--timeout', type=int, default=120)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.key:
        execute(args)
        return
    if not all((args.sdk, args.rootfs, args.disk)):
        p.error('supply --key to attach, or --sdk --rootfs --disk to boot')
    common = Path(__file__).resolve().parents[2]/'ports/common/host/cheribsd'
    sys.path.insert(0, str(common))
    from guest import Guest
    lock_path = Path(os.environ['CAPSTONE_QEMU_LOCK'])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        vm = args.out/'vm'; vm.mkdir()
        guest = Guest(args.sdk, args.rootfs, args.disk, vm, args.port,
                      disable_default_revocation=False)
        guest.argv[guest.argv.index('-m')+1] = str(args.memory_mib)
        (vm/'command.json').write_text(json.dumps(guest.argv)+'\n')
        try:
            guest.start()
            args.key = vm/'guest-key'
            execute(args)
        finally:
            guest.close()


def execute(args):
    points = json.loads(args.points.read_text())
    if args.repeat < 1 or len({p['id'] for p in points}) != len(points):
        raise ValueError('positive repeats and unique point ids required')
    (args.out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'points.json').write_bytes(args.points.read_bytes())
    options = ['-i', str(args.key), '-o', 'IdentitiesOnly=yes', '-o',
               'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
               '-o', 'LogLevel=ERROR', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
    ssh = ['ssh', *options, '-p', str(args.port), 'root@127.0.0.1']
    def call(cmd):
        return subprocess.check_output(ssh+[cmd], text=True, timeout=20).strip()
    boot = call('sysctl -n kern.boottime')
    platform = call('uname -a; sysctl security.cheri')
    files = {}
    for point in points:
        if any(k.startswith(('_RUNTIME_REVOCATION_', 'MALLOC_', 'EXP_CHERI_'))
               for k in point['environment']):
            raise ValueError('allocator policy overrides are outside the default comparison')
    for point in points:
        for host, guest in point['files'].items():
            if guest in files and files[guest]['sha256'] != digest(host):
                raise ValueError('conflicting guest input: '+guest)
            files[guest] = dict(host=host, sha256=digest(host))
    for guest, spec in files.items():
        subprocess.run(['scp', '-O', *options, '-P', str(args.port), spec['host'],
                        'root@127.0.0.1:'+guest], check=True, capture_output=True, timeout=120)
        if call('sha256 -q '+shlex.quote(guest)) != spec['sha256']:
            raise RuntimeError('staged input hash mismatch')
    manifest = dict(boot=boot, platform=platform, files=files,
                    runner_sha256=digest(__file__), repeats=args.repeat,
                    allocation_validator_sha256=digest(Path(__file__).with_name('allocation_metrics.py')),
                    timing='QEMU host elapsed seconds: diagnostic only')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for point in points:
        for repetition in range(args.repeat):
            if call('sysctl -n kern.boottime') != boot:
                raise RuntimeError('guest rebooted')
            directory = args.out/(point['id']+'-'+str(repetition))
            directory.mkdir()
            command = shlex.join(['timeout', str(args.timeout), 'env',
                        *[k+'='+str(v) for k, v in point['environment'].items()],
                        *point['argv']])
            command = 'ulimit -c 0; '+command+'; result=$?; printf "EXP-GUEST-EXIT %s\\n" "$result" >&2; exit 0'
            (directory/'command.txt').write_text(command+'\n')
            start = time.monotonic()
            # A failed host timeout aborts the campaign; it is never called a pass.
            try:
                result = subprocess.run(ssh+[command], capture_output=True,
                                        text=True, timeout=args.timeout+20)
            except subprocess.TimeoutExpired as error:
                (directory/'stdout').write_bytes(error.stdout or b'')
                (directory/'stderr').write_bytes(error.stderr or b'')
                record = dict(point=point, repetition=repetition, status='host-timeout')
                with (args.out/'runs.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                raise
            (directory/'stdout').write_text(result.stdout)
            (directory/'stderr').write_text(result.stderr)
            status = verdict(point, result.returncode, result.stdout, result.stderr)
            try: memory = samples(result.stderr)
            except (ValueError, KeyError): memory = []
            record = dict(point=point, repetition=repetition, status=status,
                          memory=memory, host_seconds=time.monotonic()-start,
                          stdout_sha256=digest(directory/'stdout'),
                          stderr_sha256=digest(directory/'stderr'))
            try: record['allocations'] = allocation_samples(result.stderr)
            except (ValueError, KeyError): record['allocations'] = []
            with (args.out/'runs.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(point['id'], repetition, status, flush=True)
            if status in ('transport-error', 'missing-exit-evidence'):
                raise RuntimeError('lost guest control')


if __name__ == '__main__':
    main()
