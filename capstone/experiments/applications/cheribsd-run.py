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


def guest_panic(serial_path):
    if not serial_path.is_file():
        return None
    matches = re.findall(rb'panic: [^\r\n]+', serial_path.read_bytes())
    return matches[-1].decode(errors='replace') if matches else None


def samples(stderr):
    return [dict((k, v if k == 'phase' else int(v))
                 for k, v in (word.split('=', 1) for word in line.split()[1:]))
            for line in stderr.splitlines() if line.startswith('EXP-CHERI ')]


def policy_environment(point):
    """Only the named on/off contrast may change the process allocator policy."""
    expected = {'cheribsd-default': 1, 'cheribsd-revocation-on': 1,
                'cheribsd-revocation-off': 0}
    arm = point['arm']
    if arm.startswith('poisoncap-'):
        if arm not in ('poisoncap-spatial', 'poisoncap-temporal'):
            raise ValueError('unknown PoisonCap arm')
        boundary = (point.get('application'), point.get('nested_allocator'))
        if boundary == ('mruby', 'mruby-gc'):
            expected[arm] = 1
        elif boundary == ('ffmpeg', 'ffmpeg-pool'):
            expected[arm] = 0
            mode = 2 if arm == 'poisoncap-temporal' else 0
            if point.get('mode') != mode or not point.get('argv') or point['argv'][-1] != str(mode):
                raise ValueError('FFmpeg pool arm and application mode disagree')
        elif boundary == ('cpython', 'cpython-pymalloc'):
            expected[arm] = 0
            if point.get('mode') != int(arm == 'poisoncap-temporal'):
                raise ValueError('CPython pymalloc arm and application mode disagree')
        else:
            raise ValueError('PoisonCap arm needs a qualified application boundary')
    if arm not in expected or point['revocation'] != expected[arm]:
        raise ValueError('arm and expected revocation state disagree')
    environment = point['environment']
    if any(k.startswith(('_RUNTIME_', 'MALLOC_', 'EXP_CHERI_')) for k in environment) or \
            'MRB_GC_POISONCAP' in environment or 'PYM_POISONCAP_MODE' in environment:
        raise ValueError('allocator overrides must come from the named study arm')
    environment = dict(environment)
    if arm.startswith('poisoncap-') and point.get('nested_allocator') == 'mruby-gc':
        environment['MRB_GC_POISONCAP'] = '1' if arm == 'poisoncap-temporal' else '0'
    if arm.startswith('poisoncap-') and point.get('nested_allocator') == 'cpython-pymalloc':
        environment['PYM_POISONCAP_MODE'] = '1' if arm == 'poisoncap-temporal' else '0'
    if arm != 'cheribsd-default':
        switch = 'ENABLE' if expected[arm] else 'DISABLE'
        environment['_RUNTIME_REVOCATION_' + switch] = '1'
    return environment


def application_command(point, timeout):
    environment = policy_environment(point)
    # Study arms inherit no allocator settings from the SSH server or shell.
    prefix = ['env'] if point['arm'] == 'cheribsd-default' else [
        'env', '-i', 'PATH=/sbin:/bin:/usr/sbin:/usr/bin', 'HOME=/root', 'LC_ALL=C']
    return shlex.join(['timeout', str(timeout), *prefix,
                      *[k+'='+str(v) for k, v in environment.items()], *point['argv']])


def nested_samples(stderr):
    return [dict((k, v if k == 'phase' else int(v))
                 for k, v in (word.split('=', 1) for word in line.split()[1:]))
            for line in stderr.splitlines() if line.startswith('MRB_GC_STUDY ')]


def verdict(point, rc, stdout, stderr, stdout_raw=None):
    if rc: return 'transport-error'
    exits = re.findall(r'^EXP-GUEST-EXIT (\d+)$', stderr, re.M)
    if len(exits) != 1: return 'missing-exit-evidence'
    if int(exits[0]) != 0: return 'guest-error'
    if 'expected_stdout_sha256' in point:
        if stdout_raw is None or hashlib.sha256(stdout_raw).hexdigest() != point['expected_stdout_sha256'] or \
                len(stdout_raw) != point['expected_stdout_bytes']:
            return 'oracle-mismatch'
    elif stdout != point['expected_stdout']: return 'oracle-mismatch'
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
    if point.get('nested_allocator') == 'mruby-gc':
        try:
            inner = nested_samples(stderr)
            mode = int(point['arm'] == 'poisoncap-temporal')
            if [s['phase'] for s in inner] != point['expected_phases'] or \
                    any(s['mode'] != mode or s['issues'] < s['releases'] or
                        s['releases'] < s['reissues'] or s['gap15'] > s['reissues'] or
                        s['slot_bytes'] <= 0 or s['metadata_bytes'] <= 0
                        for s in inner):
                return 'bad-inner-metrics'
            gaps = [line for line in stderr.splitlines() if line.startswith('MRB_GC_GAPS ')]
            if len(gaps) != 1:
                return 'bad-inner-metrics'
            bins = dict((k, int(v)) for k, v in (word.split('=', 1) for word in gaps[0].split()[1:]))
            if set(bins) != {f'b{i}' for i in range(32)} or sum(bins.values()) != inner[-1]['reissues']:
                return 'bad-inner-metrics'
            if inner[-1]['sweeps'] < point.get('expected_min_sweeps', 0):
                return 'bad-inner-metrics'
        except (KeyError, ValueError):
            return 'bad-inner-metrics'
    if point.get('nested_allocator') == 'ffmpeg-pool':
        mode = 2 if point['arm'] == 'poisoncap-temporal' else 0
        if re.findall(r'^FFPOOL-POLICY mode=(\d+)\b', stderr, re.M) != [str(mode)]:
            return 'bad-inner-metrics'
        expected = [phase for phase in point['expected_phases']
                    if phase.startswith(('before-', 'released-'))]
        phases = re.findall(r'^FFPOOL-MEM phase=([a-z]+-\d+)\b', stderr, re.M)
        if phases != expected:
            return 'bad-inner-metrics'
        totals = re.findall(r'^FF2-GAP-TOTAL issues=(\d+) reuses=(\d+) observer=(\d+)$',
                            stderr, re.M)
        pairs = re.findall(r'^FF2-GAP pair=(\d+) a=(\d+) b=(\d+)$', stderr, re.M)
        if len(totals) != 1 or [int(row[0]) for row in pairs] != list(range(16)):
            return 'bad-inner-metrics'
        issues, reuses, observer = map(int, totals[0])
        if not 0 < reuses <= issues or observer <= 0 or \
                sum(int(a) + int(b) for _, a, b in pairs) != reuses:
            return 'bad-inner-metrics'
    if point.get('nested_allocator') == 'cpython-pymalloc':
        mode = int(point['arm'] == 'poisoncap-temporal')
        policy = re.findall(r'^PYM_INTERPRETER_POLICY mode=(\d+) payload_reservation=(\d+) metadata_reservation=(\d+)$', stderr, re.M)
        report = re.findall(r'^PYM_POISONCAP mode=(\d+) sweeps=(\d+) poison_bytes=(\d+) clear_bytes=(\d+) ', stderr, re.M)
        if len(policy) != 1 or len(report) != 1 or \
                tuple(map(int, policy[0])) != (mode, 64 << 20, 16 << 20) or \
                int(report[0][0]) != mode or \
                (mode and (int(report[0][1]) == 0 or int(report[0][2]) == 0)):
            return 'bad-inner-metrics'
    return 'pass'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', type=Path, help='Attach to an owned existing guest')
    p.add_argument('--sdk', type=Path)
    p.add_argument('--rootfs', type=Path)
    p.add_argument('--disk', type=Path)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--disable-default-revocation', action='store_true',
                   help='Disable the guest-wide default at boot; arms still set their process policy')
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
                      disable_default_revocation=args.disable_default_revocation)
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
        policy_environment(point)
    for point in points:
        for host, guest in point['files'].items():
            if guest in files and files[guest]['sha256'] != digest(host):
                raise ValueError('conflicting guest input: '+guest)
            files[guest] = dict(host=host, sha256=digest(host))
    for guest, spec in files.items():
        call('mkdir -p '+shlex.quote(str(Path(guest).parent)))
        subprocess.run(['scp', '-O', *options, '-P', str(args.port), spec['host'],
                        'root@127.0.0.1:'+guest], check=True, capture_output=True, timeout=120)
        if call('sha256 -q '+shlex.quote(guest)) != spec['sha256']:
            raise RuntimeError('staged input hash mismatch')
    guest_default = call('sysctl -n security.cheri.runtime_revocation_default')
    if args.disable_default_revocation and guest_default != '0':
        raise RuntimeError('guest default revocation was not disabled')
    manifest = dict(boot=boot, platform=platform, files=files,
                    guest_default_revocation=guest_default,
                    process_environments={p['id']: policy_environment(p) for p in points},
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
            command = application_command(point, args.timeout)
            command = 'ulimit -c 0; '+command+'; result=$?; printf "EXP-GUEST-EXIT %s\\n" "$result" >&2; exit 0'
            (directory/'command.txt').write_text(command+'\n')
            start = time.monotonic()
            # Serial kernel panics and host timeouts abort this guest campaign.
            # A process exit without these signals is checked by the oracle below.
            process = subprocess.Popen(ssh+[command], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            deadline = time.monotonic() + args.timeout + 20
            serial_path = args.key.parent/'serial.log'
            failure = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = 'host-timeout'
                    break
                try:
                    stdout_raw, stderr_raw = process.communicate(timeout=min(2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    panic = guest_panic(serial_path)
                    if panic:
                        failure = 'guest-panic'
                        break
            if not failure and guest_panic(serial_path):
                failure = 'guest-panic'
            if failure:
                process.terminate()
                try:
                    stdout_raw, stderr_raw = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout_raw, stderr_raw = process.communicate()
                (directory/'stdout').write_bytes(stdout_raw)
                (directory/'stderr').write_bytes(stderr_raw)
                record = dict(point=point, repetition=repetition, status=failure,
                              host_seconds=time.monotonic()-start,
                              diagnostic=guest_panic(serial_path) if failure == 'guest-panic' else None)
                with (args.out/'runs.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                raise RuntimeError(failure + (': ' + record['diagnostic']
                                              if record['diagnostic'] else ''))
            (directory/'stdout').write_bytes(stdout_raw)
            (directory/'stderr').write_bytes(stderr_raw)
            stdout = stdout_raw.decode(errors='replace')
            stderr = stderr_raw.decode(errors='replace')
            status = verdict(point, process.returncode, stdout, stderr, stdout_raw)
            try: memory = samples(stderr)
            except (ValueError, KeyError): memory = []
            record = dict(point=point, repetition=repetition, status=status,
                          effective_environment=policy_environment(point),
                          memory=memory, host_seconds=time.monotonic()-start,
                          stdout_sha256=digest(directory/'stdout'),
                          stderr_sha256=digest(directory/'stderr'))
            try: record['allocations'] = allocation_samples(stderr)
            except (ValueError, KeyError): record['allocations'] = []
            try: record['inner_memory'] = nested_samples(stderr)
            except (ValueError, KeyError): record['inner_memory'] = []
            with (args.out/'runs.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(point['id'], repetition, status, flush=True)
            if status in ('transport-error', 'missing-exit-evidence'):
                raise RuntimeError('lost guest control')


if __name__ == '__main__':
    main()
