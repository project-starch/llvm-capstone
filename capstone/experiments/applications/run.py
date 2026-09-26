#!/usr/bin/env python3
"""Run an explicit matrix through one persistent VM; preserve every attempt.

Each JSON point specifies id, application, arm, image (host path), argv (guest
arguments), expected_stdout, environment, and optionally expected_phases.
This runner never reboots, retries, or silently skips a point. The timeout
terminates capstone_vm with SIGTERM, allowing its guest cancellation to run.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def parse_memory(text):
    samples = []
    for line in text.splitlines():
        if not line.startswith('EXP-MEM '): continue
        fields = dict(word.split('=', 1) for word in line.split()[1:])
        if not {'phase', 'live', 'peak'} <= fields.keys():
            raise ValueError('incomplete heap sample')
        sample = {k: v if k == 'phase' else int(v) for k, v in fields.items()}
        if any(v < 0 for k, v in sample.items() if k != 'phase'):
            raise ValueError('negative heap counter')
        samples.append(sample)
    return samples

def verdict(point, rc, timed_out, stdout, stderr, result):
    if timed_out: return 'timeout'
    if result.get('kind') == 'signal': return 'signal'
    if rc: return 'exit-error'
    if result.get('kind') != 'exit' or result.get('value') != 0: return 'missing-exit-evidence'
    if 'expected_values' in point:
        values = re.findall(r'\d+: oracle = "([^"\n]+)"', stdout)
        if values != point['expected_values'] or re.search(r'\b(ERROR|FATAL|PANIC):', stderr):
            return 'oracle-mismatch'
    elif stdout != point['expected_stdout']: return 'oracle-mismatch'
    try: samples = parse_memory(stderr)
    except (ValueError, KeyError): return 'bad-metrics'
    if point.get('expected_phases') is not None:
        if [x['phase'] for x in samples] != point['expected_phases']: return 'missing-phases'
    if not samples or any(x['live'] > x['peak'] for x in samples): return 'bad-metrics'
    return 'pass'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--timeout', type=int, default=90)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    points = json.loads(args.points.read_text())
    if args.repeat < 1 or len({p['id'] for p in points}) != len(points):
        raise ValueError('positive repeats and unique point ids required')
    config = json.loads((args.state / 'config.json').read_text())
    env = dict(os.environ, PYTHONPATH=str(REPO / 'capstone/runtime/host'))
    cli = [sys.executable, '-m', 'capstone_vm', '--state', str(args.state)]
    def call(words):
        return subprocess.check_output(cli + words, text=True, env=env, timeout=30).strip()
    boot = call(['exec', 'cat', '/proc/sys/kernel/random/boot_id'])
    (args.out / 'points.json').write_text(json.dumps(points, indent=2)+'\n')
    manifest = dict(boot_id=boot, platform=config['identity'], repeats=args.repeat,
                    timeout_seconds=args.timeout, points_sha256=digest(args.points),
                    runner_sha256=digest(Path(__file__)), timing='host wall time; diagnostic only')
    manifest['guest_launcher_sha256'] = call(['exec', 'sha256sum', '/usr/bin/capstone-exec']).split()[0]
    inputs = {}
    for point in points:
        for value in point['argv'] + list(point.get('environment', {}).values()):
            if value.startswith('/mnt/host/'):
                path = Path(config['share']) / value.removeprefix('/mnt/host/')
                if path.is_file(): inputs[value] = digest(path)
    manifest['workload_inputs'] = inputs
    images = {p['image']: digest(p['image']) for p in points if Path(p['image']).is_file()}
    manifest['images'] = images
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for point in points:
        for rep in range(args.repeat):
            directory = args.out / (point['id'] + '-' + str(rep))
            directory.mkdir()
            record = dict(point=point, repetition=rep, boot_id=boot)
            image = Path(point['image'])
            if not image.is_file():
                record.update(status='unavailable', reason='image missing', memory=[])
            else:
                try:
                    if digest(image) != images[point['image']]:
                        raise RuntimeError('image changed after manifest was recorded')
                    guest_image = '/mnt/host/' + str(image.resolve().relative_to(Path(config['share']).resolve()))
                    before = json.loads(call(['exec', 'capstone-exec', '--stats']))
                    if before['live_domains']:
                        raise RuntimeError('another domain is active; campaign does not own the VM')
                    if call(['exec', 'cat', '/proc/sys/kernel/random/boot_id']) != boot:
                        raise RuntimeError('boot changed during campaign')
                    result_file = directory / 'exit.json'
                    cmd = cli + ['run', '--result', str(result_file)]
                    for key, value in point.get('environment', {}).items():
                        cmd += ['-e', key + '=' + value]
                    cmd += [guest_image] + point['argv']
                    (directory / 'command.json').write_text(json.dumps(cmd)+'\n')
                    start = time.monotonic()
                    child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    timed_out = False
                    try: stdout, stderr = child.communicate(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        child.send_signal(signal.SIGTERM)
                        try: stdout, stderr = child.communicate(timeout=20)
                        except subprocess.TimeoutExpired:
                            child.kill(); stdout, stderr = child.communicate()
                    elapsed = time.monotonic()-start
                    (directory / 'stdout').write_bytes(stdout)
                    (directory / 'stderr').write_bytes(stderr)
                    stdout, stderr = stdout.decode(errors='replace'), stderr.decode(errors='replace')
                    result = json.loads(result_file.read_text()) if result_file.exists() else {}
                    after = json.loads(call(['exec', 'capstone-exec', '--stats']))
                    status = verdict(point, child.returncode, timed_out, stdout, stderr, result)
                    if digest(image) != images[point['image']]:
                        raise RuntimeError('image changed during execution')
                    if after['live_domains'] or after['live_regions'] or after['live_bytes']:
                        status = 'cleanup-failure'
                    try: memory = parse_memory(stderr)
                    except ValueError as e:
                        memory = []
                        record['metrics_error'] = str(e)
                    record.update(status=status, exit=result, returncode=child.returncode,
                                  host_seconds=elapsed, memory=memory, before=before, after=after,
                                  image_sha256=digest(image), stdout_sha256=digest(directory/'stdout'),
                                  stderr_sha256=digest(directory/'stderr'))
                    record['inner_memory'] = [dict((k, v if k == 'phase' else int(v))
                        for k, v in (word.split('=', 1) for word in line.split()[1:]))
                        for line in stderr.splitlines() if line.startswith('EXP-INNER ')]
                except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as e:
                    record.update(status='infrastructure-error', reason=str(e), memory=[])
            with (args.out / 'runs.jsonl').open('a') as f:
                f.write(json.dumps(record, sort_keys=True)+'\n')
            print(point['id'], rep, record['status'], flush=True)
            if record['status'] in ('cleanup-failure', 'infrastructure-error'):
                raise SystemExit('VM control failed; remaining points were not attempted')

if __name__ == '__main__': main()
