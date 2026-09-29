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
import shutil
import subprocess
import sys
import time
from allocation_metrics import allocation_samples, valid_allocations
from reuse_gap_metrics import parse_reuse_gap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def tree_digest(root):
    """Hash names, empty directories and contents; reject external symlinks."""
    h = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'fixture has a symlink: {path}')
        relative = path.relative_to(root).as_posix().encode()
        if path.is_dir():
            h.update(b'd\0' + relative + b'\0')
        elif path.is_file():
            h.update(b'f\0' + relative + b'\0' + digest(path).encode() + b'\0')
        else:
            raise ValueError(f'fixture has a special file: {path}')
    return h.hexdigest()

def fresh_tree_paths(share, specification):
    if not isinstance(specification, dict) or set(specification) != {'source', 'destination'}:
        raise ValueError('fresh_tree requires source and destination')
    base = share.resolve(strict=True)
    paths = []
    for name in ('source', 'destination'):
        if not isinstance(specification[name], str):
            raise ValueError(f'fresh_tree {name} must be a path string')
        relative = Path(specification[name])
        if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
            raise ValueError(f'fresh_tree {name} must be a nonempty share-relative path')
        path = base / relative
        if not path.resolve(strict=False).is_relative_to(base):
            raise ValueError(f'fresh_tree {name} escapes VM share')
        paths.append(path)
    source, destination = paths
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f'fresh_tree source is not an ordinary directory: {source}')
    source_real, destination_real = source.resolve(), destination.resolve(strict=False)
    if (source_real == destination_real or source_real in destination_real.parents or
            destination_real in source_real.parents):
        raise ValueError('fresh_tree source and destination overlap')
    return source, destination

def prepare_fresh_tree(share, specification, expected_digest, owned):
    source, destination = fresh_tree_paths(share, specification)
    if tree_digest(source) != expected_digest:
        raise RuntimeError(f'fresh_tree source changed during campaign: {source}')
    if destination.exists() or destination.is_symlink():
        if destination not in owned:
            raise RuntimeError(f'fresh_tree destination was not created by this campaign: {destination}')
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    owned.add(destination)

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

def parse_gc_study(text):
    rows = [dict((k, v if k == 'phase' else int(v))
                 for k, v in (word.split('=', 1) for word in line.split()[1:]))
            for line in text.splitlines() if line.startswith('MRB_GC_STUDY ')]
    lines = [line for line in text.splitlines() if line.startswith('MRB_GC_GAPS ')]
    if len(lines) != 1:
        raise ValueError('expected exactly one GC gap histogram')
    bins = dict((k, int(v)) for k, v in
                (word.split('=', 1) for word in lines[0].split()[1:]))
    if set(bins) != {f'b{i}' for i in range(32)}:
        raise ValueError('malformed GC gap histogram')
    return rows, bins

def verdict(point, rc, timed_out, stdout, stderr, result, stdout_raw=None):
    if timed_out: return 'timeout'
    if result.get('kind') == 'signal': return 'signal'
    if rc: return 'exit-error'
    if result.get('kind') != 'exit' or result.get('value') != 0: return 'missing-exit-evidence'
    if point.get('application') == 'postgres' and re.search(r'\b(ERROR|FATAL|PANIC):', stderr):
        return 'oracle-mismatch'
    if 'expected_values' in point:
        values = re.findall(r'\d+: oracle = "([^"\n]+)"', stdout)
        if values != point['expected_values']:
            return 'oracle-mismatch'
    elif 'expected_stdout_sha256' in point:
        if stdout_raw is None or hashlib.sha256(stdout_raw).hexdigest() != point['expected_stdout_sha256'] or \
                len(stdout_raw) != point['expected_stdout_bytes']:
            return 'oracle-mismatch'
    elif stdout != point['expected_stdout']: return 'oracle-mismatch'
    try: samples = parse_memory(stderr)
    except (ValueError, KeyError): return 'bad-metrics'
    if point.get('expected_phases') is not None:
        if [x['phase'] for x in samples] != point['expected_phases']: return 'missing-phases'
    if not samples or any(x['live'] > x['peak'] for x in samples): return 'bad-metrics'
    if point.get('allocations') and not valid_allocations(stderr, point['expected_phases']):
        return 'bad-allocation-metrics'
    if point.get('reuse_gap'):
        try: parse_reuse_gap(stderr, point['reuse_gap'])
        except ValueError: return 'bad-reuse-gap'
    if point.get('gc_gaps'):
        try:
            rows, bins = parse_gc_study(stderr)
            mode = int(point['arm'] == 'capstone-sublet')
            if [row['phase'] for row in rows] != point['expected_phases'] or \
                    any(row['mode'] != mode or row['issues'] < row['releases'] or
                        row['releases'] < row['reissues'] or row['gap15'] > row['reissues'] or
                        row['pages'] > row['peak_pages'] or row['slot_bytes'] <= 0 or
                        row['observer_bytes'] <= 0 or row['page_payload_bytes'] <= 0 or
                        row['page_metadata_bytes'] <= 0 for row in rows) or \
                    sum(bins.values()) != rows[-1]['reissues']:
                return 'bad-inner-metrics'
        except (ValueError, KeyError):
            return 'bad-inner-metrics'
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
    share = Path(config['share'])
    env = dict(os.environ, PYTHONPATH=str(REPO / 'capstone/runtime/host'))
    cli = [sys.executable, '-m', 'capstone_vm', '--state', str(args.state)]
    def call(words):
        return subprocess.check_output(cli + words, text=True, env=env, timeout=30).strip()
    boot = call(['exec', 'cat', '/proc/sys/kernel/random/boot_id'])
    (args.out / 'points.json').write_text(json.dumps(points, indent=2)+'\n')
    manifest = dict(boot_id=boot, platform=config['identity'], repeats=args.repeat,
                    timeout_seconds=args.timeout, points_sha256=digest(args.points),
                    runner_sha256=digest(Path(__file__)),
                    allocation_validator_sha256=digest(HERE / 'allocation_metrics.py'), timing='host wall time; diagnostic only')
    manifest['reuse_gap_validator_sha256'] = digest(HERE / 'reuse_gap_metrics.py')
    manifest['guest_launcher_sha256'] = call(['exec', 'sha256sum', '/usr/bin/capstone-exec']).split()[0]
    fresh_trees = {}
    for point in points:
        if 'fresh_tree' not in point:
            continue
        source, destination = fresh_tree_paths(share, point['fresh_tree'])
        if destination.exists() or destination.is_symlink():
            raise ValueError(f'fresh_tree destination exists before campaign: {destination}')
        fresh_trees[str(source)] = tree_digest(source)
    manifest['fresh_trees'] = fresh_trees
    inputs = {}
    for point in points:
        declared = point.get('inputs', [])
        if not isinstance(declared, list) or any(
                not isinstance(value, str) or not value.startswith('/mnt/host/')
                for value in declared):
            raise ValueError('point inputs must be a list of /mnt/host/ file paths')
        for value in point['argv'] + list(point.get('environment', {}).values()) + declared:
            if value.startswith('/mnt/host/'):
                path = share / value.removeprefix('/mnt/host/')
                if not path.resolve(strict=False).is_relative_to(share.resolve()):
                    raise ValueError(f'input escapes VM share: {value}')
                if path.is_file(): inputs[value] = digest(path)
                elif value in declared:
                    raise ValueError(f'declared input is not a file: {value}')
    manifest['workload_inputs'] = inputs
    images = {p['image']: digest(p['image']) for p in points if Path(p['image']).is_file()}
    manifest['images'] = images
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    owned_trees = set()
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
                    for value in point.get('inputs', []):
                        path = share / value.removeprefix('/mnt/host/')
                        if digest(path) != inputs[value]:
                            raise RuntimeError(f'declared input changed during campaign: {value}')
                    if 'fresh_tree' in point:
                        source, _ = fresh_tree_paths(share, point['fresh_tree'])
                        prepare_fresh_tree(share, point['fresh_tree'],
                                           fresh_trees[str(source)], owned_trees)
                    guest_image = '/mnt/host/' + str(image.resolve().relative_to(share.resolve()))
                    before = json.loads(call(['exec', 'capstone-exec', '--stats']))
                    expected_capacity = point.get('expected_node_capacity')
                    if expected_capacity is not None and before['node_capacity'] != expected_capacity:
                        raise RuntimeError(f"node capacity {before['node_capacity']} != {expected_capacity}")
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
                    status = verdict(point, child.returncode, timed_out, stdout, stderr, result,
                                     (directory/'stdout').read_bytes())
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
                    record['allocations'] = allocation_samples(stderr)
                    record['inner_memory'] = [dict((k, v if k == 'phase' else int(v))
                        for k, v in (word.split('=', 1) for word in line.split()[1:]))
                        for line in stderr.splitlines() if line.startswith('EXP-INNER ')]
                    if point.get('gc_gaps'):
                        record['gc_study'], record['gc_gap_bins'] = parse_gc_study(stderr)
                    if point.get('reuse_gap'):
                        try: record['reuse_gap'] = parse_reuse_gap(stderr, point['reuse_gap'])
                        except ValueError as e: record['reuse_gap_error'] = str(e)
                except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as e:
                    record.update(status='infrastructure-error', reason=str(e), memory=[])
            with (args.out / 'runs.jsonl').open('a') as f:
                f.write(json.dumps(record, sort_keys=True)+'\n')
            print(point['id'], rep, record['status'], flush=True)
            if record['status'] in ('cleanup-failure', 'infrastructure-error'):
                raise SystemExit('VM control failed; remaining points were not attempted')
    for destination in sorted(owned_trees):
        shutil.rmtree(destination)

if __name__ == '__main__': main()
