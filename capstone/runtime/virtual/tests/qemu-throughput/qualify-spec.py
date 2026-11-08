#!/usr/bin/env python3
"""Run prepared SPEC cases and compare with previously qualified output hashes.

This is a regression check, not a fresh official SPEC comparator run. It reuses
run-spec-tests.py, including guest image validators and exact command receipts.
No benchmark sources are fetched or stored in the repository.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Invalid relative filename: {name!r}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'comparators', 'baseline', 'qemu', 'images', 'firmware', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--benchmarks', nargs='+', required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--shell', type=Path)
    parser.add_argument('--null-device', action='store_true')
    parser.add_argument('--local-tmp', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0 or len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error('Positive timeout and distinct benchmarks required')
    baseline = json.loads(args.baseline.read_text())
    official = json.loads(args.comparators.read_text())
    if baseline.get('status') != 'PASS':
        raise ValueError('Reference qualification is not PASS')
    qualified = {row['benchmark']: row for row in baseline['benchmarks']}
    for name in args.benchmarks:
        relative(name)
        if name not in qualified or not qualified[name].get('passed'):
            raise ValueError(f'No passing qualified reference for {name}')
    for name, path in [('Image', args.images/'Image'), ('rootfs', args.images/'rootfs.ext2'),
                       ('firmware', args.firmware)]:
        if digest(path) != baseline['platform'][name]:
            raise ValueError(f'Guest platform differs from qualified reference: {name}')
    args.out.mkdir(parents=True, exist_ok=False)
    tests = Path(__file__).resolve().parents[1]/'native-task'
    spec = importlib.util.spec_from_file_location('spec_validation', tests/'validate-spec-tests.py')
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    report = dict(status='RUNNING', scope='Exact output equivalence to qualified baseline; not fresh official comparator execution',
                  baseline_sha256=digest(args.baseline), qemu_sha256=digest(args.qemu),
                  results=[], started_epoch=time.time())
    def save():
        (args.out/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for bench in args.benchmarks:
            start = time.monotonic()
            print('START', bench, flush=True)
            reference = qualified[bench]
            original = official[bench]
            if original['commands'] != reference['execution']['commands'] or original['image_compares'] != reference['execution']['image_commands']:
                raise ValueError(f'Command inventory changed: {bench}')
            inputs = {}
            for name, expected in baseline['inputs'][bench].items():
                path = args.shell if name == '.native-shell' and args.shell else args.prepared/bench/'stage'/relative(name)
                actual = digest(path)
                if actual != expected:
                    raise ValueError(f'Prepared artifact differs: {bench}/{name}')
                inputs[name] = actual
            manifest = args.out/f'{bench}-comparators.json'
            manifest.write_text(json.dumps({bench: original}, indent=2)+'\n')
            out = args.out/bench
            cmd = ['python3', str(tests/'run-spec-tests.py'), '--prepared', str(args.prepared),
                   '--out', str(out), '--qemu', str(args.qemu), '--images', str(args.images),
                   '--firmware', str(args.firmware), '--benchmarks', bench,
                   '--timeout', str(args.timeout), '--comparators', str(manifest)]
            if args.shell:
                cmd += ['--shell', str(args.shell)]
            if args.null_device:
                cmd += ['--null-device']
            if args.local_tmp:
                cmd += ['--local-tmp']
            (args.out/f'{bench}-command.json').write_text(json.dumps(cmd, indent=2)+'\n')
            with (args.out/f'{bench}-runner.log').open('w') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            directory = out/'runs'/bench
            execution = json.loads((directory/'execution.json').read_text())
            errors = validation.execution_errors(bench, execution, original, directory, digest(manifest))
            if result.returncode:
                errors.append(f'Runner exit {result.returncode}')
            comparisons = []
            for item in reference['comparators']:
                name = item['command'][-1]
                expected = item['output_sha256']
                actual = digest(directory/'output'/relative(name))
                comparisons.append(dict(name=name, expected=expected, actual=actual, passed=actual == expected))
            images = []
            for item in reference['image_comparators']:
                name = item['command']['argv'][-2]
                expected = item['image_sha256']
                actual = digest(directory/'output'/relative(name))
                images.append(dict(name=name, expected=expected, actual=actual, passed=actual == expected))
            passed = not errors and bool(comparisons) and all(x['passed'] for x in comparisons + images)
            row = dict(benchmark=bench, status='PASS' if passed else 'FAIL', inputs=inputs,
                       execution=execution, errors=errors, comparisons=comparisons, images=images,
                       elapsed_seconds=time.monotonic()-start)
            report['results'].append(row)
            save()
            print(row['status'], bench, f"{row['elapsed_seconds']:.1f}s", flush=True)
        report['status'] = 'PASS' if all(x['status'] == 'PASS' for x in report['results']) else 'FAIL'
    except BaseException as error:
        report['status'] = 'ERROR'
        report['error'] = repr(error)
        raise
    finally:
        report['finished_epoch'] = time.time()
        save()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
