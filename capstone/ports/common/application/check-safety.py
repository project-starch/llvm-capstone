#!/usr/bin/env python3
"""Check delegated fixtures against the ports' unchanged safety predictions.

Requires exclusive use of the selected VM while collecting QEMU diagnostics.
The application status and fault record come from capstone-job. Full return
marks come from the fixture itself, corroborated by the actual 8-bit exit code.

--expect and --verdict judge images this runner boots but the port's own
classifier cannot, such as a bug corpus's cases: the fixtures are collected as
usual (fx<n>.json, .stdout, .stderr, .qemu in --out), and the verdict script is
then run as `<verdict> <out> <expect> <arm> <images> <fixture>...`; its exit
status is the result.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PORTS = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    'classifier', PORTS / 'ffmpeg/app/host/safety-verdict.py')
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)


def classify(stdout, diagnostics, result, fixture):
    # Each port's fixtures print their own prefix; this oracle reads one. MCAPP- joined the list
    # on 2026-10-05: without it every memcached cell raised "fixture mark and actual exit status
    # disagree", not because a mark was wrong but because the mark regex matched nothing -- the
    # mirror of a clean zero, and it showed up on the three known-good controls first.
    lines = stdout.replace('TSAPP-', 'FFAPP-').replace('MCAPP-', 'FFAPP-').splitlines()
    marks = re.findall(rf'^FFAPP-FIX {fixture} mark=([0-9a-f]+)$', '\n'.join(lines), re.M)
    refusal = re.findall(r'^FFAPP-POOL fail (\d+)$', '\n'.join(lines), re.M)
    done = int(marks[0], 16) if len(marks) == 1 else None
    if done is None and len(refusal) == 1:
        done = int(refusal[0])
    if result.get('kind') == 'exit':
        if done is None or result.get('value') != done & 255 or result.get('fault'):
            raise ValueError('fixture mark and actual exit status disagree')
    elif result.get('kind') == 'signal' and result.get('value') == 11 and result.get('fault'):
        if done is not None:
            raise ValueError('fixture faulted after reporting completion')
        pc = re.search(r'\bpc=0x([0-9a-f]+)', result['fault'])
        if not pc or not re.search(r'\bpc\s*=\s*(?:0x)?' + pc[1] + r'\b', diagnostics):
            raise ValueError('fault record has no matching QEMU diagnostic')
    else:
        raise ValueError('missing clean exit or recorded domain SIGSEGV')
    # Diagnostics are a separate channel. Line-buffered fixture output must
    # contain touch, and no returned/mark line, for any attributed fault.
    outcome, info = classifier.classify(lines + diagnostics.splitlines(), fixture, done)
    if result.get('kind') == 'exit' and outcome[0].startswith('FAULT'):
        raise ValueError('fault diagnostic disagrees with normal exit')
    return outcome, info


def matches(got, detail, info, kind, value, fixture):
    if kind == 'RETURN':
        return got == kind and (int(detail, 16) >> 20 == fixture if value == '*' else detail == value)
    if kind == 'FAULT':
        return got == kind and (value == 'any' or detail == value)
    if kind == 'POOLFAIL':
        return got == kind and detail == value
    if kind == 'LEN':
        length = info['len']
        return length is not None and (length > 65536 if value == 'arena' else length == int(value))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--port', choices=['ffmpeg', 'wireshark'], required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--expect', type=Path, help="predictions (default: the port's safety-expect.txt)")
    parser.add_argument('--verdict', type=Path, help='external verdict script over the collected fixtures')
    parser.add_argument('fixtures', nargs='+', type=int)
    args = parser.parse_args()
    expectations = {}
    expect = args.expect or PORTS / args.port / 'app/host/safety-expect.txt'
    for line in expect.read_text().splitlines():
        words = line.split('#', 1)[0].split()
        if len(words) == 4 and words[0] == args.arm:
            expectations.setdefault(int(words[1]), []).append(words[2:])
    for number in args.fixtures:
        if number not in expectations:
            parser.error(f'no prediction for {args.arm} fixture {number}')
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = 'ffapp' if args.port == 'ffmpeg' else 'tsapp'
    qemu_log = args.state / 'qemu.log'
    results = []
    for number in args.fixtures:
        result_path = args.out / f'fx{number}.json'
        start = qemu_log.stat().st_size
        with (args.out / f'fx{number}.stdout').open('wb') as out, (args.out / f'fx{number}.stderr').open('wb') as err:
            subprocess.run([sys.executable, str(HERE / 'run.py'), '--state', str(args.state),
                            '--result', str(result_path), str(args.images / f'{prefix}_fx{number}.dom')],
                           stdout=out, stderr=err, check=False)
        with qemu_log.open('rb') as stream:
            stream.seek(start)
            diagnostics = stream.read().decode(errors='replace')
        (args.out / f'fx{number}.qemu').write_text(diagnostics)
        if args.verdict:
            continue
        result = json.loads(result_path.read_text())
        (got, detail, explanation), info = classify(
            (args.out / f'fx{number}.stdout').read_text(), diagnostics, result, number)
        passed = all(matches(got, detail, info, *want, number) for want in expectations[number])
        results.append(dict(fixture=number, passed=passed, outcome=got, detail=detail,
                            explanation=explanation, expected=expectations[number], **info))
        print(f'fx{number}: {"AS PREDICTED" if passed else "DIFFERS"}: {got} {detail}', flush=True)
    if args.verdict:
        return subprocess.run([sys.executable, str(args.verdict), str(args.out), str(expect), args.arm,
                               str(args.images), *map(str, args.fixtures)]).returncode
    (args.out / 'verdict.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if all(row['passed'] for row in results) else 1


if __name__ == '__main__':
    sys.exit(main())
