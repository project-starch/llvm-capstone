#!/usr/bin/env python3
"""Check the four real-decoder FATE arms and record their pool memory facts."""

import argparse
import hashlib
import json
from pathlib import Path
import re


CASES = {'xvid_vlc_trac7411': 20, 'resize_down-up': 150}
ARMS = {'capstone-pool0': 0, 'capstone-pool2': 2,
        'poisoncap-spatial': 0, 'poisoncap-temporal': 2}
FRAME = re.compile(r'^\s*0,\s*-?\d+,\s*-?\d+,\s*\d+,\s*\d+,\s*[0-9a-f]{32}\s*$', re.M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one_line(stderr, prefix):
    lines = [line for line in stderr.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(f'expected one {prefix!r}, got {len(lines)}')
    return lines[0]


def fields(line):
    return {key: int(value) for key, value in
            re.findall(r'([a-z_]+)=([0-9]+)', line)}


def gaps(stderr):
    counts = fields(one_line(stderr, 'FF2-GAP-TOTAL '))
    if set(counts) != {'issues', 'reuses', 'observer'} or counts['observer'] != 16656:
        raise ValueError('invalid pool observer totals')
    pairs = re.findall(r'^FF2-GAP pair=(\d+) a=(\d+) b=(\d+)$', stderr, re.M)
    if [int(index) for index, _, _ in pairs] != list(range(16)):
        raise ValueError('incomplete release-gap bins')
    bins = [int(value) for _, a, b in pairs for value in (a, b)]
    if sum(bins) != counts['reuses'] or counts['reuses'] > counts['issues']:
        raise ValueError('release-gap bins do not reconcile')
    return counts, bins


def collect(capstone, cheribsd, kernel):
    records = []
    for root, platform in ((capstone, 'capstone'), (cheribsd, 'cheribsd')):
        attempts = [json.loads(line) for line in (root / 'runs.jsonl').read_text().splitlines()]
        if len(attempts) != 12:
            raise ValueError(f'{platform}: expected 12 application processes')
        for attempt in attempts:
            point = attempt['point']
            name = next((case for case in CASES if case in point['id']), None)
            arm = point['arm']
            if name is None or arm not in ARMS or attempt['status'] != 'pass':
                raise ValueError(f'{platform}: bad case, arm or status')
            if platform == 'capstone' and not arm.startswith('capstone-') or \
                    platform == 'cheribsd' and not arm.startswith('poisoncap-'):
                raise ValueError('arm on wrong platform')
            rep = attempt['repetition']
            folder = root / f"{point['id']}-{rep}"
            output = (folder / 'stdout').read_bytes()
            stderr = (folder / 'stderr').read_text()
            expected = point['expected_stdout_sha256']
            if (sha(folder / 'stdout') != expected or len(output) != point['expected_stdout_bytes'] or
                    sha(folder / 'stdout') != attempt['stdout_sha256'] or
                    sha(folder / 'stderr') != attempt['stderr_sha256'] or
                    len(FRAME.findall(output.decode())) != CASES[name]):
                raise ValueError(f'{arm} {name} {rep}: frame or output oracle mismatch')
            counts, bins = gaps(stderr)
            if platform == 'capstone':
                policy = fields(one_line(stderr, 'EXP-POOL '))
                outer = fields(one_line(stderr, 'EXP-MEM phase=released-0 '))
                if (policy['mode'] != ARMS[arm] or outer['live'] != 0 or
                        (ARMS[arm] == 0 and policy['revoke'] != 0) or
                        (ARMS[arm] == 2 and policy['revoke'] == 0)):
                    raise ValueError(f'{arm} {name} {rep}: invalid Capstone policy')
                adapter = None
            else:
                policy = fields(one_line(stderr, 'FFPOOL-POLICY '))
                adapter = fields([line for line in stderr.splitlines()
                                  if line.startswith('FF2_POISONCAP ')][-1])
                outer = fields(one_line(stderr, 'EXP-CHERI phase=released-0 '))
                if (policy['mode'] != ARMS[arm] or policy['payload_reservation'] != 4194304 or
                        outer['revocation'] != 0 or outer['heap_error'] != 0 or
                        outer['shadow_error'] != 0 or
                        (ARMS[arm] == 0 and any(adapter.values())) or
                        (ARMS[arm] == 2 and adapter['sweeps'] == 0)):
                    raise ValueError(f'{arm} {name} {rep}: invalid PoisonCap policy')
            records.append(dict(case=name, frames=CASES[name], arm=arm, repetition=rep,
                                stdout_sha256=sha(folder / 'stdout'),
                                stderr_sha256=sha(folder / 'stderr'),
                                issues=counts['issues'], reuses=counts['reuses'],
                                observer_bytes=counts['observer'], bins=bins,
                                outer=outer, adapter=adapter))
    expected = {(case, arm, rep) for case in CASES for arm in ARMS for rep in range(3)}
    if {(r['case'], r['arm'], r['repetition']) for r in records} != expected:
        raise ValueError('missing or duplicate application processes')
    for case in CASES:
        group = [r for r in records if r['case'] == case]
        if len({(r['issues'], r['reuses'], tuple(r['bins'])) for r in group}) != 1:
            raise ValueError(f'{case}: matched pool lease behavior diverged')
    return dict(schema=1, application='FFmpeg 9.0.1 complete Matroska/MPEG-4 decoder',
                inputs='two adapted FATE cases; not official FATE scores',
                runs=records, valid_attempts=len(records),
                all_gap_bins_identical_per_case=True,
                patched_cheribsd_kernel_sha256=sha(kernel))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capstone', type=Path, required=True)
    p.add_argument('--cheribsd', type=Path, required=True)
    p.add_argument('--kernel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    document = collect(a.capstone, a.cheribsd, a.kernel)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(document, indent=2) + '\n')
    print(f"validated {document['valid_attempts']}/24 four-arm FATE processes")


if __name__ == '__main__':
    main()
