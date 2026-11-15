#!/usr/bin/env python3
"""One boot of the FFmpeg reuse-distance measurement on a fixed subset of FATE decode tests.

    run-ffmpeg-fate.py --arm stock|protected --rep N --image FFMPEG_DOM --kit KIT \
        --samples FATE_SUITE --refs FFMPEG_SRC/tests/ref/fate --subset SUBSET.tsv --work NEW_DIR \
        [--step-timeout S]

The workload of docs/plans/2026-11-15-ffmpeg-established-workload.md. SUBSET.tsv has one FATE test
per line: `TEST<TAB>SAMPLE<TAB>framecrc|framemd5`, TEST as FATE names it (fate-h264-conformance-...),
SAMPLE relative to the fate-suite root. The image is a reuse-hooks/ffmpeg/build.sh image (main from
experiments/applications/workloads/decode.c). Every test runs as its own process, as FATE runs it:

    ./capstone-vexec ./ffmpeg.dom /mnt/vm/fate/SAMPLE 1 framecrc|framemd5 2> /tmp/fate.err

(stderr printed after stdout) through runtime/virtual/trap_platform.py run (the run of record's FFmpeg reuse runner), all tests
of one arm and repetition in one boot. Each process prints one line per decoded frame and, at exit,
one REUSE-GAP line per pool boundary. Afterwards this script

  - compares each test's frame lines with FATE's reference file of the same source tree: the
    frame count and, line by line, the size and hash columns (the timestamp columns are the
    harness's frame index, not the ffmpeg CLI's, and are not compared);
  - sums each boundary's observer counters and bins over the processes (reuse distances never
    cross a process: every process has its own pools and its own observer);
  - writes NEW_DIR/rep.json in the form collect-reuse.py reads, with per-test records under
    "tests". "rc" counts the tests whose process did not exit 0; "oracle_ok" is true when every
    test matched its reference. "error" sums the processes' observer errors.

The console log and the stage stay in NEW_DIR/boot-1/ and NEW_DIR/stage-1/ (trap_platform's layout).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TRAP_PLATFORM = REPO / 'capstone/runtime/virtual/trap_platform.py'
COUNTERS = ('attempts', 'issues', 'releases', 'reuses', 'distinct_starts', 'drops', 'fresh_reuses', 'error')
FRAME = re.compile(r'^0,\s+-?\d+,\s+-?\d+,\s+-?\d+,\s+(\d+), (0x[0-9a-f]{8}|[0-9a-f]{32})$', re.M)
STEP = './capstone-vexec ./ffmpeg.dom /mnt/vm/fate/{sample} 1 {hash}'
# stderr goes to its own file and follows stdout, so the decoder's log lines and the exit-time
# REUSE-GAP lines never split a frame line of the fully buffered stdout.
TIMED = ('t=$(date +%s); {step} 2> /tmp/fate.err; r=$?; cat /tmp/fate.err; '
         'echo "FATE-SECONDS $(( $(date +%s) - t ))"; exit $r')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_subset(path):
    tests = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        test, sample, mode = line.split('\t')
        if mode not in ('framecrc', 'framemd5'):
            raise SystemExit(f'{test}: hash mode {mode}')
        if not re.fullmatch(r'fate-[A-Za-z0-9_.+-]+', test):
            raise SystemExit(f'{test}: not a shell-safe FATE test name')
        tests.append((test, sample, mode))
    if len({t for t, _, _ in tests}) != len(tests):
        raise SystemExit('a test is listed twice')
    return tests


def reference_frames(ref):
    """[(size, hash)] of a FATE framecrc/framemd5 reference, in order."""
    frames = []
    for line in Path(ref).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        cols = [c.strip() for c in line.split(',')]
        frames.append((int(cols[4]), cols[5]))
    return frames


def parse_reuse(text):
    found = {}
    for m in re.finditer(r'^REUSE-GAP (\S+) (.*)$', text, re.M):
        kv = dict(item.split('=', 1) for item in m.group(2).split())
        rec = {k: int(kv[k]) for k in kv if k != 'bins'}
        rec['bins'] = [int(x) for x in kv['bins'].split(',')]
        if len(rec['bins']) != 32 or m.group(1) in found:
            raise ValueError(f'malformed or repeated REUSE-GAP line for {m.group(1)}')
        found[m.group(1)] = rec
    return found


def judge(test, sample, mode, result, refs):
    """The per-test record: exit, frames, the FATE comparison and the observer lines."""
    out = result['output']
    got = [(int(s), h) for s, h in FRAME.findall(out)]
    want = reference_frames(refs / test.removeprefix('fate-'))
    first_diff = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), None)
    match = len(got) == len(want) and first_diff is None
    seconds = re.search(r'^FATE-SECONDS (\d+)$', out, re.M)
    frame_text = '\n'.join(f'{s} {h}' for s, h in got) + '\n'
    return {
        'sample': sample, 'hash': mode, 'kind': result['kind'], 'value': result['value'],
        'fault': result.get('fault'), 'kernel_faults': result.get('kernel_faults', []),
        'frames': len(got), 'reference_frames': len(want), 'fate_match': match,
        'first_difference': first_diff if first_diff is not None else (None if match else min(len(got), len(want))),
        'frames_sha256': hashlib.sha256(frame_text.encode()).hexdigest(),
        'seconds': int(seconds.group(1)) if seconds else None,
        'exp_ok': 'EXP-OK ffmpeg 1' in out,
        'reuse_gap': parse_reuse(out),
    }


def summed(tests):
    total = {}
    for rec in tests.values():
        for name, rg in rec['reuse_gap'].items():
            t = total.setdefault(name, {k: 0 for k in COUNTERS} | {'bins': [0] * 32, 'capacity': rg['capacity'],
                                                                   'processes': 0})
            for k in COUNTERS:
                t[k] += rg.get(k, 0)
            t['bins'] = [a + b for a, b in zip(t['bins'], rg['bins'])]
            t['processes'] += 1
            if rg['capacity'] != t['capacity']:
                raise ValueError(f'{name}: capacity differs between processes')
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('stock', 'protected'), required=True)
    p.add_argument('--rep', type=int, required=True)
    for name in ('image', 'kit', 'samples', 'refs', 'subset', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--step-timeout', type=int, default=1800)
    a = p.parse_args()
    if a.work.exists():
        raise SystemExit(f'use a new output directory: {a.work}')
    tests = read_subset(a.subset)

    cmd = [sys.executable, str(TRAP_PLATFORM), 'run', '--kit', str(a.kit), '--work', str(a.work),
           '--step-timeout', str(a.step_timeout), '--file', f'ffmpeg.dom={a.image}']
    for sample in sorted({s for _, s, _ in tests}):
        cmd += ['--file', f'fate/{sample}={a.samples / sample}']
    for test, sample, mode in tests:
        cmd += ['--step', f'{test}=' + TIMED.format(step=STEP.format(sample=sample, hash=mode))]
    t0 = time.time()
    run = subprocess.run(cmd, env=dict(os.environ))
    wall = time.time() - t0

    results = {r['name']: r for r in json.loads((a.work / 'results.json').read_text())}
    per_test = {}
    for test, sample, mode in tests:
        r = results.get(test)
        if r is None:
            per_test[test] = {'sample': sample, 'hash': mode, 'kind': 'missing', 'value': None, 'fate_match': False,
                              'reuse_gap': {}}
            continue
        per_test[test] = judge(test, sample, mode, r, a.refs)
    failed = [t for t, r in per_test.items() if r['kind'] != 'exit' or r['value'] != 0 or not r.get('exp_ok')]
    mismatched = [t for t, r in per_test.items() if not r['fate_match']]
    boots = sorted(a.work.glob('boot-*'))
    rec = {
        'program': 'ffmpeg', 'workload': 'FATE decode subset', 'arm': a.arm, 'rep': a.rep,
        'completed': run.returncode == 0 and len(boots) == 1 and not any(
            per_test[t]['kind'] in ('missing', 'harness', 'none') for t in per_test),
        'boots': len(boots), 'boot_wall_seconds': round(wall, 1),
        'guest_seconds': sum(r.get('seconds') or 0 for r in per_test.values()),
        'rc': len(failed), 'failed_tests': failed,
        'oracle_ok': not mismatched, 'fate_mismatches': mismatched,
        'frames': sum(r.get('frames', 0) for r in per_test.values()),
        'reuse_gap': summed(per_test),
        'tests': per_test,
        'staged_sha256': {'ffmpeg.dom': sha(a.image), 'capstone-vexec': sha(a.kit / 'adapter/capstone-vexec'),
                          'subset': sha(a.subset),
                          'samples': hashlib.sha256(''.join(f'{s}\0{sha(a.samples / s)}\n' for s in
                                                            sorted({s for _, s, _ in tests})).encode()).hexdigest()},
        'platform_sha256': {n: sha(a.kit / 'images' / n) for n in ('Image', 'fw_jump.elf', 'initramfs.cpio.gz')}
                           | {'qemu-system-riscv64': sha(a.kit / 'qemu-system-riscv64')},
        'run_staged_sha256': sha(TRAP_PLATFORM),
    }
    (a.work / 'rep.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    for name, rg in rec['reuse_gap'].items():
        print(f'ffmpeg {a.arm} rep{a.rep} {name}: processes={rg["processes"]} issues={rg["issues"]} '
              f'releases={rg["releases"]} reuses={rg["reuses"]} distinct_starts={rg["distinct_starts"]} '
              f'error={rg["error"]}')
    print(f'ffmpeg {a.arm} rep{a.rep}: completed={rec["completed"]} failed={len(failed)} '
          f'fate_mismatches={len(mismatched)} frames={rec["frames"]} {wall:.0f}s')
    ok = rec['completed'] and not failed and not mismatched and all(
        rg['error'] == 0 and rg['issues'] == rg['attempts'] for rg in rec['reuse_gap'].values())
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
