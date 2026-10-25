#!/usr/bin/env python3
"""Positive control for the lookaside-slot check (slot-check-negative/control.c), both arms.

    slot-check-negative.py --builds <dir from build-controls.sh, with release/ and protected/>
                           --kit <platform kit> --llvm-bin <dir> --work <new dir>

Not part of the pre-registered run, and kept out of build-controls.sh and run-controls.py so
those stay as they were for the run of record. For each arm it compiles slot-check-negative/
control.c with the EXACT command build-controls.sh recorded for la-control-parent (first line of
<arm>/la-control-parent.log; only the source and the output differ), boots it once through
run-controls.py's own run_one, and requires:
  CHECK small-is-lookaside-slot ok, CHECK oversize-is-lookaside-slot FAILED,
  CHECK disabled-is-lookaside-slot FAILED, the program returned, exit 0, no fault.
"""
import argparse
import importlib.util
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run_controls', HERE / 'run-controls.py')
rc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc_mod)
TAG = 'la-slot-check-negative'
EXPECT = {'small-is-lookaside-slot': 'ok', 'oversize-is-lookaside-slot': 'FAILED',
          'disabled-is-lookaside-slot': 'FAILED'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ('--builds', '--kit', '--llvm-bin', '--work'):
        p.add_argument(a, type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    a.locks = a.work / 'locks'
    a.locks.mkdir()
    ok_all = True
    for arm in ('release', 'protected'):
        cmd = shlex.split((a.builds / arm / 'la-control-parent.log').read_text().splitlines()[0])
        src = [i for i, w in enumerate(cmd) if w.endswith('/control-parent/control.c')]
        out = [i for i, w in enumerate(cmd) if w.endswith('/la-control-parent.dom')]
        assert len(src) == 1 and len(out) == 1, 'recorded command has no unique source/output'
        cmd[src[0]] = str(HERE / 'slot-check-negative/control.c')
        image = a.work / arm / f'{TAG}.dom'
        image.parent.mkdir()
        cmd[out[0]] = str(image)
        b = subprocess.run(cmd, capture_output=True, text=True)
        if b.returncode:
            print(b.stderr[-2000:])
            raise SystemExit(f'build {arm} failed')
        a.builds_saved, a.builds = a.builds, a.work
        _, _, _, staged_rc, wall, log = rc_mod.run_one(a, arm, TAG)
        a.builds = a.builds_saved
        (a.work / 'boot' / f'{arm}-{TAG}' / 'body.txt').write_text(log)
        body = log.split(f'PORT_BEGIN:{TAG}\n', 1)[-1]
        checks = dict(re.findall(r'^CHECK (\S+) (ok|FAILED)$', body, re.M))
        end = re.search(rf'^PORT_END:{TAG}:(\d+)', body, re.M)
        held = (all(checks.get(k) == v for k, v in EXPECT.items()) and f'{TAG} RETURNED' in body
                and end is not None and end.group(1) == '0' and 'domain fault' not in body)
        ok_all &= held
        slots = [l for l in body.splitlines() if l.startswith('SLOT ')]
        print(f'{arm:<9} {TAG} sha256={rc_mod.sha256(image)} held={held} wall={wall}s')
        for k in EXPECT:
            print(f'  CHECK {k} {checks.get(k)} (expected {EXPECT[k]})')
        for l in slots:
            print('  ' + l)
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
