#!/usr/bin/env python3
"""test-batch.py KIT PROGRAMS WORK: the corpus tools' Batch (bug-corpora/tools/virtualvm.py) on a trap-route kit.

PROGRAMS holds the ordinary static test programs segv, ok, loop, three and the script die (`poweroff -f`).
One batch runs segv, ok, loop (bound 10 s), after-loop, die, three:
  segv       signal 11, and the boot goes on                  (a fault ends one program)
  ok         exit 0 after the fault, in the same boot
  loop       timeout: the watchdog kills it, the boot goes on
  after-loop exit 0
  die        kind none, harness "guest died"                   (only a dead guest ends a boot)
  three      exit 3, in a second boot                          (control: the status is read, not assumed)
The batch must report completed=False, because one boot did not run every step.

The kernel of an ordinary kit may not run protected programs, so the batch's `--platform` admission would stop it.
The test therefore copies the kit and wraps its launcher so that `--platform` passes; every other call goes to the
real trap-vexec.sh. Control: the unwrapped kit's batch, on a kernel without the .dom loader, begins no step.
"""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / 'capstone/bug-corpora/tools'))
import virtualvm  # noqa: E402

WRAPPER = """#!/bin/sh
[ "$1" = --platform ] && exit 0
exec sh "$(dirname "$0")/real-vexec" "$@"
"""


def main():
    kit, programs, work = (Path(x) for x in sys.argv[1:4])
    work.mkdir(parents=True)
    test_kit = work / 'kit'
    shutil.copytree(kit, test_kit)
    shutil.move(test_kit / 'adapter/capstone-vexec', test_kit / 'adapter/real-vexec')
    (test_kit / 'adapter/capstone-vexec').write_text(WRAPPER)
    (test_kit / 'adapter/capstone-vexec').chmod(0o755)

    batch = virtualvm.Batch(work / 'stage')
    batch.put('real-vexec', test_kit / 'adapter/real-vexec')
    for name in ('segv', 'ok', 'loop', 'die', 'three'):
        batch.put(name, programs / name)
    plan = [('segv', './capstone-vexec ./segv'), ('ok', './ok'), ('loop', './loop'), ('after-loop', './ok'),
            ('die', './die'), ('three', './three')]
    for name, command in plan:
        batch.step(name, command)
    batch.bounds['loop'] = 10
    steps, serial, completed = batch.execute(test_kit, work / 'guest', 900)

    want = {'segv': ('signal', 11), 'ok': ('exit', 0), 'loop': ('none', None), 'after-loop': ('exit', 0),
            'die': ('none', None), 'three': ('exit', 3)}
    failed = 0
    for name, (kind, value) in want.items():
        result = steps.get(name, ({}, {'kind': 'missing'}))[1]
        ok = result.get('kind') == kind and result.get('value') == value
        if name == 'loop':
            ok = ok and '[runner] TIMEOUT' in steps[name][0] and 'harness' not in result
        if name == 'die':
            ok = ok and result.get('harness') == 'guest died'
        print(f"{'PASS' if ok else 'FAIL'} {name}: {result}")
        failed += not ok
    print(f"{'PASS' if not completed else 'FAIL'} completed={completed} (one boot died)")
    failed += bool(completed)

    control = virtualvm.Batch(work / 'control-stage')
    control.put('ok', programs / 'ok')
    control.step('ok', './ok')
    control_steps, _, control_completed = control.execute(kit, work / 'control-guest', 300)
    ok = not control_steps
    print(f"{'PASS' if ok else 'FAIL'} control: without the .dom loader the batch begins no step "
          f"({control_steps or 'none'})")
    failed += not ok
    print(serial)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
