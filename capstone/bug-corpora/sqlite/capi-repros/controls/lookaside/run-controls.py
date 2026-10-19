#!/usr/bin/env python3
"""Run the LOOKASIDE lifetime controls on both arms, one boot per image, and judge each against the plan.

A copy of ../run-controls.py (the memsys5 controls): the same boot, judge and negative-test
options; only the image names (la-control-*), the repository depth and the address lines differ.

    run-controls.py --builds <dir with release/ and protected/ from build-controls.sh>
                    --kit <platform kit: qemu-system-riscv64, images/, adapter/capstone-vexec,
                           adapter/module/capstone_vm.ko> --llvm-bin <dir> --work <new dir>
                    [--jobs 8] [--locks <dir>]

A local runner, not the corpus's: shared/run-virtual.py in engine-repros judges bug cases by
fault function and has no notion of a probe line, a check line or an address. This one boots
each image in its own VM through runtime/virtual/run-staged.py (its own lock file per boot, so
boots run in parallel), and reads the serial log:

  PROBE <label> <op> <file>:<line> addr=..  printed before each access, `PROBE <label> done`
                                           after it (control_common.h)
  CHECK <what> ok|FAILED                   each prediction the program itself tests
  BLOCK / SUCCESSOR / FREED / LINK lines   addresses and bounds
  `<tag> RETURNED`                         the program returned from its body
  capstone-exec: domain fault cause=..     the launcher's fault line; its pc is resolved to
                                           function and file:line with llvm-symbolizer

A run COMPLETES when it returned, exited 0, logged no fault, and every probe reached `done`.
A stale run FAULTS AT THE PROBE when the last PROBE line is stale-A-read without `done` AND the
fault pc symbolizes to that probe's file:line. Anything else is reported as what it is.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
REPO = HERE.parents[6]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import sdk_builtins  # noqa: E402  (C-81: refuse a builtins build before anything starts)
RUN_STAGED = REPO / 'capstone/runtime/virtual/run-staged.py'
PROGRAMS = {  # image -> (source dir, prediction per arm)
    'la-control-parent': ('control-parent', {'release': 'completed', 'protected': 'completed'}),
    'la-control-sibling': ('control-sibling', {'release': 'completed', 'protected': 'completed'}),
    'la-control-successor': ('control-successor', {'release': 'completed', 'protected': 'completed'}),
    'la-control-successor-stale': ('control-successor',
                                {'release': 'completed', 'protected': 'fault-25-at-stale-A-read'}),
}
FAULT = re.compile(r'domain fault cause=(\d+) pc=(0x[0-9a-f]+) address=(0x[0-9a-f]+).*?'
                   r'code=(0x[0-9a-f]+)-(0x[0-9a-f]+)')
PROBE = re.compile(r'^PROBE (\S+) (read|write) (\S+):(\d+) addr=(0x[0-9a-f]+) len=(\d+)', re.M)

GATE = """#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export HOME=/tmp TMPDIR=/tmp CAPSTONE_EXEC_DIAGNOSTICS=1
echo PORT_BEGIN:{tag}
./capstone-vexec ./{tag}.dom {tag}
echo PORT_END:{tag}:$?
rmmod capstone_vm
echo VIRTUAL_STAGED_DONE
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbolize(llvm_bin, image, pc, code_start):
    hdr = subprocess.run([str(llvm_bin / 'llvm-readelf'), '-l', str(image)], capture_output=True,
                         text=True, check=True).stdout
    base = min(int(m.group(1), 16) for m in re.finditer(r'^\s*LOAD\s+\S+\s+(0x[0-9a-f]+)', hdr, re.M))
    addr = pc - code_start + base
    out = subprocess.run([str(llvm_bin / 'llvm-symbolizer'), f'--obj={image}', hex(addr)],
                         capture_output=True, text=True, check=True).stdout.split('\n')
    loc = out[1].strip() if len(out) > 1 else '?'
    m = re.match(r'(.*):(\d+):(\d+)$', loc)
    # The function from the sized function symbols: the symbolizer names the nearest symbol,
    # and the assembler's .Lpcrel_hi labels are in this image's symbol table.
    func = None
    nm = subprocess.run([str(llvm_bin / 'llvm-nm'), '--print-size', '--defined-only', str(image)],
                        capture_output=True, text=True, check=True).stdout
    for line in nm.splitlines():
        w = line.split()
        if len(w) == 4 and w[2] in 'tTwW' and int(w[0], 16) <= addr < int(w[0], 16) + int(w[1], 16):
            func = w[3]
    return {'image_offset': hex(addr), 'function': func,
            'file': Path(m.group(1)).name if m else loc, 'line': int(m.group(2)) if m else None}


def run_one(a, arm, tag):
    image = a.builds / arm / f'{tag}.dom'
    stage = a.work / 'stage' / f'{arm}-{tag}'
    stage.mkdir(parents=True)
    shutil.copy(image, stage / image.name)
    shutil.copy(a.kit / 'adapter/capstone-vexec', stage / 'capstone-vexec')
    shutil.copy(a.kit / 'adapter/module/capstone_vm.ko', stage / 'capstone_vm.ko')
    (stage / 'gate.sh').write_text(GATE.format(tag=tag))
    (stage / 'gate.sh').chmod(0o755)
    boot = a.work / 'boot' / f'{arm}-{tag}'
    env = dict(os.environ, CAPSTONE_QEMU_LOCK=str(a.locks / f'qemu-{arm}-{tag}.lock'))
    t0 = time.monotonic()
    p = subprocess.run(['python3', str(RUN_STAGED), '--qemu', str(a.kit / 'qemu-system-riscv64'),
                        '--images', str(a.kit / 'images'), '--stage', str(stage), '--work', str(boot),
                        '--timeout', '900'], env=env, capture_output=True, text=True)
    wall = round(time.monotonic() - t0, 1)
    log = (boot / 'serial.log').read_bytes().decode('utf-8', 'replace').replace('\r', '')
    return arm, tag, image, p.returncode, wall, log


def judge(a, arm, tag, image, staged_rc, wall, log):
    srcdir, predicted = PROGRAMS[tag][0], PROGRAMS[tag][1][arm]
    body = log.split(f'PORT_BEGIN:{tag}\n', 1)[-1] if f'PORT_BEGIN:{tag}\n' in log else ''
    end = re.search(rf'^PORT_END:{re.escape(tag)}:(\d+)', body, re.M)
    body = body[:end.start()] if end else body
    probes = [dict(label=m.group(1), op=m.group(2), at=f'{srcdir}/{m.group(3)}:{m.group(4)}',
                   line=int(m.group(4)), addr=m.group(5), len=int(m.group(6)),
                   done=bool(re.search(rf'^PROBE {re.escape(m.group(1))} done$', body[m.end():], re.M)))
              for m in PROBE.finditer(body)]
    checks = dict(re.findall(r'^CHECK (\S+) (ok|FAILED)$', body, re.M))
    info = [l for l in body.splitlines() if re.match(r'(LA|SLOT|SIBLING|BLOCK|FREED|LINK|PARENT|AGAIN|TRY|SUCCESSOR|repro) ', l)]
    fault = None
    m = FAULT.search(body)
    if m:
        fault = dict(cause=int(m.group(1)), pc=m.group(2), address=m.group(3), code_start=m.group(4))
        fault.update(symbolize(a.llvm_bin, image, int(m.group(2), 16), int(m.group(4), 16)))
    returned = f'{tag} RETURNED' in body
    exit_code = int(end.group(1)) if end else None
    if not body or f'{tag} BEGIN' not in body:
        outcome = 'no-reading'
    elif fault:
        outcome = 'fault'
    elif returned and exit_code == 0 and all(p['done'] for p in probes):
        outcome = 'completed'
    else:
        outcome = 'neither'
    checks_ok = bool(checks) and all(v == 'ok' for v in checks.values())
    if predicted == 'completed':
        held = outcome == 'completed' and checks_ok
        if tag.endswith('-stale'):
            held = held and 'stale-read-returns-C-bytes' in checks and \
                any(p['label'] == 'stale-A-read' and p['done'] for p in probes)
    else:
        last = probes[-1] if probes else None
        held = (outcome == 'fault' and fault['cause'] == 25 and last is not None
                and last['label'] == 'stale-A-read' and not last['done']
                and fault['file'] == 'control.c' and fault['line'] == last['line']
                and fault['address'] == last['addr']
                and checks_ok and 'stale-read-returns-C-bytes' not in checks)
    return dict(arm=arm, image=tag, image_sha256=sha256(image), predicted=predicted, outcome=outcome,
                prediction_held=held, exit_code=exit_code, returned=returned, fault=fault,
                probes=probes, checks=checks, lines=info, wall_s=wall, staged_rc=staged_rc,
                completed_boot='VIRTUAL_STAGED_DONE' in log)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--builds', type=Path, required=True)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--llvm-bin', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--locks', type=Path)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--only', help='comma-separated image names')
    p.add_argument('--rejudge', action='store_true',
                   help='Judge the boot logs already in --work again, without booting')
    p.add_argument('--swap-arms', action='store_true',
                   help="With --rejudge: judge each log against the OTHER arm's prediction. The "
                        "judge's negative test: the stale pair must then come out not held")
    a = p.parse_args()
    sdk_builtins.gate([a.builds / arm / "stage" for arm in ("release", "protected") if (a.builds / arm).is_dir()], who="sqlite:run-controls.py")
    if a.rejudge:
        runs = []
        for arm in ('release', 'protected'):
            for tag in PROGRAMS:
                log = (a.work / 'boot' / f'{arm}-{tag}' / 'body.txt').read_text()
                other = {'release': 'protected', 'protected': 'release'}[arm] if a.swap_arms else arm
                r = judge(a, other, tag, a.builds / arm / f'{tag}.dom', None, None, log)
                runs.append(r)
                print(f"log {arm:<9} judged as {other:<9} {tag:<27} {r['outcome']:<10} held={r['prediction_held']}")
        return 0
    a.work.mkdir(parents=True, exist_ok=False)
    a.locks = a.locks or a.work / 'locks'
    a.locks.mkdir(parents=True, exist_ok=True)
    tags = [t for t in PROGRAMS if not a.only or t in a.only.split(',')]
    jobs = [(arm, tag) for arm in ('release', 'protected') for tag in tags]
    for arm, tag in jobs:
        if not (a.builds / arm / f'{tag}.dom').is_file():
            raise SystemExit(f'no image {arm}/{tag}.dom: refusing to report a partial set')
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(a.jobs) as pool:
        raw = list(pool.map(lambda j: run_one(a, *j), jobs))
    runs = []
    for arm, tag, image, rc, wall, log in raw:
        (a.work / 'boot' / f'{arm}-{tag}' / 'body.txt').write_text(log)
        r = judge(a, arm, tag, image, rc, wall, log)
        runs.append(r)
        f = r['fault']
        print(f"{arm:<9} {tag:<27} {r['outcome']:<10} held={r['prediction_held']} wall={wall}s "
              + (f"cause={f['cause']} pc={f['pc']} {f['function']} {f['file']}:{f['line']}" if f else ''))
    (a.work / 'runs.json').write_text(json.dumps(dict(total_wall_s=round(time.monotonic() - t0, 1),
                                                      runs=runs), indent=1) + '\n')
    return 0 if all(r['prediction_held'] for r in runs) else 1


if __name__ == '__main__':
    raise SystemExit(main())
