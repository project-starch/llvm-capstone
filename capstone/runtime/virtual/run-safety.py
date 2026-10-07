#!/usr/bin/env python3
"""Run virtual application fixtures against unchanged physical-port predictions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PORTS = HERE.parents[1]/'ports'
spec = importlib.util.spec_from_file_location('judge', PORTS/'common/application/check-safety.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', choices=('ffmpeg', 'wireshark'), required=True)
    p.add_argument('--arm', required=True)
    for name in ('qemu', 'platform-images', 'adapter', 'images', 'work'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--fixtures', nargs='+', type=int,
                   help='Explicit configured fixture subset; default is every prediction for the arm')
    a = p.parse_args()
    expect = PORTS/a.port/'app/host/safety-expect.txt'
    predictions = {}
    for line in expect.read_text().splitlines():
        words = line.split('#')[0].split()
        if len(words) == 4 and words[0] == a.arm:
            predictions.setdefault(int(words[1]), []).append(words[2:])
    if not predictions:
        p.error('arm has no registered predictions')
    if a.fixtures:
        if any(n not in predictions for n in a.fixtures):
            p.error('requested fixture has no prediction')
        predictions = {n: predictions[n] for n in sorted(set(a.fixtures))}
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work/'stage'; stage.mkdir()
    shutil.copy2(a.adapter/'capstone-vexec', stage/'capstone-vexec')
    shutil.copy2(a.adapter/'module/capstone_vm.ko', stage/'capstone_vm.ko')
    prefix = 'ffapp' if a.port == 'ffmpeg' else 'tsapp'
    images = {}
    script = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1',
              'insmod capstone_vm.ko || exit 1', 'export CAPSTONE_EXEC_DIAGNOSTICS=1']
    for number in sorted(predictions):
        name = f'{prefix}_fx{number}.dom'
        shutil.copy2(a.images/name, stage/name)
        images[number] = subprocess.check_output(['sha256sum', str(a.images/name)], text=True).split()[0]
        script += [f'echo SAFETY_BEGIN:{number}', f'./capstone-vexec ./{name}', f'echo SAFETY_END:{number}:$?']
    script += ['rmmod capstone_vm', 'echo SAFETY_CLEANUP:$?', 'echo VIRTUAL_STAGED_DONE']
    (stage/'gate.sh').write_text('\n'.join(script)+'\n')
    run = subprocess.run([sys.executable, str(HERE/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.platform_images.resolve()), '--stage', str(stage),
                          '--work', str(a.work/'guest'), '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    parts = {int(n): (text, int(code)) for n, text, code in
             re.findall(r'^SAFETY_BEGIN:(\d+)\n(.*?)^SAFETY_END:\1:(\d+)\n', log, re.M | re.S)}
    results = []
    for n, wants in sorted(predictions.items()):
        passed = False
        detail = 'fixture did not complete'
        try:
            text, code = parts[n]
            faults = re.findall(r'^capstone-exec: domain fault [^\n]+', text, re.M)
            if len(faults) == 1 and code == 139:
                result = dict(kind='signal', value=11, fault=faults[0])
            elif not faults:
                result = dict(kind='exit', value=code)
            else:
                raise ValueError('fault count/status mismatch')
            outcome, info = judge.classify(text, text, result, n)
            got, detail, explanation = outcome
            passed = all(judge.matches(got, detail, info, *want, n) for want in wants)
        except (KeyError, ValueError) as e:
            detail = str(e)
        results.append(dict(fixture=n, passed=passed, detail=detail, image_sha256=images[n]))
    completed = run.returncode == 0 and '\nSAFETY_CLEANUP:0\n' in log
    success = completed and all(r['passed'] for r in results)
    record = dict(status='PASS' if success else 'FAIL', port=a.port, arm=a.arm,
                  completed=completed, results=results,
                  selected_fixtures=sorted(predictions),
                  predictions_sha256=subprocess.check_output(['sha256sum', str(expect)], text=True).split()[0],
                  gate_sha256=hashlib.sha256(('\n'.join(script)+'\n').encode()).hexdigest(),
                  platform_sha256={name: subprocess.check_output(['sha256sum', str(path)], text=True).split()[0]
                                   for name, path in dict(qemu=a.qemu, launcher=a.adapter/'capstone-vexec',
                                                         module=a.adapter/'module/capstone_vm.ko',
                                                         Image=a.platform_images/'Image').items()})
    (a.work/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
