#!/usr/bin/env python3
"""run-c8.py --application X.dom --work DIR [kit options]: run the c8-copy test (this folder) in one virtual context on the
kit's QEMU (runtime/virtual run-staged.py) and judge it. PASS needs every expected C8_CHECK line, all PASS, the
control fired, the C8_DONE line with zero failures, and exit 0. A missing line is a FAIL, never a pass."""
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[2] / 'runtime/virtual',
               help='runtime/virtual checkout with run-staged.py (default: this repo\'s)')
for n in ('adapter', 'application', 'qemu', 'images', 'work'):
    p.add_argument('--' + n, type=Path, required=True)
a = p.parse_args()
R = a.runtime
a.work.mkdir(parents=True, exist_ok=False)
st = a.work / 'stage'; st.mkdir()
shutil.copy2(a.adapter / 'capstone-vexec', st / 'capstone-vexec')
shutil.copy2(a.adapter / 'module/capstone_vm.ko', st / 'capstone_vm.ko')
shutil.copy2(a.application, st / 'c8.dom')
(st / 'gate.sh').write_text('''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
./capstone-vexec c8.dom
echo C8_EXIT:$?
rmmod capstone_vm
echo VIRTUAL_STAGED_DONE
''')
subprocess.run([sys.executable, str(R / 'run-staged.py'), '--qemu', str(a.qemu.resolve()), '--images',
                str(a.images.resolve()), '--stage', str(st.resolve()), '--work', str((a.work / 'guest').resolve()),
                '--timeout', '600', '--no-exact-bounds'])
serial = a.work / 'guest/serial.log'
log = serial.read_text(errors='replace').replace('\r', '') if serial.exists() else ''
got = dict(re.findall(r'^C8_CHECK:(\w+):(PASS|FAIL)$', log, re.M))
want = [f'{c}_{n}_{k}' for c in ('both_aligned', 'dst_misaligned', 'src_misaligned', 'both_8_mod_16')
        for n in (16, 32) for k in ('matches_memcpy', 'tags')] + ['control_word_copy_drops_tag']
tests = {w: got.get(w) == 'PASS' for w in want}
tests['no_unexpected_checks'] = set(got) <= set(want)
tests['done_zero_failures'] = re.search(rf'^C8_DONE:{len(want)}:0$', log, re.M) is not None
tests['exit_zero'] = 'C8_EXIT:0' in log
res = dict(status='PASS' if all(tests.values()) else 'FAIL', passed=sum(tests.values()), total=len(tests), tests=tests)
(a.work / 'result.json').write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps({k: res[k] for k in ('status', 'passed', 'total')}))
sys.exit(0 if res['status'] == 'PASS' else 1)
