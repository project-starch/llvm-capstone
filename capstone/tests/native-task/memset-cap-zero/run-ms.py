#!/usr/bin/env python3
"""run-ms.py --application X.dom --work DIR [kit options]: run the memset-cap-zero test ms-zero (this folder) in one virtual context on the
kit's QEMU (runtime/virtual run-staged.py) and judge it. PASS needs every expected MS_CHECK line, all PASS, the
control fired, the MS_DONE line with zero failures, and exit 0. A missing line is a FAIL, never a pass."""
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[3] / 'runtime/virtual',
               help='runtime/virtual checkout with run-staged.py (default: this repo\'s)')
for n in ('adapter', 'application', 'qemu', 'images', 'work'):
    p.add_argument('--' + n, type=Path, required=True)
a = p.parse_args()
R = a.runtime
a.work.mkdir(parents=True, exist_ok=False)
st = a.work / 'stage'; st.mkdir()
shutil.copy2(a.adapter / 'capstone-vexec', st / 'capstone-vexec')
shutil.copy2(a.adapter / 'module/capstone_vm.ko', st / 'capstone_vm.ko')
shutil.copy2(a.application, st / 'ms.dom')
(st / 'gate.sh').write_text('''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
./capstone-vexec ms.dom
echo MS_EXIT:$?
rmmod capstone_vm
echo VIRTUAL_STAGED_DONE
''')
subprocess.run([sys.executable, str(R / 'run-staged.py'), '--qemu', str(a.qemu.resolve()), '--images',
                str(a.images.resolve()), '--stage', str(st.resolve()), '--work', str((a.work / 'guest').resolve()),
                '--timeout', '600', '--no-exact-bounds'])
serial = a.work / 'guest/serial.log'
log = serial.read_text(errors='replace').replace('\r', '') if serial.exists() else ''
got = dict(re.findall(r'^MS_CHECK:(\w+):(PASS|FAIL)$', log, re.M))
want = ['control_no_memset_tags_set', 'fill_after_fence', 'fill_after_eviction', 'all_cases_ran']
tests = {w: got.get(w) == 'PASS' for w in want}
tests['no_unexpected_checks'] = set(got) <= set(want)
tests['done_zero_failures'] = re.search(rf'^MS_DONE:{len(want)}:0:832$', log, re.M) is not None
tests['exit_zero'] = 'MS_EXIT:0' in log
res = dict(status='PASS' if all(tests.values()) else 'FAIL', passed=sum(tests.values()), total=len(tests), tests=tests)
(a.work / 'result.json').write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps({k: res[k] for k in ('status', 'passed', 'total')}))
sys.exit(0 if res['status'] == 'PASS' else 1)
