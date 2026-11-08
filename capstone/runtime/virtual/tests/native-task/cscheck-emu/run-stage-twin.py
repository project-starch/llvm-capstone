#!/usr/bin/env python3
"""MEASUREMENT ONLY. run-stage-twin.py STAGE OUT --arm exchange|direct ...: the QEMU twin of a native board stage.

STAGE is a make-native-stage.py stage (plan.json + stage/); its own run.sh runs unchanged, as the board runs it. The
twin is capstone-qemu virtual/flat-index-20 with the rtl lane's detach/adopt and x-capstone-cscheck-frame, the C.5a
device tree, the C.5a-route adapter (c5a_model, flat 20-bit table, slot transport), one slot, icount 0. Outputs come
back in-band (@@SPEC-FILE, gzip+base64) into OUT/results; the plan's host_validation runs separately
(validate-plan.py). The twin's ADOPT-KIND-MISMATCH / ADOPT-UNPARKED lines must be absent, and the adapter's
adopt_kind_refused must read 0."""
import argparse, base64, gzip, json, os, re, shutil, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('stage', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--arm', choices=['exchange', 'direct'], required=True)
for name in ['qemu', 'images', 'firmware', 'dtb', 'runner']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--memory', default='936M', help="the board's guest RAM")
p.add_argument('--records', type=int, required=True, help='node_flat_records: the flat table size for this stage')
p.add_argument('--timeout', type=int, default=43200)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
st = a.out / 'stage'
shutil.copytree(a.stage / 'stage', st, symlinks=False)
# Memory high-water, sampled every second while run.sh runs: the largest MemTotal - MemAvailable (kB) and the
# largest VmHWM (kB) of any *_base* process (comm is cut to 15 characters).
SAMPLER = r'''( mx=0; hw=0; while [ ! -f /tmp/sampler.stop ]; do
  u=$(awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} END{print t-a}' /proc/meminfo); [ "$u" -gt "$mx" ] && mx=$u
  for f in /proc/[0-9]*/status; do case "$(head -1 $f 2>/dev/null)" in *_base*) h=$(awk '/^VmHWM/{print $2}' $f 2>/dev/null); [ -n "$h" ] && [ "$h" -gt "$hw" ] && hw=$h;; esac; done
  echo "$mx $hw" > /tmp/sampler.out; sleep 1; done ) &
'''
(st / 'gate.sh').write_text('#!/bin/sh\nexport SPEC_INBAND_MAX=67108864\nrm -f /tmp/sampler.stop\n' + SAMPLER + 'sh /mnt/vm/run.sh\n'
                            'touch /tmp/sampler.stop; sleep 2; echo "@@MEM-PEAK used_kB vmhwm_kB $(cat /tmp/sampler.out)"\n'
                            'echo "@@COUNTERS-FINAL $(cat /sys/module/capstone_vm/parameters/native_counters)"\n'
                            'echo VIRTUAL_STAGED_DONE\n')
flags = ('capstone_vm.native_task=1 capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 '
         'capstone_vm.c5a_model=1 capstone_vm.checked_copy_emu=1 capstone_vm.native_slot_transport=1 '
         f'cma=256M capstone_vm.node_flat_records={a.records}')
if a.arm == 'direct':
    flags += ' capstone_vm.native_io=1' + ''.join(f' capstone_vm.native_{k}=1' for k in
                                                  ['locks', 'stat', 'seek', 'fd', 'close', 'path', 'clock'])
assert len(('root=/dev/vda ro ' + flags).encode()) < 512
cpu = 'x-capstone-supervisor-slots=1,svadu=false,x-capstone-cscheck-frame=true'
cmd = ['python3', str(a.runner), '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
       '--firmware', str(a.firmware.resolve()), '--stage', str(st.resolve()), '--work', str((a.out / 'vm').resolve()),
       '--memory', a.memory, '--disk-mib', '1024', '--timeout', str(a.timeout), '--icount', '0',
       '--kernel-arg', flags, '--dtb', str(a.dtb.resolve())]
(a.out / 'run.json').write_text(json.dumps(dict(arm=a.arm, flags=flags, cpu=cpu, command=cmd), indent=1) + '\n')
with (a.out / 'runner.log').open('w') as log:
    rc = subprocess.run(cmd, env=dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=cpu), stdout=log, stderr=subprocess.STDOUT).returncode
text = (a.out / 'vm/serial.log').read_text(errors='replace').replace('\r', '')
res = a.out / 'results'
res.mkdir()
for name, body in re.findall(r'^@@SPEC-FILE \S+ (\S+)\n(.*?)^@@SPEC-FILE-END$', text, re.M | re.S):
    (res / name).write_bytes(gzip.decompress(base64.b64decode(''.join(body.split()))))
ends = re.findall(r'^@@SPEC-END (\d+) .*? rc=(\d+)', text, re.M)
bad = [l for l in text.splitlines() if 'ADOPT-KIND-MISMATCH' in l or 'ADOPT-UNPARKED' in l]
refused = re.findall(r'adopt_kind_refused=(\d+)', text)
summary = dict(runner_rc=rc, commands=[(int(i), int(r)) for i, r in ends], files=sorted(x.name for x in res.iterdir()),
               twin_adopt_lines=len(bad), adopt_kind_refused=refused[-1] if refused else None,
               done='@@SPEC-DONE' in text,
               mem_peak=re.findall(r'^@@MEM-PEAK used_kB vmhwm_kB (\d+) (\d+)', text, re.M),
               oom=bool(re.search(r'Out of memory|oom-kill|Killed process', text)))
(a.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary))
sys.exit(0 if rc == 0 and summary['done'] and not bad and refused and refused[-1] == '0' and
         all(r == 0 for _, r in summary['commands']) else 1)
