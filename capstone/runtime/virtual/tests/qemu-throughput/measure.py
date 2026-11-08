#!/usr/bin/env python3
"""Matched host throughput controls; unchanged GCC test input, no TCG plugin."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['stock', 'fork', 'images', 'firmware', 'plain', 'native', 'input', 'reference', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    arms = ['stock-plain', 'fork-plain', 'platform-plain', 'platform-native']
    p.add_argument('--arm', choices=arms, action='append', help='Explicit order; default forward then reverse')
    p.add_argument('--iterations', type=int, default=50)
    p.add_argument('--icount', type=int, choices=range(11))
    p.add_argument('--host-cpu', type=int, help='Pin entire emulator process tree to this allowed host CPU')
    p.add_argument('--host-perf', type=Path, help='Diagnostic sampling; results are excluded from clean timing comparisons')
    p.add_argument('--pcrel', choices=['true', 'false'], help='Explicit scalar-only PC-relative translation control')
    a = p.parse_args()
    if a.pcrel is not None and (not a.arm or 'platform-native' in a.arm):
        p.error('--pcrel requires explicit scalar arms; native profile stays unchanged')
    if a.iterations < 1:
        p.error('iterations must be positive')
    if a.host_cpu is not None:
        if a.host_cpu not in os.sched_getaffinity(0):
            p.error('host CPU is not in the allowed affinity mask')
        os.sched_setaffinity(0, {a.host_cpu})
    a.out.mkdir(parents=True, exist_ok=False)
    sha = lambda f: hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()
    pins = {k: sha(getattr(a, k)) for k in ['stock','fork','firmware','plain','native','input','reference']}
    pins.update({k:sha(a.images/k) for k in ['Image','rootfs.ext2']})
    report = dict(status='RUNNING', inputs=pins, iterations=a.iterations, icount=a.icount,
                  profiled=bool(a.host_perf), pcrel_override=a.pcrel,
                  host=dict(uname=list(platform.uname()), affinity=sorted(os.sched_getaffinity(0))),
                  samples=[], limitations=[
                      'Host serial-delivery timing, not target cycles or hardware overhead.',
                      'One untimed warm-up compilation per fresh guest, then a batch including exec/exit.',
                      'Only stock-plain versus fork-plain uses identical machine/CPU options and binary.',
                      'Platform arms intentionally change discovery/CPU flags and Linux capability paths.',
                      'Native versus plain binaries differ in compiler/libc/ABI; not isolated ISA cost.',
                      'No profiling plugin; build flags and host contention still require auditing.'])
    def save():
        (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    runner = Path(__file__).resolve().parents[2]/'run-staged.py'
    common_cpu = 'rv64,sstc=false,h=false,sv48=false,sv57=false'
    cap_cpu = common_cpu + ',x-capstone-u-mode=true,x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true'
    cap_flags = ' '.join('capstone_vm.'+x for x in ['native_task=1','native_io=1','native_locks=1','native_stat=1','native_seek=1','native_fd=1','native_close=1','native_path=1','native_clock=1','dom_binfmt=1','native_dispatch=1','checked_copy=1','native_slot_transport=1','unpinned=1','supervisor_slots=32'])
    try:
        for i, arm in enumerate(a.arm or arms + arms[::-1]):
            d = a.out/f'{i:02d}-{arm}';stage=d/'stage';stage.mkdir(parents=True)
            for src,name in [(a.native if arm=='platform-native' else a.plain,'gcc.bin'),(a.input,'t1.c'),(a.reference,'reference.s')]:
                shutil.copy2(src,stage/name)
            script = ['#!/bin/sh','set -eu',"trap 'echo THROUGHPUT_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT",'cp -a /mnt/vm /tmp/throughput','cd /tmp/throughput','chmod +x gcc.bin','echo 0 > /proc/sys/kernel/randomize_va_space','killall syslogd klogd 2>/dev/null || true','./gcc.bin t1.c -O3 -finline-limit=50000 -o output.s 2>warmup.err','cmp output.s reference.s','echo HOST_WINDOW_BEGIN']
            # Distinct outputs let us validate every timed compilation outside the window.
            script += [f'./gcc.bin t1.c -O3 -finline-limit=50000 -o output-{j}.s 2>compile-{j}.err' for j in range(a.iterations)]
            script += ['echo HOST_WINDOW_END']
            script += [f'cmp output-{j}.s reference.s' for j in range(a.iterations)]
            script += ['echo PASS:ALL_EXACT_OUTPUTS']
            (stage/'gate.sh').write_text('\n'.join(script)+'\n')
            cap = arm.startswith('platform-')
            flags = cap_flags if cap else ''
            assert len(('root=/dev/vda ro '+flags).encode()) < 512
            cpu = cap_cpu if cap else common_cpu
            if a.pcrel is not None:
                cpu = re.sub(r',x-pcrel=(?:true|false)', '', cpu) + ',x-pcrel=' + a.pcrel
            cmd = ['python3',str(runner),'--qemu',str(a.stock if arm=='stock-plain' else a.fork),'--images',str(a.images),'--firmware',str(a.firmware),'--stage',str(stage),'--work',str(d/'vm'),'--memory','4G','--disk-mib','128','--timeout','180','--machine','virt-capstone' if cap else 'virt','--cpu',cpu,'--host-window']
            if flags:cmd+=['--kernel-arg',flags]
            if a.host_perf:cmd+=['--host-perf',str(a.host_perf)]
            if a.icount is not None:cmd+=['--icount',str(a.icount)]
            (d/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            env=dict(os.environ);env.pop('CAPSTONE_QEMU_CPU_EXTRA',None);env.pop('CAPSTONE_QEMU_DOMAIN_LOCK',None)
            print('START',i,arm,flush=True)
            load=list(os.getloadavg())
            with (d/'runner.log').open('w') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            log=(d/'vm/serial.log').read_text(errors='replace').replace('\r','')
            assert 'THROUGHPUT_EXIT:0\n' in log and 'PASS:ALL_EXACT_OUTPUTS\n' in log
            assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic',log)
            timing=json.loads((d/'vm/host-window.json').read_text())
            row=dict(arm=arm,run=i,host_loadavg_before=load,reference_match=True,**timing)
            report['samples'].append(row);save();print('PASS',json.dumps(row),flush=True)
        report['medians']={arm:{k:statistics.median(x[k] for x in report['samples'] if x['arm']==arm) for k in ['wall_seconds','cpu_seconds']} for arm in sorted({x['arm'] for x in report['samples']})}
        report['status']='PASS';save()
    except BaseException as e:
        report['status']='FAIL';report['error']=repr(e);save();raise


if __name__ == '__main__':
    main()
