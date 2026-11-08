#!/usr/bin/env python3
"""Read/write fault-policy diagnostic; whole-invocation attempts, not timing."""
import argparse, hashlib, json, os, re, shutil, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ['qemu','images','firmware','plugin','profrun','probes','out']:
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--access',type=int,choices=[0,1],required=True)
p.add_argument('--fast',type=int,choices=[0,1],default=0)
p.add_argument('--ahead',type=int,default=0)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);stage=a.out/'stage';stage.mkdir();profile=a.out/'profile';profile.mkdir()
inputs={}
for path,name in [(a.probes/'memory-plain','memory-plain'),(a.probes/'memory.dom','memory.dom'),(a.profrun,'profrun')]:
    shutil.copy2(path,stage/name);inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
tag_probe=a.probes/'memory-tag-store.dom'
if not tag_probe.is_file(): raise FileNotFoundError(tag_probe)
shutil.copy2(tag_probe,stage/tag_probe.name);inputs[tag_probe.name]=hashlib.sha256(tag_probe.read_bytes()).hexdigest()
for name,path in [('qemu',a.qemu),('Image',a.images/'Image'),('rootfs',a.images/'rootfs.ext2'),('firmware',a.firmware),('plugin',a.plugin)]:inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
lines=['#!/bin/sh','set -eu',"trap 'echo MEMORY_GATE_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT",'cp -a /mnt/vm /tmp/memory-access','cd /tmp/memory-access','chmod +x memory* profrun','echo 0 > /proc/sys/kernel/randomize_va_space','killall syslogd klogd 2>/dev/null || true']
cases=[]
for arm in ['plain','native']:
    exe='memory-plain' if arm=='plain' else 'memory.dom'
    for mode in ['read','write','read-write','tagged']:
        cases.append(dict(arm=arm,mode=mode));lines += [f'echo CASE:{arm}:{mode}',f'./profrun ./{exe} {mode}']
for mode in ['denied','stale']:
    cases.append(dict(arm='native',mode=mode));lines += [f'echo CASE:native:{mode}','set +e',f'./profrun ./memory.dom {mode}','rc=$?','set -e','echo NEGATIVE_RC:$rc','[ "$rc" -eq 139 ]']
lines += ['./memory-tag-store.dom']
lines += ['for key in unpinned_fast_fault fault_ahead_max pin_ahead_pages; do echo SETTING:$key:$(cat /sys/module/capstone_vm/parameters/$key); done', 'echo FAST_COUNT:$(cat /sys/module/capstone_vm/parameters/unpinned_fast_faults)', 'echo AHEAD_COUNT:$(cat /sys/module/capstone_vm/parameters/fault_ahead_pages)']
lines += ['echo ACCESS_READS:$(cat /sys/module/capstone_vm/parameters/unpinned_access_reads)','echo POLICY:$(cat /sys/module/capstone_vm/parameters/unpinned_access_fault)','echo MEMORY_DONE']
(stage/'gate.sh').write_text('\n'.join(lines)+'\n')
flags='capstone_vm.native_task=1 capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=32 capstone_vm.native_io=1'
flags+=f' capstone_vm.unpinned_access_fault={a.access} capstone_vm.unpinned_fast_fault={a.fast} capstone_vm.fault_ahead_max={a.ahead} capstone_vm.pin_ahead_pages=0'
assert len(('root=/dev/vda ro '+flags).encode()) < 512, 'kernel command line would be truncated'
cmd=['python3',str(Path(__file__).resolve().parents[2]/'run-staged.py'),'--qemu',str(a.qemu),'--images',str(a.images),'--firmware',str(a.firmware),'--stage',str(stage),'--work',str(a.out/'vm'),'--memory','4G','--disk-mib','128','--timeout','180','--icount','0','--kernel-arg',flags,'--plugin',str(a.plugin),'--plugin-arg','out='+str(profile)]
(a.out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env=dict(os.environ,CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
with (a.out/'runner.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
log=(a.out/'vm/serial.log').read_text(errors='replace').replace('\r','');assert 'MEMORY_GATE_EXIT:0\n' in log and 'MEMORY_DONE\n' in log
assert 'TAG_FIRST_PASS\n' in log
assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic',log)
assert not list(profile.glob('*.invalid'))
assert len(list(profile.glob('window-*.tsv')))==len(cases)
for i,c in enumerate(cases):
    marker='CASE:'+c['arm']+':'+c['mode']+'\n';body=log.split(marker,1)[1].split('CASE:',1)[0]
    if c['mode'] in ['denied','stale']:assert 'ARMED_'+c['mode'].upper()+'\n' in body and 'NEGATIVE_RC:139\n' in body
    else:assert 'MEMORY_ACCESS_PASS:'+c['mode']+'\n' in body
    usage=re.search(r'PROFRUN_RUSAGE minflt=(\d+) majflt=(\d+) maxrss_kb=(\d+)',body)
    assert usage, 'missing child rusage'
    c['rusage']=dict(zip(['minflt','majflt','maxrss_kb'],map(int,usage.groups())))
    header=(profile/f'window-{i}.tsv').read_text();c['total_attempts']=int(re.search(r'^# total (\d+)',header,re.M)[1])
reads=int(re.search(r'ACCESS_READS:(\d+)',log)[1]);assert (reads>0)==bool(a.access)
assert 'POLICY:'+('Y' if a.access else 'N')+'\n' in log
settings=dict(re.findall(r'SETTING:(\w+):(\w+)',log))
assert settings==dict(unpinned_fast_fault='Y' if a.fast else 'N',fault_ahead_max=str(a.ahead),pin_ahead_pages='0')
fast_count=int(re.search(r'FAST_COUNT:(\d+)',log)[1]); ahead_count=int(re.search(r'AHEAD_COUNT:(\d+)',log)[1])
assert (fast_count>0)==bool(a.fast) and (ahead_count>0)==bool(a.ahead)
result=dict(status='PASS',settings=settings,fast_count=fast_count,ahead_count=ahead_count,access=a.access,fast=a.fast,ahead=a.ahead,inputs=inputs,cases=cases,tag_first_pass=True,eligible_read_attempts=reads,limitations=['Whole invocation includes exec/libc startup and exit; not isolated handler cost. RSS is whole-process ru_maxrss.', 'Denial/stale smoke checks require an armed marker and SIGSEGV, not a verified fault PC/cause.','Plain and Capstone compilers/libc differ; compare policy off/on within each arm.','QEMU instruction attempts, not time or physical-page counts.'])
(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
