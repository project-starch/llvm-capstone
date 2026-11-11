#!/usr/bin/env python3
"""Qualify mallocng's public behavior, tag preservation and object lifetimes.

The former slot-registry/recycled-mapping implementation tests are retired with
that implementation. These tests exercise the allocator's externally visible
contract and retain the same-address reuse and service-boundary controls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import memory_mode
import native_platform
HERE=Path(__file__).resolve().parent

def image_abi(path):
    data = path.read_bytes()
    if data.count(b'CPONVVM7') != 1:
        raise ValueError(f'{path}: missing or ambiguous virtual image ABI magic')
    return 7

def fault_offset(application, config):
    bindir = Path(config['cc']).parent
    symbols = subprocess.check_output([str(bindir/'llvm-nm'), str(application)], text=True)
    symbol = re.search(r'^([0-9a-fA-F]+) [A-Za-z] cap_malloc_fault_store$', symbols, re.M)
    headers = subprocess.check_output([str(bindir/'llvm-readelf'), '-l', str(application)], text=True)
    loads = [int(line.split()[2], 16) for line in headers.splitlines() if line.strip().startswith('LOAD ')]
    if not symbol or not loads:
        raise ValueError('cannot identify the denial instruction and image load base')
    return int(symbol[1], 16) - min(loads)

def grade(log, cases, lifetimes, store_offset=None):
    results={}
    for name in cases:
        rows=re.findall(r'MALLOC_CASE:'+re.escape(name)+r'\n(.*?)MALLOC_EXIT:'+re.escape(name)+r':(\d+)\n',log,re.S)
        if len(rows)!=1: results[name]=False; continue
        body,rc=rows[0]; rc=int(rc)
        if name in ('ok','population','churn','hot','hot-service-control','nested-ok','native'):
            marker={'ok':'MALLOC_TAG_COPY_OK','population':'MALLOC_POPULATION_OK','churn':'MALLOC_CHURN_OK','nested-ok':'MALLOC_SURVIVED:nested-ok','native':'MALLOC_TAG_COPY_OK'}.get(name,'MALLOC_HOT_END')
            results[name]=rc==0 and marker in body and 'MALLOC_FAIL' not in body and 'domain fault' not in body
            continue
        expected=28 if name=='bounds' else 2 if name=='interior-free' else 25
        if not lifetimes:
            if name in ('stale','inplace-stale','reuse-stale','reuse-free'):
                results[name]=rc==0 and 'MALLOC_SURVIVED:'+name in body and 'domain fault' not in body
                continue
            # mallocng invalidates the freed slot header; its second metadata
            # lookup dereferences an untagged pointer (not a lifetime denial).
            if name=='stale-free': expected=24
        # The intended action must be reached; an allocation/setup fault is not a catch.
        ready='MALLOC_FAULT_READY:'+name+'\n'
        after=body.split(ready,1)[1] if ready in body else ''
        causes=[int(n) for n in re.findall(r'domain fault cause=(\d+)',after)]
        results[name]=rc==139 and causes==[expected] and 'MALLOC_FAIL' not in body
        if store_offset is not None and name in ('stale','inplace-stale','remap-stale','bounds','reuse-stale','nested-stale','nested-grandchild'):
            sites = re.findall(r'domain fault cause=\d+ pc=0x([0-9a-f]+).*?code=0x([0-9a-f]+)-', after)
            results[name] = results[name] and len(sites)==1 and int(sites[0][0],16)-int(sites[0][1],16)==store_offset
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('adapter','application','native','qemu','images','work'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--skip-churn',action='store_true')
    p.add_argument('--malloc-count',type=int,default=70000)
    p.add_argument('--exact-bounds',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--unpinned',action='store_true')
    a=p.parse_args()
    if a.malloc_count<1:p.error('--malloc-count must be positive')
    config=json.loads((a.sdk/'sdk.json').read_text())
    lifetimes=config['heap_lifetimes']
    abi=image_abi(a.application)
    offset=fault_offset(a.application,config)
    if not config['heap_object_bounds']:
        p.error('this denial suite requires object bounds; use public/nested controls for the mapping-authority arm')
    cases=['native','ok','population','hot','hot-service-control']+([] if a.skip_churn else ['churn'])
    cases+=['stale','stale-free','inplace-stale','remap-stale','bounds','interior-free','reuse-stale','reuse-free','nested-ok','nested-stale','nested-grandchild','nested-double']
    a.work.mkdir(parents=True,exist_ok=False); stage=a.work/'stage';stage.mkdir()
    inputs={native_platform.LAUNCHER:a.adapter/native_platform.LAUNCHER,'malloc.dom':a.application,'native':a.native}
    for name,path in inputs.items():shutil.copy2(path,stage/name)
    gate=['#!/bin/sh','cd /mnt/vm || exit 1','dmesg -n 1',native_platform.admit(),memory_mode.echo('MALLOC_'),f'export CAPSTONE_MALLOC_COUNT={a.malloc_count}']
    for name in cases:
        gate+=['echo MALLOC_CASE:'+name, './native ok' if name=='native' else ('CAPSTONE_VM_TRACE=1 ' if name in ('hot','hot-service-control') else '')+'./capstone-vexec malloc.dom '+name+(' may-survive' if not lifetimes else ''),'echo MALLOC_EXIT:'+name+':$?']
    gate+=[native_platform.leave('MALLOC_'),'echo VIRTUAL_STAGED_DONE']
    (stage/'gate.sh').write_text('\n'.join(gate)+'\n')
    run=subprocess.run([sys.executable,str(HERE/'run-staged.py'),'--qemu',str(a.qemu.resolve()),'--images',str(a.images.resolve()),'--stage',str(stage.resolve()),'--work',str((a.work/'guest').resolve()),'--timeout',str(a.timeout),'--exact-bounds' if a.exact_bounds else '--no-exact-bounds'])
    log=(a.work/'guest/serial.log').read_text(errors='replace').replace('\r','')
    checks=grade(log,cases,lifetimes,offset)
    checks['cleanup']=run.returncode==0 and 'MALLOC_CLEANUP:0\n' in log
    checks['memory_mode']=memory_mode.verified(log,a.unpinned,1)
    checks['no_kernel_fault']='Kernel panic' not in log and 'Oops' not in log
    checks['population_count']=f'MALLOC_COUNT_USED {a.malloc_count}\n' in log
    intervals=re.findall(r'MALLOC_HOT_BEGIN\n(.*?)MALLOC_HOT_END\n',log,re.S)
    kinds=[re.findall(r'CAPSTONE_VM_BOUNDARY kind=(\d+)',x) for x in intervals]
    # These two checks read the launcher's boundary trace (CAPSTONE_VM_BOUNDARY). A native task
    # has no launcher loop and prints no such trace. The hot cases still run and are graded by
    # their own markers, and the record names the two observations as not available.
    not_observable=[]
    if 'CAPSTONE_VM_BOUNDARY' in log:
        checks['local_fast_path']=len(kinds)==2 and kinds[0]==['10']
        checks['service_control']=len(kinds)==2 and '7' in kinds[1]
    else:
        not_observable=['local_fast_path','service_control']
    result={'status':'PASS' if all(checks.values()) else 'FAIL','tests':checks,'checks':checks,'pass':all(checks.values()),'not_observable':not_observable,'sdk':config,'virtual_vm_abi':abi,'fault_store_offset':offset,'exact_bounds':a.exact_bounds,'inputs':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in inputs.items()}}
    (a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,ok in checks.items():print(('PASS ' if ok else 'FAIL ')+name)
    for name in not_observable:print('NOT OBSERVABLE '+name)
    return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
