#!/usr/bin/env python3
"""One repetition of the 2026-10-11 memory-behaviour measurement: one program, one arm, one boot.

    mm-run.py --program memcached|mruby --arm stock|protected --rep N --work NEW_DIR \
        --image IMAGE --qemu QEMU --images IMAGES --adapter ADAPTER --job CAPSTONE_JOB \
        [--harness MC_HARNESS --stats-client MC_STATS] [--ao AO_RB --smoke SMOKE_RB --driver MSTATS_RB]

Stages the image, the launcher and module, and the program's driver into a fresh directory, writes
its gate.sh, boots it once through runtime/virtual/run-staged.py (CAPSTONE_QEMU_LOCK must name this
VM's own lock), and parses the console into NEW_DIR/rep.json: every `capstone-vexec --stats` reading,
the launcher process's /proc status lines, and the program's own statistics. Files the guest sends
back (base64 between markers) land in NEW_DIR/files/. The console log stays in NEW_DIR/guest/ and is
not part of the results.

memcached: the server runs under capstone-job as uid 65534 with the gate's flags (-U 0 -m 64 -t 4);
CAPSTONE_VM_STATS=1 makes the launcher print its own per-process counters at exit; the unchanged mc-harness drives the protocol oracle's script against it (its own SERVER-COMMAND is a
`sleep` it later stops, so the real server stays up), then the process is read, `stats`, `stats slabs`
and `stats items` are taken by mc-stats, and the server is stopped with TERM.
mruby: mstats.rb runs the AO render benchmark, then (a second process) the port's smoke.rb, each
between two --stats readings, its stdout kept as the oracle.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PORT = 21299
MC_FLAGS = f'-l 127.0.0.1 -p {PORT} -U 0 -m 64 -t 4'

COMMON = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
chmod 666 /dev/capstone-vm
O=/tmp/mm; mkdir -p $O; chmod 777 $O
vstats() { echo "MM_VSTATS:$1 $(./capstone-vexec --stats)"; }
procst() { echo "MM_PROC:$1 $(grep -E '^(State|VmPeak|VmSize|VmHWM|VmRSS|RssAnon|RssFile|VmData)' /proc/$2/status 2>/dev/null | tr -s ' \t' ' ' | tr '\n' ';')"; }
emit() { [ -f "$2" ] || return 0; echo "MM_FILE:$1"; gzip -c "$2" | base64; echo "MM_FILE_END"; }
'''

MEMCACHED = COMMON + r'''
mkdir -p $O/h; chmod 777 $O/h
vstats before
CAPSTONE_VM_STATS=1 ./capstone-job $O/job.json --user 65534:65534 -- ./capstone-vexec ./memcached.dom FLAGS > $O/server.out 2> $O/server.err &
JOB=$!
./mc-harness --out $O/h --port PORT --stop TERM -- sleep 100000
echo "MM_HARNESS:$?"
L=""
for s in /proc/[0-9]*/status; do
  grep -q "^PPid:[[:space:]]*$JOB\$" $s 2>/dev/null && { p=${s#/proc/}; L="$L ${p%/status}"; }
done
echo "MM_LAUNCHER:$JOB:$L"
for p in $L; do procst "after_workload:$p" $p; done
vstats during
./mc-stats PORT > $O/stats.txt; echo "MM_MCSTATS:$?"
for p in $L; do procst "after_stats:$p" $p; done
vstats during2
kill -TERM $JOB; wait $JOB; echo "MM_JOB:$?"
vstats after
for f in job.json server.out server.err stats.txt; do emit $f $O/$f; done
for f in transcript.norm status.txt identity.txt server.out server.err; do emit h-$f $O/h/$f; done
rmmod capstone_vm; echo "MM_CLEANUP:$?"
echo VIRTUAL_STAGED_DONE
'''.replace('FLAGS', MC_FLAGS).replace('PORT', str(PORT))

MRUBY = COMMON + r'''
run_one() {   # label, mstats.rb arguments
  label=$1; shift
  vstats "$label:before"
  CAPSTONE_VM_STATS=1 ./capstone-vexec ./mruby ./mstats.rb "$@" > $O/$label.out 2> $O/$label.err &
  pid=$!
  last=""; done_read=""; samples=0
  while :; do
    s=$(grep -E '^(State|VmPeak|VmSize|VmHWM|VmRSS|RssAnon|RssFile|VmData)' /proc/$pid/status 2>/dev/null | tr -s ' \t' ' ' | tr '\n' ';')
    case "$s" in ""|*"State: Z"*) break ;; esac
    last=$s; samples=$((samples+1))
    if [ -z "$done_read" ] && grep -q MM_DONE $O/$label.err 2>/dev/null; then done_read=$s; fi
    usleep 500000
  done
  wait $pid; rc=$?
  echo "MM_PROC:$label:last $last"
  echo "MM_PROC:$label:done $done_read"
  echo "MM_SAMPLES:$label:$samples"
  echo "MM_RC:$label:$rc"
  vstats "$label:after"
  emit $label.out $O/$label.out
  emit $label.err $O/$label.err
}
mkdir -p $O/gc; cp smoke.rb $O/gc/; chmod -R 777 $O/gc
run_one ao ./bm_ao_render.rb
run_one smoke $O/gc/smoke.rb $O/gc
rmmod capstone_vm; echo "MM_CLEANUP:$?"
echo VIRTUAL_STAGED_DONE
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(log):
    rec = {'vexec_stats': {}, 'proc': {}, 'markers': {}, 'files': {}}
    for tag, js in re.findall(r'^MM_VSTATS:(\S+) (.*)$', log, re.M):
        try:
            rec['vexec_stats'][tag] = json.loads(js)
        except ValueError:
            rec['vexec_stats'][tag] = {'unparsed': js}
    for tag, text in re.findall(r'^MM_PROC:(\S+) ?(.*)$', log, re.M):
        fields = {}
        for item in text.split(';'):
            if ':' in item:
                k, v = item.split(':', 1)
                fields[k.strip()] = v.strip()
        rec['proc'][tag] = fields
    for name, value in re.findall(r'^MM_(HARNESS|MCSTATS|JOB|CLEANUP|LAUNCHER|SAMPLES:\w+|RC:\w+):(.*)$', log, re.M):
        rec['markers'][name] = value.strip()
    for name, data in re.findall(r'^MM_FILE:(\S+)\n(.*?)^MM_FILE_END$', log, re.M | re.S):
        rec['files'][name] = gzip.decompress(base64.b64decode(data))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--program', choices=('memcached', 'mruby'), required=True)
    p.add_argument('--arm', choices=('stock', 'protected'), required=True)
    p.add_argument('--rep', type=int, required=True)
    for name in ('work', 'image', 'qemu', 'images', 'adapter', 'run-staged'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('job', 'harness', 'stats-client', 'ao', 'smoke', 'driver'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--timeout', type=int, default=3600)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    staged = {'capstone-vexec': a.adapter / 'capstone-vexec', 'capstone_vm.ko': a.adapter / 'module/capstone_vm.ko'}
    if a.program == 'memcached':
        staged.update({'capstone-job': a.job, 'memcached.dom': a.image, 'mc-harness': a.harness,
                       'mc-stats': a.stats_client})
        gate = MEMCACHED
    else:
        staged.update({'mruby': a.image, 'mstats.rb': a.driver, 'bm_ao_render.rb': a.ao, 'smoke.rb': a.smoke})
        gate = MRUBY
    for name, src in staged.items():
        shutil.copy2(src, stage / name)
    (stage / 'gate.sh').write_text(gate)
    t0 = time.time()
    run = subprocess.run([sys.executable, str(a.run_staged), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work / 'guest'), '--timeout', str(a.timeout)])
    wall = time.time() - t0
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    rec = parse(log)
    files = a.work / 'files'
    files.mkdir()
    for name, data in rec['files'].items():
        (files / name).write_bytes(data)
    rec['files'] = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                    for name, data in rec['files'].items()}
    rec.update(program=a.program, arm=a.arm, rep=a.rep, completed=run.returncode == 0,
               boot_wall_seconds=round(wall, 1),
               staged_sha256={name: sha(stage / name) for name in sorted(staged)},
               gate_sha256=hashlib.sha256(gate.encode()).hexdigest(),
               platform_sha256={name: sha(a.images / name) for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')}
               | {'qemu-system-riscv64': sha(a.qemu.resolve())})
    (a.work / 'rep.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    print(f'{a.program} {a.arm} rep{a.rep}: completed={rec["completed"]} {wall:.0f}s')
    return 0 if rec['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
