#!/usr/bin/env python3
"""One boot of the memcached reuse-distance row: one arm, REPS servers in turn.

    run-reuse.py --arm stock|protected --image MEMCACHED_DOM --work NEW_DIR --qemu QEMU --images IMAGES \
        --adapter ADAPTER --job CAPSTONE_JOB --harness MC_HARNESS --run-staged RUN_STAGED [--reps 3]

Each repetition is the protocol oracle unchanged: host/mc-harness/mc-harness.c (one connection
through phases 1 and 2, eight concurrent connections in phase 3, the stats of phase 4) against a
fresh server started as host/run-virtual.py starts it (capstone-job as uid 65534, capstone-vexec,
`-l 127.0.0.1 -p 21299 -U 0 -m 64 -t 4`), stopped with TERM. memcached's SIGTERM handler ends the
event loop and main returns, so the observer's atexit handler prints the REUSE-GAP lines on the
server's stderr. CAPSTONE_QEMU_LOCK must name this VM's own lock. NEW_DIR/result.json holds, per
repetition, the parsed REUSE-GAP lines, the harness and job status and the sha256 of the normalised
transcript; NEW_DIR/rep<N>/ the files the guest sent back. The console log stays in NEW_DIR/guest/.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

FLAGS = '-l 127.0.0.1 -p 21299 -U 0 -m 64 -t 4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate(reps):
    lines = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1', 'insmod capstone_vm.ko || exit 1',
             'chmod 666 /dev/capstone-vm']
    for rep in range(1, reps + 1):
        out = f'/tmp/rg-{rep}'
        lines += [f'echo RG_BEGIN:{rep}', f'mkdir -p {out}; chmod 777 {out}',
                  f'./mc-harness --out {out} --port 21299 --stop TERM -- ./capstone-job {out}/job.json '
                  f'--user 65534:65534 -- ./capstone-vexec ./memcached.dom {FLAGS}',
                  'rc=$?',
                  f'for f in transcript.norm status.txt identity.txt job.json server.out server.err; do '
                  f'[ -f {out}/$f ] || continue; echo RG_FILE:$f; gzip -c {out}/$f | base64; echo RG_FILE_END; done',
                  f'echo RG_END:{rep}:$rc']
    lines += ['rmmod capstone_vm', 'echo RG_CLEANUP:$?', 'echo VIRTUAL_STAGED_DONE']
    return '\n'.join(lines) + '\n'


def parse_reuse(text):
    out = {}
    for name, fields in re.findall(r'^REUSE-GAP (\S+) (.*)$', text, re.M):
        kv = dict(item.split('=', 1) for item in fields.split())
        rec = {k: (kv[k] if k == 'arm' else int(kv[k]))
               for k in ('arm', 'attempts', 'issues', 'releases', 'reuses', 'distinct_starts', 'error', 'capacity')}
        rec['bins'] = [int(kv[f'b{k}']) for k in range(1, 33)]
        out[name] = rec
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('stock', 'protected'), required=True)
    for name in ('image', 'work', 'qemu', 'images', 'adapter', 'job', 'harness', 'run-staged'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--timeout', type=int, default=3600)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    staged = {'capstone-vexec': a.adapter / 'capstone-vexec', 'capstone_vm.ko': a.adapter / 'module/capstone_vm.ko',
              'capstone-job': a.job, 'memcached.dom': a.image, 'mc-harness': a.harness}
    for name, src in staged.items():
        shutil.copy2(src, stage / name)
    script = gate(a.reps)
    (stage / 'gate.sh').write_text(script)
    t0 = time.time()
    run = subprocess.run([sys.executable, str(a.run_staged), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work / 'guest'), '--timeout', str(a.timeout)])
    wall = time.time() - t0
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    reps = []
    for rep, text, rc in re.findall(r'^RG_BEGIN:(\d+)\n(.*?)^RG_END:\1:(\d+)\n', log, re.M | re.S):
        files = {name: gzip.decompress(base64.b64decode(data)) for name, data in
                 re.findall(r'^RG_FILE:(\S+)\n(.*?)^RG_FILE_END\n', text, re.M | re.S)}
        d = a.work / f'rep{rep}'
        d.mkdir()
        for name, data in files.items():
            (d / name).write_bytes(data)
        err = files.get('server.err', b'').decode(errors='replace')
        reps.append(dict(rep=int(rep), harness_status=int(rc),
                         status=files.get('status.txt', b'').decode().strip(),
                         job=json.loads(files.get('job.json', b'{}') or b'{}'),
                         identity=files.get('identity.txt', b'').decode().strip(),
                         transcript_sha256=hashlib.sha256(files['transcript.norm']).hexdigest()
                         if 'transcript.norm' in files else None,
                         other_stderr=[l for l in err.splitlines() if not l.startswith('REUSE-GAP ')],
                         reuse=parse_reuse(err)))
    rec = dict(arm=a.arm, completed=run.returncode == 0 and '\nRG_CLEANUP:0\n' in log, boot_wall_seconds=round(wall, 1),
               repetitions=reps, flags=FLAGS,
               staged_sha256={name: sha(stage / name) for name in sorted(staged)},
               gate_sha256=hashlib.sha256(script.encode()).hexdigest(),
               platform_sha256={name: sha(a.images / name) for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')}
               | {'qemu-system-riscv64': sha(a.qemu.resolve())})
    (a.work / 'result.json').write_text(json.dumps(rec, indent=1) + '\n')
    ok = rec['completed'] and len(reps) == a.reps and all(r['harness_status'] == 0 and r['reuse'] for r in reps)
    print(f'{a.arm}: completed={rec["completed"]} reps={len(reps)} ok={ok} {wall:.0f}s')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
