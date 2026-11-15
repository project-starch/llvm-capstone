#!/usr/bin/env python3
"""One boot of the memcached reuse-distance row: one arm, REPS servers in turn.

    run-reuse.py --arm stock|protected --image MEMCACHED_DOM --work NEW_DIR --qemu QEMU --images IMAGES \
        --adapter ADAPTER --job CAPSTONE_JOB --harness MC_HARNESS --run-staged RUN_STAGED [--reps 3] \
        [--flags SERVER_FLAGS] [--load-client MC_CRUSHER --load-conf CONF --load-expect N [--load-poll S] [--load-stall S]]

With --load-client (the established workload, docs/plans/2026-11-15-memcached-established-workload.md), each
server runs the protocol oracle first and then mc-crusher (build-load-client.sh) with CONF against it, natively
in the guest over loopback TCP, through mc-harness's --load phase: the harness waits until the server has
processed N commands (cmd_get + cmd_set), ends the client and then sends `shutdown`. The client runs with
libevent's default backend (EVENT_NOEPOLL is set for the server only). NEW_DIR/rep<N>/load.txt says whether the
load completed; load-stats.txt holds the server's stats, slab and item counters after it.

Each repetition is the protocol oracle unchanged: host/mc-harness/mc-harness.c (one connection
through phases 1 and 2, eight concurrent connections in phase 3, the stats of phase 4) against a
fresh server started as host/run-virtual.py starts it (capstone-job as uid 65534, capstone-vexec,
`-l 127.0.0.1 -p 21299 -U 0 -m 64 -t 4 -A`), stopped with the `shutdown` command (mc-harness
--stop shutdown). On the trap route a delivered signal ends a protected program even with a handler,
so memcached-1.6.45-shutdown-without-signal.patch makes `shutdown` set the main loop's stop flag
directly, as the SIGINT handler would; main returns and the observer's atexit handler prints the
REUSE-GAP lines on the server's stderr. Both arms carry the patch. CAPSTONE_QEMU_LOCK must name this VM's own lock. NEW_DIR/result.json holds, per
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

FLAGS = '-l 127.0.0.1 -p 21299 -U 0 -m 64 -t 4 -A'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


LOAD_FILES = 'load.txt load-progress.txt load-stats.txt load.out load.err'


def gate(reps, flags, load_expect=None, load_poll=10, load_stall=300):
    # The trap libc returns ENOSYS for epoll until its opaque-reference side tables exist, so libevent
    # takes its poll backend (EVENT_NOEPOLL); the slab allocator the study observes is the same.
    lines = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1', './capstone-vexec --platform > /dev/null || exit 1',
             'export EVENT_NOEPOLL=1']
    load = ''
    if load_expect:
        load = (f'--load "unset EVENT_NOEPOLL; exec ./mc-crusher --conf ./load.conf --ip 127.0.0.1 --port 21299" '
                f'--load-expect {load_expect} --load-poll {load_poll} --load-stall {load_stall} ')
    for rep in range(1, reps + 1):
        out = f'/tmp/rg-{rep}'
        lines += [f'echo RG_BEGIN:{rep}', f'mkdir -p {out}; chmod 777 {out}',
                  f'./mc-harness --out {out} --port 21299 --stop shutdown {load}-- ./capstone-job {out}/job.json '
                  f'--user 65534:65534 -- ./capstone-vexec ./memcached.dom {flags}',
                  'rc=$?',
                  f'for f in transcript.norm status.txt identity.txt job.json server.out server.err'
                  f'{" " + LOAD_FILES if load_expect else ""}; do '
                  f'[ -f {out}/$f ] || continue; echo RG_FILE:$f; gzip -c {out}/$f | base64; echo RG_FILE_END; done',
                  f'echo RG_END:{rep}:$rc']
    lines += ["! grep -q '^capstone_vm ' /proc/modules", 'echo RG_CLEANUP:$?', 'echo VIRTUAL_STAGED_DONE']
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
    p.add_argument('--flags', default=FLAGS, help='memcached flags (default: those of the protocol-oracle row)')
    p.add_argument('--load-client', type=Path, help='mc-crusher, static for the guest (build-load-client.sh)')
    p.add_argument('--load-conf', type=Path, help="mc-crusher's configuration file")
    p.add_argument('--load-expect', type=int, help='commands (cmd_get + cmd_set) the configuration sends')
    p.add_argument('--load-poll', type=int, default=10)
    p.add_argument('--load-stall', type=int, default=300)
    a = p.parse_args()
    if bool(a.load_client) != bool(a.load_conf) or bool(a.load_client) != bool(a.load_expect):
        p.error('--load-client, --load-conf and --load-expect go together')
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    staged = {'capstone-vexec': a.adapter / 'capstone-vexec',
              'capstone-job': a.job, 'memcached.dom': a.image, 'mc-harness': a.harness}
    if a.load_client:
        staged |= {'mc-crusher': a.load_client, 'load.conf': a.load_conf}
    for name, src in staged.items():
        shutil.copy2(src, stage / name)
    script = gate(a.reps, a.flags, a.load_expect, a.load_poll, a.load_stall)
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
        load = None
        if a.load_client:
            line = files.get('load.txt', b'').decode(errors='replace').strip()
            load = dict(item.split('=', 1) for item in line.split()) if line else {}
            load['client_stderr_lines'] = len(files.get('load.err', b'').splitlines())
        reps.append(dict(rep=int(rep), harness_status=int(rc), load=load,
                         status=files.get('status.txt', b'').decode().strip(),
                         job=json.loads(files.get('job.json', b'{}') or b'{}'),
                         identity=files.get('identity.txt', b'').decode().strip(),
                         transcript_sha256=hashlib.sha256(files['transcript.norm']).hexdigest()
                         if 'transcript.norm' in files else None,
                         other_stderr=[l for l in err.splitlines() if not l.startswith('REUSE-GAP ')],
                         reuse=parse_reuse(err)))
    rec = dict(arm=a.arm, completed=run.returncode == 0 and '\nRG_CLEANUP:0\n' in log, boot_wall_seconds=round(wall, 1),
               repetitions=reps, flags=a.flags,
               load=dict(conf=a.load_conf.read_text(), expect=a.load_expect, poll=a.load_poll, stall=a.load_stall)
               if a.load_client else None,
               staged_sha256={name: sha(stage / name) for name in sorted(staged)},
               gate_sha256=hashlib.sha256(script.encode()).hexdigest(),
               platform_sha256={name: sha(a.images / name) for name in ('Image', 'fw_jump.elf', 'initramfs.cpio.gz')}
               | {'qemu-system-riscv64': sha(a.qemu.resolve())})
    (a.work / 'result.json').write_text(json.dumps(rec, indent=1) + '\n')
    ok = rec['completed'] and len(reps) == a.reps and all(r['harness_status'] == 0 and r['reuse'] for r in reps) \
        and all(r['load'] is None or r['load'].get('load') == 'complete' for r in reps)
    print(f'{a.arm}: completed={rec["completed"]} reps={len(reps)} ok={ok} {wall:.0f}s')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
