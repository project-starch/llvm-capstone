#!/usr/bin/env python3
"""One boot of the mruby reuse-distance measurement: one arm, one repetition.

    run-mruby-reuse.py --arm stock|protected --rep N --root BUILD_ROOT --kit KIT \
        --run-staged RUN_STAGED --work NEW_DIR

BUILD_ROOT is a build-mruby-reuse.sh root. Stages its interpreter, the AO render benchmark of the
same mruby tree (benchmark/bm_ao_render.rb), the kit's launcher and module, and a gate that runs
`mruby bm_ao_render.rb` once, between two `capstone-vexec --stats` readings; the benchmark's
stdout (a 64x64 PPM) comes back gzip+base64 for the cross-arm comparison. Boots it through runtime/virtual/run-staged.py
(CAPSTONE_QEMU_LOCK must name this VM's own lock) and writes NEW_DIR/rep.json with the parsed
REUSE-GAP line, the oracle, the exit status, the --stats readings and the staged hashes. The
console log stays in NEW_DIR/guest/.
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

GATE = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export HOME=/tmp TMPDIR=/tmp
echo "RG_VSTATS:before $(./capstone-vexec --stats)"
t0=$(date +%s)
./capstone-vexec ./mruby ./bm_ao_render.rb AO_ARGS > /tmp/rg.out 2> /tmp/rg.err
rc=$?
echo "RG_TIME:$(( $(date +%s) - t0 ))"
echo "RG_RC:$rc"
echo "RG_OUT_SHA256:$(sha256sum < /tmp/rg.out)"
echo "RG_OUT_BYTES:$(wc -c < /tmp/rg.out)"
echo RG_OUT_FILE; gzip -c /tmp/rg.out | base64; echo RG_OUT_FILE_END
sed 's/^/RG_ERR:/' /tmp/rg.err
echo "RG_VSTATS:after $(./capstone-vexec --stats)"
rmmod capstone_vm; echo "RG_CLEANUP:$?"
echo VIRTUAL_STAGED_DONE
'''

FIELDS = ('capacity', 'attempts', 'issues', 'releases', 'reuses', 'distinct_starts', 'error')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tree_sha(root):
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode() + b'\0' + bytes.fromhex(sha(f)))
    return h.hexdigest()


def parse_reuse(text):
    found = {}
    for m in re.finditer(r'^REUSE-GAP (\S+) (.*)$', text, re.M):
        name, rest = m.group(1), m.group(2)
        kv = dict(item.split('=', 1) for item in rest.split())
        rec = {k: int(kv[k]) for k in FIELDS}
        bins = [int(x) for x in kv['bins'].split(',')]
        if len(bins) != 32:
            raise ValueError(f'{name}: {len(bins)} bins')
        rec['bins'] = bins
        if name in found:
            raise ValueError(f'two REUSE-GAP lines for {name}')
        found[name] = rec
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('stock', 'protected'), required=True)
    p.add_argument('--rep', type=int, required=True)
    for name in ('root', 'kit', 'run-staged', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--ao-width', type=int, help='IMAGE_WIDTH = IMAGE_HEIGHT (the benchmark\'s ARGV[0]); '
                   'default: none passed, the benchmark\'s own 64')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    tree = a.root / 'src/mruby'
    shutil.copy2(tree / 'build/capstone/bin/mruby', stage / 'mruby')
    shutil.copy2(tree / 'benchmark/bm_ao_render.rb', stage / 'bm_ao_render.rb')
    shutil.copy2(a.kit / 'adapter/capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(a.kit / 'adapter/module/capstone_vm.ko', stage / 'capstone_vm.ko')
    (stage / 'gate.sh').write_text(GATE.replace('AO_ARGS', str(a.ao_width) if a.ao_width else ''))
    staged = {n: sha(stage / n) for n in ('mruby', 'bm_ao_render.rb', 'capstone-vexec', 'capstone_vm.ko', 'gate.sh')}
    t0 = time.time()
    run = subprocess.run([sys.executable, str(a.run_staged), '--qemu', str(a.kit / 'qemu-system-riscv64'),
                          '--images', str(a.kit / 'images'), '--stage', str(stage),
                          '--work', str(a.work / 'guest'), '--timeout', str(a.timeout)])
    wall = time.time() - t0
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    m = re.search(r'^RG_OUT_FILE\n(.*?)^RG_OUT_FILE_END$', log, re.M | re.S)
    out = gzip.decompress(base64.b64decode(m.group(1))) if m else None
    if out is not None:
        (a.work / 'ao.out').write_bytes(out)
    err = re.findall(r'^RG_ERR:(.*)$', log, re.M)
    rec = {
        'program': 'mruby', 'ao_width': a.ao_width or 64, 'arm': a.arm, 'rep': a.rep, 'completed': run.returncode == 0,
        'boot_wall_seconds': round(wall, 1),
        'rc': int(m.group(1)) if (m := re.search(r'^RG_RC:(\d+)$', log, re.M)) else None,
        'guest_seconds': int(m.group(1)) if (m := re.search(r'^RG_TIME:(\d+)$', log, re.M)) else None,
        'stdout_bytes': None if out is None else len(out),
        'stdout_sha256': None if out is None else hashlib.sha256(out).hexdigest(),
        'stdout_sha256_guest': (m.group(1).split()[0] if (m := re.search(r'^RG_OUT_SHA256:(.*)$', log, re.M)) else None),
        'stderr_other': [l for l in err if not l.startswith('REUSE-GAP ')],
        'reuse_gap': parse_reuse('\n'.join(err)),
        'vexec_stats': {t: json.loads(j) for t, j in re.findall(r'^RG_VSTATS:(\S+) (.*)$', log, re.M)},
        'cleanup': (m.group(1) if (m := re.search(r'^RG_CLEANUP:(\d+)$', log, re.M)) else None),
        'staged_sha256': staged,
        'platform_sha256': {n: sha(a.kit / 'images' / n) for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}
                           | {'qemu-system-riscv64': sha(a.kit / 'qemu-system-riscv64')},
        'run_staged_sha256': sha(a.run_staged),
    }
    (a.work / 'rep.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    rg = rec['reuse_gap'].get('gc-slots')
    print(f'mruby {a.arm} rep{a.rep}: completed={rec["completed"]} rc={rec["rc"]} out={rec["stdout_sha256"]} '
          f'error={rg and rg["error"]} issues={rg and rg["issues"]} {wall:.0f}s')
    ok = rec['completed'] and rec['rc'] == 0 and out and out.startswith(b'P6\n%d %d\n255\n' % (((a.ao_width or 64),) * 2)) and rg and rg['error'] == 0
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
