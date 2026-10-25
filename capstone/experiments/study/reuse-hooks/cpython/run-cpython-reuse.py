#!/usr/bin/env python3
"""One boot of the CPython reuse-distance measurement: one arm, one repetition.

    run-cpython-reuse.py --arm stock|protected --rep N --build BUILD_OUT --kit KIT \
        --run-staged RUN_STAGED --work NEW_DIR

BUILD_OUT is a build-cpython-reuse.sh output. Stages its cpython.dom, the build's own Lib/ as
/mnt/vm/cpy/lib/python3.13 (with its _sysconfigdata__linux_.py and an empty lib-dynload, compiled
by the build's native 3.13.7 with guest paths, as the memory-behaviour run did), the kit's
launcher and module, and a gate that runs the run-ports.py JSON/GC one-liner verbatim once,
between two `capstone-vexec --stats` readings. Boots it through runtime/virtual/run-staged.py
(CAPSTONE_QEMU_LOCK must name this VM's own lock) and writes NEW_DIR/rep.json with the parsed
REUSE-GAP line, the oracle, the exit status, the --stats readings and the staged hashes. The
console log stays in NEW_DIR/guest/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ONE_LINER = ('import json,gc; a=[{"n":n} for n in range(1000)]; '
             'assert json.loads(json.dumps(a))==a; del a; gc.collect(); print("CPYTHON-OK 1000")')

GATE = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export HOME=/tmp TMPDIR=/tmp
echo "RG_VSTATS:before $(./capstone-vexec --stats)"
t0=$(date +%s)
PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S -c 'ONE_LINER' > /tmp/rg.out 2> /tmp/rg.err
rc=$?
echo "RG_TIME:$(( $(date +%s) - t0 ))"
echo "RG_RC:$rc"
sed 's/^/RG_OUT:/' /tmp/rg.out
sed 's/^/RG_ERR:/' /tmp/rg.err
echo "RG_VSTATS:after $(./capstone-vexec --stats)"
rmmod capstone_vm; echo "RG_CLEANUP:$?"
echo VIRTUAL_STAGED_DONE
'''.replace('ONE_LINER', ONE_LINER)

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
    for name in ('build', 'kit', 'run-staged', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    image = a.build / 'image/cpython.dom'
    manifest = json.loads((a.build / 'image/manifest.json').read_text())
    shutil.copy2(image, stage / 'cpython.dom')
    if sha(stage / 'cpython.dom') != manifest['image_sha256']:
        sys.exit('staged image does not match its manifest')
    shutil.copy2(a.kit / 'adapter/capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(a.kit / 'adapter/module/capstone_vm.ko', stage / 'capstone_vm.ko')
    src = a.build / 'source'
    lib = stage / 'cpy/lib/python3.13'
    shutil.copytree(src / 'src/Python-3.13.7/Lib', lib, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    bdir = (src / 'build/pybuilddir.txt').read_text().strip()
    shutil.copy2(src / 'build' / bdir / '_sysconfigdata__linux_.py', lib / '_sysconfigdata__linux_.py')
    (lib / 'lib-dynload').mkdir(exist_ok=True)
    py_tree = tree_sha(lib)
    native = src / 'build-python/python'
    ver = subprocess.check_output([str(native), '-B', '-c', 'import platform; print(platform.python_version())'],
                                  text=True).strip()
    if ver != '3.13.7':
        sys.exit('native python is ' + ver)
    comp = subprocess.run([str(native), '-B', '-m', 'compileall', '-q', '-f', '-x', '/test/',
                           '-d', '/mnt/vm/cpy/lib/python3.13', str(lib)], capture_output=True, text=True)
    (stage / 'gate.sh').write_text(GATE)
    staged = {n: sha(stage / n) for n in ('cpython.dom', 'capstone-vexec', 'capstone_vm.ko', 'gate.sh')}
    t0 = time.time()
    run = subprocess.run([sys.executable, str(a.run_staged), '--qemu', str(a.kit / 'qemu-system-riscv64'),
                          '--images', str(a.kit / 'images'), '--stage', str(stage),
                          '--work', str(a.work / 'guest'), '--timeout', str(a.timeout)])
    wall = time.time() - t0
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    out = re.findall(r'^RG_OUT:(.*)$', log, re.M)
    err = re.findall(r'^RG_ERR:(.*)$', log, re.M)
    rec = {
        'program': 'cpython', 'arm': a.arm, 'rep': a.rep, 'completed': run.returncode == 0,
        'boot_wall_seconds': round(wall, 1),
        'rc': int(m.group(1)) if (m := re.search(r'^RG_RC:(\d+)$', log, re.M)) else None,
        'guest_seconds': int(m.group(1)) if (m := re.search(r'^RG_TIME:(\d+)$', log, re.M)) else None,
        'stdout': out,
        'stderr_other': [l for l in err if not l.startswith('REUSE-GAP ')],
        'reuse_gap': parse_reuse('\n'.join(err)),
        'vexec_stats': {t: json.loads(j) for t, j in re.findall(r'^RG_VSTATS:(\S+) (.*)$', log, re.M)},
        'cleanup': (m.group(1) if (m := re.search(r'^RG_CLEANUP:(\d+)$', log, re.M)) else None),
        'staged_sha256': staged, 'manifest_image_sha256': manifest['image_sha256'],
        'manifest_nested': manifest.get('nested'), 'manifest_runtime_revision': manifest.get('runtime_revision'),
        'manifest_runtime_dirty': manifest.get('runtime_dirty'),
        'stdlib_py_tree_sha256': py_tree, 'stdlib_with_bytecode_tree_sha256': tree_sha(lib),
        'compileall_rc': comp.returncode, 'native_python': ver,
        'platform_sha256': {n: sha(a.kit / 'images' / n) for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}
                           | {'qemu-system-riscv64': sha(a.kit / 'qemu-system-riscv64')},
        'run_staged_sha256': sha(a.run_staged), 'one_liner': ONE_LINER,
    }
    (a.work / 'rep.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    rg = rec['reuse_gap'].get('pymalloc')
    print(f'cpython {a.arm} rep{a.rep}: completed={rec["completed"]} rc={rec["rc"]} out={out} '
          f'error={rg and rg["error"]} issues={rg and rg["issues"]} {wall:.0f}s')
    ok = rec['completed'] and rec['rc'] == 0 and out == ['CPYTHON-OK 1000'] and rg and rg['error'] == 0
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
