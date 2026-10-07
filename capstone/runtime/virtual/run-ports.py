#!/usr/bin/env python3
"""Qualify the remaining SDK application ports against independent workload oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SCRIPT = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1
echo PORT_BEGIN:postgres
cp -a pgcluster /tmp/pgcluster
chown -R nobody /tmp/pgcluster
chmod 666 /dev/capstone-vm
su nobody -s /bin/sh -c 'cd /mnt/vm; ./capstone-vexec ./bin/postgres --single -D /tmp/pgcluster -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT -c log_timezone=GMT -c dynamic_shared_memory_type=sysv -c exit_on_error=true postgres < work.sql'
echo PORT_END:postgres:$?
echo PORT_BEGIN:cpython
CPY_SUBLET_MODE=1 PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S -c 'import json,gc; a=[{"n":n} for n in range(1000)]; assert json.loads(json.dumps(a))==a; del a; gc.collect(); print("CPYTHON-OK 1000")'
echo PORT_END:cpython:$?
for input in input input.flip; do
  echo PORT_BEGIN:ffmpeg-$input
  ./capstone-vexec ./ffmpeg.dom /mnt/vm/$input.mkv 1
  echo PORT_END:ffmpeg-$input:$?
done
echo PORT_BEGIN:tshark
./capstone-vexec ./tshark.dom -r /mnt/vm/dns_port.pcap -T fields -e frame.number -e dns.qry.name
echo PORT_END:tshark:$?
echo PORT_BEGIN:tshark-missing
./capstone-vexec ./tshark.dom -r /mnt/vm/absent.pcap -T fields -e frame.number
echo PORT_END:tshark-missing:$?
echo PORT_BEGIN:cleanup
./capstone-vexec --stats
rmmod capstone_vm
echo PORT_END:cleanup:$?
echo VIRTUAL_STAGED_DONE
'''


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def frames(text):
    return re.findall(r'^0,.*[0-9a-f]{32}$', text, re.M)


def tree_sha(root):
    h = hashlib.sha256()
    for file in sorted(root.rglob('*')):
        if file.is_file():
            h.update(str(file.relative_to(root)).encode()+b'\0')
            h.update(bytes.fromhex(sha(file)))
    return h.hexdigest()


def pg_rows(text):
    return re.findall(r'\t(?: \d+: [^\n]+|----)', text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'images', 'adapter', 'cpython', 'cpython-lib', 'cpython-native', 'postgres',
                 'pg-fixture', 'ffmpeg', 'ffmpeg-fixture', 'tshark', 'tshark-native', 'capture', 'work'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--cpython-inner', action='store_true',
                   help='Require the inner pymalloc adapter to report revocation on every free')
    p.add_argument('--ffmpeg-staged', action='store_true',
                   help='Use the source recipe M5 image, whose successful exit status is 5')
    p.add_argument('--omit-application', choices=('cpython', 'postgres', 'ffmpeg', 'tshark'))
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work/'stage'
    stage.mkdir()
    paths = {name: getattr(a, name) for name in ('qemu', 'cpython', 'cpython_native', 'postgres', 'ffmpeg', 'tshark', 'tshark_native', 'capture')}
    paths.update(launcher=a.adapter/'capstone-vexec', module=a.adapter/'module/capstone_vm.ko')
    paths.update({name: a.images/name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    paths.update(input=a.ffmpeg_fixture/'input.mkv', flipped_input=a.ffmpeg_fixture/'input.flip.mkv',
                 ffmpeg_reference=a.ffmpeg_fixture/'native.out', ffmpeg_flipped_reference=a.ffmpeg_fixture/'native.flip.out',
                 pg_reference=a.pg_fixture/'native-work.stdout',
                 sql=REPO/'capstone/ports/postgres/app/work.sql')
    hashes = {name: sha(path) for name, path in paths.items()}
    for name in ('cpython', 'postgres', 'ffmpeg', 'tshark'):
        if name != a.omit_application:
            target = stage/'bin/postgres' if name == 'postgres' else stage/(name+'.dom')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths[name], target)
    shutil.copy2(paths['launcher'], stage/'capstone-vexec')
    shutil.copy2(paths['module'], stage/'capstone_vm.ko')
    shutil.copytree(a.cpython_lib, stage/'cpy/lib/python3.13', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    version = subprocess.check_output([str(a.cpython_native), '-c', 'import platform; print(platform.python_version())'], text=True).strip()
    if version != '3.13.7':
        p.error('bytecode preparation requires native CPython 3.13.7')
    # Distribution bytecode contains no native pointers. Compile from the staged
    # sources, with guest paths, using exactly the pinned interpreter version.
    subprocess.run([str(a.cpython_native), '-m', 'compileall', '-q', '-f',
                    '-x', '/test/', '-d', '/mnt/vm/cpy/lib/python3.13', str(stage/'cpy/lib/python3.13')], check=True)
    hashes['cpython_distribution'] = tree_sha(stage/'cpy')
    hashes['postgres_pristine_cluster'] = tree_sha(a.pg_fixture/'cluster')
    hashes['postgres_share'] = tree_sha(a.pg_fixture/'install/share')
    hashes['gate_script'] = hashlib.sha256(SCRIPT.encode()).hexdigest()
    shutil.copytree(a.pg_fixture/'cluster', stage/'pgcluster')
    shutil.copytree(a.pg_fixture/'install/share/postgresql', stage/'share')
    for key, name in [('input', 'input.mkv'), ('flipped_input', 'input.flip.mkv'), ('capture', 'dns_port.pcap'), ('sql', 'work.sql')]:
        shutil.copy2(paths[key], stage/name)
    (stage/'gate.sh').write_text(SCRIPT)
    native = subprocess.run([str(a.tshark_native), '-r', str(a.capture), '-T', 'fields',
                             '-e', 'frame.number', '-e', 'dns.qry.name'], capture_output=True, text=True, check=True).stdout
    (a.work/'tshark-native.stdout').write_text(native)
    run = subprocess.run([sys.executable, str(HERE/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work/'guest'), '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    parts = {name: (int(code), text) for name, text, code in
             re.findall(r'^PORT_BEGIN:([^\n]+)\n(.*?)^PORT_END:\1:(\d+)\n', log, re.M | re.S)}
    checks = {name+'_exit': parts.get(name, (-1, ''))[0] == 0
              for name in ('cpython', 'postgres', 'ffmpeg-input', 'ffmpeg-input.flip', 'tshark', 'cleanup')}
    body = lambda name: parts.get(name, (-1, ''))[1]
    if a.ffmpeg_staged:
        for name in ('ffmpeg-input', 'ffmpeg-input.flip'):
            checks[name+'_exit'] = (parts.get(name, (-1, ''))[0] == 5 and
                                   'STAGE M5 frames=30 packets=30' in body(name))
    checks['cpython_json_gc'] = 'CPYTHON-OK 1000' in body('cpython')
    if a.cpython_inner:
        checks['cpython_inner_revocation'] = 'CPY-SUBLET mode=1 (sublet: every free revokes)' in body('cpython')
    expected_pg = pg_rows(paths['pg_reference'].read_text())
    checks['postgres_native_rows'] = len(expected_pg) == 22 and pg_rows(body('postgres')) == expected_pg
    ff = frames(body('ffmpeg-input'))
    flip = frames(body('ffmpeg-input.flip'))
    checks['ffmpeg_native_frames'] = len(ff) == 30 and ff == frames(paths['ffmpeg_reference'].read_text())
    checks['ffmpeg_flipped_frames'] = len(flip) == 30 and flip == frames(paths['ffmpeg_flipped_reference'].read_text())
    checks['ffmpeg_input_sensitivity'] = len(ff) == len(flip) == 30 and sum(x != y for x, y in zip(ff, flip)) == 10
    actual_dns = '\n'.join(re.findall(r'^\d+\t[^\n]*$', body('tshark'), re.M))+'\n'
    checks['tshark_native_dns'] = bool(native.strip()) and actual_dns == native
    checks['tshark_missing_input'] = parts.get('tshark-missing', (-1, ''))[0] not in (-1, 0, 139) and 'absent.pcap' in body('tshark-missing')
    stats = re.search(r'^\{"version":1,.*\}$', body('cleanup'), re.M)
    stats = json.loads(stats.group()) if stats else {}
    checks['resources_released'] = bool(stats) and all(stats.get(k) == 0 for k in ('live_domains', 'live_regions', 'live_bytes', 'tag_pages'))
    checks['no_unexpected_faults'] = 'domain fault cause=' not in log
    checks['completed'] = run.returncode == 0
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', tests=checks, sha256=hashes,
                  omitted_application=a.omit_application, stats=stats,
                  scope='Rebuilt virtual SDK application workloads; one hart, private anonymous mappings; Linux trusted')
    (a.work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
