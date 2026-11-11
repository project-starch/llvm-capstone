#!/usr/bin/env python3
"""Recompile every logged Perl object in an external, separately patched tree."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tree', type=Path, required=True)
p.add_argument('--log', type=Path, action='append', required=True)
p.add_argument('--cc', required=True)
p.add_argument('--record', type=Path, required=True)
p.add_argument('--host-test', action='store_true')
p.add_argument('--sanitize', action='store_true', help='Host ASan/LSan qualification')
p.add_argument('--jobs', type=int, default=4)
a = p.parse_args()
a.record.mkdir(parents=True, exist_ok=False)
commands = {}
link = None
for log in a.log:
    for line in log.read_text().splitlines():
        if 'capstone-cc' not in line or ' -o ' not in line:
            continue
        argv = shlex.split(line)
        target = argv[argv.index('-o')+1]
        argv[0] = a.cc
        if a.sanitize:
            argv[1:1] = ['-fsanitize=address', '-fno-omit-frame-pointer']
        if a.host_test and '-c' in argv:
            argv[1:1] = ['-DCAPSTONE_PERL_PROVENANCE_TEST=1', '-fno-strict-aliasing', '-fwrapv']
            argv = ['-DSPEC_LINUX_X64' if arg == '-DSPEC_LINUX_AARCH64' else arg for arg in argv]
        if '-c' in argv:
            commands[target] = argv
        else:
            link = argv
if not commands or not link:
    raise SystemExit('Missing compile or link commands')


def run(item):
    target, argv = item
    path = a.record / (target.replace('/', '_')+'.log')
    with path.open('w') as out:
        out.write(shlex.join(argv)+'\n')
        out.flush()
        result = subprocess.run(argv, cwd=a.tree, stdout=out, stderr=subprocess.STDOUT)
    return target, result.returncode


with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
    outcomes = dict(pool.map(run, commands.items()))
(a.record/'commands.json').write_text(json.dumps({'compile':commands,'link':link,'outcomes':outcomes},indent=2)+'\n')
failed = [name for name, code in outcomes.items() if code]
if failed:
    raise SystemExit('Compile failed: '+', '.join(failed))
target, rc = run((link[link.index('-o')+1], link))
if rc:
    raise SystemExit('Link failed: '+target)
inventory = {name: hashlib.sha256((a.tree/name).read_bytes()).hexdigest() for name in [*commands, target]}
(a.record/'artifacts.json').write_text(json.dumps(inventory,indent=2)+'\n')
print('Built',len(commands),'objects and',target,flush=True)
