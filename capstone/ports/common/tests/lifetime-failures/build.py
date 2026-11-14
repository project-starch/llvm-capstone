#!/usr/bin/env python3
"""Build short allocator tests and controls from configured, patched sources."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('port', choices=['postgres', 'mruby', 'perl'])
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--sdk', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--spawn-object', type=Path)
parser.add_argument('--baseline', default='0133b9e21cd5f250fbbb02e756dbb9b3037a7e33')
a = parser.parse_args()
src, out, sdk = a.source.resolve(), a.out.resolve(), a.sdk.resolve()
patches = {
    'postgres': 'postgres/memory-contexts/patches/postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch',
    'mruby': 'mruby/app/patches/4.0.0-rc2/0008-gc-slots-as-sublet-lifetimes.patch',
    'perl': 'perl/sublet/patches/5.36.3/0001-sv-heads-as-sublet-lifetimes.patch',
}
patch_path = 'capstone/ports/' + patches[a.port]
patch = (repo / patch_path).read_text()
old = subprocess.check_output(['git', '-c', 'submodule.recurse=false', 'show',
                               a.baseline + ':' + patch_path], cwd=repo, text=True)
control = out / 'control-source'
control.mkdir(parents=True, exist_ok=True)
# Perl's quoted header includes resolve beside perl.h. Copy all root headers
# so the control's callers really expand the original uproot_SV macro.
if a.port == 'perl':
    for p in src.glob('*.h'):
        shutil.copy2(p, control / p.name)
for name in re.findall(r'^--- a/(\S+)', patch, re.M):
    target = control / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / name, target)
for data, reverse in [(patch, True), (old, False)]:
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-d', str(control)] +
                   (['-R'] if reverse else []), input=data, text=True, check=True)
env = dict(os.environ, CAPSTONE_SDK=str(sdk))
cc = str(sdk / 'capstone-cc')
common = ['-O1', '-w', '-I' + str(repo / 'capstone/runtime/include'),
          '-include', str(here / 'inject.h')]

def run(args):
    subprocess.run([cc, *map(str, args)], env=env, check=True)

for arm in ['fixed', 'unfixed']:
    dest = out / arm
    dest.mkdir(parents=True, exist_ok=True)
    tree = src if arm == 'fixed' else control
    objects = []

    def compile(source, name, flags):
        obj = dest / (name + '.o')
        run([*common, *flags, '-c', source, '-o', obj])
        objects.append(obj)

    if a.port == 'postgres':
        flags = ['-I' + str(tree / 'src/include'), '-I' + str(src / 'src/include')]
        for name in ['aset', 'mcxt', 'slab', 'generation', 'bump', 'alignedalloc', 'memdebug']:
            relative = 'src/backend/utils/mmgr/' + name + '.c'
            compile(tree / relative if (tree / relative).exists() else src / relative, name, flags)
        port = repo / 'capstone/ports/postgres/memory-contexts'
        compile(port / 'src/shared/postgres-compat.c', 'compat', flags)
        compile(port / 'src/native/replay/printf.c', 'printf', flags)
        compile(here / 'pg-test.c', 'test', flags)
        run([*objects, '-o', dest / 'postgres.dom'])
    elif a.port == 'perl':
        flags = ['-DPERL_CORE', '-DPERL_CAPSTONE_SUBLET', '-D_GNU_SOURCE',
                 '-I' + str(tree), '-I' + str(src)]
        compile(tree / 'sv.c', 'sv', flags)
        compile(here / 'perl-test.c', 'test', flags)
        run(['-Wl,--wrap=Perl_croak_no_mem', *objects, src / 'libperl.a', '-o', dest / 'perl.dom'])
    else:
        if not a.spawn_object:
            parser.error('mruby needs --spawn-object from its application build')
        flags = ['-DMRB_NO_BOXING', '-DMRB_CAPSTONE_SUBLET', '-I' + str(src / 'include'),
                 '-I' + str(src / 'build/capstone/include'), '-I' + str(src / 'src')]
        compile(tree / 'src/gc.c', 'gc', flags)
        compile(here / 'mruby-test.c', 'test', flags)
        run([*objects, src / 'build/capstone/lib/libmruby.a', a.spawn_object,
             '-o', dest / 'mruby.dom'])
    print(a.port, arm, 'build PASS', flush=True)
