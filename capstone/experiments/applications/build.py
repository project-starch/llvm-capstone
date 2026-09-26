#!/usr/bin/env python3
"""Relink already-built application objects with an instrumented application SDK.

Upstream builds stay immutable. This is a link adapter, not a replacement for
the source preparation/build recipes in ports/. Paths are explicit inputs.
"""
import argparse
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def inputs(app, root):
    if app == 'tshark':
        build = root / 'xbuild'
        command = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'tshark'], text=True).splitlines()[-1]
        return [(build / word).resolve() for word in shlex.split(command) if word.endswith(('.o', '.a'))]
    if app == 'sqlite':
        return [root / 'obj' / n for n in ('sqlite3.o', 'sqlite_vfs.o', 'sqlite_os.o')]
    if app == 'ffmpeg':
        return [root / 'domain/ffapp_decode.o'] + [root / 'domain/ffmpeg-build' / lib / (lib+'.a')
                    for lib in ('libavformat', 'libavcodec', 'libavutil')]
    if app == 'perl':
        src = root / 'src/perl-5.36.3'
        return [src / 'perlmain.o', src / 'libperl.a', *sorted((src / 'lib/auto').rglob('*.a'))]
    if app == 'mruby':
        src = root / 'src/mruby/build/capstone'
        return [src / 'mrbgems/mruby-bin-mruby/tools/mruby/mruby.o', src / 'lib/libmruby.a']
    if app == 'cpython':
        path = REPO / 'capstone/ports/cpython/interpreter/survey-cpython-capstone.py'
        spec = importlib.util.spec_from_file_location('survey', path)
        survey = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(survey)
        names = []
        for _, var in survey.GROUPS:
            names += survey.make_var(root / 'build', var)
        for _, objects in survey.EXTRA:
            names += objects
        return [root / 'build' / name for name in dict.fromkeys(names)]
    if app == 'postgres':
        src = root / 'domain/postgresql-17.5'
        names = set()
        for directory in ('src/backend', 'src/timezone'):
            for listing in (src / directory).rglob('objfiles.txt'):
                names.update(listing.read_text().split())
        # Module names are renamed by the port's build recipe.
        for directory in ('src/backend/snowball', 'src/pl/plpgsql/src'):
            names.update(str(p.relative_to(src)) for p in (src / directory).rglob('*.o'))
        return ([root / 'link/static_modules.o'] + [src / n for n in sorted(names)] +
                [src / 'src/port/libpgport_srv.a', src / 'src/common/libpgcommon_srv.a'])
    raise ValueError(app)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', choices=['perl', 'mruby', 'cpython', 'postgres', 'sqlite', 'ffmpeg', 'tshark'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--toolchain', type=Path, required=True)
    p.add_argument('--heap', choices=['level0', 'sublet'], default='level0')
    p.add_argument('--arena', type=int, default=64*1024*1024)
    p.add_argument('--heap-log', type=int, default=26)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input-revision', required=True, help='Port recipe revision; cached objects are identified separately by hashes')
    p.add_argument('--libc-root', type=Path, help='Separate root containing musl-src and musl-build')
    p.add_argument('--include', type=Path, action='append', default=[])
    p.add_argument('--nested', choices=['none', 'cpython', 'mruby'], default='none')
    args = p.parse_args()
    input_revision = subprocess.check_output(['git', '-C', REPO, 'rev-parse', args.input_revision+'^{commit}'], text=True).strip()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    def run(cmd):
        cmd = list(map(str, cmd)); commands.append(cmd)
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        subprocess.run(cmd, check=True)
    libc_root = args.libc_root.resolve() if args.libc_root else root
    musl = libc_root / 'musl-src/musl-1.2.5'
    libc = libc_root / 'musl-build/libc-capstone.a'
    if args.app == 'cpython':
        libc = root / 'libc-and-builtins.a'
    objects = inputs(args.app, root)
    if args.nested == 'cpython':
        objects += [root / 'runtime' / n for n in ('pym_backing.o', 'pym_block_lifetimes.o', 'pym_sublet_glue.o')]
    if args.nested != 'none' and (args.heap != 'level0' or args.app != args.nested):
        raise ValueError('nested discovery arms use level0 for their separate outer heap')
    for f in [musl / 'include/stdlib.h', libc, *objects]:
        if not f.is_file(): raise FileNotFoundError(f)
    sdk = out / 'sdk'
    run(['cmake', '-S', REPO / 'capstone/runtime/application', '-B', sdk, '-G', 'Ninja',
         '-DCMAKE_TOOLCHAIN_FILE=' + str(REPO / 'capstone/ports/common/cmake/toolchains/capstone-domain.cmake'),
         '-DCAPSTONE_LLVM_BUILD_DIR=' + str(args.toolchain.resolve()),
         '-DPORT_HEADER_PROVIDER=musl', '-DPORT_C11_ATOMICS=ON',
         '-DPORT_MUSL_ROOT=' + str(musl), '-DCAPSTONE_MUSL_ARCHIVE=' + str(libc),
         '-DCAPSTONE_APPLICATION_SDK=ON', '-DCAPSTONE_APPLICATION_HEAP=' + args.heap,
         '-DCAPSTONE_APPLICATION_HEAP_LOG=' + str(args.heap_log),
         '-DCAPSTONE_APPLICATION_DATA_BYTES=33554432',
         '-DCAPSTONE_APPLICATION_ARENA_BYTES=' + str(args.arena),
         '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_C_FLAGS_RELEASE=-O1 -DCAPSTONE_LEVEL0_STATS -DCAPSTONE_SUBLET_HEAP_STATS' +
         (' -DCAPSTONE_APPLICATION_HEAP_BYTES=' + str((80 if args.nested == 'cpython' else 32) << 20)
          if args.nested != 'none' else '')])
    run(['cmake', '--build', sdk, '-j8'])
    probe = out / 'memory.o'
    defines = (['-DEXP_SUBLET'] if args.heap == 'sublet' else []) + (['-DEXP_PYMALLOC'] if args.nested == 'cpython' else [])
    run([sdk / 'capstone-cc', '-O1', *defines,
         '-c', HERE / 'memory.c', '-o', probe])
    image = out / (args.app + '.dom')
    extra = []
    if args.nested != 'none':
        extra += ['-O1', '-Wl,--wrap=__capstone_region', *defines, '-I',
                  REPO / 'capstone/runtime/include', HERE / 'regions.c']
    if args.app in ('sqlite', 'ffmpeg'):
        source = HERE / 'workloads' / ('sqlite.c' if args.app == 'sqlite' else 'decode.c')
        extra += ['-O1', source]
        for inc in args.include: extra += ['-I', inc.resolve()]
        if args.app == 'ffmpeg': extra += ['-I', REPO / 'capstone/ports/ffmpeg/app/src/shared']
    run([sdk / 'capstone-cc', '-Wl,--wrap=main,--wrap=write', probe, *extra, *objects, '-o', image])
    run([args.toolchain / 'bin/llvm-objcopy', '--strip-debug', image])
    manifest = dict(application=args.app, heap=args.heap, nested=args.nested, recipe_revision=input_revision,
                    runtime_revision=subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
                    runtime_dirty=bool(subprocess.check_output(['git', '-C', REPO, 'status', '--porcelain'])),
                    arena_bytes=args.arena, heap_log=args.heap_log,
                    image_sha256=sha(image), compiler_sha256=sha(args.toolchain / 'bin/clang'),
                    runtime_archive_sha256=sha(sdk / 'libapplication-runtime.a'),
                    builtins_archive_sha256=sha(sdk / 'libcapstone-application-builtins.a'),
                    inputs=[dict(path=str(f.relative_to(root)) if f.is_relative_to(root) else f.name,
                                 sha256=sha(f)) for f in [libc, *objects]],
                    instrument_sha256=sha(HERE / 'memory.c'))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(image)

if __name__ == '__main__':
    main()
