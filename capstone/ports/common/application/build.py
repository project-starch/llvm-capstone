#!/usr/bin/env python3
"""Relink already-built application objects with a delegated application SDK.

Upstream builds stay immutable. This is a link adapter, not a replacement for
the source preparation/build recipes in ports/. Paths are explicit inputs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXPERIMENTS = REPO / "capstone/experiments/applications"

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
        return [src / 'mrbgems/mruby-bin-mruby/tools/mruby/mruby.o', src / 'lib/libmruby.a',
                root / 'runtime/spawn-shell.o']
    if app == 'cpython':
        path = REPO / 'capstone/ports/cpython/app/survey-cpython-capstone.py'
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
        listing = root / 'link/backend-objects.txt'
        if not listing.is_file():
            raise ValueError('PostgreSQL link inputs missing; rerun build-domain.sh with PGSU_FROM=link')
        names = set(listing.read_text().split())
        if not names:
            raise ValueError('PostgreSQL backend object list is empty')
        return ([root / 'link/static_modules.o'] + [src / n for n in sorted(names)] +
                [src / 'src/port/libpgport_srv.a', src / 'src/common/libpgcommon_srv.a'])
    raise ValueError(app)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', choices=['perl', 'mruby', 'cpython', 'postgres', 'sqlite', 'ffmpeg', 'tshark'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--toolchain', type=Path, required=True)
    p.add_argument('--heap', choices=['level0', 'sublet'], default='level0')
    p.add_argument('--profile', choices=['physical', 'virtual'], default='physical')
    p.add_argument('--arena', type=int, default=64*1024*1024)
    p.add_argument('--heap-log', type=int, default=26)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input-revision', required=True, help='Port recipe revision; cached objects are identified separately by hashes')
    p.add_argument('--libc-root', type=Path, help='Separate root containing musl-src and musl-build')
    p.add_argument('--musl', type=Path, help='Explicit musl source directory')
    p.add_argument('--libc', type=Path, help='Explicit Capstone libc archive')
    p.add_argument('--include', type=Path, action='append', default=[])
    p.add_argument('--nested', choices=['none', 'cpython', 'mruby', 'postgres', 'perl'], default='none')
    p.add_argument('--gc-gaps', action='store_true', help='mruby GC-slot aggregate observer is present in the input archive')
    p.add_argument('--reuse-gap', action='store_true',
                   help='retain the PostgreSQL inner-chunk reuse report in the final image')
    p.add_argument('--instrument', action='store_true', help='Enable the application-memory study observers')
    p.add_argument('--allocations', action='store_true', help='Count application allocation sizes and address reuse')
    args = p.parse_args()
    if (args.allocations or args.gc_gaps or args.reuse_gap) and not args.instrument:
        p.error('study observers require --instrument')
    if args.profile == 'virtual' and args.instrument:
        p.error('virtual migration does not qualify historical memory-study observers')
    if 'cpython' in (args.app, args.nested) and args.profile != 'virtual':
        p.error('CPython builds only for the virtual profile: its heap is musl mallocng')
    input_revision = subprocess.check_output(['git', '-C', REPO, 'rev-parse', args.input_revision+'^{commit}'], text=True).strip()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    def run(cmd):
        cmd = list(map(str, cmd)); commands.append(cmd)
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        env = os.environ.copy()
        # This adapter owns the SDK it just built. An upstream recipe's SDK
        # must not redirect capstone-cc to another runtime at the final link.
        env.pop('CAPSTONE_SDK', None)
        subprocess.run(cmd, check=True, env=env)
    libc_root = args.libc_root.resolve() if args.libc_root else root
    musl = libc_root / 'musl-src/musl-1.2.5'
    libc = libc_root / 'musl-build/libc-capstone.a'
    if args.app == 'cpython':
        libc = root / 'libc-and-builtins.a'
    if args.musl: musl = args.musl.resolve()
    if args.libc: libc = args.libc.resolve()
    objects = inputs(args.app, root)
    if args.nested == 'cpython':
        objects += [root / 'runtime' / n for n in ('pym_backing.o', 'pym_block_lifetimes.o', 'pym_sublet_glue.o')]
    if args.nested == 'postgres':
        objects += [root / 'link/context-pools.o']
    if args.nested == 'perl':
        # Built by ports/perl/musl/build-perl-domain.sh with PERLD_SV_HEADS=1.
        objects += [root / 'link/perl-sv-heads.o']
    if args.reuse_gap and args.app == 'postgres' and args.nested == 'none':
        objects += [root / 'link/spatial-reuse-gap.o']
    if args.nested != 'none' and (args.heap != 'level0' or args.app != args.nested):
        raise ValueError('nested discovery arms use level0 for their separate outer heap')
    if args.gc_gaps and args.app != 'mruby':
        raise ValueError('--gc-gaps is only for the instrumented mruby archives')
    if args.reuse_gap and args.app != 'postgres':
        raise ValueError('--reuse-gap requires an instrumented PostgreSQL archive')
    for f in [musl / 'include/stdlib.h', libc, *objects]:
        if not f.is_file(): raise FileNotFoundError(f)
    sdk = out / 'sdk'
    run(['cmake', '-S', REPO / 'capstone/runtime/application', '-B', sdk, '-G', 'Ninja',
         '-DCMAKE_TOOLCHAIN_FILE=' + str(REPO / 'capstone/ports/common/cmake/toolchains/capstone-domain.cmake'),
         '-DCAPSTONE_LLVM_BUILD_DIR=' + str(args.toolchain.resolve()),
         '-DPORT_HEADER_PROVIDER=musl', '-DPORT_C11_ATOMICS=ON',
         '-DPORT_MUSL_ROOT=' + str(musl), '-DCAPSTONE_MUSL_ARCHIVE=' + str(libc),
         '-DCAPSTONE_APPLICATION_SDK=ON', '-DCAPSTONE_APPLICATION_HEAP=' + args.heap,
         '-DCAPSTONE_APPLICATION_VIRTUAL=' + ('ON' if args.profile == 'virtual' else 'OFF'),
         '-DCAPSTONE_APPLICATION_HEAP_LOG=' + str(args.heap_log),
         '-DCAPSTONE_APPLICATION_DATA_BYTES=33554432',
         '-DCAPSTONE_APPLICATION_ARENA_BYTES=' + str(args.arena),
         '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_C_FLAGS_RELEASE=-O1' +
         (' -DCAPSTONE_LEVEL0_STATS -DCAPSTONE_SUBLET_HEAP_STATS' if args.instrument else ''), '-DCAPSTONE_APPLICATION_GRANT_BYTES=' + str({'postgres': 64, 'mruby': 32, 'perl': 32}.get(args.nested, 0) << 20)])
    run(['cmake', '--build', sdk, '-j8'])
    probe = out / 'memory.o'
    defines = (['-DEXP_SUBLET'] if args.heap == 'sublet' else []) + (['-DEXP_PYMALLOC'] if args.nested == 'cpython' else []) + (['-DEXP_MRB_GC_SUBLET'] if args.nested == 'mruby' else []) + (['-DEXP_PG_CONTEXT_SUBLET'] if args.nested == 'postgres' else []) + (['-DEXP_MRB_GC_GAPS'] if args.gc_gaps else [])
    if args.reuse_gap: defines += ['-DEXP_PG_REUSE_GAP']
    if args.allocations: defines += ['-DEXP_ALLOCATIONS', '-DEXP_CAPSTONE']
    memory_includes = (['-I', REPO / 'capstone/ports/postgres/memory-contexts/src/allocators/sublet',
                        '-I', REPO / 'capstone/runtime/include']
                       if args.nested == 'postgres' else [])
    if args.instrument:
        run([sdk / 'capstone-cc', '-O1', *defines, *memory_includes,
             '-c', EXPERIMENTS / 'memory.c', '-o', probe])
    image = out / (args.app + '.dom')
    extra = []
    if args.allocations:
        extra += ['-O1', *defines, EXPERIMENTS / 'allocations.c',
                  '-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free']
    if args.nested != 'none':
        extra += ['-O1', '-Wl,--wrap=__capstone_region', *defines, '-I',
                  REPO / 'capstone/runtime/include', HERE / 'regions.c']
    if args.reuse_gap:
        extra += ['-Wl,--undefined=pg_reuse_gap_report']
    if args.app in ('sqlite', 'ffmpeg'):
        source = EXPERIMENTS / 'workloads' / ('sqlite.c' if args.app == 'sqlite' else 'decode.c')
        extra += ['-O1', source]
        for inc in args.include: extra += ['-I', inc.resolve()]
        if args.app == 'ffmpeg': extra += ['-I', REPO / 'capstone/ports/ffmpeg/app/src/shared']
    if args.instrument:
        extra += ['-Wl,--wrap=main,--wrap=write', probe]
    elif args.nested in ('cpython', 'postgres'):
        extra += ['-O1', '-Wl,--wrap=main', *defines, HERE / 'initialize.c']
    run([sdk / 'capstone-cc', *extra, *objects, '-o', image])
    run([args.toolchain / 'bin/llvm-objcopy', '--strip-debug', image])
    manifest = dict(application=args.app, application_abi=2, profile=args.profile,
                    virtual_vm_abi=3 if args.profile == 'virtual' else None,
                    instrument=args.instrument, allocations=args.allocations,
                    gc_gaps=args.gc_gaps,
                    reuse_gap=args.reuse_gap,
                    allocations_sha256=sha(EXPERIMENTS / 'allocations.c') if args.allocations else None,
                    heap='virtual' if args.profile == 'virtual' else args.heap,
                    nested=args.nested, recipe_revision=input_revision,
                    runtime_revision=subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
                    runtime_dirty=bool(subprocess.check_output(['git', '-C', REPO, 'status', '--porcelain'])),
                    arena_bytes=args.arena, heap_log=args.heap_log,
                    image_sha256=sha(image), compiler_sha256=sha(args.toolchain / 'bin/clang'),
                    runtime_archive_sha256=sha(sdk / 'libapplication-runtime.a'),
                    builtins_archive_sha256=sha(sdk / 'libcapstone-application-builtins.a'),
                    inputs=[dict(path=str(f.relative_to(root)) if f.is_relative_to(root) else f.name,
                                 sha256=sha(f)) for f in [libc, *objects]],
                    instrument_sha256=sha(EXPERIMENTS / 'memory.c') if args.instrument else None)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(image)

if __name__ == '__main__':
    main()
