#!/usr/bin/env python3
"""Build real mruby/FFmpeg applications from explicitly prepared port sources.

Use app/host/prepare-source.sh for FFmpeg and the mruby port's prepared
4.0.0-rc2 tree. No sources are fetched or stored in the repository. The compiler
argument is a configured driver (CheriBSD SDK flags or application capstone-cc).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(root):
    entries = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file(): continue
        relative = path.relative_to(root)
        if relative.parts[0] in ('.git', 'build', 'bin'): continue
        if path.suffix in ('.c', '.h', '.S', '.s', '.rb') or path.name in ('configure', 'Makefile'):
            entries[str(relative)] = digest(path)
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', choices=['ffmpeg', 'mruby'], required=True)
    p.add_argument('--platform', choices=['cheribsd', 'poisoncap'], required=True)
    for name in ('source', 'cc', 'ar', 'ranlib', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--allocations', action='store_true')
    p.add_argument('--nested-pool', choices=['poisoncap'])
    p.add_argument('--nested-gc', choices=['poisoncap'])
    p.add_argument('--pool-reuse-gaps', action='store_true')
    p.add_argument('--jobs', type=int, default=16)
    args = p.parse_args()
    args.out = args.out.resolve(); args.source = args.source.resolve()
    if args.nested_pool and (args.app != 'ffmpeg' or args.platform != 'poisoncap'):
        p.error('--nested-pool poisoncap requires --app ffmpeg --platform poisoncap')
    if args.nested_gc and (args.app != 'mruby' or args.platform != 'poisoncap'):
        p.error('--nested-gc poisoncap requires --app mruby --platform poisoncap')
    if args.platform == 'poisoncap' and not (args.nested_pool or args.nested_gc):
        p.error('the PoisonCap build must select its nested allocator adapter')
    if args.nested_pool and not (args.source/'libavutil/trace.h').exists():
        p.error('the PoisonCap build requires prepare-source.sh --pool')
    if args.nested_pool:
        for filename, hook in (('buffer.c', 'ff2_payload_reusable'),
                               ('refstruct.c', 'ff2_ref_reusable')):
            if hook not in (args.source/'libavutil'/filename).read_text():
                p.error('prepared FFmpeg sources lack quarantine-aware selection; rerun prepare-source.sh --pool')
    if args.pool_reuse_gaps and not args.nested_pool:
        p.error('--pool-reuse-gaps requires --nested-pool poisoncap')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'source-inputs.json').write_text(json.dumps(source_manifest(args.source), indent=2)+'\n')
    commands = []
    def run(command, cwd=args.out, env=None):
        command = list(map(str, command)); commands.append(dict(argv=command, cwd=str(cwd)))
        (args.out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with (args.out/'build.log').open('a') as log:
            subprocess.run(command, cwd=cwd, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    cc = str(args.cc.absolute())
    if args.app == 'mruby':
        src = args.out/'source'
        def ignore(directory, names):
            return [name for name in names if name == '.git' or
                    (Path(directory) == args.source and name in ('build', 'bin'))]
        shutil.copytree(args.source, src, ignore=ignore)
        if args.nested_gc:
            run(['python3', REPO/'capstone/experiments/study/prepare-mruby-poisoncap.py',
                 '--source', src])
        config = args.out/'mruby-cheribsd.rb'
        shutil.copy2(HERE/'mruby-cheribsd.rb', config)
        env = dict(os.environ, EXP_ALLOCATIONS=str(int(args.allocations)),
                   EXP_GC_STUDY=str(int(bool(args.nested_gc))),
                   EXP_ALLOC_SOURCE=str(HERE/'allocations.c'), EXP_CC=cc, EXP_AR=str(args.ar.absolute()),
                   EXP_MEMORY_SOURCE=str(HERE/'cheribsd-memory.c'),
                   MRUBY_CONFIG=str(config))
        run(['rake', '-j'+str(args.jobs)], src, env)
        image = args.out/'mruby'
        shutil.copy2(src/'build/cheribsd/bin/mruby', image)
    else:
        build = args.out/'build'; build.mkdir()
        options = ['--disable-'+name for name in ('everything','autodetect','doc',
            'network','asm','pthreads','programs','debug','iconv',
            'swresample','swscale','avfilter','avdevice','shared')]
        run([args.source/'configure', '--cc='+cc, '--ld='+cc,
             '--ar='+str(args.ar.absolute()), '--ranlib='+str(args.ranlib.absolute()),
             '--enable-cross-compile', '--target-os=freebsd',
             '--arch=riscv64', *options, '--enable-demuxer=matroska',
             '--enable-decoder=mpeg4', '--enable-parser=mpeg4video',
             '--enable-protocol=file', '--enable-static', '--extra-cflags=-O1'], build)
        run(['make', '-j'+str(args.jobs)], build)
        app = REPO/'capstone/ports/ffmpeg/app/src/shared'
        flags = ['-O1', '-Wl,--wrap=main,--wrap=write']
        for path in (build, args.source, app): flags += ['-I', str(path)]
        sources = [HERE/'workloads/decode.c', HERE/'cheribsd-memory.c']
        if args.nested_pool:
            pool = REPO/'capstone/ports/ffmpeg/buffer-pool/src'
            sources[0] = REPO/'capstone/experiments/study/ffmpeg-poisoncap-decode.c'
            flags += ['-DFFPOOL_CHERI', '-DFFPOOL_POISONCAP', '-DFFPOOL_APP_MEMORY',
                      '-DFFPOOL_APP_QUARANTINE',
                      '-I', str(pool/'shared'), '-I', str(REPO/'capstone/runtime/include')]
            if args.pool_reuse_gaps:
                flags += ['-DFFPOOL_STUDY_GAPS']
            sources += [pool/'shared/pool-allocator.c',
                        pool/'cheribsd/poisoncap-payload.c']
        if args.allocations:
            flags += ['-DEXP_ALLOCATIONS', '-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=posix_memalign,--wrap=aligned_alloc']
            sources += [HERE/'allocations.c']
        image = args.out/'ffmpeg'
        run([cc, *flags, *sources, app/'ffapp_decode.c', *[build/lib/(lib+'.a')
             for lib in ('libavformat','libavcodec','libavutil')], '-lm', '-o', image])
    adapter_paths = []
    if args.nested_gc:
        adapter_paths += [REPO/'capstone/experiments/study'/name for name in
                          ('prepare-mruby-poisoncap.py', 'mruby-poisoncap-gc.inc')]
    if args.nested_pool:
        adapter_paths += [REPO/'capstone/ports/ffmpeg/buffer-pool/src'/name for name in
                          ('shared/pool-allocator.c', 'shared/payload-backend.h',
                           'shared/trace.h', 'cheribsd/poisoncap-payload.c')]
        adapter_paths += [REPO/'capstone/experiments/study/ffmpeg-poisoncap-decode.c']
    if adapter_paths:
        adapter_paths += [REPO/'capstone/ports/common/include/poisoncap-quarantine-policy.h']
    manifest = dict(app=args.app, platform=args.platform, allocations=args.allocations,
                    nested_policy=('published-sqlite-thresholds-corrected-v1'
                                   if args.nested_gc or args.nested_pool else None),
                    quarantine_policy_sha256=(digest(REPO/'capstone/ports/common/include/poisoncap-quarantine-policy.h')
                                               if args.nested_gc or args.nested_pool else None),
                    nested_pool=args.nested_pool, nested_gc=args.nested_gc,
                    pool_reuse_gaps=args.pool_reuse_gaps,
                    image_sha256=digest(image), compiler_driver_sha256=digest(args.cc),
                    source=str(args.source), source_manifest_sha256=digest(args.out/'source-inputs.json'),
                    commands_sha256=digest(args.out/'commands.json'),
                    builder_sha256=digest(Path(__file__)),
                    adapter_sources=({name: digest(REPO/'capstone/experiments/study'/name)
                                      for name in ('prepare-mruby-poisoncap.py',
                                                   'mruby-poisoncap-gc.inc')}
                                     if args.nested_gc else {}),
                    nested_adapter_inputs={str(path.relative_to(REPO)): digest(path)
                                           for path in adapter_paths},
                    experiment_sources=source_manifest(HERE),
                    repository_revision=subprocess.check_output(['git','-C',str(REPO),
                        'rev-parse','HEAD'], text=True).strip())
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(image)


if __name__ == '__main__':
    main()
