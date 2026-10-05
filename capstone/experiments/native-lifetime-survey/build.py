#!/usr/bin/env python3
"""Build one pinned native application in the experiment's Linux container."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('CAPSTONE_TMP_ROOT', '/tmp/capstone')) / 'native-survey'

def run(argv, cwd, env):
    print('+', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('application')
    parser.add_argument('variant', choices=['baseline', 'observed'])
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    name, variant = args.application, args.variant
    # Perl requires distinct makefile and Makefile on a case-sensitive filesystem.
    build_root = Path('/var/tmp/native-survey-build') if name == 'perl' else WORK / 'build'
    source = build_root / variant / name
    prefix = WORK / 'install' / variant / name
    prefix.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        shutil.copytree(WORK / 'original' / name, source)
        if variant == 'observed':
            run([sys.executable, HERE / 'instrument.py', name, source], HERE, os.environ.copy())
    run([sys.executable, HERE / 'instrument.py', '--check', name, source,
         WORK / 'original' / name, variant], HERE, os.environ.copy())
    env = os.environ.copy()
    env.pop('NS_OUT', None)
    env.pop('LD_PRELOAD', None)
    cflags = '-O2 -g'
    ldflags = ''
    if variant == 'observed':
        cflags += ' -I' + str(HERE)
        lib = WORK / 'lib'
        ldflags = f'-Wl,--no-as-needed -L{lib} -lnativesurvey -Wl,-rpath,{lib}'
    env.update(CFLAGS=cflags, CXXFLAGS=cflags, LDFLAGS=ldflags)
    commands = []
    make = ['make', f'-j{args.jobs}']
    if name == 'sqlite':
        speedtest = WORK / 'original/sqlite-src/test/speedtest1.c'
        for program, unit in [('speedtest1', speedtest), ('sqlite3', source / 'shell.c')]:
            commands.append(['cc', *cflags.split(), '-DSQLITE_ENABLE_MEMSYS5',
                             '-DSQLITE_ENABLE_RTREE', '-I.', 'sqlite3.c', unit,
                             *ldflags.split(), '-lm', '-ldl', '-lpthread', '-o', prefix / program])
    elif name == 'cpython':
        commands = [['./configure', '--without-ensurepip', '--without-static-libpython', f'--prefix={prefix}'],
                    make]
    elif name == 'postgresql':
        commands = [['./configure', '--without-icu', '--without-readline', '--without-zlib', f'--prefix={prefix}'],
                    make, make + ['install']]
    elif name == 'perl':
        commands = [['sh', 'Configure', '-des', f'-Dprefix={prefix}', f'-Dccflags={cflags}',
                     f'-Dldflags={ldflags}', '-Dman1dir=none', '-Dman3dir=none'], make]
    elif name == 'mruby':
        config = source / 'survey_config.rb'
        config.write_text("MRuby::Build.new do |conf|\n  toolchain :gcc\n  conf.gembox 'default'\n"
                          + f"  conf.cc.flags << {json.dumps(cflags)}\n"
                          + f"  conf.linker.flags << {json.dumps(ldflags)}\n"
                          + "  conf.enable_test\nend\n")
        env['MRUBY_CONFIG'] = str(config)
        commands = [['rake', f'-j{args.jobs}']]
    elif name == 'ffmpeg':
        commands = [['./configure', '--disable-everything', '--disable-autodetect', '--disable-doc',
                     '--disable-network', '--disable-x86asm', '--disable-debug', '--disable-ffplay',
                     '--disable-ffprobe', '--enable-ffmpeg', '--enable-protocol=file,pipe',
                     '--enable-indev=lavfi', '--enable-filter=testsrc2,scale,hflip,format,null,anull',
                     '--enable-encoder=mpeg4,rawvideo', '--enable-decoder=mpeg4,h264,wrapped_avframe',
                     '--enable-parser=mpeg4video,h264', '--enable-muxer=matroska,framemd5,framecrc,null',
                     '--enable-demuxer=matroska,mov,avi,h264,m4v', '--enable-static', '--disable-shared',
                     f'--extra-cflags={cflags}', f'--extra-ldflags={ldflags}', f'--prefix={prefix}'],
                    make + ['ffmpeg']]
    elif name == 'wireshark':
        build = source / 'out'
        commands = [['cmake', '-S', '.', '-B', build, '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_wireshark=OFF', '-DBUILD_tshark=ON',
                     '-DBUILD_stratoshark=OFF', '-DENABLE_LUA=OFF', '-DENABLE_GNUTLS=OFF',
                     '-DENABLE_KERBEROS=OFF', '-DBUILD_dumpcap=OFF', '-DENABLE_SMI=OFF', '-DBUILD_sharkd=OFF',
                     f'-DCMAKE_INSTALL_PREFIX={prefix}'],
                    ['cmake', '--build', build, '--target', 'tshark', '-j', str(args.jobs)]]
    elif name == 'memcached':
        commands = [['./configure', '--disable-docs', f'--prefix={prefix}'], make]
    else:
        raise ValueError(name)
    for command in commands:
        run(command, source, env)
    manifest = dict(application=name, variant=variant, commands=[[str(x) for x in c] for c in commands],
                    cflags=cflags, ldflags=ldflags, source=json.loads((WORK/'sources.json').read_text())[name],
                    instrumentation_sha256=hashlib.sha256((HERE/'instrument.py').read_bytes()).hexdigest())
    (prefix/'build.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    main()
