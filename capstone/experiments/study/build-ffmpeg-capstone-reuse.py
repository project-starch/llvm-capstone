#!/usr/bin/env python3
"""Link the full FFmpeg pool port with the study observer and application SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'library-build', 'capstone-cc', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    source, build = args.source.resolve(), args.library_build.resolve()
    cc, out = args.capstone_cc.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bp = REPO/'capstone/ports/ffmpeg/buffer-pool/src'
    exp = REPO/'capstone/experiments/applications'
    app = REPO/'capstone/ports/ffmpeg/app/src/shared'
    common = ['-O1', '-DFFPOOL_STUDY_GAPS', '-DFFPOOL_STUDY_GAPS_STDERR']
    for path in (build, source, bp/'shared', bp/'capstone-domain',
                 bp/'allocators/sublet', REPO/'capstone/runtime/include', app):
        common += ['-I', str(path)]
    decode = out/'decode.o'
    commands = [[str(cc), *common, '-Dmain=ffdecode_main', '-c',
                 str(exp/'workloads/decode.c'), '-o', str(decode)],
                [str(cc), *common, '-Wl,--wrap=main,--wrap=write',
                 str(exp/'memory.c'), str(HERE/'ffmpeg-capstone-reuse.c'),
                 str(decode), str(app/'ffapp_decode.c'),
                 str(bp/'shared/pool-allocator.c'),
                 str(bp/'capstone-domain/payload-capabilities.c'),
                 str(bp/'allocators/sublet/pool-leases.c'),
                 *[str(build/lib/(lib+'.a')) for lib in
                   ('libavformat', 'libavcodec', 'libavutil')],
                 '-lm', '-o', str(out/'ffmpeg.dom')]]
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    with (out/'build.log').open('w') as log:
        for command in commands:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    manifest = dict(source=str(source), library_build=str(build),
                    source_version=(source/'VERSION').read_text().strip(),
                    compiler_sha256=digest(cc), image_sha256=digest(out/'ffmpeg.dom'),
                    commands_sha256=digest(out/'commands.json'),
                    observer_sources={str(path.relative_to(REPO)): digest(path) for path in
                        (HERE/'ffmpeg-capstone-reuse.c', exp/'workloads/decode.c',
                         bp/'shared/pool-allocator.c',
                         bp/'capstone-domain/payload-capabilities.c',
                         bp/'allocators/sublet/pool-leases.c')},
                    libraries={lib: digest(build/lib/(lib+'.a')) for lib in
                               ('libavformat', 'libavcodec', 'libavutil')})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(out/'ffmpeg.dom')


if __name__ == '__main__':
    main()
