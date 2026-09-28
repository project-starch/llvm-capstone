#!/usr/bin/env python3
"""Link the full FFmpeg pool port with the study observer and application SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def program_pool_bytes(path):
    """Read the launch descriptor, not the SDK's unrelated malloc arena size."""
    data = Path(path).read_bytes()
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected a little-endian ELF64 application')
    offset = struct.unpack_from('<Q', data, 40)[0]
    stride, count, names_index = struct.unpack_from('<HHH', data, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * stride)
                for i in range(count)]
    names = sections[names_index]
    strings = data[names[4]:names[4] + names[5]]
    for section in sections:
        name = strings[section[0]:].split(b'\0', 1)[0]
        if name == b'.capstone_application':
            magic, version, _, _, heap = struct.unpack_from('<5Q', data, section[4])
            if magic != 0x315050414e4f5043 or version != 1:
                raise ValueError('unsupported application descriptor')
            return heap
    raise ValueError('missing application launch descriptor')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'library-build', 'capstone-cc', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    source, build = args.source.resolve(), args.library_build.resolve()
    cc, out = args.capstone_cc.resolve(), args.out.resolve()
    for filename, hook in (('buffer.c', 'ff2_payload_reusable'),
                           ('refstruct.c', 'ff2_ref_reusable')):
        if hook not in (source/'libavutil'/filename).read_text():
            p.error('prepared FFmpeg sources lack quarantine-aware selection; rebuild the prepared source and libraries')
    out.mkdir(parents=True, exist_ok=False)
    bp = REPO/'capstone/ports/ffmpeg/buffer-pool/src'
    exp = REPO/'capstone/experiments/applications'
    app = REPO/'capstone/ports/ffmpeg/app/src/shared'
    common = ['-O1', '-DFFPOOL_STUDY_GAPS', '-DFFPOOL_STUDY_GAPS_STDERR',
              '-DFFPOOL_APP_QUARANTINE']
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
    heap_bytes = program_pool_bytes(out/'ffmpeg.dom')
    if heap_bytes < 256 << 20:
        p.error('application SDK needs CAPSTONE_APPLICATION_HEAP_BYTES >= 268435456; '
                'CAPSTONE_APPLICATION_ARENA_BYTES controls the separate malloc arena')
    manifest = dict(source=str(source), library_build=str(build),
                    program_pool_bytes=heap_bytes,
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
