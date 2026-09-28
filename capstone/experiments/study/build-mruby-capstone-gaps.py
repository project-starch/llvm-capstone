#!/usr/bin/env python3
"""Replace only mruby's GC object in a copied Capstone archive.

The source and original archive remain immutable. Link the resulting scratch
root with applications/build.py --app mruby --gc-gaps; use --nested mruby for
the Sublet arm. The caller supplies a previously configured application SDK.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['spatial', 'sublet'], required=True)
    p.add_argument('--source-root', type=Path, required=True,
                   help='Pinned, built mruby port root containing src/mruby')
    p.add_argument('--sdk-cc', type=Path, required=True,
                   help='capstone-cc from a configured application SDK')
    p.add_argument('--llvm-ar', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    source = args.source_root.resolve() / 'src/mruby'
    original = source / 'build/capstone/lib/libmruby.a'
    entry = source / 'build/capstone/mrbgems/mruby-bin-mruby/tools/mruby/mruby.o'
    if not original.is_file() or not entry.is_file():
        raise FileNotFoundError('mruby must first be built by its port recipe')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prepared = out / 'prepared'
    subprocess.run([sys.executable, str(HERE / 'prepare-mruby-capstone-gaps.py'),
                    '--source', str(source / 'src/gc.c'), '--out', str(prepared),
                    '--arm', args.arm], check=True)
    root = out / 'root'
    archive = root / 'src/mruby/build/capstone/lib/libmruby.a'
    binary = root / 'src/mruby/build/capstone/mrbgems/mruby-bin-mruby/tools/mruby/mruby.o'
    archive.parent.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    shutil.copy2(original, archive)
    binary.symlink_to(entry)
    obj = out / 'gc.o'
    command = [str(args.sdk_cc.resolve()), '-O1', '-std=gnu99',
               '-DPOOL_ALIGNMENT=16', '-DMRB_NO_DIRECT_THREADING',
               '-DMRB_NO_BOXING', '-DMRB_NO_IO_POPEN',
               '-DMRB_WITH_IO_PREAD_PWRITE', '-DMRB_STR_LENGTH_MAX=0',
               '-DMRB_GC_STUDY_GAPS', '-I', str(source / 'include'),
               '-I', str(source / 'build/capstone/include'), '-I', str(prepared)]
    if args.arm == 'sublet':
        command += ['-DMRB_CAPSTONE_GC_SUBLET', '-I', str(REPO / 'capstone/sublet')]
    command += ['-c', str(prepared / 'gc.c'), '-o', str(obj)]
    subprocess.run(command, check=True)
    subprocess.run([str(args.llvm_ar.resolve()), 'r', str(archive), str(obj)], check=True)
    (out / 'manifest.json').write_text(json.dumps({
        'arm': args.arm, 'source_gc_sha256': sha(source / 'src/gc.c'),
        'original_archive_sha256': sha(original), 'original_entry_sha256': sha(entry),
        'prepared_gc_sha256': sha(prepared / 'gc.c'),
        'observer_sha256': sha(prepared / 'mruby-gc-gap.inc'),
        'compiler_sha256': sha(args.sdk_cc.resolve()),
        'object_sha256': sha(obj), 'archive_sha256': sha(archive),
        'command': command,
    }, indent=2) + '\n')
    print(root)


if __name__ == '__main__':
    main()
