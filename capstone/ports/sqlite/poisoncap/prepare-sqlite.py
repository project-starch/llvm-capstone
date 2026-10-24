#!/usr/bin/env python3
"""Prepare PoisonCap's published SQLite for the engine corpus, outside the repository.

    prepare-sqlite.py WORK [--artifact ARCHIVE]

The source is the paper artifact pinned in
ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/platform.json (the same pin the
platform recipe builds QEMU, CheriBSD and LLVM from), downloaded and checked by
that recipe's own archive(), so there is one pin and one check. Its sqlite/ tree is
CTSRD-CHERI/sqlite 3.22.0-cheri plus PoisonCap's memsys5 (src/mem5.c) and driver
(test/speedtest1.c); nothing in it is changed here.

Output, under WORK:
  sqlite-src/                 the artifact's sqlite/ tree, as published
  amalgamation-published/     sqlite3.c and sqlite3.h generated from it, and sqlite3ext.h (mem5.c as
                              published: NESTED_TEMPORAL defined, i.e. poison on free,
                              quarantine and revocation)
  amalgamation-spatial/       the same with mem5.c's one `#define NESTED_TEMPORAL 1`
                              removed: the artifact's own switch, so the two arms differ
                              in exactly that line (bounds stay; free goes straight to
                              memsys5FreeUnsafe)
  prepare-sqlite.json         hashes of the archive, mem5.c and both amalgamations

The amalgamation is SQLite's own `make sqlite3.c` (tclsh and a host C compiler).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RECIPE = REPO / 'capstone/ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap'
# Read from the artifact on 2026-10-24 and matched against the source audit
# experiments/study/results/20260927-poisoncap-source-audit.json.
MEM5_SHA256 = 'd8cb4c0e688350992c480b7956b5b6d47496700447ded6386e4c02e12d338df8'
TEMPORAL_DEFINE = '#define NESTED_TEMPORAL 1\n'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe():
    spec = importlib.util.spec_from_file_location('poisoncap_prepare', RECIPE / 'prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_sqlite(archive, dest):
    if dest.exists():
        shutil.rmtree(dest)
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts[1:]
            if len(parts) < 2 or parts[0] != 'sqlite':
                continue
            if '..' in parts or member.name.startswith('/'):
                sys.exit(f'unsafe archive path: {member.name}')
            target = dest.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                target.chmod(member.mode & 0o755)
                count += 1
    if not count:
        sys.exit(f'no sqlite/ tree in {archive}')
    return count


# configure carries CFLAGS/CPPFLAGS feature macros into the parser and keyword generators, and
# OPTS into both, so an inherited environment could change the generated sqlite3.c. Run with a
# fixed, minimal one, and record the tools it used.
CLEAN_ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'HOME': '/nonexistent'}


def tool_versions():
    out = {}
    for name, argv in (('cc', ['cc', '--version']), ('make', ['make', '--version'])):
        done = subprocess.run(argv, env=CLEAN_ENV, capture_output=True, text=True)
        out[name] = done.stdout.splitlines()[0] if done.returncode == 0 else f'unavailable ({done.returncode})'
    done = subprocess.run(['tclsh'], input='puts [info patchlevel]', env=CLEAN_ENV,
                          capture_output=True, text=True)
    out['tclsh'] = done.stdout.strip() or 'unavailable'
    return out


def amalgamate(src, build, out):
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    for argv in ([str(src / 'configure'), '--disable-tcl', '--disable-readline'],
                 ['make', 'sqlite3.c']):
        done = subprocess.run(argv, cwd=build, env=CLEAN_ENV, capture_output=True, text=True)
        if done.returncode:
            sys.exit(f'{" ".join(argv)} failed in {build}:\n{done.stdout[-2000:]}{done.stderr[-2000:]}')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('sqlite3.c', 'sqlite3.h'):
        shutil.copyfile(build / name, out / name)
    # The official amalgamation ships the extension header beside them (spellfix.c needs it).
    shutil.copyfile(src / 'src/sqlite3ext.h', out / 'sqlite3ext.h')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('work', type=Path)
    parser.add_argument('--artifact', type=Path,
                        help='an already downloaded artifact archive (checked against the pin)')
    a = parser.parse_args()
    work = a.work.resolve()
    if work.is_relative_to(REPO):
        parser.error('WORK must be outside the repository')
    work.mkdir(parents=True, exist_ok=True)

    pin = json.loads((RECIPE / 'platform.json').read_text())['artifact']
    archive = a.artifact or work / 'artifact.tar.gz'
    recipe().archive(pin, archive)   # downloads if absent; refuses content that is not the pin

    src = work / 'sqlite-src'
    files = extract_sqlite(archive, src)
    mem5 = src / 'src/mem5.c'
    if sha256(mem5) != MEM5_SHA256:
        sys.exit(f'{mem5}: sha256 {sha256(mem5)} is not the audited {MEM5_SHA256}')

    amalgamate(src, work / 'build-published', work / 'amalgamation-published')

    # The spatial arm: the same tree with the one define removed, nothing else.
    spatial_src = work / 'sqlite-src-spatial'
    if spatial_src.exists():
        shutil.rmtree(spatial_src)
    shutil.copytree(src, spatial_src, symlinks=True)
    text = (spatial_src / 'src/mem5.c').read_text()
    if text.count(TEMPORAL_DEFINE) != 1:
        sys.exit(f'expected exactly one {TEMPORAL_DEFINE.strip()!r} in mem5.c')
    (spatial_src / 'src/mem5.c').write_text(text.replace(TEMPORAL_DEFINE, ''))
    amalgamate(spatial_src, work / 'build-spatial', work / 'amalgamation-spatial')

    record = {
        'artifact': {**pin, 'archive_sha256_local': sha256(archive)},
        'sqlite_files': files,
        'environment': CLEAN_ENV,
        'tools': tool_versions(),
        'mem5_sha256': MEM5_SHA256,
        'amalgamations': {
            arm: {name: sha256(work / f'amalgamation-{arm}' / name)
                  for name in ('sqlite3.c', 'sqlite3.h', 'sqlite3ext.h')}
            for arm in ('published', 'spatial')
        },
    }
    (work / 'prepare-sqlite.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
