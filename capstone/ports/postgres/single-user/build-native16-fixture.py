#!/usr/bin/env python3
"""Build a pristine PostgreSQL 17.5 cluster with Capstone's 16-byte MAXALIGN.

Source capstone/tests/capstone-test-env.sh before invoking this script. Build
sources and the cluster stay under CAPSTONE_TMP_ROOT; an existing root is never
reused. This is an input fixture builder, not a benchmark runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.request


ARCHIVE_SHA256 = 'fcb7ab38e23b264d1902cb25e6adafb4525a6ebcbd015434aeef9eda80f528d8'
ARCHIVE_URL = 'https://ftp.postgresql.org/pub/source/v17.5/postgresql-17.5.tar.bz2'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree_digest(root):
    h = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix().encode()
        if path.is_dir():
            h.update(b'd\0' + relative + b'\0')
        elif path.is_file() and not path.is_symlink():
            h.update(b'f\0' + relative + b'\0' + digest(path).encode() + b'\0')
        else:
            raise ValueError(f'nonregular cluster entry: {path}')
    return h.hexdigest()


def replace_once(path, pattern, replacement):
    text = path.read_text()
    changed, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'expected one {pattern!r} in {path}, found {count}')
    path.write_text(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--jobs', type=int, default=12)
    args = parser.parse_args()
    temporary = Path(os.environ['CAPSTONE_TMP_ROOT'])
    root = args.root or temporary/'pg-native16-fixture'
    archive = args.archive or temporary/'pg-mmgr-host/pg.tar.bz2'
    if root.exists() or args.jobs < 1 or not root.resolve().is_relative_to(temporary.resolve()):
        parser.error('use a fresh root under CAPSTONE_TMP_ROOT and positive --jobs')
    if not archive.resolve().is_relative_to(temporary.resolve()):
        parser.error('keep fetched PostgreSQL sources under CAPSTONE_TMP_ROOT')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError('PostgreSQL 17.5 archive SHA-256 mismatch')
    root.mkdir(parents=True)
    with tarfile.open(archive) as source_tar:
        source_tar.extractall(root, filter='data')
    source = root/'postgresql-17.5'
    port = Path(__file__).resolve().parent
    patch = port/'patches/0004-memorychunk-header-16.patch'
    with patch.open('rb') as stream:
        subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '-p1', '-d', str(source)],
                       stdin=stream, check=True, stdout=subprocess.DEVNULL)

    install = root/'install'
    commands = [
        [str(source/'configure'), f'--prefix={install}', '--without-readline',
         '--without-zlib', '--without-icu'],
        ['make', '-j'+str(args.jobs)],
        ['make', 'install'],
        [str(install/'bin/initdb'), '-D', str(root/'cluster'), '--locale=C',
         '--encoding=UTF8', '-A', 'trust'],
    ]
    environment = {**os.environ, 'CFLAGS': '-O1', 'LC_ALL': 'C'}
    with (root/'configure.log').open('w') as log:
        subprocess.run(commands[0], cwd=source, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    replace_once(source/'src/include/pg_config.h', r'^#define MAXIMUM_ALIGNOF 8$',
                 '#define MAXIMUM_ALIGNOF 16')
    for index, name in ((1, 'make.log'), (2, 'install.log'), (3, 'initdb.log')):
        with (root/name).open('w') as log:
            subprocess.run(commands[index], cwd=source, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    cluster_config = root/'cluster/postgresql.conf'
    for name in ('log_timezone', 'timezone'):
        replace_once(cluster_config, rf"^{name} = '[^']+'$", f"{name} = 'GMT'")
    replace_once(cluster_config, r'^dynamic_shared_memory_type = posix\b',
                 'dynamic_shared_memory_type = sysv')
    manifest = {
        'schema': 1,
        'postgres_version': '17.5',
        'source_archive_sha256': digest(archive),
        'compat_patch_sha256': digest(patch),
        'builder_sha256': digest(__file__),
        'compiler_flags': '-O1',
        'maxalign_bytes': 16,
        'configure': commands[0],
        'initdb': commands[3],
        'initdb_sha256': digest(install/'bin/initdb'),
        'postgres_sha256': digest(install/'bin/postgres'),
        'cluster_tree_sha256': tree_digest(root/'cluster'),
        'cluster_config_sha256': digest(cluster_config),
        'cluster_policy': {'locale': 'C', 'timezone': 'GMT',
                           'dynamic_shared_memory_type': 'sysv'},
    }
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(root/'cluster')


if __name__ == '__main__':
    main()
