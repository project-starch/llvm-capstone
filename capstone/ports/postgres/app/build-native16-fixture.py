#!/usr/bin/env python3
"""Build a pristine PostgreSQL 17.5 cluster with Capstone's 16-byte MAXALIGN.

Source capstone/tests/capstone-test-env.sh before invoking this script. Build
sources and the cluster stay under CAPSTONE_TMP_ROOT; an existing root is never
reused. This is an input fixture builder, not a benchmark runner.

TRIMMING, for a board payload with a size cap (the vp2 DECLARED-bake cap, 26,000,000 bytes of firmware). Both are
applied with the cluster's own native build, so the catalog stays consistent:
  --drop-templates     DROP DATABASE template0 and template1 (single-user, on database postgres). Valid only for a
                       consumer that never opens or copies them: sql-repros' run-arm.py runs `postgres --single` on
                       database postgres, and no case creates a database.
  --wal-segsize-mb N   pg_resetwal --wal-segsize=N (initdb's segment is 16 MiB of mostly zeros).
  --trim-from ROOT     do not build: trim a COPY of a previous build root's cluster with that root's own binaries
                       (checked against its manifest), into --root. This is how the sql0 fixture was made from the
                       2026-10-18 native16 root.

REPRODUCIBILITY. PostgreSQL writes the time into three files: global/pg_control, pg_stat/pgstat.stat and the WAL
segment's checkpoint record. So two trims of the same cluster differ in those files and in any whole-tree hash
(run-arm.py's fixture_record included), while every catalog and data file is byte-identical. The manifest therefore
also records cluster_stable_tree_sha256, the tree digest without those time-bearing files: two trims of the same
source agree on it, and that is the identity to check a rebuilt fixture against.
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


# Files PostgreSQL stamps with the time, which make a cluster's whole-tree hash differ between identical runs.
TIME_BEARING = ('global/pg_control', 'pg_stat/', 'pg_wal/0')


def stable_tree_digest(root):
    h = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative.startswith(TIME_BEARING):
            continue
        if path.is_dir():
            h.update(b'd\0' + relative.encode() + b'\0')
        elif path.is_file() and not path.is_symlink():
            h.update(b'f\0' + relative.encode() + b'\0' + digest(path).encode() + b'\0')
        else:
            raise ValueError(f'nonregular cluster entry: {path}')
    return h.hexdigest()


def trim(install, cluster, drop_templates, wal_segsize_mb):
    """Apply the requested trims to CLUSTER with INSTALL's own binaries; returns the commands run."""
    done = []
    if drop_templates:
        sql = ("UPDATE pg_database SET datistemplate = false WHERE datname IN ('template0', 'template1');\n"
               'DROP DATABASE template0;\nDROP DATABASE template1;\n'
               'SELECT datname FROM pg_database ORDER BY 1;\n')
        command = [str(install/'bin/postgres'), '--single', '-D', str(cluster), '-c', 'timezone=GMT',
                   '-c', 'dynamic_shared_memory_type=sysv', 'postgres']
        r = subprocess.run(command, input=sql, capture_output=True, text=True, check=True)
        names = re.findall(r'datname = "([^"]+)"', r.stdout)
        if names != ['postgres'] or 'ERROR' in r.stdout + r.stderr:
            raise RuntimeError(f'template drop left {names}: {r.stdout[-400:]} {r.stderr[-400:]}')
        done.append({'command': command, 'stdin': sql})
    if wal_segsize_mb:
        command = [str(install/'bin/pg_resetwal'), f'--wal-segsize={wal_segsize_mb}', '-D', str(cluster)]
        subprocess.run(command, capture_output=True, text=True, check=True)
        control = subprocess.run([str(install/'bin/pg_controldata'), str(cluster)], capture_output=True,
                                 text=True, check=True).stdout
        m = re.search(r'^Bytes per WAL segment:\s+(\d+)$', control, re.M)
        if not m or int(m.group(1)) != wal_segsize_mb * 1048576:
            raise RuntimeError(f'pg_resetwal did not set a {wal_segsize_mb} MiB segment')
        done.append({'command': command})
    return done


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
    parser.add_argument('--drop-templates', action='store_true')
    parser.add_argument('--wal-segsize-mb', type=int, choices=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument('--trim-from', type=Path, metavar='ROOT')
    args = parser.parse_args()
    temporary = Path(os.environ['CAPSTONE_TMP_ROOT'])
    root = args.root or temporary/'pg-native16-fixture'
    if args.trim_from:
        return trim_existing(args, root, temporary, parser)
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
    trims = trim(install, root/'cluster', args.drop_templates, args.wal_segsize_mb)
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
        'cluster_stable_tree_sha256': stable_tree_digest(root/'cluster'),
        'cluster_config_sha256': digest(cluster_config),
        'trim': {'drop_templates': args.drop_templates, 'wal_segsize_mb': args.wal_segsize_mb, 'commands': trims},
        'cluster_policy': {'locale': 'C', 'timezone': 'GMT',
                           'dynamic_shared_memory_type': 'sysv'},
    }
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(root/'cluster')


def trim_existing(args, root, temporary, parser):
    """--trim-from: a trimmed copy of a previous root's cluster, made with that root's own binaries."""
    import shutil
    source = args.trim_from.resolve()
    if root.exists() or not root.resolve().is_relative_to(temporary.resolve()):
        parser.error('use a fresh --root under CAPSTONE_TMP_ROOT')
    if not (args.drop_templates or args.wal_segsize_mb):
        parser.error('--trim-from needs --drop-templates and/or --wal-segsize-mb')
    origin = json.loads((source/'manifest.json').read_text())
    install = source/'install'
    if digest(install/'bin/postgres') != origin['postgres_sha256']:
        raise ValueError(f'{install}/bin/postgres is not the binary {source}/manifest.json records')
    root.mkdir(parents=True)
    shutil.copytree(source/'cluster', root/'cluster', symlinks=True)
    trims = trim(install, root/'cluster', args.drop_templates, args.wal_segsize_mb)
    manifest = {**origin,
                'builder_sha256': digest(__file__),
                'trimmed_from': {'root': str(source), 'manifest_sha256': digest(source/'manifest.json'),
                                 'cluster_tree_sha256': origin['cluster_tree_sha256'],
                                 'cluster_stable_tree_sha256': stable_tree_digest(source/'cluster')},
                'trim': {'drop_templates': args.drop_templates, 'wal_segsize_mb': args.wal_segsize_mb,
                         'install': str(install), 'commands': trims},
                'cluster_tree_sha256': tree_digest(root/'cluster'),
                'cluster_stable_tree_sha256': stable_tree_digest(root/'cluster')}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(root/'cluster')


if __name__ == '__main__':
    main()
