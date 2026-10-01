#!/usr/bin/env python3
"""Run SQLite 3.22.0, built as an ordinary program, in a capstone_vm guest and compare it with
the native oracle built from the same release (build-native.sh).

Checks, each against the native programs' output byte for byte:
  work     the shell on a database file: work.sql (types, floating point, transactions, an
           index, a recursive CTE, string functions, update/delete, integrity_check, .tables,
           .schema) and a second writer started with .system while the first holds
           BEGIN IMMEDIATE, which must be refused ("database is locked")
  again    a second shell on the same file: the data persisted and is intact
  speedtest1  the main testset at --size N on a database file; the per-phase result oracle
           (row count and FNV-1a hash of every result cell, ../study/speedtest1-oracle.patch)

  usage: run-app.py --state <vm state> --build <dir with sqlite3.dom, speedtest1.dom>
                    --native <dir with sqlite3, speedtest1> --share <the VM's share dir>
                    [--label L] [--size N] [--report out.json]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ORACLE = re.compile(r'^STUDY-ORACLE phase=\d+ rows=\d+ hash=[0-9a-f]{16}$', re.M)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def work_sql(sqlite3, db, helper):
    text = (HERE / 'work.sql.in').read_text()
    return text.replace('@SQLITE3@', sqlite3).replace('@DB@', db).replace('@HELPER@', helper)


def native(args, tmp):
    """The oracle's outputs, run in a scratch directory."""
    sqlite3 = str((args.native / 'sqlite3').resolve())
    (tmp / 'work.sql').write_text(work_sql(sqlite3, 'work.db', str(HERE / 'second-writer.sh')))
    out = {}
    for name, command, stdin in [
            ('work', [sqlite3, 'work.db'], tmp / 'work.sql'),
            ('again', [sqlite3, 'work.db'], HERE / 'again.sql'),
            ('speedtest1', [str((args.native / 'speedtest1').resolve()), '--size', str(args.size),
                            'st.db'], None)]:
        with open(stdin if stdin else os.devnull) as source:
            r = subprocess.run(command, cwd=tmp, stdin=source, capture_output=True, text=True,
                               timeout=3600)
        out[name] = (r.returncode, r.stdout, r.stderr)
    return out


def guest(args):
    share = args.share
    image = f'sqlite3-{args.label}.dom'
    speed = f'speedtest1-{args.label}.dom'
    shutil.copy2(args.build / 'sqlite3.dom', share / image)
    shutil.copy2(args.build / 'speedtest1.dom', share / speed)
    shutil.copy2(HERE / 'second-writer.sh', share / 'second-writer.sh')
    shutil.copy2(HERE / 'again.sql', share / 'again.sql')
    (share / f'work-{args.label}.sql').write_text(
        work_sql(f'/mnt/host/{image}', '/tmp/work.db', '/mnt/host/second-writer.sh'))
    cli = [sys.executable, '-m', 'capstone_vm', '--state', str(args.state), 'exec']
    runs = {
        'work': f'cd /tmp && rm -f work.db work.db-journal && /mnt/host/{image} work.db '
                f'< /mnt/host/work-{args.label}.sql',
        'again': f'cd /tmp && /mnt/host/{image} work.db < /mnt/host/again.sql',
        'speedtest1': f'cd /tmp && rm -f st.db st.db-journal && /mnt/host/{speed} '
                      f'--size {args.size} st.db',
    }
    out = {}
    for name, script in runs.items():
        # a domain fault's record goes to stderr only when asked for; stderr is shown on failure
        r = subprocess.run([*cli, 'sh', '-c', 'export CAPSTONE_EXEC_DIAGNOSTICS=1; ' + script],
                           capture_output=True, text=True, timeout=7200)
        out[name] = (r.returncode, r.stdout, r.stderr)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--share', type=Path, required=True)
    p.add_argument('--label', default='app')
    p.add_argument('--size', type=int, default=1)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='sqlite-322-native-') as tmp:
        expected = native(args, Path(tmp))
    got = guest(args)
    results, failed = {}, 0
    for name in ('work', 'again', 'speedtest1'):
        (erc, eout, _), (grc, gout, gerr) = expected[name], got[name]
        if name == 'speedtest1':
            eout = '\n'.join(ORACLE.findall(eout))
            gout = '\n'.join(ORACLE.findall(gout))
        ok = erc == grc == 0 and eout == gout and eout != ''
        failed += not ok
        results[name] = {
            'verdict': 'PASS' if ok else 'FAIL', 'status': grc, 'native_status': erc,
            'lines': len(eout.splitlines()),
            'output_sha256': hashlib.sha256(gout.encode()).hexdigest(),
            'native_output_sha256': hashlib.sha256(eout.encode()).hexdigest(),
        }
        if not ok:
            results[name]['guest_stderr_tail'] = gerr.strip().splitlines()[-5:]
            e_lines, g_lines = eout.splitlines(), gout.splitlines()
            diff = [f'{i}: native {a!r} guest {b!r}' for i, (a, b)
                    in enumerate(zip(e_lines, g_lines)) if a != b][:5]
            if len(e_lines) != len(g_lines):
                diff.append(f'native {len(e_lines)} lines, guest {len(g_lines)} lines')
            results[name]['first_differences'] = diff
        print(f'{name}: {results[name]["verdict"]} ({results[name]["lines"]} lines)')
        for line in results[name].get('guest_stderr_tail', []) + results[name].get('first_differences', []):
            print(f'  {line}')
    if args.report:
        args.report.write_text(json.dumps({
            'label': args.label, 'size': args.size,
            'images': {'sqlite3.dom': sha256(args.build / 'sqlite3.dom'),
                       'speedtest1.dom': sha256(args.build / 'speedtest1.dom')},
            'results': results}, indent=1) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
