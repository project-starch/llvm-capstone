#!/usr/bin/env python3
"""Host ASan reference for the SQLite 3.22 engine corpus: does each case make its invalid access?

    run-host-asan.py --amalgamation <dir with sqlite3.c and sqlite3.h> --ext-src <3.22 source tree>
                     OUT [--jobs N]

memsys5 makes ASan blind (one array it never sees carved up), so this builds every case natively
with SQLite on the system allocator instead: no memsys5, lookaside off
(SQLITE_DEFAULT_LOOKASIDE=0,0), no page-cache bulk pool (SQLITE_DEFAULT_PCACHE_INITSZ=0), and a
4 GiB quarantine, so every SQLite object is its own ASan allocation and freed memory is not reissued
during a case. A report names the access, the free and the allocation; a SILENT case that RETURNED
made no invalid heap access on this trigger, which no allocator protection can then catch. The page
cache still recycles unpinned pages without a free, so a stale cached page stays invisible here.

Same staged amalgamation as ports/sqlite/repro322/build-virtual.py (sqlite3.c with
fts5-azarg-patch.py and fts3-doclist-null-patch.py), whose group and case tables this imports, and the same case sources. A case
that configures memsys5's heap itself (SQLITE_CONFIG_HEAP over sqlite_heap) is built as a VARIANT
with that one call replaced by SQLITE_OK, so its allocator wrapper sits on the system allocator;
the variant source is written beside the result and the row says so. Both controls must report
(heap-use-after-free, heap-buffer-overflow) or the run exits 75.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
REPRO = REPO / 'capstone/ports/sqlite/repro322'
spec = importlib.util.spec_from_file_location('build_virtual', REPRO / 'build-virtual.py')
bv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv)

CC = ['clang', '-O0', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-w']
# SQLITE_OMIT_COMPILEOPTION_DIAGS: the compile-option table cannot stringify "0,0".
ENGINE = [*bv.CONFIG, '-DSQLITE_DEFAULT_LOOKASIDE=0,0', '-DSQLITE_DEFAULT_PCACHE_INITSZ=0',
          '-DSQLITE_OMIT_COMPILEOPTION_DIAGS']
EXT = {'expertrem': ('expert/sqlite3expert.c', ['-Iexpert']),
       'spellfixoom': ('misc/spellfix.c', ['-DSQLITE_CORE', '-Imisc'])}
HEAP_CALL = re.compile(r'sqlite3_config\(\s*SQLITE_CONFIG_HEAP\s*,\s*sqlite_heap\s*,[^;]*\)(?=\s*;)')
CONTROLS = {'control_uaf_mem5': 'heap-use-after-free', 'control_bounds_mem5': 'heap-buffer-overflow'}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('output', type=Path)
    p.add_argument('--amalgamation', type=Path, required=True)
    p.add_argument('--ext-src', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=16)
    p.add_argument('--symbolizer', default='/usr/bin/llvm-symbolizer-21')
    a = p.parse_args()
    if a.output.exists():
        sys.exit(f'{a.output} exists; a reference run is never written over')
    out = a.output.resolve()
    (out / 'src').mkdir(parents=True)
    source = out / 'src/sqlite3-capstone.c'
    source.write_bytes((a.amalgamation / 'sqlite3.c').read_bytes())
    for fix in ('fts5-azarg-patch.py', 'fts3-doclist-null-patch.py'):
        fixed = subprocess.run([sys.executable, str(REPO / 'capstone/ports/sqlite' / fix),
                                str(source)], capture_output=True, text=True)
        if fixed.returncode:
            sys.exit(f'{fix} failed: {fixed.stdout}{fixed.stderr}')
    flags, cases = bv.group_flags(), bv.case_groups()

    def run(cmd, log):
        with open(log, 'w') as stream:
            stream.write(' '.join(map(str, cmd)) + '\n')
            stream.flush()
            return subprocess.run(list(map(str, cmd)), stdout=stream,
                                  stderr=subprocess.STDOUT).returncode

    def engine(group):
        obj = out / f'sqlite3-{group}.o'
        ok = run([*CC, *ENGINE, *flags[group], f'-I{a.amalgamation}', '-c', source, '-o', obj],
                 out / f'engine-{group}.log') == 0
        return group, obj if ok else None

    groups = sorted({g for _, g, _ in cases.values()} | {'core'})
    with ThreadPoolExecutor(a.jobs) as ex:
        objects = dict(ex.map(engine, groups))
    if not all(objects.values()):
        sys.exit(f'engine build failed: {[g for g, o in objects.items() if not o]}')

    def case(item):
        name, src, tag, group, extra = item
        text = Path(src).read_text()
        row = {}
        if HEAP_CALL.search(text):
            src = out / 'src' / f'{name}.variant.c'
            src.write_text(HEAP_CALL.sub('SQLITE_OK /* host ASan variant: system allocator */', text))
            row['variant'] = 'SQLITE_CONFIG_HEAP over sqlite_heap replaced by SQLITE_OK'
        sources, inc = [str(src)], []
        if 'repro_memfs' in text:
            sources.append(str(REPRO / 'repro322_memfs.c'))
        if tag in EXT:
            rel, opts = EXT[tag]
            sources.append(str(a.ext_src / 'ext' / rel))
            inc = [o if not o.startswith('-I') else f'-I{a.ext_src}/ext/{o[2:]}' for o in opts]
        image = out / f'{name}.bin'
        if run([*CC, '-DREPRO322_VIRTUAL', *bv.CONFIG, *flags[group], *extra, *inc, f'-I{REPRO}',
                f'-I{a.amalgamation}', *sources, objects[group], '-lm', '-lpthread', '-ldl',
                '-o', image], out / f'{name}.build.log'):
            return name, dict(row, result='BUILD-FAILED')
        env = dict(os.environ, ASAN_SYMBOLIZER_PATH=a.symbolizer,
                   ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:quarantine_size_mb=4096:'
                                'malloc_context_size=12')
        try:
            r = subprocess.run([str(image), tag], capture_output=True, text=True, timeout=600,
                               env=env, cwd=out)
        except subprocess.TimeoutExpired:
            return name, dict(row, result='TIMEOUT')
        log = (r.stdout + r.stderr).replace(str(Path.home()), '~')
        (out / f'{name}.run.log').write_text(log)
        m = re.search(r'ERROR: AddressSanitizer: ([\w-]+)', log)
        row.update(result=m.group(1) if m else 'SILENT', exit=r.returncode,
                   began=f'{tag} BEGIN' in log, returned=f'{tag} RETURNED' in log)
        if m:
            where = re.search(r'is located (\d+) bytes (inside of|after|before) (\d+)-byte region', log)
            if where:
                row['located'] = f'{where.group(1)} bytes {where.group(2)} {where.group(3)}-byte region'
            sections = re.split(r'\n(?=freed by thread|previously allocated by thread|allocated by thread)',
                                log[m.start():])
            names = lambda s: re.findall(r'#\d+ 0x[0-9a-f]+ in (\S+)', s)[:6]
            row['access'] = names(sections[0])
            for s in sections[1:]:
                row['freed_by' if s.startswith('freed') else 'allocated_by'] = names(s)
        return name, row

    items = [(d.name, str(d / 'case.c'), t, g, e) for d, (t, g, e) in sorted(cases.items())]
    items += [(c, str(CORPUS / 'controls' / f'{c}.c'), c, 'core', []) for c in CONTROLS]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = dict(ex.map(case, items))
    (out / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    for name, r in results.items():
        print(f"{name[:52]:<52} {r['result']:<22} {' <- '.join(r.get('access', [])[:3])}"
              f"{'  [variant]' if r.get('variant') else ''}")
    bad = [c for c, want in CONTROLS.items() if results[c]['result'] != want]
    if bad:
        print(f'CONTROL FAILED: {bad}; no row of this run is a reading')
        return 75
    return 0


if __name__ == '__main__':
    sys.exit(main())
