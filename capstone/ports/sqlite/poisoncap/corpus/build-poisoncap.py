#!/usr/bin/env python3
"""Build the SQLite 3.22.0 engine corpus for PoisonCap's published platform (CheriBSD purecap).

    build-poisoncap.py --platform DIR --amalgamation DIR --ext-src DIR --out NEW_DIR [--no-lookaside]

The sibling of ports/sqlite/repro322/build-virtual.py, and deliberately built from it: the case
list, the tag-to-group manifest and the per-group feature flags come from that file's own
case_groups() and group_flags() (which read corpus322.sh), so the cases, triggers and flags cannot
drift between our arms and this one. What differs, and why:

  * the compiler is PoisonCap's own (DIR/work/sdk), for CheriBSD riscv64 purecap, with their
    image's rootfs as sysroot;
  * -DREPRO322_POISONCAP selects the block in repro322_common.h that hands SQLite the static heap
    without CHERI_PERM_POISON (what their malloc strips) and reports a fault's signal, code and PC;
  * CONFIG is FreeBSD's: SQLite asks malloc_usable_size() for a block's size (the reason
    build-virtual.py gives for its own CONFIG); memsys5 serves every case, so this only keeps
    the unused system-allocator path compiling;
  * -lpthread: SQLite is built thread-safe, as on our arms, and FreeBSD keeps pthreads out of libc;
  * the amalgamation is PoisonCap's (prepare-sqlite.py: published or spatial), staged as a copy
    with the same two source fixes every arm compiles.
The memsys5 arm only: SQLITE_CONFIG_HEAP over the static array, as --memsys5 there.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REPRO = REPO / 'capstone/ports/sqlite/repro322'
spec = importlib.util.spec_from_file_location('build_virtual', REPRO / 'build-virtual.py')
build_virtual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_virtual)

PURECAP = ['--target=riscv64-unknown-freebsd', '-march=rv64imafdcxcheri', '-mabi=l64pc128d',
           '-mno-relax', '-fuse-ld=lld']
CONFIG = ['-DSQLITE_MALLOCSIZE=malloc_usable_size', '-include', 'malloc_np.h']
FIXES = (('fts5-azarg-patch.py', '(azArg ? &azArg[1] : 0)'),
         ('fts3-doclist-null-patch.py', 'pDL->aAll ? &pDL->aAll[pDL->nAll] : 0'))


def run(command, log):
    with log.open('w') as stream:
        stream.write(' '.join(command) + '\n')
        stream.flush()
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--platform', type=Path, required=True, help='POISONCAP_PLATFORM (holds work/)')
    p.add_argument('--amalgamation', type=Path, required=True, help='dir with sqlite3.c and sqlite3.h')
    p.add_argument('--ext-src', type=Path, required=True, help='a SQLite 3.22.0 source tree (ext/)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--no-lookaside', action='store_true')
    p.add_argument('--opt', default='-O0')
    p.add_argument('--only', help='comma-separated case number prefixes')
    p.add_argument('--heap-bytes', type=int,
                   help='SENSITIVITY ONLY: every case\'s memsys5 heap at this size (a power of two), '
                        'overriding the corpus\'s sizes; never for the main row')
    a = p.parse_args()
    work = a.platform.resolve() / 'work'
    cc = work / 'sdk/bin/clang'
    sysroot = work / 'output/rootfs-riscv64-purecap'
    for need in (cc, sysroot, a.amalgamation / 'sqlite3.c', a.ext_src / 'ext'):
        if not need.exists():
            sys.exit(f'missing {need}')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'obj').mkdir()
    target = [*PURECAP, f'--sysroot={sysroot}']

    flags, cases = build_virtual.group_flags(), build_virtual.case_groups()
    if a.only:
        keep = set(a.only.split(','))
        cases = {c: v for c, v in cases.items() if c.name[:2] in keep}

    staged = a.out / 'src'
    staged.mkdir()
    source = staged / 'sqlite3.c'
    source.write_bytes((a.amalgamation / 'sqlite3.c').read_bytes())
    for script, marker in FIXES:
        if marker not in source.read_text(errors='replace'):
            done = subprocess.run([sys.executable, str(REPO / 'capstone/ports/sqlite' / script), str(source)],
                                  capture_output=True, text=True)
            if done.returncode:
                sys.exit(f'{script} failed: {done.stdout}{done.stderr}')

    engine = {}
    # core always: the controls link against it, whatever --only selects.
    for group in sorted({g for _, g, _ in cases.values()} | {'core'}):
        obj = a.out / 'obj' / f'sqlite3-{group}.o'
        command = [str(cc), *target, a.opt, *CONFIG, *flags[group], '-DSQLITE_ENABLE_MEMSYS5',
                   f'-I{a.amalgamation}', '-c', str(source), '-o', str(obj)]
        if run(command, a.out / f'engine-{group}.log'):
            engine[group] = obj
        print(f'  engine {group:<10} {"OK" if group in engine else "FAIL"}')

    extension = {'expertrem': ('expert/sqlite3expert.c', ['-Iexpert']),
                 'spellfixoom': ('misc/spellfix.c', ['-DSQLITE_CORE', '-Imisc'])}
    built, failed = [], []
    for case, (tag, group, extra) in sorted(cases.items()):
        if group not in engine:
            failed.append(case.name)
            continue
        body = (case / 'case.c').read_text()
        sources = [str(case / 'case.c')]
        if 'repro_memfs' in body:
            sources.append(str(REPRO / 'repro322_memfs.c'))
        extra_inc = []
        if tag in extension:
            relative, options = extension[tag]
            sources.append(str(a.ext_src / 'ext' / relative))
            extra_inc = [o if not o.startswith('-I') else f'-I{a.ext_src}/ext/{o[2:]}' for o in options]
        binary = a.out / case.name
        if a.heap_bytes:
            extra = [e for e in extra if not e.startswith('-DSQLITE_HEAP_SIZE=')] + [f'-DSQLITE_HEAP_SIZE={a.heap_bytes}']
        command = [str(cc), *target, a.opt, '-DREPRO322_VIRTUAL', '-DREPRO322_VIRTUAL_MEMSYS5',
                   '-DREPRO322_POISONCAP', *(['-DREPRO322_NO_LOOKASIDE'] if a.no_lookaside else []),
                   f'-I{REPO}/capstone/runtime/include', *CONFIG, *flags[group], *extra, *extra_inc,
                   f'-I{REPRO}', f'-I{a.amalgamation}', *sources, str(engine[group]), '-lm', '-lpthread',
                   '-o', str(binary)]
        ok = run(command, a.out / f'{case.name}.log')
        (built if ok else failed).append(case.name)
        print(f'  {case.name:<52} {"OK" if ok else "FAIL"} [{group}]')
    # The controls (controls/*.c): same harness, flags and core-group engine as the cases.
    controls = []
    for control in sorted((HERE / 'controls').glob('*.c')):
        if 'core' not in engine:
            failed.append(control.name)
            continue
        binary = a.out / control.stem
        command = [str(cc), *target, a.opt, '-DREPRO322_VIRTUAL', '-DREPRO322_VIRTUAL_MEMSYS5',
                   '-DREPRO322_POISONCAP', *(['-DREPRO322_NO_LOOKASIDE'] if a.no_lookaside else []),
                   f'-I{REPO}/capstone/runtime/include', *CONFIG, *flags['core'],
                   f'-I{REPRO}', f'-I{a.amalgamation}', str(control), str(engine['core']),
                   '-lm', '-lpthread', '-o', str(binary)]
        ok = run(command, a.out / f'{control.stem}.log')
        (controls if ok else failed).append(control.stem)
        print(f'  control {control.stem:<44} {"OK" if ok else "FAIL"}')
    (a.out / 'build.json').write_text(json.dumps(dict(
        compiler=str(cc), target=target, amalgamation=str(a.amalgamation), opt=a.opt, heap_bytes=a.heap_bytes,
        no_lookaside=a.no_lookaside, groups={g: flags[g] for g in sorted(engine)},
        cases={c.name: dict(tag=t, group=g, extra=e) for c, (t, g, e) in cases.items()},
        built=built, controls=controls, failed=failed), indent=2) + '\n')
    print(f'built={len(built)} failed={len(failed)} ({a.out})')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
