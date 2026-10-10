#!/usr/bin/env python3
"""Build the SQLite 3.22.0 engine corpus for the VIRTUAL address space.

    build-virtual-sqlite322.py --sdk <virtual SDK> --amalgamation <dir with
        sqlite3.c and sqlite3.h> --out <new dir> [--memsys5 | --sublet] [--only 00,07]

WHY THIS CORPUS NEEDS ITS OWN BUILDER. The other C corpora are one translation
unit plus a driver, or one call into a port's CMake seam. These 33 cases are
neither: each one needs the AMALGAMATION COMPILED WITH ITS OWN FEATURE FLAGS --
FTS3, FTS3 with parentheses, FTS4, FTS5, JSON1, R*Tree, DBSTAT, the shared-cache
and incremental-blob switches -- because a case whose feature is missing does
not fail, it runs a different query and passes having tested nothing. The groups
and their flags are corpus322.sh's, and this tool READS THEM OUT OF THAT FILE
rather than restating them, so the two cannot drift. It also reads which group
each case belongs to from the same manifest, matching on the tag the case itself
declares in `REPRO322_MAIN`.

WHAT IT CHANGES AGAINST THE PHYSICAL ARM, and why:

  * `-DREPRO322_VIRTUAL` selects the libc scaffolding in repro322_common.h.
    The physical arm's output path is a shared hostcall region and its entry
    point is domain_main(); neither exists in a Linux process.
  * the freestanding math and qsort declarations (`repro322_math_decl.h`, and
    `repro322_fts_stubs.c` behind it) are NOT used. They exist because the
    freestanding amalgamation has no libm or stdlib declaration; musl supplies
    both, and injecting the shim would declare over the real headers.
  * two `ext` cases pull in their own extension translation unit, as
    corpus322.sh's own table does: `--ext-src` names the 3.22.0 FULL source
    tree whose `ext/expert` and `ext/misc` they need. Without it those two are
    reported as not built rather than quietly dropped.
  * one case, `07_129371553c_fts3_destroy_oom_stale_table`, engineers an OOM by
    reaching for `sqlite_heap` directly. That array only exists when memsys5 is
    configured, so the case is inherently nested and the platform arm reports
    it not-applicable instead of pretending to measure it.
  * `--memsys5` selects the nested arm, SQLITE_CONFIG_HEAP over one static
    array, and then the amalgamation is compiled with SQLITE_ENABLE_MEMSYS5.
    Without it SQLite allocates through the platform -- here the virtual
    runtime's own allocator, one bounded object per allocation and a revoke on
    every free. That pair IS the experiment; see repro322_common.h.
  * `--sublet` selects the THIRD arm: memsys5 again, over the same static
    array, under its own Sublet port. The amalgamation is copied and
    `ports/sqlite/sublet/sublet-3220000-memsys5.patch` is applied to the copy
    with `-F0`, so a source it was not written for is refused rather than
    patched somewhere near. The patch issues every block as a child lifetime
    of the pool (CDERIVE) and revokes it on free (CREVOKE); it includes
    <capstone/capability.h> from `capstone/runtime/include`.
  * `--pagecache` adds the THIRD nested allocator on top of `--sublet`: the page
    cache, whose pins are lifetimes (`sublet/sublet-3220000-pagecache.patch`).
    A page is unpinned without being freed or reassigned, so that end is one no
    malloc-level mechanism can see.

-O0 for both the amalgamation and the case, as corpus322.sh uses: these are
defects that turn on one specific access, and an optimiser that moves it moves
the evidence with it.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
CORPUS = REPO / 'capstone/bug-corpora/sqlite/engine-repros'
REPRO = REPO / 'capstone/ports/sqlite/repro322'
DRIVER = REPO / 'capstone/ports/sqlite/repro322/corpus322.sh'
# What configure defines on Linux and the one configuration the application
# port depends on: SQLite asks the heap for a block's size instead of putting an
# 8-byte header in front of every block, which would leave every structure
# holding a capability 8 bytes off its 16-byte boundary. Same two as
# ports/sqlite/app/build-domain.sh.
CONFIG = ['-DHAVE_MALLOC_H=1', '-DHAVE_MALLOC_USABLE_SIZE=1']


def group_flags():
    """{group: [flags]} read out of corpus322.sh's own group table."""
    text = DRIVER.read_text()
    body = re.search(r'case "\$GROUP" in\n(.*?)\n\s*\*\) echo "unknown group',
                     text, re.S)
    if not body:
        sys.exit(f'{DRIVER} no longer has a recognisable group table')
    groups = {}
    for name, flags in re.findall(r'^\s*([a-zA-Z0-9]+)\)\s*CF="([^"]*)"', body.group(1), re.M):
        # $MATHINC is the freestanding math/qsort shim; see the module docstring.
        groups[name] = [w for w in flags.split() if not w.startswith('$')]
    if not groups:
        sys.exit(f'{DRIVER} group table parsed to nothing')
    return groups


def case_groups():
    """{case directory: (group, extra flags)}, by the tag the case declares."""
    manifest = {}
    text = DRIVER.read_text()
    for group, block in re.findall(r'^\s*([a-zA-Z0-9]+)\)\s*cat <<EOF\n(.*?)^EOF\n',
                                   text, re.M | re.S):
        for line in block.splitlines():
            words = line.split()
            if len(words) >= 2:
                manifest[words[1]] = (group, words[2:])
    cases = {}
    for case in sorted(CORPUS.glob('[0-9][0-9]_*')):
        source = case / 'case.c'
        if not source.is_file():
            continue
        tag = re.search(r'REPRO322_MAIN\("([^"]+)"\)', source.read_text())
        if not tag:
            sys.exit(f'{source} declares no REPRO322_MAIN tag')
        tag = tag.group(1)
        # The R2 images are named fzNN_r2 in the manifest and fzNN by their own
        # REPRO322_MAIN, exactly as corpus322.sh's resolve_src allows for.
        entry = manifest.get(tag) or manifest.get(tag + '_r2')
        if not entry:
            sys.exit(f'{case.name} (tag {tag}) is in no group of {DRIVER.name}; '
                     f'a case with no feature flags would run a different query '
                     f'and pass having tested nothing')
        cases[case] = (tag, *entry)
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--amalgamation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--memsys5', action='store_true', help='The nested arm')
    p.add_argument('--sublet', action='store_true',
                   help='The nested arm with memsys5 under its own Sublet port')
    p.add_argument('--pagecache', action='store_true',
                   help="--sublet plus SQLite's THIRD nested allocator, the page cache, on its "
                        "own Sublet port: a page's data is a child lifetime of the pin, revoked "
                        "when the last reference goes away. The page is neither freed nor "
                        "reassigned there, so no malloc-level mechanism can see that end")
    p.add_argument('--no-lookaside', action='store_true',
                   help="With --sublet: turn SQLite's SECOND nested allocator off, so "
                        "the small allocations it would take go to memsys5, which this "
                        "arm's patch does cover. A probe, not an arm: it answers whether "
                        "a silence is the unprotected layer or memsys5.")
    p.add_argument('--ext-src', type=Path,
                   help='SQLite 3.22.0 full source tree, for the two ext cases')
    p.add_argument('--only', help='comma-separated case number prefixes')
    p.add_argument('--observe', type=Path, action='append', default=[],
                   help='An instrument, not a case: built against the same '
                        'engine object and the same arm as the cases, with the '
                        'core group flags, and reported separately')
    p.add_argument('--opt', default='-O0')
    a = p.parse_args()

    cc = a.sdk / 'capstone-cc'
    if not cc.is_file():
        sys.exit(f'no capstone-cc in {a.sdk}')
    if not (a.amalgamation / 'sqlite3.c').is_file():
        sys.exit(f'no sqlite3.c in {a.amalgamation}')
    flags, cases = group_flags(), case_groups()
    if a.only:
        keep = {w.zfill(2) for w in a.only.split(',')}
        cases = {c: v for c, v in cases.items() if c.name[:2] in keep}
    if not cases:
        sys.exit('no cases selected')

    if a.pagecache:
        a.sublet = True
    if a.sublet:
        a.memsys5 = True
    arm = ('memsys5-sublet-pagecache' if a.pagecache else
           'memsys5-sublet' if a.sublet else 'memsys5' if a.memsys5 else 'platform')
    if a.no_lookaside:
        arm += '-no-lookaside'
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'obj').mkdir(exist_ok=True)
    # Every arm compiles a COPY carrying the same source fixes: fts5-azarg-patch.py, which the
    # physical build also applies after adapt-sqlite-322.sh (build-sqlite-row322.sh), and
    # fts3-doclist-null-patch.py, the same NULL-plus-offset class in fts3EvalDlPhraseNext. Without the first
    # FTS5's default-tokenizer path computes &azArg[1] with azArg == NULL, which is CINCOFFSET on a
    # NULL capability and traps (cause 24) at CREATE VIRTUAL TABLE -- before any FTS5 case reaches
    # its defect. The virtual arms ran without it until 2026-10-10 and every FTS5 row faulted there
    # on all of them. The copy is named sqlite3-capstone.c, the name the Sublet patch is written for.
    staged = a.out / 'src'
    staged.mkdir(exist_ok=True)
    source = staged / 'sqlite3-capstone.c'
    source.write_bytes((a.amalgamation / 'sqlite3.c').read_bytes())
    if '(azArg ? &azArg[1] : 0)' not in source.read_text(errors='replace'):
        fixed = subprocess.run([sys.executable, str(REPO / 'capstone/ports/sqlite/fts5-azarg-patch.py'),
                                str(source)], capture_output=True, text=True)
        if fixed.returncode:
            sys.exit(f'fts5-azarg-patch.py failed: {fixed.stdout}{fixed.stderr}')
        print('  applied fts5-azarg-patch.py to the copy')
    if 'pDL->aAll ? &pDL->aAll[pDL->nAll] : 0' not in source.read_text(errors='replace'):
        fixed = subprocess.run([sys.executable, str(REPO / 'capstone/ports/sqlite/fts3-doclist-null-patch.py'),
                                str(source)], capture_output=True, text=True)
        if fixed.returncode:
            sys.exit(f'fts3-doclist-null-patch.py failed: {fixed.stdout}{fixed.stderr}')
        print('  applied fts3-doclist-null-patch.py to the copy')
    sublet_includes = []
    if a.sublet:
        patches = ['sublet/sublet-3220000-memsys5.patch']
        if a.pagecache:
            patches.append('sublet/sublet-3220000-pagecache.patch')
        for name in patches:
            patch = REPO / 'capstone/ports/sqlite' / name
            done = subprocess.run(['patch', '-s', '-F0', '-p1', '-d', str(staged)],
                                  stdin=patch.open('rb'), capture_output=True, text=True)
            if done.returncode:
                sys.exit(f'{patch.name} did not apply: {done.stdout}{done.stderr}')
            print(f'  patched {source.name} with {patch.name}')
        sublet_includes = [f'-I{REPO}/capstone/runtime/include']
    engine = {}
    for group in sorted({g for _, g, _ in cases.values()}):
        if group not in flags:
            sys.exit(f'group {group!r} has no flag line in {DRIVER.name}')
        obj = a.out / 'obj' / f'sqlite3-{group}.o'
        command = [str(cc), a.opt, *CONFIG, *flags[group],
                   *(['-DSQLITE_ENABLE_MEMSYS5'] if a.memsys5 else []),
                   *sublet_includes,
                   f'-I{a.amalgamation}', '-c', str(source), '-o', str(obj)]
        log = a.out / f'engine-{group}.log'
        with log.open('w') as stream:
            stream.write(' '.join(command) + '\n')
            stream.flush()
            if subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode:
                print(f'  engine {group:<10} FAIL ({log})')
                continue
        print(f'  engine {group:<10} OK  {obj.stat().st_size} bytes')
        engine[group] = obj

    built, failed, skipped = [], [], []
    for case, (tag, group, extra) in sorted(cases.items()):
        if group not in engine:
            print(f'  {case.name:<52} SKIP (no {group} engine object)')
            failed.append(case.name)
            continue
        body = (case / 'case.c').read_text()
        configures_heap = 'sqlite3_config(SQLITE_CONFIG_HEAP' in body
        if configures_heap and not a.memsys5:
            print(f'  {case.name:<52} not-applicable (reaches for sqlite_heap, '
                  f'which only the memsys5 arm has)')
            skipped.append(case.name)
            continue
        sources = [str(case / 'case.c')]
        extra_inc = []
        if 'repro_memfs' in body:
            sources.append(str(REPRO / 'repro322_memfs.c'))
        # The two extension cases, as corpus322.sh's own EXTRA_SRC table has them.
        extension = {'expertrem': ('expert/sqlite3expert.c', ['-Iexpert']),
                     'spellfixoom': ('misc/spellfix.c', ['-DSQLITE_CORE', '-Imisc'])}
        if tag in extension:
            if not a.ext_src:
                print(f'  {case.name:<52} not built (needs --ext-src for '
                      f'{extension[tag][0]})')
                failed.append(case.name)
                continue
            relative, options = extension[tag]
            sources.append(str(a.ext_src / 'ext' / relative))
            extra_inc = [o if not o.startswith('-I') else f'-I{a.ext_src}/ext/{o[2:]}'
                         for o in options]
        image = a.out / f'{case.name}.dom'
        command = [str(cc), a.opt, '-DREPRO322_VIRTUAL',
                   *(['-DREPRO322_VIRTUAL_MEMSYS5'] if a.memsys5 else []),
                   *(['-DREPRO322_VIRTUAL_SUBLET'] if a.sublet else []),
                   *(['-DREPRO322_NO_LOOKASIDE'] if a.no_lookaside else []),
                   *sublet_includes,
                   *CONFIG, *flags[group], *extra,
                   *extra_inc, f'-I{REPRO}', f'-I{a.amalgamation}',
                   *sources, str(engine[group]), '-lm', '-o', str(image)]
        log = a.out / f'{case.name}.log'
        with log.open('w') as stream:
            stream.write(' '.join(command) + '\n')
            stream.flush()
            ok = not subprocess.run(command, stdout=stream,
                                    stderr=subprocess.STDOUT).returncode
        if ok:
            print(f'  {case.name:<52} OK  {image.stat().st_size} bytes [{group}]')
            built.append(case.name)
        else:
            print(f'  {case.name:<52} FAIL [{group}]')
            sys.stdout.write(''.join(f'      {l}' for l in log.read_text().splitlines(True)
                                     if 'error' in l.lower())[:600])
            failed.append(case.name)
    observed = []
    for source in a.observe:
        group = 'core'
        if group not in engine:
            print(f'  {source.name:<52} SKIP (no {group} engine object)')
            continue
        image = a.out / f'{source.stem}.dom'
        command = [str(cc), a.opt, '-DREPRO322_VIRTUAL',
                   *(['-DREPRO322_VIRTUAL_MEMSYS5'] if a.memsys5 else []),
                   *(['-DREPRO322_VIRTUAL_SUBLET'] if a.sublet else []),
                   *sublet_includes,
                   *CONFIG, *flags[group], f'-I{REPO}/capstone/runtime/include',
                   f'-I{REPRO}', f'-I{a.amalgamation}',
                   str(source), str(engine[group]), '-lm', '-o', str(image)]
        log = a.out / f'{source.stem}.log'
        with log.open('w') as stream:
            stream.write(' '.join(command) + '\n')
            stream.flush()
            ok = not subprocess.run(command, stdout=stream,
                                    stderr=subprocess.STDOUT).returncode
        print(f'  {source.name:<52} {"OK " if ok else "FAIL"} (instrument)')
        if ok:
            observed.append(source.stem)
        else:
            sys.stdout.write(''.join(f'      {l}' for l in
                                     log.read_text().splitlines(True)
                                     if 'error' in l.lower())[:600])
    (a.out / 'build.json').write_text(json.dumps(dict(
        arm=arm, sdk=str(a.sdk), amalgamation=str(a.amalgamation), opt=a.opt,
        groups={g: flags[g] for g in sorted(engine)},
        cases={c.name: dict(tag=t, group=g, extra=e) for c, (t, g, e) in cases.items()},
        built=built, failed=failed, not_applicable=skipped,
        instruments=observed), indent=2) + '\n')
    print(f'arm={arm} built={len(built)} failed={len(failed)} '
          f'not-applicable={len(skipped)}  ({a.out})')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
