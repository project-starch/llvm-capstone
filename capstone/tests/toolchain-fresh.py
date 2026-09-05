#!/usr/bin/env python3
"""Is the Capstone toolchain BINARY up to date with the compiler SOURCE it sits next to?

Why this exists (2026-09-05, C-44): dev was fast-forwarded onto a branch that changed the
jump-table lowering AND, in the same range, retired the `-fno-jump-tables` pin from the
SQLite build scripts. The main checkout's llvm/cmake-build-debug had last been built the
evening before. Running the new script with the old binary produced an image whose 26 jump
tables used the pre-change integer dispatch, which faults on every corpus file -- and it was
filed as a compiler defect in the new lowering, with a bisection that held "the compiler
binary" constant without asking which binary it was. A script change and a compiler change
that land together are only consistent together.

Definition. ninja's own dependency graph is asked (a dry run, ~0.5 s) what it would do for
the targets we actually use (llc, clang, lld -- ld.lld is a symlink the lld target creates,
not a target). The answer is classified:
  fresh          nothing pending
  fresh-relink   every pending step is part of the VCS-revision chain: a commit touches
                 .git/logs/HEAD, which regenerates VCSRevision.h / VCSVersion.inc, recompiles
                 the handful of objects embedding it (AsmPrinter, LTO, IRSymtab, the two
                 Version.cpp) and relinks; the code is the same
  STALE          any other compile or tablegen step is pending: a compiler source is newer
                 than its object, or a .td newer than its .inc

Exit status: 0 fresh / fresh-relink, 1 STALE, 2 cannot check (no ninja, no build dir,
ninja failed). "cannot check" is an error on purpose: silence must never read as fresh.
Identity is printed either way (CodeGen .so hash + mtime, clang's embedded revision) so a
result row can carry it.

Usage: toolchain-fresh.py [--build DIR] [--targets llc clang ld.lld] [--quiet-if-fresh]
  --build defaults to $CAPSTONE_LLVM_BUILD_DIR, else the parent of $CAPSTONE_LLVM_BIN.
"""
import argparse, hashlib, os, re, shutil, subprocess, sys, time

# Objects that embed the VCS revision and are legitimately recompiled after every commit.
VCS_OBJECTS = ('AsmPrinter/AsmPrinter.cpp.o', 'lib/LTO/LTO.cpp.o', 'Object/IRSymtab.cpp.o',
               'clang/lib/Basic/Version.cpp.o', 'lld/Common/Version.cpp.o')
VCS_GENERATED = ('VCSRevision.h', 'VCSVersion.inc')


def identity(build):
    so = os.path.join(build, 'lib', 'libLLVMCapstoneCodeGen.so')
    try:
        st = os.stat(so)
        h = hashlib.sha256(open(so, 'rb').read()).hexdigest()[:16]
        ident = f'libLLVMCapstoneCodeGen.so {h} built {time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}'
    except OSError:
        ident = 'libLLVMCapstoneCodeGen.so MISSING'
    try:
        v = subprocess.run([os.path.join(build, 'bin', 'clang'), '--version'], capture_output=True,
                           text=True, timeout=30).stdout
        m = re.search(r'\b([0-9a-f]{40})\b', v)
        ident += f'; clang embeds {m.group(1)[:12]}' if m else '; clang embeds no revision'
    except (OSError, subprocess.TimeoutExpired):
        ident += '; clang not runnable'
    return ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', default=os.environ.get('CAPSTONE_LLVM_BUILD_DIR') or
                    (os.path.dirname(os.environ['CAPSTONE_LLVM_BIN'].rstrip('/'))
                     if os.environ.get('CAPSTONE_LLVM_BIN') else None))
    ap.add_argument('--targets', nargs='+', default=['llc', 'clang', 'lld'])
    ap.add_argument('--quiet-if-fresh', action='store_true')
    a = ap.parse_args()
    if not a.build or not os.path.isfile(os.path.join(a.build, 'build.ninja')):
        print(f'toolchain-fresh: cannot check: no build.ninja under {a.build!r}', file=sys.stderr)
        return 2
    if not shutil.which('ninja'):
        print('toolchain-fresh: cannot check: ninja not on PATH', file=sys.stderr)
        return 2
    try:
        r = subprocess.run(['ninja', '-C', a.build, '-n', *a.targets], capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f'toolchain-fresh: cannot check: ninja dry run failed: {e}', file=sys.stderr)
        return 2
    if r.returncode != 0:
        print(f'toolchain-fresh: cannot check: ninja -n rc={r.returncode}: {r.stderr.strip()[:200]}', file=sys.stderr)
        return 2
    steps = [m.group(1) for m in re.finditer(r'^\[\d+/\d+\] (.*)$', r.stdout, re.M)]
    offending = []
    for s in steps:
        if s.startswith('Linking') or s.startswith('Creating') or s.startswith('Re-running'):
            continue
        if s.startswith('Generating') and s.endswith(VCS_GENERATED):
            continue
        if s.startswith('Building') and s.endswith(VCS_OBJECTS):
            continue
        offending.append(s)
    ident = identity(a.build)
    if offending:
        print(f'toolchain-fresh: STALE: {a.build}\n  {ident}\n  {len(steps)} ninja steps pending, '
              f'{len(offending)} of them compile/tablegen outside the revision chain, e.g.:', file=sys.stderr)
        for s in offending[:6]:
            print('    ' + re.sub(r'CMakeFiles/[^ ]*\.dir/', '', s)[:110], file=sys.stderr)
        print(f'  rebuild before producing any artifact: ninja -j90 -C {a.build} ' + ' '.join(a.targets)
              + '\n  (never while a QEMU suite or twin run is in flight)', file=sys.stderr)
        return 1
    if not a.quiet_if_fresh:
        note = f' ({len(steps)} pending steps are revision-chain relinks only)' if steps else ''
        print(f'toolchain-fresh: fresh: {ident}{note}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
