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

# Every binary the Capstone lit suites actually invoke, not just the compiler.
#
# THIS LIST USED TO BE ('llc', 'clang', 'lld'), AND THAT WAS A HOLE IN THE GATE ITSELF.
# This is a shared-library build, so a partial rebuild leaves llc fresh while the tools a test
# pipes into are not. Measured 2026-09-10 after a merge: rebuilding only llc/clang/lld/llvm-mc
# left llvm-readelf at its six-day-old build, and MC/Capstone/obj-relocs-cap-constant.ll failed
# with `Machine: 103` -- EM_CR, an unrelated vendor's e_machine -- which reads as an object-writer
# defect in whatever was merged last rather than as a stale reader. This gate reported FRESH
# throughout, because it only ever asked ninja about those three targets.
#
# A PARTIAL REBUILD IS WORSE THAN NO REBUILD. No rebuild leaves a consistently old toolchain whose
# failures are attributable to being old; a partial one leaves an INCONSISTENT toolchain whose
# failures point at whatever changed most recently -- after a merge, someone else's work.
#
# Tell for this class, worth knowing: a backend that got its own e_machine wrong would emit a wrong
# value of its own family, or zero. It would never emit another vendor's specific, valid identifier.
# A nonsensical-but-valid-elsewhere constant means a stale reader, not a live defect.
DEFAULT_TARGETS = ('llc', 'clang', 'lld', 'llvm-mc', 'llvm-readobj', 'llvm-readelf',
                   'llvm-objdump', 'FileCheck', 'not', 'split-file')

# Scanned to keep DEFAULT_TARGETS honest as the tests change; see drift_check.
SUITE_DIRS = ('llvm/test/MC/Capstone', 'llvm/test/CodeGen/Capstone')
RUN_TOOL_RE = re.compile(r'\b(llc|clang|ld\.lld|lld|llvm-[a-z0-9-]+|FileCheck|split-file|not)\b')
# Aliases: the tool a test runs vs the ninja target that builds it.
TOOL_TARGET = {'ld.lld': 'lld'}


def drift_check(repo_root, suites, targets):
    """Tools the suites invoke that `targets` does not cover.

    A hardcoded list silently falls behind the tests it is meant to cover, so this makes that
    failure loud instead of letting the gate narrow itself over time. Returns None when no suite
    directory exists (nothing to check), else a sorted list of uncovered tool names.
    """
    seen, found_any = set(), False
    for rel in suites:
        d = os.path.join(repo_root, rel)
        if not os.path.isdir(d):
            continue
        found_any = True
        for dirpath, _, names in os.walk(d):
            for n in names:
                try:
                    with open(os.path.join(dirpath, n), 'r', errors='ignore') as fh:
                        for line in fh:
                            if 'RUN:' not in line:
                                continue
                            seen.update(RUN_TOOL_RE.findall(line.split('RUN:', 1)[1]))
                except OSError:
                    continue
    if not found_any:
        return None
    covered = set(targets)
    return sorted(t for t in seen if TOOL_TARGET.get(t, t) not in covered)



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
    ap.add_argument('--targets', nargs='+', default=list(DEFAULT_TARGETS))
    ap.add_argument('--suite', nargs='+', default=None,
                    help='lit dirs whose RUN lines are scanned for tools not covered by --targets')
    ap.add_argument('--no-drift-check', action='store_true')
    ap.add_argument('--quiet-if-fresh', action='store_true')
    a = ap.parse_args()
    if not a.build or not os.path.isfile(os.path.join(a.build, 'build.ninja')):
        print(f'toolchain-fresh: cannot check: no build.ninja under {a.build!r}', file=sys.stderr)
        return 2
    if not a.no_drift_check:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        missing = drift_check(root, a.suite or SUITE_DIRS, a.targets)
        if missing:
            print('toolchain-fresh: cannot check: the lit suites invoke tools this gate does not '
                  f'verify: {" ".join(missing)}. Add them to DEFAULT_TARGETS (or pass --targets); '
                  'reporting fresh without them is how a stale reader reads as a codegen bug.',
                  file=sys.stderr)
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
        # ninja prints objects as <dir>/CMakeFiles/<target>.dir/<file>.o; drop the CMakeFiles
        # segment before matching, or the revision-chain objects never match and every commit
        # without a rebuild reads STALE (it did, 2026-09-07: AsmPrinter/LTO/IRSymtab/Version).
        if s.startswith('Building') and re.sub(r'CMakeFiles/[^/]+\.dir/', '', s).endswith(VCS_OBJECTS):
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
