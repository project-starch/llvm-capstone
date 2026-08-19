#!/usr/bin/env python3
"""Compile every musl .c file with the Capstone pure-cap clang and tally the result.

This is the runnable check for the port: it says how many of musl's sources the
compiler accepts, and it FAILS if that number regresses.

Three properties it deliberately has, because a survey that cannot fail is not
evidence (CLAUDE.md, "A CLEAN result is not evidence until the check is known to
fire"):

  * "no data" is an ERROR, not a zero. An unprepared tree, a missing clang or an
    empty file list exits non-zero and says where it looked.
  * POSITIVE CONTROLS. One file that must compile and one that must not. If
    either flips, the harness reports ERROR rather than a number, because a
    flipped control means the flags or the tree are not what we think.
  * Foreign-architecture sources are excluded explicitly and the count of what
    was dropped is printed, so the denominator is never quietly inflated.

Usage:  survey-musl-capstone.py <musl-src-dir> [--expect-ok N] [--jobs N]
"""

import argparse
import collections
import concurrent.futures
import os
import pathlib
import re
import subprocess
import sys

# Pinned baseline: the OK count this port is known to reach. Raise it when a
# patch improves things (that is the point), never lower it to make a run pass.
#
# LOWERED ONCE, deliberately, from 1270 to 1242: syscall_arg_t had to become
# `void *`, because no capability-carrying integer type exists on this target and
# an integer typedef silently stripped the tag off every pointer argument (see
# arch-capstone64/syscall_arch.h). That is a correctness fix, and it costs 28
# files. Trading 28 compilable files for a libc whose write() actually works is
# the right way round; trading the other way would have kept a bigger number and
# a broken ABI.
#
# RAISED to 1280 when the C-21 frontend fix landed (a negative integer constant
# cast to a capability crashed clang's constant evaluator). 38 files came back,
# including open.c, fopen.c and the *at() family.
#
# RAISED to 1321 once C-25 and C-26 moved onto the shared compiler branch. Both
# had been sitting on this port branch alone, so anything built from a sibling
# branch was measured against a compiler that did not have them. 41 more files.
BASELINE_OK = 1321

# A file that MUST compile, and a file that MUST NOT, with the reason it fails.
#
# strlen.c WAS the must-fail control and is retired here deliberately, which is
# what the old comment asked for: it now compiles. Its blocker was `s - a`, a
# pointer difference, not the `(uintptr_t)s % ALIGN` the comment blamed -- C-25
# fixed exactly that. A control tied to a bug we intend to fix expires the day
# the fix lands, so the replacement is tied to something we do NOT intend to fix.
#
# mallocng computes a table size from sizeof(void*) and underflows it to
# SIZE_MAX on a 16-byte pointer (meta.h:20, "array is too large"). That is a
# design assumption of mallocng, not a compiler defect, and this port uses a
# different allocator -- so it will keep failing, which is the property a
# negative control needs. All six mallocng files fail the same way; malloc.c is
# the one named here.
CONTROL_MUST_PASS = "src/stdlib/abs.c"
CONTROL_MUST_FAIL = "src/malloc/mallocng/malloc.c"

FOREIGN_ARCH_KEEP = {"riscv64", "capstone64", "generic"}


def compile_flags(musl: pathlib.Path) -> list[str]:
    return [
        "-target", "capstone64-unknown-elf",
        "-Xclang", "-target-feature", "-Xclang", "+m",
        # The A extension: musl's riscv64 atomics are lr.d/sc.d. Without this
        # 35 files fail on "instruction requires the following: 'Zalrsc'",
        # which is a missing flag, not a porting problem.
        "-Xclang", "-target-feature", "-Xclang", "+a",
        "-std=c99", "-nostdinc", "-ffreestanding", "-fno-builtin",
        # LOAD-BEARING, NOT COSMETIC. Without it the backend takes the
        # sibling-call path for `return callee(...)`, emits `cjalr ra` (WITH
        # link) and then drops the epilogue and the return entirely, so control
        # falls into whatever basic block follows. Reproduced at -O0/-O1/-O2 with
        # a 3-line function; measured in musl as write() succeeding and the
        # caller then executing its own default case and returning -ENOSYS.
        # build-sqlite-capstone.sh already carries this flag, undocumented.
        "-fno-optimize-sibling-calls",
        # ALSO LOAD-BEARING, and it was missing until 2026-08-15. A jump table is
        # .rodata reached through `scc gp`, which under the gp-captable ABI lands
        # outside gp's bounds and faults -- the same mechanism as C-4a's constant
        # pools, where ISSUES.md already records -fno-jump-tables as mandatory.
        # Every application build here carries it (build-file-probe.sh,
        # build-lua-probe.sh); the ARCHIVE did not, so 15 members shipped with
        # absolute-addressed switch tables, vfscanf and strftime among them. They
        # compile and archive cleanly and fault only when called, which is why a
        # survey counting compiles could never have caught it.
        "-fno-jump-tables",
        "-D_XOPEN_SOURCE=700",
        f"-I{musl}/arch/capstone64",
        f"-I{musl}/arch/generic",
        f"-I{musl}/obj/src/internal",
        f"-I{musl}/src/include",
        f"-I{musl}/src/internal",
        f"-I{musl}/obj/include",
        f"-I{musl}/include",
        "-O1", "-w",
        # int -> pointer conversion is an ERROR in current clang, not a warning,
        # so -w does not cover it. It is also exactly what this ABI requires:
        # syscall_arg_t is `void *` because no capability-carrying integer type
        # exists on this target, so every integer argument (fd, count, flags) is
        # deliberately cast to a pointer and rides in the cursor. Without this
        # 497 of 1361 files fail on that one diagnostic alone.
        "-Wno-int-conversion", "-Wno-error=int-conversion",
    ]


def bucket(message: str) -> str:
    # An LLVM assertion labels itself; use its text rather than a rule per case.
    # The assertion sits ~190 chars into the line (after the source path), which
    # is why the captured line must not be truncated before bucketing.
    assertion = re.search(r"Assertion `(.+?)' failed", message)
    if assertion:
        return "backend assert: " + assertion.group(1)[:80]
    if "materialize arbitrary" in message:
        return "backend: cannot materialize >64-bit constant (pointer via integer)"
    if "Cannot select" in message:
        return "backend: Cannot select (i128 integer op on a capability)"
    if "long_double_incorrectly" in message:
        return "long double: compiler and bits/float.h disagree"
    if "array is too large" in message:
        return "static assert: sizeof(void*) assumption (mallocng)"
    if "Unable to legalize non-vector shift" in message:
        return "backend: i128 shift (long double)"
    if "Too many bits for" in message:
        return "backend: APInt >64 bits (long double / capability constant)"
    if "exit code 134" in message:
        return "backend: crash, assertion text not captured"
    stripped = re.sub(r"^.*?error: ", "", message)
    return "other: " + stripped[:70]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("musl_dir")
    parser.add_argument("--expect-ok", type=int, default=BASELINE_OK)
    parser.add_argument("--jobs", type=int, default=min(16, (os.cpu_count() or 4)))
    parser.add_argument("--list-failures", action="store_true")
    parser.add_argument("--objects", metavar="DIR",
                        help="also keep the object files, so an archive can be "
                             "built from whatever compiles. The flags and the "
                             "file set then have exactly one definition, here.")
    parser.add_argument("--print-flags", action="store_true",
                        help="print the compile flags, one per line, and exit. "
                             "For the sources that are OURS rather than musl's "
                             "(libc-ext/): they must be built exactly as the "
                             "archive is, and two copies of this list would "
                             "drift.")
    args = parser.parse_args()

    musl = pathlib.Path(args.musl_dir).resolve()
    if args.print_flags:
        print("\n".join(compile_flags(musl)))
        return 0
    clang = os.environ.get("CAPSTONE_CLANG")
    if not clang or not pathlib.Path(clang).exists():
        print(f"ERROR: CAPSTONE_CLANG not set or missing: {clang!r}", file=sys.stderr)
        return 2
    for needed in ("arch/capstone64/syscall_arch.h", "obj/include/bits/alltypes.h"):
        if not (musl / needed).is_file():
            print(f"ERROR: tree not prepared, missing {musl / needed}\n"
                  f"       run prepare-musl-capstone.sh first", file=sys.stderr)
            return 2

    foreign = set(os.listdir(musl / "arch")) - FOREIGN_ARCH_KEEP
    every = sorted((musl / "src").rglob("*.c"))
    if not every:
        print(f"ERROR: no .c files found under {musl / 'src'}", file=sys.stderr)
        return 2
    files = [f for f in every
             if not (set(f.relative_to(musl).parts) & foreign)]
    dropped = len(every) - len(files)
    if not files:
        print(f"ERROR: every source was filtered out as foreign-arch", file=sys.stderr)
        return 2

    flags = compile_flags(musl)

    objdir = pathlib.Path(args.objects).resolve() if args.objects else None

    def compile_one(path: pathlib.Path) -> tuple[str, bool, str]:
        rel = str(path.relative_to(musl))
        if objdir is None:
            out = "/dev/null"
        else:
            obj = objdir / (rel[:-2].replace("/", "_") + ".o")
            obj.parent.mkdir(parents=True, exist_ok=True)
            out = str(obj)
        done = subprocess.run([clang, *flags, "-c", str(path), "-o", out],
                              capture_output=True, text=True)
        if done.returncode == 0:
            return rel, True, ""
        # An LLVM assertion failure prints "Assertion `...' failed." and only
        # then a generic "error: ... exit code 134". Matching on "error:" alone
        # buckets every backend crash as "unknown", which is how 62 files ended
        # up in one meaningless pile on the first run. Prefer the specific line.
        lines = done.stdout.splitlines() + done.stderr.splitlines()
        first = next((l for l in lines
                      if "error in backend:" in l or "Assertion" in l
                      or "UNREACHABLE" in l), None)
        if first is None:
            first = next((l for l in lines if "error:" in l), "<no error line>")
        return rel, False, first[:400]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(compile_one, files))

    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]
    by_path = {r[0]: r[1] for r in results}

    print(f"musl tree      {musl}")
    print(f"compiler       {clang}")
    print(f"surveyed       {len(files)} files ({dropped} foreign-arch sources excluded)")
    print(f"compiled       {len(ok)}")
    print(f"failed         {len(bad)}   ({100 * len(ok) / len(files):.1f}% ok)")

    print("\nfailures by cause:")
    for cause, count in collections.Counter(bucket(r[2]) for r in bad).most_common():
        print(f"  {count:5d}  {cause}")
    print("\nfailures by directory:")
    for where, count in collections.Counter(
            str(pathlib.PurePath(r[0]).parent) for r in bad).most_common(15):
        print(f"  {count:5d}  {where}")
    if args.list_failures:
        print("\nfailing files:")
        for r in sorted(bad):
            print(f"  {r[0]}\n      {r[2]}")

    status = 0
    for control, must_pass in ((CONTROL_MUST_PASS, True), (CONTROL_MUST_FAIL, False)):
        got = by_path.get(control)
        if got is None:
            print(f"\nERROR: control {control} was not surveyed at all", file=sys.stderr)
            status = 2
        elif got is not must_pass:
            print(f"\nERROR: control {control} expected "
                  f"{'PASS' if must_pass else 'FAIL'} but got "
                  f"{'PASS' if got else 'FAIL'}.\n"
                  f"       The harness is not measuring what it claims. If this is an "
                  f"intended improvement, retire the control deliberately.",
                  file=sys.stderr)
            status = 2
    if status:
        return status

    if len(ok) < args.expect_ok:
        print(f"\nREGRESSION: {len(ok)} compiled, baseline is {args.expect_ok}",
              file=sys.stderr)
        return 1
    if len(ok) > args.expect_ok:
        print(f"\nIMPROVED: {len(ok)} compiled, baseline is {args.expect_ok}. "
              f"Raise BASELINE_OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
