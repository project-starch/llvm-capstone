#!/usr/bin/env python3
"""Check that each shim reproduces the same memory-safety event as the real
defect it models.

Not a bug hunt: xlang/repro/<key>/ already reproduces every defect in the real
software. This checks the layer in between.

The real software is not the vehicle. Purecap mruby does run (mruby-port/
records the four changes that got it there), but the corpus pins NINE mruby
versions spanning 2017-2026 and only one is proven to boot, and rows 1-3 and 7
are Rust, for which no purecap toolchain exists at all. So each defect is
re-expressed as a SHIM: a small C program doing the same allocate / free /
offending-access with the same geometry. The shim is what runs under CHERI and
under Capstone, so it has to be faithful:

    real mruby (xlang/repro/4/asan.txt)  heap-use-after-free, WRITE of size 8,
    shim (mruby_vmstack_uaf.c)     16 bytes inside a 1024-byte region

The failure this prevents is a FALSE result, not a missing one. A shim that
quietly does nothing runs clean under CHERI and is scored MISS — which reads
as "CHERI failed to catch this defect". So this runs first and the CHERI boot
is refused if it fails.

It does NOT establish that the shim triggers the CVE or takes the same code
path. It is an equivalence of MEMORY EVENTS only, which suffices because a
capability verdict depends solely on what is allocated, what is freed, and
whether the offending access touches it. That argument dies if a shim's
ALLOCATOR stops matching reality — hence plain malloc/free, and hence the
upper-bound caveat in RESULTS.md.

Two checks:
  DEFECTS  same ASan error class, access kind and access size as the real trace.
  GUARDS   the two spatial defects also have IN-BOUNDS parameter regimes where
           nothing happens at all; those must be REFUSED at compile time, or
           they would score a spurious MISS. Checked in both directions.

Needs a C compiler with AddressSanitizer and nothing else — no CHERI, no SDK,
no QEMU, no root; seconds on x86-64. CHERI enters only downstream, in
compile-purecap.sh, run-in-guest.sh and the QEMU drivers.

Layout required, relative to this file:
    shims/<defect>.c         one per DEFECTS entry; stem MUST equal Defect.name
    shims/vm_stack_uaf.h     template shared by the six same-shape defects
    mock-mruby/mock_mruby.c  harness linked into every shim
Each compile is shim + harness -> one binary in --out, with a .log beside it.

Adding a shim: express geometry as explicit byte constants, never from
sizeof(void*) or a struct holding a pointer. This gate validates at 8-byte
pointers; the measurement runs at 16-byte capabilities, so anything else is
validated at one layout and measured at another.

  ./check_shim_fidelity.py                    # every defect, plus guards
  ./check_shim_fidelity.py mruby_vmstack_uaf  # one defect (name or table key)
  ./check_shim_fidelity.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "mock-mruby" / "mock_mruby.c"

# Defaults, overridable by flag. The environment variables are honoured because
# the shell callers (compile-purecap.sh, run-cheri-baseline-xlang.sh) set them.
DEFAULT_CC = os.environ.get("CC", "clang")
DEFAULT_OUT = Path(
    os.environ.get(
        "OUT", Path(os.environ.get("TMPDIR", "/tmp")) / "cheri-baseline-xlang-native"
    )
)

# Exit code a shim uses to say "I could not create the condition under test"
# (e.g. realloc grew the VM stack in place, so no stale pointer exists). It is
# a validity failure, never a MISS.
SHIM_INVALID = 66

ASAN_OPTIONS = ":".join(
    [
        "detect_leaks=0",  # shims leak by design; not the defect
        "detect_stack_use_after_return=1",  # rlua_escaped_handle_uar IS one
    ]
)

CFLAGS = ["-fsanitize=address", "-g", "-O0", "-fno-omit-frame-pointer"]


@dataclass(frozen=True)
class Defect:
    """One reproduced upstream defect, and what its shim must reproduce.

    `key` is the corpus/paper table number. It is kept because it is the join
    across five places — the paper's table, xlang/repro/<key>/, rows.tsv, and the
    "ROW <key>" lines the guest emits that the shared classify.py parses — but
    it names a table cell, not a defect, so everything a human reads uses
    `name` and `upstream` instead.
    """

    key: str  # corpus table number, e.g. "4"
    name: str  # what it IS, and the shim's filename stem
    upstream: str  # CVE or issue it reproduces
    asan_class: str  # e.g. heap-use-after-free
    access: str  # e.g. "READ of size 43"


@dataclass(frozen=True)
class Guard:
    """One compile that must be REFUSED — or must succeed — proving a spatial
    row cannot be built in a regime where its defect does not actually happen.

    The hazard: the two spatial defects only overflow above a trigger
    threshold. mruby_bytecode_overflow's corrupted instruction stores to
    register 38 of a 4-register frame, so where it lands depends on depth:

        depth 28 -> slot 126 -> byte 1008 -> INSIDE the 1024-byte VM stack
        depth 29 -> slot 129 -> byte 1032 -> past it

    Below the threshold there is no memory error at all. Such a build would run
    clean under CHERI and be scored MISS, which reads as "CHERI failed to catch
    this spatial bug" when in truth the bug never happened. So each shim carries
    a _Static_assert refusing the in-bounds regime, and these guards check that
    the refusal works.

    BOTH DIRECTIONS ARE REQUIRED. Only checking that in-bounds builds are
    refused would still pass if the shim were broken badly enough to compile
    for nothing at all; only checking that above-threshold builds succeed would
    not prove any guard exists. Together they pin the threshold exactly where
    the corpus says it is — depth 29 is the first crossing, and 80 outer locals
    is what the upstream trigger.rb uses — and catch a later edit that moves
    it.
    """

    label: str  # shown in the results table
    defect: str  # name of the shim to compile
    define: str  # -D that overrides the trigger parameter
    must_compile: bool  # True: above threshold. False: must be refused.

    @property
    def expected(self) -> str:
        return "built" if self.must_compile else "refused"


# Data tables: the column alignment IS the readability, so they are kept
# out of the formatter's hands. Note black only honours a bare marker.
# fmt: off
DEFECTS = [
    Defect("1",  "rlua_userdata_uaf",       "rlua#19",        "heap-use-after-free",    "READ of size 43"),
    Defect("2",  "rlua_escaped_handle_uar", "rlua#97",        "stack-use-after-return", "READ of size 8"),
    Defect("3",  "libpulse_iterator_uaf",   "GHSA-f56g",      "heap-use-after-free",    "READ of size 8"),
    Defect("4",  "mruby_vmstack_uaf",       "CVE-2022-1071",  "heap-use-after-free",    "WRITE of size 8"),
    Defect("5",  "mruby_kwargs_uaf",        "CVE-2022-1934",  "heap-use-after-free",    "READ of size 8"),
    Defect("6",  "mruby_bytecode_overflow", "CVE-2026-1979",  "heap-buffer-overflow",   "WRITE of size 8"),
    Defect("7",  "secp256k1_preallocated_uaf", "RUSTSEC-2022-0070", "heap-use-after-free", "READ of size 8"),
    Defect("8",  "mruby_values_at_uaf",     "CVE-2020-6838",     "heap-use-after-free",    "READ of size 8"),
    Defect("9",  "mruby_irep_pool_uaf",     "mruby#3829",     "heap-use-after-free",    "READ of size 4"),
    Defect("10", "mruby_range_uaf",         "CVE-2022-1106",  "heap-use-after-free",    "WRITE of size 8"),
    Defect("11", "mruby_upvar_overflow",    "CVE-2018-10191", "heap-buffer-overflow",   "READ of size 16"),
    Defect("12", "mruby_io_dataptr_uaf",    "CVE-2018-10199",     "heap-use-after-free",    "READ of size 4"),
    Defect("13", "mruby_hash_slice_uaf",    "CVE-2020-6840",     "heap-use-after-free",    "READ of size 16"),
    Defect("14", "mruby_gc_stackroot_uaf",  "CVE-2017-9527",     "heap-use-after-free",    "READ of size 4"),
    Defect("15", "mruby_sprintf_argv_uaf",  "mruby#3722",     "heap-use-after-free",    "READ of size 16"),
]

GUARDS = [
    # below threshold -> no overflow happens -> the build must be refused
    Guard("depth 28 (in bounds)",  "mruby_bytecode_overflow", "-DDEPTH=28",        must_compile=False),
    Guard("40 locals (in bounds)", "mruby_upvar_overflow",    "-DOUTER_LOCALS=40", must_compile=False),
    # at/above threshold -> the overflow is real -> the build must succeed
    Guard("depth 29 (crosses)",    "mruby_bytecode_overflow", "-DDEPTH=29",        must_compile=True),
    Guard("80 locals (pinned)",    "mruby_upvar_overflow",    "-DOUTER_LOCALS=80", must_compile=True),
]
# fmt: on

RULE = "-" * 100


@dataclass(frozen=True)
class Toolchain:
    """Where to compile with, and where to put the artefacts."""

    cc: str
    out: Path


def compile_shim(
    tc: Toolchain, defect: str, binary: Path, *extra: str
) -> subprocess.CompletedProcess:
    """Build one shim. Diagnostics come back on .stdout, status on .returncode.

    Keeping those two separate is the point: the bash version expressed this as
    `compile ... | grep -q`, and under `set -o pipefail` a refused build's
    non-zero status won over grep's match, so every refusal was misreported as
    a successful build.
    """
    return subprocess.run(
        [
            tc.cc,
            *CFLAGS,
            *extra,
            "-o",
            str(binary),
            str(HERE / "shims" / f"{defect}.c"),
            str(HARNESS),
        ],
        capture_output=True,
        text=True,
    )


def check_defect(tc: Toolchain, d: Defect) -> bool:
    def report(result: str) -> None:
        print(
            f"{d.name:<24} {d.upstream:<15} {d.asan_class:<22} "
            f"{d.access:<18} {result}"
        )

    binary, log = tc.out / d.name, tc.out / f"{d.name}.log"

    # Delete first: otherwise a build that silently produces nothing leaves the
    # PREVIOUS run's binary in place, we execute that, and report PASS for a
    # shim we did not build. Observed with --cc /bin/true.
    binary.unlink(missing_ok=True)

    build = compile_shim(tc, d.name, binary)
    if build.returncode != 0 or not binary.is_file():
        log.write_text(build.stdout + build.stderr)
        why = "BUILD-FAIL" if build.returncode else "BUILD-PRODUCED-NOTHING"
        report(f"{why} (see {log})")
        return False

    run = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        env={**os.environ, "ASAN_OPTIONS": ASAN_OPTIONS},
    )
    output = run.stdout + run.stderr
    log.write_text(output)

    if run.returncode == SHIM_INVALID:
        report("SHIM-INVALID (condition not created)")
        return False

    if d.asan_class in output and d.access in output:
        report("PASS")
        return True

    got = next(
        (
            line.split("AddressSanitizer: ")[1].split()[0]
            for line in output.splitlines()
            if "ERROR: AddressSanitizer: " in line
        ),
        "none",
    )
    report(f"FAIL (rc={run.returncode} got={got} see {log})")
    return False


def check_guard(tc: Toolchain, guard: Guard) -> bool:
    # A refused build is the shim's _Static_assert firing; its message names
    # the in-bounds regime, which is what we match on.
    (tc.out / "guard").unlink(missing_ok=True)
    build = compile_shim(tc, guard.defect, tc.out / "guard", guard.define)
    refused = "IN-BOUNDS regime" in build.stdout + build.stderr
    got = "refused" if refused else "built"
    ok = (not refused) == guard.must_compile
    print(
        f"  {guard.label:<28} {got} "
        f"{'PASS' if ok else f'FAIL (wanted {guard.expected})'}"
    )
    return ok


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="check_shim_fidelity.py",
        description="Check that each shim faithfully reproduces the real "
        "defect it models, by comparing its AddressSanitizer "
        "verdict against the corpus reproduction. Needs only a C "
        "compiler — no CHERI toolchain.",
        epilog="A defect may be given by name (mruby_vmstack_uaf) or by its "
        "corpus/paper table key (4). Naming any defect selects only "
        "those and skips the threshold guards, which are a property of "
        "the whole suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "defects",
        nargs="*",
        metavar="DEFECT",
        help="defect name or table key (default: all, plus guards)",
    )
    ap.add_argument(
        "--cc", default=DEFAULT_CC, help="C compiler to build the shims with"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for binaries and per-defect logs",
    )
    ap.add_argument("--list", action="store_true", help="list the defects and exit")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.list:
        for d in DEFECTS:
            print(f"{d.key:>3}  {d.name:<24} {d.upstream:<15} {d.asan_class}")
        return 0

    selectors = set(args.defects)
    unknown = selectors - {d.name for d in DEFECTS} - {d.key for d in DEFECTS}
    if unknown:
        print(
            f"unknown defect(s): {', '.join(sorted(unknown))}; try --list",
            file=sys.stderr,
        )
        return 2

    # Layout the script depends on: one shim per DEFECTS entry, named for the
    # defect, plus the harness every shim links against. Checked up front so a
    # mistyped table entry says so, instead of surfacing as a clang "no such
    # file or directory" twenty lines into the output.
    missing = [
        str(HERE / "shims" / f"{d.name}.c")
        for d in DEFECTS
        if not (HERE / "shims" / f"{d.name}.c").is_file()
    ]
    if not HARNESS.is_file():
        missing.append(str(HARNESS))
    if missing:
        print("missing source file(s):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("Defect.name must match shims/<name>.c", file=sys.stderr)
        return 2

    # The compiler is the other prerequisite. Probe it once, so a missing
    # clang says so instead of raising FileNotFoundError, and a compiler
    # without AddressSanitizer says THAT instead of failing every defect with
    # an identical confusing BUILD-FAIL.
    probe = OUT_PROBE = args.out / "probe.c"
    args.out.mkdir(parents=True, exist_ok=True)
    probe.write_text("int main(void){return 0;}\n")
    try:
        built = subprocess.run(
            [args.cc, *CFLAGS, "-o", str(args.out / "probe"), str(probe)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(f"compiler not found: {args.cc}  (set --cc or $CC)", file=sys.stderr)
        return 2
    finally:
        probe.unlink(missing_ok=True)
    if built.returncode != 0 or not (args.out / "probe").is_file():
        print(f"{args.cc} cannot build with AddressSanitizer:", file=sys.stderr)
        print((built.stderr or built.stdout).strip()[:400], file=sys.stderr)
        return 2
    (args.out / "probe").unlink(missing_ok=True)

    tc = Toolchain(cc=args.cc, out=args.out)
    tc.out.mkdir(parents=True, exist_ok=True)

    results: list[bool] = []
    print(f"{'DEFECT':<24} {'UPSTREAM':<15} {'CLASS':<22} " f"{'ACCESS':<18} RESULT")
    print(RULE)
    for d in DEFECTS:
        if not selectors or d.name in selectors or d.key in selectors:
            results.append(check_defect(tc, d))
    print(RULE)

    # Guards are a property of the whole suite, so they run only on a full pass.
    if not selectors:
        print("threshold guards (in-bounds regimes must not build):")
        results.extend(check_guard(tc, g) for g in GUARDS)
        (tc.out / "guard").unlink(missing_ok=True)
        print(RULE)

    passed = sum(results)
    failed = len(results) - passed
    print(f"pass={passed} fail={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
