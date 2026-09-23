#!/usr/bin/env python3
"""Build CPython's own object list with the Capstone pure-cap clang and tally it.

Input is a tree prepare-cpython-capstone.sh configured. The survey asks CPython's
Makefile which objects the interpreter consists of, deletes them, runs `make -k`
on exactly those, and reads back one line per compile from capstone-cc's log.
Every object is compiled with the flags CPython itself chose for it (-std=c11,
-O3, -DPy_BUILD_CORE*, per-module include paths), which a hand-written loop over
*.c files would not reproduce.

What it deliberately does, because a survey that cannot fail is not evidence
(CLAUDE.md, "A CLEAN result is not evidence until the check is known to fire"):

  * The denominator is the Makefile's list, not what happened to get compiled.
    An object make never attempted -- because a generated header it needs
    failed first -- is NOTRUN, printed and counted, never silently dropped.
  * "No data" is an ERROR: no object list, no log lines, or a Makefile whose CC
    is not capstone-cc all exit 2 and say where they looked.
  * POSITIVE CONTROLS: one object that must compile and one that must not, each
    with the reason. A flipped control is an ERROR rather than a number.
  * Modules configure left out (a library this target lacks, or disabled) are
    listed with their state, since they are outside the denominator by choice.

Usage:  survey-cpython-capstone.py <build-dir> [--expect-ok N] [--jobs N]
                                   [--list-failures]
"""

import argparse
import collections
import os
import pathlib
import re
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
# Pinned baseline: the OK count this port is known to reach. Raise it when a
# change improves things, never lower it to make a run pass.
# SET 2026-09-23: 222 of 253, clang d030df93d4a4 (= origin/dev), patches 0001-0006.
# RAISED 2026-09-23: 250 of 253, clang 97d5978c6402 (dev + the C-50, C-51, C-52,
# C-54 and C-55 fixes), patches 0001-0009. The three left are _multiprocessing's
# two objects (no integer is as wide as a pointer) and getbuildinfo.o (NOTRUN).
# A compiler without those fixes is expected to fall below this and fail.
BASELINE_OK = 250

# MUST_PASS: a core object that includes every header patches 0001-0006 touch
# (longobject.h, pycore_pymem.h, pycore_obmalloc.h, pycore_pyhash.h,
# pycore_qsbr.h, pyport.h) and, built with -g -O3, failed on C-50 until the
# compiler fix. If any of those regresses, this flips first.
CONTROL_MUST_PASS = "Objects/boolobject.o"
# MUST_FAIL: a synthetic object, toolchain/survey-control-must-fail.c, compiled
# after the make run through the same capstone-cc and log, with CPython's core
# flags. It asserts that some integer is as wide as a pointer, which is false
# only on a capability target. Until 2026-09-23 the control was
# Objects/longobject.o, failing on that same fact; patch 0007 made it compile,
# and a port on its way to linking should not keep a real object broken.
CONTROL_MUST_FAIL = "survey-control-must-fail.o"
CONTROL_MUST_FAIL_SRC = HERE / "toolchain" / "survey-control-must-fail.c"

# Variables of CPython's generated Makefile that together are the interpreter:
# libpython's objects, the program's main, and the bundled static libraries the
# enabled modules link (libmpdec for _decimal, expat for pyexpat, HACL* for the
# SHA-2 family). The group is how the report attributes a failure.
GROUPS = [
    ("parser", "PARSER_OBJS"),
    ("objects", "OBJECT_OBJS"),
    ("python", "PYTHON_OBJS"),
    ("core-modules", "MODULE_OBJS"),
    ("stdlib-modules", "MODOBJS"),
    ("lib:mpdec", "LIBMPDEC_OBJS"),
    ("lib:expat", "LIBEXPAT_OBJS"),
    ("lib:hacl-sha2", "LIBHACL_SHA2_OBJS"),
]
EXTRA = [("core-modules", ["Modules/getbuildinfo.o", "Modules/getpath.o"]),
         ("python", ["Python/frozen.o"]),
         ("program", ["Programs/python.o"])]


def bucket(message: str) -> str:
    assertion = re.search(r"Assertion `(.+?)' failed", message)
    if assertion:
        return "backend assert: " + assertion.group(1)[:80]
    if "materialize arbitrary" in message:
        return "backend: cannot materialize >64-bit constant (pointer via integer)"
    if "Cannot select" in message:
        # "t13: c128 = ..." names a DAG node by its number in one function;
        # the same defect in two files differs only there.
        what = re.sub(r"\bt\d+:", "tN:", message.split("Cannot select:", 1)[-1].strip())
        return "backend: Cannot select " + what[:60]
    if "error in backend" in message:
        return "backend: " + message.split("error in backend:", 1)[-1].strip()[:70]
    if "UNREACHABLE executed" in message:
        return "backend unreachable: " + message.split("UNREACHABLE executed", 1)[-1].strip()[:70]
    if "static assertion failed" in message or "static_assert" in message:
        return "static assert: " + re.sub(r"^.*?error: ", "", message)[:70]
    if "array is too large" in message or "negative size" in message:
        return "static assert via array size: " + re.sub(r"^.*?error: ", "", message)[:60]
    if "incompatible integer to pointer" in message or "incompatible pointer to integer" in message:
        return "int-conversion: pointer and integer mixed"
    if "<no error line" in message:
        return "no diagnostic captured (see the .o.err file)"
    return "other: " + re.sub(r"^.*?error: ", "", message)[:80]


ERROR_SITE = re.compile(r"^(?P<file>/[^:]+):(?P<line>\d+):\d+: error: (?P<msg>.*)$")


def error_sites(err_file: pathlib.Path, source_root: pathlib.Path) -> set[tuple[str, str]]:
    """Every distinct error in one object's diagnostics, as (where, what).

    The first error alone under-reports: a file that includes two broken headers
    is attributed to whichever comes first, and the second stays invisible until
    the first is patched. Counting every site shows the whole layer at once.
    A backend failure has no source location; it is keyed by its assertion.
    """
    sites = set()
    if not err_file.is_file():
        return sites
    for line in err_file.read_text(errors="replace").splitlines():
        m = ERROR_SITE.match(line)
        if m:
            where = m.group("file")
            try:
                where = str(pathlib.Path(where).relative_to(source_root))
            except ValueError:
                pass
            sites.add((f"{where}:{m.group('line')}", m.group("msg")[:90]))
            continue
        a = re.search(r"Assertion `(.+?)' failed", line)
        if a:
            sites.add(("<backend>", "assert " + a.group(1)[:90]))
        elif "error in backend:" in line or "UNREACHABLE executed" in line:
            sites.add(("<backend>", re.sub(r"\bt\d+:", "tN:", line.strip())[:100]))
    return sites


def make_var(build: pathlib.Path, name: str) -> list[str]:
    printer = build / "survey-print.mk"
    printer.write_text("survey-print-%:\n\t@echo $($*)\n")
    done = subprocess.run(["make", "-s", "-f", "Makefile", "-f", str(printer),
                           f"survey-print-{name}"], cwd=build,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"ERROR: cannot read {name} from {build}/Makefile:\n{done.stderr}")
    return done.stdout.split()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir")
    parser.add_argument("--expect-ok", type=int, default=None,
                        help=f"default {BASELINE_OK}, the baseline at CPython's own -O3; "
                             f"with --opt there is no default and no regression gate")
    parser.add_argument("--opt", metavar="LEVEL",
                        help="replace the -O3 in CPython's OPT with LEVEL; write --opt=-Os, not --opt -Os; "
                             "everything else CPython chose stays")
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--list-failures", action="store_true")
    parser.add_argument("--sites", type=int, default=30,
                        help="how many error sites to print (default 30)")
    args = parser.parse_args()

    build = pathlib.Path(args.build_dir).resolve()
    for needed in ("Makefile", "pyconfig.h", "Modules/Setup.stdlib"):
        if not (build / needed).is_file():
            print(f"ERROR: {build} is not a configured tree (no {needed}); "
                  f"run prepare-cpython-capstone.sh", file=sys.stderr)
            return 2
    cc = make_var(build, "CC")
    if not cc or not cc[0].endswith("/capstone-cc"):
        print(f"ERROR: the Makefile's CC is {cc!r}, not capstone-cc", file=sys.stderr)
        return 2
    clang = os.environ.get("CAPSTONE_CLANG", "")
    for var in ("CAPSTONE_CLANG", "CAPSTONE_LD_LLD", "CPY_MUSL", "CPY_RUNTIME_DIR",
                "CPY_LIBC_ARCHIVE", "CPY_LINKER_SCRIPT"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set; source {build}/capstone-env.sh", file=sys.stderr)
            return 2

    group_of: dict[str, str] = {}
    for group, var in GROUPS:
        for obj in make_var(build, var):
            group_of.setdefault(obj, group)
    for group, objs in EXTRA:
        for obj in objs:
            group_of.setdefault(obj, group)
    expected = sorted(group_of)
    if len(expected) < 100:
        print(f"ERROR: only {len(expected)} objects listed by {build}/Makefile; "
              f"the variable names in GROUPS no longer match it", file=sys.stderr)
        return 2

    # Start from nothing, or make would skip up-to-date objects and they would
    # read as NOTRUN.
    for obj in expected:
        for stale in (build / obj, build / (obj + ".err")):
            stale.unlink(missing_ok=True)
    log = build / "survey-compile.log"
    log.unlink(missing_ok=True)
    env = dict(os.environ, CPY_SURVEY_LOG=str(log))
    overrides = []
    if args.opt:
        opt = make_var(build, "OPT")
        if "-O3" not in opt:
            print(f"ERROR: CPython's OPT is {opt!r}, no -O3 to replace", file=sys.stderr)
            return 2
        overrides = ["OPT=" + " ".join(args.opt if f == "-O3" else f for f in opt)]
    if args.expect_ok is None:
        args.expect_ok = BASELINE_OK if not args.opt else 0
    with open(build / "survey-make.log", "w") as out:
        subprocess.run(["make", "-k", f"-j{args.jobs}", *overrides, *expected], cwd=build,
                       env=env, stdout=out, stderr=subprocess.STDOUT)
        # The MUST_FAIL control: same wrapper, same log, CPython's core flags.
        (build / CONTROL_MUST_FAIL).unlink(missing_ok=True)
        subprocess.run([*cc, *make_var(build, "PY_CORE_CFLAGS"), "-c",
                        str(CONTROL_MUST_FAIL_SRC), "-o", CONTROL_MUST_FAIL], cwd=build,
                       env=env, stdout=out, stderr=subprocess.STDOUT)

    if not log.is_file() or log.stat().st_size == 0:
        print(f"ERROR: make ran but capstone-cc logged nothing ({log}); "
              f"see {build}/survey-make.log", file=sys.stderr)
        return 2
    outcome: dict[str, tuple[bool, str]] = {}
    for line in log.read_text().splitlines():
        obj, rc, _src, first = (line.split("\t") + ["", "", "", ""])[:4]
        obj = str(pathlib.Path(obj)) if not obj.startswith("/") else \
            str(pathlib.Path(obj).relative_to(build))
        outcome[obj] = (rc == "0", first)

    ok = [o for o in expected if outcome.get(o, (None,))[0] is True]
    bad = [o for o in expected if outcome.get(o, (None,))[0] is False]
    notrun = [o for o in expected if o not in outcome]
    extra = sorted(set(outcome) - set(expected) - {CONTROL_MUST_FAIL})

    print(f"build tree     {build}")
    print(f"compiler       {clang}")
    try:
        ver = subprocess.run([clang, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
        rev = re.search(r"\(([^()]*\s+)?([0-9a-f]{40})\)", ver)
        print(f"compiler rev   {rev.group(2) if rev else 'unknown'}")
    except Exception as e:  # the number is still valid; its attribution is not
        print(f"compiler rev   unavailable ({e})")
    applied = build / "applied-patches.txt"
    names = applied.read_text().split() if applied.is_file() else None
    print(f"patches        {'UNKNOWN (no applied-patches.txt)' if names is None else len(names)}"
          + (f"  {' '.join(names)}" if names else ""))
    print(f"optimisation   {overrides[0] if overrides else 'CPython default OPT'}"
          + ("  (no baseline gate)" if args.opt and args.expect_ok == 0 else ""))
    print(f"surveyed       {len(expected)} objects (CPython's Makefile list)")
    print(f"compiled       {len(ok)}")
    print(f"failed         {len(bad)}")
    print(f"notrun         {len(notrun)}   (a prerequisite failed first)")
    print(f"               {100 * len(ok) / len(expected):.1f}% ok")
    if extra:
        print(f"also compiled  {len(extra)} objects outside the list (not counted): "
              + " ".join(extra[:8]) + (" ..." if len(extra) > 8 else ""))

    print("\nby group:          ok  fail notrun")
    for group in dict.fromkeys(g for g, _ in GROUPS + [(g, None) for g, _ in EXTRA]):
        members = [o for o in expected if group_of[o] == group]
        if members:
            print(f"  {group:15s} {sum(o in ok for o in members):5d} "
                  f"{sum(o in bad for o in members):5d} {sum(o in notrun for o in members):6d}"
                  f"   of {len(members)}")

    print("\nfailures by FIRST error (one cause per object):")
    for cause, count in collections.Counter(bucket(outcome[o][1]) for o in bad).most_common():
        print(f"  {count:5d}  {cause}")
    source_root = pathlib.Path(make_var(build, "srcdir")[0])
    if not source_root.is_absolute():
        source_root = (build / source_root).resolve()
    per_site = collections.Counter()
    unexplained = []
    for o in bad:
        sites = error_sites(build / (o + ".err"), source_root)
        if not sites:
            unexplained.append(o)
        per_site.update(sites)
    print(f"\nerror sites by objects affected (every error in every failing object; "
          f"{len(per_site)} distinct):")
    for (where, what), count in per_site.most_common(args.sites):
        print(f"  {count:5d}  {where}\n         {what}")
    if unexplained:
        print(f"  {len(unexplained):5d}  failing objects with no parsable error site: "
              + " ".join(unexplained[:6]))

    # The objects that COMPILE are not thereby correct. clang's
    # -Wcapstone-pointer-roundtrip marks each explicit integer->pointer
    # conversion, whose result is untagged and faults when dereferenced. Every
    # object's diagnostics are kept, so this reads them all, and counts source
    # SITES (a header included by 200 files is one site). It sees explicit
    # casts only, not a pointer carried through memcpy or a union.
    roundtrip = re.compile(r"^(/[^:]+):(\d+):\d+: warning: .*\[-Wcapstone-pointer-roundtrip\]")
    rt_sites: dict[tuple[str, int], set[str]] = {}
    for o in expected:
        err = build / (o + ".err")
        if not err.is_file():
            continue
        for line in err.read_text(errors="replace").splitlines():
            m = roundtrip.match(line)
            if m:
                where = m.group(1)
                try:
                    where = str(pathlib.Path(where).relative_to(source_root))
                except ValueError:
                    pass
                rt_sites.setdefault((where, int(m.group(2))), set()).add(o)
    rt_files = collections.Counter(f for f, _ in rt_sites)
    print(f"\ninteger->pointer round trips (-Wcapstone-pointer-roundtrip): {len(rt_sites)} "
          f"sites in {len(rt_files)} source files, reached by "
          f"{len(set().union(*rt_sites.values())) if rt_sites else 0} objects")
    for f, n in rt_files.most_common(20):
        print(f"  {n:5d}  {f}")
    if not rt_sites:
        print("WARNING: no round-trip site at all. Include/internal/pycore_gc.h has known "
              "ones; check that the warning is still emitted before reading this as clean.",
              file=sys.stderr)

    if args.list_failures or len(bad) + len(notrun) <= 40:
        print("\nfailing objects:")
        for o in bad:
            print(f"  {o}\n      {outcome[o][1][:300]}")
        for o in notrun:
            print(f"  {o}\n      NOTRUN")

    states = collections.defaultdict(list)
    for line in (build / "Makefile").read_text().splitlines():
        m = re.match(r"MODULE_([A-Z0-9_]+)_STATE=(\S+)", line)
        if m:
            states[m.group(2)].append(m.group(1).lower())
    print("\nstdlib modules by configure state (only 'yes' is in the denominator):")
    for state in sorted(states):
        print(f"  {state:9s} {len(states[state]):3d}  {' '.join(sorted(states[state]))}")

    status = 0
    for control, must_pass in ((CONTROL_MUST_PASS, True), (CONTROL_MUST_FAIL, False)):
        if control is None:
            print(f"\nWARNING: no MUST_{'PASS' if must_pass else 'FAIL'} control set yet; "
                  f"this run is not a gate", file=sys.stderr)
            continue
        got = outcome.get(control, (None,))[0]
        if got is None:
            print(f"\nERROR: control {control} was not compiled at all", file=sys.stderr)
            status = 2
        elif got is not must_pass:
            print(f"\nERROR: control {control} expected {'PASS' if must_pass else 'FAIL'} "
                  f"but got {'PASS' if got else 'FAIL'}. The harness is not measuring what "
                  f"it claims; if this is an intended improvement, retire the control "
                  f"deliberately.", file=sys.stderr)
            status = 2
    if status:
        return status
    if len(ok) < args.expect_ok:
        print(f"\nREGRESSION: {len(ok)} compiled, baseline is {args.expect_ok}", file=sys.stderr)
        return 1
    if len(ok) > args.expect_ok and not args.opt:
        print(f"\nIMPROVED: {len(ok)} compiled, baseline is {args.expect_ok}. Raise BASELINE_OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
