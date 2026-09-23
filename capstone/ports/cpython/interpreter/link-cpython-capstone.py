#!/usr/bin/env python3
"""Link what the survey compiled into one domain image, and say what is missing.

Input is a tree survey-cpython-capstone.py has just run on. The link is the one
CPython's Makefile would do for a static interpreter (Programs/python.o, every
LIBRARY_OBJS object, the bundled libmpdec/expat/HACL* archives) and the one
every musl domain here does (my_first_domain/link.ld, the port runtime,
libc-capstone.a and compiler-rt's builtins). Objects the survey could not
compile are left out, and two get a measured per-file workaround first:
Python/compile.o (C-52: -mllvm -regalloc=basic) and Modules/getbuildinfo.o
(NOTRUN only because its Makefile rule depends on every other object).

Every undefined symbol is attributed with a NATIVE build of the same CPython
(configure; make, on the host): the native object that defines it says whether
the symbol is missing because a known-broken CPython object is absent, because
configure left a module out here, or because libc, the port runtime or the
compiler builtins do not provide it. Only the last kind is new information;
the first is the survey's result seen from the link.

A second link that ignores undefined symbols gives an image, whose segment size
is a LOWER bound (the absent objects are not in it) to compare with the 4 MiB a
domain can have.

Controls: PyLong_FromVoidPtr must come out undefined and attributed to
Objects/longobject.o, which the survey requires to fail; Py_BytesMain, defined
by Modules/main.o which compiles, must not. Either flipping exits 2.

Usage:  link-cpython-capstone.py <build-dir> --native <native-build-dir> [--out DIR]
"""

import argparse
import collections
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("survey", HERE / "survey-cpython-capstone.py")
survey = importlib.util.module_from_spec(spec)
spec.loader.exec_module(survey)

MUST_BE_UNDEFINED = ("PyLong_FromVoidPtr", "Objects/longobject.o")
MUST_BE_DEFINED = "Py_BytesMain"

# The bundled static libraries CPython links for enabled modules, by Makefile
# variable, with the object list that makes each.
ARCHIVES = [("LIBMPDEC_A", "LIBMPDEC_OBJS"), ("LIBEXPAT_A", "LIBEXPAT_OBJS"),
            ("LIBHACL_SHA2_A", "LIBHACL_SHA2_OBJS")]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def native_symbol_map(native: pathlib.Path, nm: str) -> dict[str, str]:
    """symbol -> the native object (path relative to the build) defining it."""
    defined: dict[str, str] = {}
    objs = [p for d in ("Parser", "Objects", "Python", "Modules", "Programs")
            for p in (native / d).rglob("*.o")]
    if len(objs) < 100:
        raise SystemExit(f"ERROR: only {len(objs)} objects under {native}; "
                         f"is it a built native CPython tree?")
    for o in objs:
        out = run([nm, "-g", "--defined-only", str(o)]).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3:          # --defined-only: every line is a definition
                defined.setdefault(parts[2], str(o.relative_to(native)))
    return defined


def parse_undefined(text: str) -> dict[str, set[str]]:
    """lld's 'undefined symbol: X' blocks -> X -> the objects referencing it.

    Each block is the error line, then '>>> referenced by' lines naming the
    object as 'path.o:(function)'."""
    undefined: dict[str, set[str]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"ld\.lld: error: undefined symbol: (.+)$", line)
        if m:
            current = m.group(1).strip()
            undefined.setdefault(current, set())
        elif line.startswith(">>>") and current is not None:
            ref = re.search(r"(\S+\.o):", line)
            if ref:
                undefined[current].add(ref.group(1))
        else:
            current = None
    return undefined


def section_sizes(readobj_sections: str) -> dict[str, int]:
    """llvm-readobj --sections -> {name: size}, one 'Section {' block at a time."""
    sizes, name = {}, None
    for line in readobj_sections.splitlines():
        line = line.strip()
        if line.startswith("Name: "):
            name = line.split()[1]
        elif line.startswith("Size: ") and name is not None:
            sizes[name] = int(line.split()[1])
            name = None
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--native", required=True, help="a built native CPython 3.13.7 tree")
    ap.add_argument("--out", help="where the link happens (default <build>/link-attempt)")
    args = ap.parse_args()

    build = pathlib.Path(args.build_dir).resolve()
    native = pathlib.Path(args.native).resolve()
    out = pathlib.Path(args.out).resolve() if args.out else build / "link-attempt"
    for var in ("CAPSTONE_CLANG", "CAPSTONE_LD_LLD", "CPY_RUNTIME_DIR", "CPY_LIBC_ARCHIVE",
                "CPY_LINKER_SCRIPT"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set; source {build}/capstone-env.sh", file=sys.stderr)
            return 2
    bindir = pathlib.Path(os.environ["CAPSTONE_CLANG"]).parent
    ar, nm, readobj = (str(bindir / t) for t in ("llvm-ar", "llvm-nm", "llvm-readobj"))
    for tool in (ar, nm, readobj):
        if not os.access(tool, os.X_OK):
            print(f"ERROR: {tool} missing (ninja llvm-ar llvm-nm llvm-readobj)", file=sys.stderr)
            return 2
    log = build / "survey-compile.log"
    if not log.is_file():
        print(f"ERROR: no {log}; run the survey on this tree first", file=sys.stderr)
        return 2
    status = {}
    for line in log.read_text().splitlines():
        obj, rc = line.split("\t")[:2]
        status[obj] = rc == "0"

    group_of = {}
    for group, var in survey.GROUPS:
        for obj in survey.make_var(build, var):
            group_of.setdefault(obj, group)
    for group, objs in survey.EXTRA:
        for obj in objs:
            group_of.setdefault(obj, group)
    archive_objs = {o for _, var in ARCHIVES for o in survey.make_var(build, var)}

    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "CPY_SURVEY_LOG"}
    workarounds = {}

    # C-52: compile.c at -O1+ crashes the Greedy allocator; the basic one compiles it.
    if not status.get("Python/compile.o"):
        make_log = (build / "survey-make.log").read_text()
        line = next((l for l in make_log.splitlines()
                     if "capstone-cc" in l and "-o Python/compile.o" in l), None)
        if line is None:
            print("ERROR: compile.c's command is not in survey-make.log", file=sys.stderr)
            return 2
        target = out / "compile.o"
        cmd = line.replace("-o Python/compile.o", f"-o {shlex.quote(str(target))}") \
            + " -mllvm -regalloc=basic"
        done = run(["bash", "-c", cmd], cwd=build, env=env)
        if done.returncode == 0:
            workarounds["Python/compile.o"] = (str(target), "-mllvm -regalloc=basic (C-52)")
        else:
            print(f"compile.c workaround failed:\n{done.stderr[-2000:]}", file=sys.stderr)
    # getbuildinfo.o: CPython's own rule, with every prerequisite taken as up to date.
    if not status.get("Modules/getbuildinfo.o"):
        assume = [a for o in group_of if o != "Modules/getbuildinfo.o" for a in ("-o", o)]
        done = run(["make", *assume, "Modules/getbuildinfo.o"], cwd=build, env=env)
        if done.returncode == 0 and (build / "Modules/getbuildinfo.o").is_file():
            workarounds["Modules/getbuildinfo.o"] = (str(build / "Modules/getbuildinfo.o"),
                                                     "its own rule, prerequisites assumed built")
        else:
            print(f"getbuildinfo.o failed:\n{done.stdout[-1000:]}{done.stderr[-1000:]}",
                  file=sys.stderr)

    def have(o):
        return o in workarounds or (status.get(o) and (build / o).is_file())

    core = [o for o in group_of if o not in archive_objs]
    present = [workarounds[o][0] if o in workarounds else str(build / o)
               for o in core if have(o)]
    absent = sorted(o for o in group_of if not have(o))

    archives = []
    for avar, ovar in ARCHIVES:
        members = [str(build / o) for o in survey.make_var(build, ovar) if have(o)]
        name = out / pathlib.Path(survey.make_var(build, avar)[0]).name
        name.unlink(missing_ok=True)
        if members:
            done = run([ar, "rcs", str(name), *members])
            if done.returncode:
                print(done.stderr, file=sys.stderr)
                return 2
            archives.append(str(name))

    runtime = sorted(str(p) for p in pathlib.Path(os.environ["CPY_RUNTIME_DIR"]).glob("*.o"))
    base = [os.environ["CAPSTONE_LD_LLD"], "--gc-sections", "-T", os.environ["CPY_LINKER_SCRIPT"],
            "--error-limit=0", *runtime, *present, *archives, os.environ["CPY_LIBC_ARCHIVE"]]
    strict = run([*base, "-o", str(out / "python.dom")])
    (out / "link-strict.log").write_text(strict.stdout + strict.stderr)
    undefined = parse_undefined(strict.stderr)
    other_errors = [l for l in strict.stderr.splitlines()
                    if l.startswith("ld.lld: error:") and "undefined symbol" not in l]

    defined_by = native_symbol_map(native, nm)
    failed_builtins = set()
    fb = pathlib.Path(os.environ["CPY_LIBC_ARCHIVE"]).parent / "builtins" / "failed.txt"
    if fb.is_file():
        failed_builtins = {"__" + l.split("\t")[0] for l in fb.read_text().splitlines() if l}
    first_error = {}
    for line in log.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[1] != "0":
            first_error[parts[0]] = survey.bucket(parts[3])

    by_absent = collections.defaultdict(list)
    not_configured = collections.defaultdict(list)
    builtins, platform = [], []
    for sym in sorted(undefined):
        where = defined_by.get(sym)
        if where in group_of and where in absent:
            by_absent[where].append(sym)
        elif where is not None and where not in group_of:
            not_configured[where].append(sym)
        elif sym in failed_builtins:
            builtins.append(sym)
        else:
            platform.append(sym)

    print(f"build tree     {build}")
    print(f"linked         {len(present)} objects + {len(archives)} archives "
          f"({len(absent)} of {len(group_of)} survey objects absent)")
    for o, (_, how) in sorted(workarounds.items()):
        print(f"workaround     {o}: {how}")
    print(f"strict link    exit {strict.returncode}; {len(undefined)} undefined symbols, "
          f"{len(other_errors)} other errors")

    print(f"\nundefined because a CPython object is absent: "
          f"{sum(map(len, by_absent.values()))} symbols from {len(by_absent)} objects")
    for o in sorted(by_absent, key=lambda o: -len(by_absent[o])):
        print(f"  {len(by_absent[o]):4d}  {o:42s} {first_error.get(o, 'not compiled')[:60]}")
    print(f"\nundefined because configure left the defining module out here: "
          f"{sum(map(len, not_configured.values()))}")
    for o in sorted(not_configured):
        print(f"  {o}: {' '.join(not_configured[o][:6])}")
    print(f"\nundefined compiler-rt builtins (their source did not compile): {len(builtins)}")
    for s in builtins:
        print(f"  {s}")
    print(f"\nundefined, provided by NOTHING in this link (libc, runtime or unknown): {len(platform)}")
    for s in platform:
        refs = sorted(undefined[s])
        print(f"  {s:34s} <- {', '.join(refs[:3])}{' ...' if len(refs) > 3 else ''}")
    if other_errors:
        print("\nother link errors:")
        for l in other_errors[:30]:
            print(f"  {l}")

    loose = run([*base, "--unresolved-symbols=ignore-all", "--noinhibit-exec",
                 "-o", str(out / "python-lowerbound.dom")])
    (out / "link-lowerbound.log").write_text(loose.stdout + loose.stderr)
    image = out / "python-lowerbound.dom"
    if image.is_file():
        phdr = run([readobj, "--program-headers", str(image)]).stdout
        memsz = [int(x) for x in re.findall(r"MemSize:\s*(\d+)", phdr)]
        filesz = [int(x) for x in re.findall(r"FileSize:\s*(\d+)", phdr)]
        sections = run([readobj, "--sections", str(image)]).stdout
        sizes = section_sizes(sections)
        print(f"\nimage without the absent objects (a LOWER bound):")
        print(f"  PT_LOAD memsz {sum(memsz):,} bytes ({sum(memsz) / 2**20:.2f} MiB), "
              f"filesz {sum(filesz):,}; a domain gets at most 4 MiB")
        for name in (".text", ".rodata", ".data", ".bss", ".init_array", ".fini_array",
                     ".capstone_cap_init", ".gct"):
            if name in sizes:
                print(f"  {name:20s} {int(sizes[name]):>12,}")
        # The absent objects, sized: capstone64 code is larger than x86-64 code,
        # so their native size is scaled by the ratio the present objects show.
        # An ESTIMATE; the objects themselves cannot be built yet.
        size = run(["size", "--version"])
        if size.returncode == 0:
            def tdb(tool, path):
                rows = run([tool, str(path)]).stdout.splitlines()
                f = rows[1].split() if len(rows) > 1 else None
                return int(f[0]) + int(f[1]) + int(f[2]) if f else None
            cap = nat = gone = 0
            for o in group_of:
                n = tdb("size", native / o) if (native / o).is_file() else None
                if n is None:
                    continue
                if o in absent:
                    gone += n
                elif (build / o).is_file():
                    cap += tdb(str(bindir / "llvm-size"), build / o) or 0
                    nat += n
            if nat:
                est = gone * cap / nat
                print(f"  absent objects: {gone:,} bytes native x {cap / nat:.2f} "
                      f"(capstone/native over the present ones) = ~{est / 2**20:.1f} MiB more, "
                      f"so ~{(sum(memsz) + est) / 2**20:.1f} MiB for the whole image (estimate)")
        # Linking is not working: which linked libc calls need a syscall the
        # domain's hostcall does not serve (musl-capstone's own checker).
        checker = HERE.parent.parent / "musl-capstone" / "check-domain-support.py"
        root = pathlib.Path(os.environ["CPY_RUNTIME_DIR"]).parent
        baseline = root / "linkcheck" / "has.dom"
        if checker.is_file() and baseline.is_file() and os.environ.get("CPY_MUSL"):
            chk = run([sys.executable, str(checker), str(image), "--musl", os.environ["CPY_MUSL"],
                       "--archive", str(root / "musl-build" / "libc-capstone.a"),
                       "--baseline", str(baseline), "--nm", nm, "--port-objects", *runtime])
            (out / "domain-support.txt").write_text(chk.stdout + chk.stderr)
            need = collections.Counter()
            rows = 0
            for line in chk.stdout.splitlines():
                m = re.match(r"\s+(\S+)\s+needs (.+?)\s+src/", line)
                if m:
                    rows += 1
                    need.update(m.group(2).split())
            print(f"\nlinked libc calls that need an UNSERVED syscall (check-domain-support.py, "
                  f"exit {chk.returncode}): {rows} symbols, {len(need)} syscalls")
            print("  " + " ".join(sorted(need)))
            print(f"  full list: {out / 'domain-support.txt'}")
        else:
            print("\nWARNING: check-domain-support.py not run (checker, baseline image or CPY_MUSL "
                  "missing)", file=sys.stderr)
    else:
        print(f"\nno lower-bound image; see {out / 'link-lowerbound.log'}")

    ok = True
    sym, obj = MUST_BE_UNDEFINED
    if sym not in undefined or defined_by.get(sym) != obj:
        print(f"\nERROR: control {sym} should be undefined and attributed to {obj}; got "
              f"{'defined' if sym not in undefined else defined_by.get(sym)}", file=sys.stderr)
        ok = False
    if MUST_BE_DEFINED in undefined:
        print(f"\nERROR: control {MUST_BE_DEFINED} came out undefined", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
