#!/usr/bin/env python3
"""Bake a selection of MicroPython's own tests into a C table, with their expected output.

WHY A TABLE AND NOT ONE IMAGE PER TEST. A domain runs a program that was compiled into it, so a
test per image means a boot per test, and MicroPython ships 576 tests in basics/ alone. The loader
already calls a domain repeatedly and prints each call's return value, and the entry glue's
reentry path keeps globals alive between calls -- so N tests become N domain switches inside one
boot. This script produces the table; port/mpy_domain.c walks it one test per call.

EXPECTED OUTPUT comes from the test's own .exp file when it has one, and otherwise from running
the test under the HOST python3. A test whose expectation cannot be produced is SKIPPED with its
reason recorded, never silently included -- an unverifiable pass is worse than an absent test.
The one expectation that is kept without a hash is MicroPython's regex .exp template, because the
runner matches the captured output against it line by line and so can still score the test.

SELECTION is stated in the generated header, and it follows what the build actually enables.
A test is excluded when it imports a module this build does not have, when it needs a construct
the port does not compile, or when no expectation can be produced. The caller names the modules
it built in with --have-module, so raising the feature level widens the corpus by itself instead
of leaving the selection behind. The exclusion list is emitted alongside the table so a coverage
number can never be read as if it covered the whole suite.
"""
import argparse
import ast
import base64
import pathlib
import re
import subprocess
import sys

# Constructs this port does not build that are NOT imports, so the module rule cannot see them.
# Everything here is anchored: a bare substring is how sys1.py was excluded for threading when
# what it contains is hasattr(sys.implementation, '_thread'), and how the selection used to drop
# 83 tests for containing the word import.
#
# `_thread` and `vfs` were in this list and are gone: both are modules, so a test that actually
# wants one says `import` and the module rule refuses it by name, against what the build
# registers. Keeping them here as text only added false positives.
UNSUPPORTED = [
    ("native emitter", ("@micropython.native", "@micropython.viper", "@micropython.asm")),
]

# The fallback for a file this host python cannot parse. Anchored against a method call, a name
# that merely ends in open, and a definition of a method called open, each of which a bare
# "open(" counted as the builtin.
BUILTIN_OPEN_RE = re.compile(r"(?<!def )(?<![\w.])open\s*\(")

# Only consulted when the build did not pass --have-float.
FLOAT_PATTERNS = ("float(", "1.0", "0.5", "math.", "complex(")

# Reachable without the caller saying so: the module every test already runs in.
ALWAYS_AVAILABLE = {"__main__"}

FROM_RE = re.compile(r"^[ \t]*from[ \t]+([\w.]+)[ \t]+import", re.M)
IMPORT_RE = re.compile(r"^[ \t]*import[ \t]+([^\n#]+)", re.M)


def imported_modules(src: str) -> set:
    """Top-level module names a test imports.

    Parsed, not grepped. A substring match on "import " throws away every test that does nothing
    worse than import sys, and 73 of the 83 excluded that way import only modules this port has
    built in. A test whose syntax this host python cannot parse falls back to the line-anchored
    regex, which sees the same imports for every shape the suite uses.
    """
    mods = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        for m in FROM_RE.finditer(src):
            mods.add(m.group(1).split(".")[0])
        for m in IMPORT_RE.finditer(src):
            for part in m.group(1).split(","):
                part = part.strip().split(" as ")[0].strip()
                if part:
                    mods.add(part.split(".")[0])
        return mods
    # A branch on sys.implementation is CPython's path and MicroPython never reaches it. Three
    # async tests import `types` there and nowhere else, and dropping them for a module the
    # target never asks for is a skip with nothing behind it.
    cpython_only = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "sys.implementation" in ast.unparse(node.test):
            cpython_only.update(id(sub) for sub in ast.walk(node))
    for node in ast.walk(tree):
        if id(node) in cpython_only:
            continue
        if isinstance(node, ast.Import):
            mods |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module.split(".")[0])
    return mods


# MicroPython's runner treats these .exp files as line-oriented regex templates rather than
# literal output. Preserve the template in the expected table so the resumable runner can apply
# the same normalization when exact output capture is enabled.
REGEX_OUTPUT = {
    "micropython/meminfo.py",
    "basics/bytes_compare3.py",
    "basics/builtin_help.py",
    "basics/weakref_callback_exception.py",
    "misc/sys_settrace_cov.py",
    "net_inet/tls_text_errors.py",
    "ports/unix/extra_coverage.py",
    "thread/thread_exc2.py",
    "ports/esp32/partition_ota.py",
}


def expectation(test: pathlib.Path, python: str, timeout: float):
    exp = test.with_suffix(test.suffix + ".exp")
    if exp.exists():
        data = exp.read_bytes()
        test_key = f"{test.parent.name}/{test.name}"
        if test_key in REGEX_OUTPUT:
            encoded = base64.b64encode(data).decode("ascii")
            return None, f"regex-exp:{encoded}"
        return data, "exp-file"
    try:
        out = subprocess.run([python, "-BS", str(test)], capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, f"host run failed: {exc}"
    if out.returncode != 0:
        return None, "host python exits non-zero"
    return out.stdout, "host python"


def expected_retval(idx: int, data: bytes) -> int:
    """The exact 32-bit word the domain must return for this test.

    Must match mpy_domain.c's MPY_TEST_RUNNER block bit for bit -- if the two drift, every test
    reports FAIL, which is at least loud. Bit 31 (raised) is NOT set here: whether a test raises
    is a property of MicroPython, not of the expectation, and the traceback text is part of the
    output the hash already covers. The scorer therefore compares with bit 31 masked off."""
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return ((idx & 0x7FF) << 20) | ((len(data) & 0xF) << 16) | (h & 0xFFFF)


def c_string(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        esc = line.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        out.append(f'        "{esc}"')
    return "\n".join(out) if out else '        ""'


def uses_builtin_open(src: str) -> bool:
    """True when the test CALLS the builtin open(), which needs a filesystem under it.

    Parsed, not grepped, for the same reason the imports are: `open(` also appears in
    `def open(self)` and in `f.open()`, and neither is the builtin. Not a module, so the module
    rule cannot see it.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return bool(BUILTIN_OPEN_RE.search(src))
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"
               for n in ast.walk(tree))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tests_dir")
    ap.add_argument("out_header")
    ap.add_argument("--add-tests-dir", action="append", default=[],
                    help="additional non-recursive test directory (repeatable)")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many sorted candidates before applying --limit")
    ap.add_argument("--max-bytes", type=int, default=1500,
                    help="skip tests larger than this; 0 = no limit")
    ap.add_argument("--have-module", action="append", default=[],
                    help="a module this build has compiled in (repeatable)")
    ap.add_argument("--exclude-module", action="append", default=[],
                    help="a module the build registers but which cannot work here (repeatable)")
    ap.add_argument("--have-float", action="store_true",
                    help="this build has float objects, so float tests are candidates")
    ap.add_argument("--expect-timeout", type=float, default=20,
                    help="seconds allowed for the host-Python oracle")
    ap.add_argument("--include-unsupported", action="store_true",
                    help="execute every candidate; retain tests without an output oracle")
    args = ap.parse_args()

    tests_dirs = [pathlib.Path(args.tests_dir), *(pathlib.Path(p) for p in args.add_tests_dir)]
    multiple_dirs = len(tests_dirs) > 1
    candidates = sorted(
        ((f"{d.name}/{t.name}" if multiple_dirs else t.name), t)
        for d in tests_dirs
        for t in d.glob("*.py")
    )
    if not candidates:
        sys.exit(f"no tests under {', '.join(str(d) for d in tests_dirs)}")
    if args.offset < 0:
        sys.exit("--offset must be non-negative")
    candidates = candidates[args.offset:]

    excluded = set(args.exclude_module)
    have = (ALWAYS_AVAILABLE | set(args.have_module)) - excluded
    kept, skipped = [], []
    for label, t in candidates:
        src = t.read_text(encoding="utf8", errors="replace")
        if not args.include_unsupported and args.max_bytes and len(src) > args.max_bytes:
            skipped.append((label, f"larger than {args.max_bytes} B"))
            continue
        missing = sorted(imported_modules(src) - have)
        if missing and not args.include_unsupported:
            how = ", ".join(m + " (excluded)" if m in excluded else m for m in missing)
            skipped.append((label, "needs module " + how))
            continue
        why = next((name for name, pats in UNSUPPORTED if any(p in src for p in pats)), None)
        if why is None and uses_builtin_open(src):
            why = "the builtin open()"
        if why is None and not args.have_float and any(p in src for p in FLOAT_PATTERNS):
            why = "float"
        if why and not args.include_unsupported:
            skipped.append((label, f"needs {why}"))
            continue
        exp, how = expectation(t, args.python, args.expect_timeout)
        # A regex .exp is an expectation, just not a hashable one. The runner reconstructs the
        # template from this row and matches the captured output against it line by line, so
        # dropping these was the selection refusing a test the scoring can handle.
        if exp is None and not how.startswith("regex-exp:") and not args.include_unsupported:
            skipped.append((label, how))
            continue
        kept.append((label, src, exp, how))
        if args.limit and len(kept) >= args.limit:
            break

    with open(args.out_header, "w") as f:
        f.write("/* GENERATED by tools/gen-test-table.py -- do not edit.\n")
        f.write(f" * {len(kept)} tests kept, {len(skipped)} skipped.\n")
        f.write(f" * Modules the selection was told this build has: {' '.join(sorted(have)) or 'none'}\n")
        if excluded:
            f.write(f" * Registered but excluded: {' '.join(sorted(excluded))}\n")
        f.write(f" * Size limit: {args.max_bytes or 'none'}; float tests: "
                f"{'candidates' if args.have_float else 'excluded'}\n")
        f.write(" * Skipped, with the reason, so a coverage number cannot be read as full coverage:\n")
        for name, why in skipped:
            f.write(f" *   {name:<40s} {why}\n")
        f.write(" */\n#pragma once\n\n")
        f.write(f"#define MPY_TEST_COUNT {len(kept)}\n\n")
        f.write("static const char *const mpy_tests[MPY_TEST_COUNT] = {\n")
        for name, src, _exp, _how in kept:
            f.write(f"    /* {name} */\n{c_string(src)},\n")
        f.write("};\n")

    side = pathlib.Path(args.out_header).with_suffix(".expected")
    with open(side, "w") as f:
        for i, (name, _src, exp, how) in enumerate(kept):
            if exp is None:
                f.write(f"{i}\t{name}\t-\t-\t{how}\n")
            else:
                f.write(f"{i}\t{name}\t{len(exp)}\t{expected_retval(i, exp):#010x}\t{how}\n")
    print(f"{len(kept)} tests -> {args.out_header}; expectations -> {side}; {len(skipped)} skipped")


if __name__ == "__main__":
    main()
