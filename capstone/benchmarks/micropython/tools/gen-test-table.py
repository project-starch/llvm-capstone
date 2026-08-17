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

SELECTION is deliberately narrow and stated in the generated header: tests that need float,
imports, a filesystem or threading are excluded, because this port is built without them. The
exclusion list is emitted alongside the table so a coverage number can never be read as if it
covered the whole suite.
"""
import argparse
import pathlib
import subprocess
import sys

# Constructs this port does not build. Matching is on the source text, which is crude but
# fails SAFE: a test wrongly excluded shows up as a smaller corpus, never as a false pass.
UNSUPPORTED = [
    ("float", ("float(", "1.0", "0.5", "math.", "complex(")),
    ("import", ("import ",)),
    ("filesystem", ("open(", "os.", "vfs")),
    ("thread", ("_thread",)),
    ("native emitter", ("@micropython.native", "@micropython.viper", "@micropython.asm")),
]


def expectation(test: pathlib.Path, python: str, timeout: float):
    exp = test.with_suffix(test.suffix + ".exp")
    if exp.exists():
        return exp.read_bytes(), "exp-file"
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
    ap.add_argument("--max-bytes", type=int, default=1500, help="skip tests larger than this")
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

    kept, skipped = [], []
    for label, t in candidates:
        src = t.read_text(encoding="utf8", errors="replace")
        if not args.include_unsupported and len(src) > args.max_bytes:
            skipped.append((label, f"larger than {args.max_bytes} B"))
            continue
        why = next((name for name, pats in UNSUPPORTED if any(p in src for p in pats)), None)
        if why and not args.include_unsupported:
            skipped.append((label, f"needs {why}"))
            continue
        exp, how = expectation(t, args.python, args.expect_timeout)
        if exp is None and not args.include_unsupported:
            skipped.append((label, how))
            continue
        kept.append((label, src, exp, how))
        if args.limit and len(kept) >= args.limit:
            break

    with open(args.out_header, "w") as f:
        f.write("/* GENERATED by tools/gen-test-table.py -- do not edit.\n")
        f.write(f" * {len(kept)} tests kept, {len(skipped)} skipped.\n")
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
