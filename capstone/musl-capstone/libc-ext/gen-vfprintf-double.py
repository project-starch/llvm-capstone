#!/usr/bin/env python3
"""Generate a `long double`-free vfprintf from musl's own vfprintf.c.

WHY THIS EXISTS. `src/stdio/vfprintf.c` is the ONLY file in musl's src/stdio
that does not compile for capstone64 -- 118 of 119 do. Everything else in the
family (printf, fprintf, snprintf, sprintf, dprintf, asprintf, perror, puts,
fwrite) funnels through it, so this one file is the difference between "no
formatted output at all" and "all of musl's stdio".

It fails for C-20: `long double` is 128-bit on this target and so is a
capability, so every long-double operation hits the backend's i128 wall.

WHAT MAKES THE TRANSFORM CHEAP. musl's float formatter does NOT need 128-bit
arithmetic. `fmt_fp` expands the mantissa into `uint32_t big[]` limbs and does
base-1e9 long division on them; the only long-double dependence is the type of
the value, one call to frexpl, and three LDBL_ constants. Narrow those to double
and the whole formatter compiles and stays correctly rounded, because it is
still musl's algorithm. That is why this is a substitution and not a
reimplementation: a hand-written dtoa is exactly the kind of "plausible output,
subtly wrong digits" that has to be retracted later.

WHAT IT COSTS. `%Lf` and friends pop a `double` instead of a `long double`. On
this target that is not a narrowing of anything usable: no program can produce a
long double, since every 128-bit float builtin fails to compile (C-20), so no
caller can pass one.

WHY GENERATED RATHER THAN VENDORED. prepare-musl-capstone.sh keeps the upstream
tree byte-identical -- arch/capstone64 is the entire delta, auditable with one
diff -r. A patched src/stdio/vfprintf.c would break that invariant, and a
700-line vendored copy would rot. This transforms upstream at build time and
FAILS LOUDLY if the source drifts: every substitution below must fire, and none
may be left over.
"""

import argparse
import pathlib
import re
import sys

# (pattern, replacement, why). For musl's own vfprintf.c each MUST match at least
# once: a rule that fires zero times means upstream changed under us, and
# silently generating an untransformed file would hand the compiler back the long
# double we are here to remove. That is the positive control for this generator.
#
# With --source the file is not musl's and that per-rule check does not apply:
# mruby's src/fmt_fp.c, the same algorithm with the same provenance, has no
# `frexpl` at all. Only the first rule stays mandatory there, because a file with
# no `long double` had no business being passed in. THE LEFTOVER CHECK BELOW IS
# THE REAL GATE either way -- it is what guarantees nothing 128-bit survives, and
# it does not care which file this is.
SUBSTITUTIONS = [
    (r"long double", "double",
     "the formatted value itself, in pop_arg, fmt_fp and its rounding locals"),
    (r"\bfrexpl\b", "frexp",
     "the one libm call; src/math/frexp.c compiles, frexpl.c does not"),
    (r"\bLDBL_", "DBL_",
     "MANT_DIG, MAX_EXP and EPSILON size the limb array and the round trip"),
    # musl's <math.h> classification macros are ternary chains over sizeof(x)
    # whose LAST arm is the long-double one (__signbitl, __fpclassifyl). The
    # arm is dead for a double argument, but only after constant folding: at
    # -O0 clang emits all three arms and the link then wants the long-double
    # functions plus __extenddftf2 to reach them. vfprintf is built at -O0
    # (see build-musl-capstone.sh), so the macros have to go. The clang
    # builtins are type-generic and expand inline, with no libm call at any
    # optimisation level.
    (r"\bsignbit\(", "__builtin_signbit(",
     "otherwise __signbitl at -O0"),
    (r"\bisfinite\(", "__builtin_isfinite(",
     "otherwise __fpclassifyl at -O0"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("musl_dir")
    parser.add_argument("output")
    parser.add_argument("--source", help="narrow this file instead of musl's "
                                         "vfprintf.c (musl_dir is then unused)")
    args = parser.parse_args()

    if args.source:
        src = pathlib.Path(args.source)
        if not src.is_file():
            print(f"ERROR: no such source {src}", file=sys.stderr)
            return 2
    else:
        src = pathlib.Path(args.musl_dir) / "src" / "stdio" / "vfprintf.c"
        if not src.is_file():
            print(f"ERROR: no musl vfprintf at {src}", file=sys.stderr)
            return 2

    text = src.read_text()
    counts = []
    for i, (pattern, replacement, _why) in enumerate(SUBSTITUTIONS):
        text, n = re.subn(pattern, replacement, text)
        # Rule 0 is mandatory everywhere; the rest only for musl's own file.
        if n == 0 and (i == 0 or not args.source):
            where = ("this file" if args.source else
                     "musl's vfprintf.c; re-read it before trusting this generator")
            print(f"ERROR: substitution {pattern!r} matched nothing in {src}.\n"
                  f"       {where}.", file=sys.stderr)
            return 2
        if n:
            counts.append((pattern, n))

    leftover = re.findall(r"long double|\bfrexpl\b|\bLDBL_", text)
    if leftover:
        print(f"ERROR: {len(leftover)} long-double references survived: "
              f"{sorted(set(leftover))}", file=sys.stderr)
        return 2

    header = [
        "/* GENERATED by libc-ext/gen-vfprintf-double.py -- do not edit.",
        f" * Source: {src}",
        " * Transform (each rule fired, or generation would have failed):",
    ]
    header += [f" *   {pattern!r:24} x{n}" for pattern, n in counts]
    header += [
        " * See the generator for why narrowing to double is sound here.",
        " */",
        "",
    ]
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + text)
    print(f"generated {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
