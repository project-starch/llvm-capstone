#!/usr/bin/env python3
"""Make mruby's generated parser compile for capstone64, with one substitution.

THE DEFECT, in mruby's own terms (parse.y:504, new_call):

    node *n = list4(nint(pass?NODE_CALL:NODE_SCALL), a, nsym(b), c);

`nint(x)` is `((node*)(intptr_t)(x))`, and NODE_CALL / NODE_SCALL are adjacent
enum values (26, 27). clang folds `pass ? 26 : 27` into `27 ^ (pass != 0)` --
and because the result is cast to `node *`, the xor is emitted at i128, the
capability width, for which there is no instruction:

    Cannot select: t89: i128 = xor t103, Constant:i128<27>
      t103: i128 = zero_extend t97
        t97: i64 = setcc (load i32 pass), 0, setne

Routing the ternary through an `enum node_type` first keeps the fold at integer
width and leaves a single int-to-pointer conversion, which this target accepts.

WHY THIS IS THE WHOLE PORT. It was the ONLY error in the file. mruby's parser
stores small integers in `node *` fields throughout (`nint`, `intn`, `nsym`,
`sym`, `typen`), and every other one of those hundreds of sites compiles: the
store becomes `mv` plus `stc` and the read becomes `lw`, so the integer travels
in the low half of an untagged capability and comes back intact. Only the folded
ternary produced arithmetic ON that value.

NOT A PATCH AGAINST THE MRUBY TREE. The tree stays byte-identical, as
prepare-musl-capstone.sh keeps musl's; this writes a fixed copy into the build
directory. If the substitution ever stops matching, generation FAILS rather than
silently producing the original.

Upstreamable as a portability fix, and worth offering: it costs mruby nothing.
"""

import argparse
import pathlib
import sys

OLD = "node *n = list4(nint(pass?NODE_CALL:NODE_SCALL), a, nsym(b), c);"
NEW = ("enum node_type nt = pass ? NODE_CALL : NODE_SCALL;\n"
       "  node *n = list4(nint(nt), a, nsym(b), c);")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("y_tab_c", help="mruby's generated mrbgems/mruby-compiler/core/y.tab.c")
    parser.add_argument("output")
    args = parser.parse_args()

    src = pathlib.Path(args.y_tab_c)
    if not src.is_file():
        print(f"ERROR: no parser source at {src}", file=sys.stderr)
        return 2

    text = src.read_text()
    n = text.count(OLD)
    if n != 1:
        print(f"ERROR: expected exactly 1 occurrence of new_call's node-type "
              f"ternary in {src}, found {n}.\n"
              f"       mruby's parser has changed; re-read parse.y around "
              f"new_call before trusting this.", file=sys.stderr)
        return 2

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text.replace(OLD, NEW))
    print(f"generated {out} (1 substitution in new_call)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
