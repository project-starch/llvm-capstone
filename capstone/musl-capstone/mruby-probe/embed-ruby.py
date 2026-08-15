#!/usr/bin/env python3
"""Embed a Ruby file into the domain as a C string, VERBATIM.

The point of running a corpus row on the real interpreter is that the trigger is
the corpus's file and not a paraphrase of it, so this copies bytes and escapes
them; it does not reformat, strip comments, or "simplify". If the embedded text
ever stops matching the source file, that is a different experiment.
"""
import os
import pathlib
import sys

if len(sys.argv) != 4:
    print("usage: embed-ruby.py <in.rb> <out.c> <symbol>", file=sys.stderr)
    sys.exit(2)

src, out, sym = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
if not src.is_file():
    print(f"ERROR: no such Ruby file: {src}", file=sys.stderr)
    sys.exit(2)

text = src.read_text()

# MRUBY_PROBE_ROW_DEPTH rewrites the trigger's recursion depth. The corpus file
# says recurse(150); a non-reclaiming allocator cannot afford that (the VM stack
# is re-extended on the way down and the old buffer never comes back), so the
# question is what the SMALLEST depth is that still arms the bug.
#
# This is a DEVIATION from the corpus's own trigger and must be reported as one.
# What makes it honest is the matched pair: if the revoke arm faults and the
# control arm completes at the same depth, the stale access demonstrably
# happened and revocation is demonstrably what stopped it. A depth that does not
# arm shows up as BOTH arms completing, which is not a result and reads as one.
depth = os.environ.get("MRUBY_PROBE_ROW_DEPTH")
if depth:
    old = "recurse(150)"
    if text.count(old) != 1:
        print(f"ERROR: expected exactly one {old} in {src}, found "
              f"{text.count(old)}; this trigger is not depth-parameterised",
              file=sys.stderr)
        sys.exit(2)
    text = text.replace(old, f"recurse({int(depth)})")

if not text.strip():
    print(f"ERROR: {src} is empty; nothing would be triggered", file=sys.stderr)
    sys.exit(2)

lines = []
for line in text.splitlines():
    esc = line.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'  "{esc}\\n"')

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    f"/* GENERATED from {src} by embed-ruby.py -- do not edit. */\n"
    f"static const char {sym}[] =\n" + "\n".join(lines) + ";\n")
print(f"embedded {src} ({len(text.splitlines())} lines) as {sym}")
