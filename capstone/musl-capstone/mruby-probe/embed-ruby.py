#!/usr/bin/env python3
"""Embed a Ruby file into the domain as a C string, VERBATIM.

The point of running a corpus row on the real interpreter is that the trigger is
the corpus's file and not a paraphrase of it, so this copies bytes and escapes
them; it does not reformat, strip comments, or "simplify". If the embedded text
ever stops matching the source file, that is a different experiment.
"""
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
