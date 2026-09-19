#!/usr/bin/env python3
"""Did ggml ever expose a per-object free? Read the public headers, tag by tag.

The pickaxe over 11k commits timed out and was the wrong tool anyway: the
question is about the PUBLIC API surface, which is a handful of header files at a
handful of tags. Every whisper.cpp release is probed, because whisper vendors
ggml and its tags are the pins we could actually adopt.
"""
import re
import subprocess

W = "/tmp/claude-1000/corpus-research/whispercpp.git"
FREE_LIKE = re.compile(r"\bggml_\w*free\w*\s*\(", re.I)
ALLOC_HDRS = ("ggml.h", "ggml-alloc.h", "ggml-backend.h")


def git(*a):
    return subprocess.run(["git", "-C", W, *a], capture_output=True,
                          text=True).stdout


def key(t):
    m = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", t)
    return tuple(int(x) for x in m.groups()) if m else None


tags = sorted({t for t in git("tag", "-l").split() if key(t)}, key=key)
print(f"{len(tags)} whisper.cpp release tags, {tags[0]} .. {tags[-1]}\n")

print(f"{'tag':<10} {'ggml headers present':<44} free-like symbols declared")
print("-" * 110)
prev = None
for t in tags:
    names = [p for p in git("ls-tree", "-r", "--name-only", t).split()
             if p.rsplit("/", 1)[-1] in ALLOC_HDRS]
    syms = set()
    for p in names:
        for m in FREE_LIKE.finditer(git("show", f"{t}:{p}")):
            syms.add(m.group(0).rstrip("(").strip())
    short = sorted({n.rsplit("/", 1)[-1] for n in names})
    cur = tuple(sorted(syms))
    mark = "" if cur == prev else "   <-- CHANGED"
    print(f"{t:<10} {str(short):<44} {', '.join(sorted(syms))}{mark}")
    prev = cur
