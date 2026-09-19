#!/usr/bin/env python3
"""The consumer-side question for ggml, asked directly.

A corpus case needs a defect in a USER of the allocator, not in the allocator.
For CPython that meant Modules/ and Objects/ code holding a pointer across a
free. For ggml it means whisper.cpp or llama.cpp code holding a ggml_tensor* or
a tensor->data pointer across a ggml_free, a graph reset, or a buffer
reallocation.

So: survey the consumer sources, not ggml/, and do it without relying on the
authors having used the word "use-after-free" in a subject line.
"""
import os
import re
import subprocess

WHISPER = "/tmp/claude-1000/corpus-research/whispercpp.git"
LLAMA = "/tmp/claude-1000/corpus-research/llamacpp.git"

LIFETIME = re.compile(
    r"use.after.free|use-after-free|double.free|dangling|freed|stale|lifetime|"
    r"premature|invalid.*read|after.*(free|destroy|release)|segfault|segv|crash",
    re.I)


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                          text=True).stdout


def survey(repo, label, paths):
    print("=" * 74)
    print(f"{label}: consumer sources {paths}")
    print("=" * 74)
    out = git(repo, "log", "--format=%h%x02%s", "--", *paths)
    rows = [r.split("\x02") for r in out.strip().split("\n") if r.strip()]
    hits = [(h, s) for h, s in rows if LIFETIME.search(s)]
    print(f"{len(rows)} commits touch those paths; {len(hits)} mention a "
          f"lifetime or crash word\n")
    for h, s in hits:
        print(f"    {h}  {s[:92]}")
    return hits


survey(WHISPER, "whisper.cpp", ["src/whisper.cpp"])
print()
survey(WHISPER, "whisper.cpp examples", ["examples"])
print()

# How the consumer actually uses the ported allocator. If contexts are only ever
# freed at teardown, the shape a corpus case needs cannot arise.
print("=" * 74)
print("HOW whisper.cpp USES ggml_free AT THE PINNED TAG")
print("=" * 74)
src = git(WHISPER, "show", "v1.9.4:src/whisper.cpp")
frees = [(i + 1, l.strip()) for i, l in enumerate(src.split("\n"))
         if "ggml_free" in l]
print(f"{len(frees)} ggml_free call sites in src/whisper.cpp at v1.9.4:\n")
for ln, text in frees:
    print(f"    {ln:>6}  {text[:88]}")
