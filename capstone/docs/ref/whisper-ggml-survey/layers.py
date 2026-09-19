#!/usr/bin/env python3
"""whisper.cpp / ggml, surveyed the way CPython was -- not by subject grep.

The earlier negative result came from grepping commit SUBJECTS for temporal
words. That is a fair instrument for CPython, whose fixes are titled "Fix
use-after-free in X". It is a weak one here: ggml's history is titled "ggml :
fix ..." with the detail in the body, or not at all. A clean zero from a weak
instrument is the failure mode this project keeps paying for, so this asks a
different question.

Three questions, each answered from the FILES a commit touches rather than from
how its author chose to describe it:

  1. WHICH allocators does ggml actually have? The port covers ggml_context.
     ggml-alloc.c (the graph allocator) and ggml-backend.cpp (the buffer
     scheduler) are nested allocators too, and are NOT covered.
  2. What is the whole history of each of those files, regardless of wording?
     Small enough to read rather than grep.
  3. Of the commits touching them, which are temporal defects in a CONSUMER?
"""
import os
import re
import subprocess

REPO = os.environ.get("GGML_REPO", "/tmp/claude-1000/corpus-research/llamacpp.git")
WHISPER = os.environ.get("WHISPER_REPO",
                         "/tmp/claude-1000/corpus-research/whispercpp.git")

# The allocator layers ggml stacks, and whether our port covers them.
LAYERS = {
    "ggml/src/ggml.c": "ggml_context bump arena -- PORTED",
    "ggml/src/ggml-alloc.c": "ggml_gallocr graph allocator -- not ported",
    "ggml/src/ggml-backend.cpp": "backend buffer scheduler -- not ported",
    "ggml/src/ggml-backend.c": "backend buffer scheduler (older name)",
    "ggml/src/ggml-impl.h": "shared internals",
}

TEMPORAL = re.compile(
    r"use.after.free|use-after-free|double.free|dangling|freed|free.*before|"
    r"stale|leak|lifetime|premature|already.*freed|invalid.*read|"
    r"after.*destroy|realloc|reuse", re.I)


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                          text=True).stdout


def history(repo, path):
    out = git(repo, "log", "--format=%h%x02%s", "--", path)
    return [r.split("\x02") for r in out.strip().split("\n") if r.strip()]


print("=" * 74)
print("1. THE LAYERS, AND HOW BIG EACH ONE'S HISTORY IS")
print("=" * 74)
for path, what in LAYERS.items():
    rows = history(REPO, path)
    print(f"{len(rows):>5} commits  {path:<32} {what}")

print()
print("=" * 74)
print("2. EVERY COMMIT TOUCHING THE TWO UNPORTED ALLOCATORS WHOSE SUBJECT")
print("   CARRIES ANY MEMORY-LIFETIME WORD AT ALL (a wide net, on purpose)")
print("=" * 74)
for path in ("ggml/src/ggml-alloc.c", "ggml/src/ggml-backend.cpp",
             "ggml/src/ggml-backend.c"):
    rows = history(REPO, path)
    hits = [(h, s) for h, s in rows if TEMPORAL.search(s)]
    print(f"\n--- {path}: {len(hits)} of {len(rows)} ---")
    for h, s in hits:
        print(f"    {h}  {s[:92]}")

print()
print("=" * 74)
print("3. THE PORTED FILE, EVERY COMMIT, NO KEYWORD FILTER AT ALL")
print("   (small enough to read; a filter here would be the weak instrument)")
print("=" * 74)
rows = history(REPO, "ggml/src/ggml.c")
print(f"{len(rows)} commits touch ggml/src/ggml.c in llama.cpp.")
hits = [(h, s) for h, s in rows if TEMPORAL.search(s)]
print(f"{len(hits)} have any lifetime word in the subject:\n")
for h, s in hits:
    print(f"    {h}  {s[:92]}")
