#!/usr/bin/env python3
"""At v1.4.3, ggml_backend_buffer_free_tensor was a real public per-tensor free.

Two things decide whether an older pin gives the corpus anything:
  1. did it actually RETURN storage that a later allocation could reuse, or was
     it an empty hook most backends ignored;
  2. did any CONSUMER call it -- a free nobody calls cannot be misused.

Scanning by basename earlier produced a two-release-too-wide answer, because
bindings/ruby/ext/ carries a stale vendored copy of the header. Real paths only
here.
"""
import subprocess

W = "/tmp/claude-1000/corpus-research/whispercpp.git"
TAG = "v1.4.3"


def git(*a):
    return subprocess.run(["git", "-C", W, *a], capture_output=True,
                          text=True).stdout


def show(p):
    return git("show", f"{TAG}:{p}")


print("=" * 78)
print(f"{TAG}: the implementation")
print("=" * 78)
src = show("ggml-backend.c").split("\n")
for i, l in enumerate(src):
    if "ggml_backend_buffer_free_tensor" in l and "(" in l:
        lo, hi = max(0, i - 1), min(len(src), i + 10)
        for j in range(lo, hi):
            print(f"  {j+1:>5}  {src[j][:94]}")
        print()

print("=" * 78)
print(f"{TAG}: what the CPU backend does with it")
print("=" * 78)
for i, l in enumerate(src):
    if "free_tensor" in l and ("cpu" in l.lower() or "/*" in l or "NULL" in l):
        print(f"  {i+1:>5}  {l[:94]}")

print()
print("=" * 78)
print(f"{TAG}: every caller, anywhere in the tree")
print("=" * 78)
files = [p for p in git("ls-tree", "-r", "--name-only", TAG).split()
         if p.endswith((".c", ".cpp", ".h", ".m", ".cu"))]
callers = {}
for p in files:
    n = show(p).count("ggml_backend_buffer_free_tensor")
    if n:
        callers[p] = n
for p, n in sorted(callers.items()):
    kind = "DECLARATION/IMPL" if p.rsplit("/", 1)[-1].startswith("ggml-backend") else "CALLER"
    print(f"  {n:>3}x  {p:<52} {kind}")
