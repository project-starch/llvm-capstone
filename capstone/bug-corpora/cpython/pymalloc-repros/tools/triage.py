#!/usr/bin/env python3
"""Triage the temporal defects live in the pinned CPython 3.13.7, on three axes.

The live set is two disjoint groups, and they are proven live in different ways:

  A  fixed on the 3.13 branch AFTER our v3.13.7 tag. The backport itself is the
     proof: a fix cherry-picked into 3.13 says the defect was in 3.13.
  B  fixed on main and never backported to 3.13 at all.

Axis 1  consumer-side C defect, not documentation, tests, build files, or the
        allocator's own hardening
Axis 2  reproducible without true concurrency -- one hart, no free-threading,
        no subinterpreters
Axis 3  WHICH ALLOCATOR the freed object came from, because CPython stacks
        three of them and the port covers only the middle one:

          freelist  ten per-type free lists; a free stops there and never
                    reaches pymalloc.  NOT covered by the port.
          arena     PyArena, the parser/AST bump arena.  NOT covered.
          pymalloc  everything else at or below the 512-byte threshold.
                    COVERED.

Axis 3 is assigned from the file the fix touches, which is a proxy: a defect in
listobject.c is very likely about a list, and lists are freelist-managed. Where
the proxy is weak the case is marked 'unclear' rather than counted.
"""
import os
import re
import subprocess

# An input, not a constant: point it at any CPython clone. The default is where
# the 2026-09-18 run had it.
REPO = os.environ.get("CPYTHON_REPO", "/tmp/claude-1000/corpus-research/cpython.git")
PAT = ("use.after.free|use-after-free|double free|dangling|freed memory|"
       "stale pointer")
GH = re.compile(r"gh-(\d+)|bpo-(\d+)", re.I)
CONC = re.compile(r"thread|race|free-threaded|freethread|interpreter finaliz"
                  r"|across interpreters|concurrent|semaphore", re.I)

# Files implementing a per-type free list in 3.13 (pycore_freelist.h names them).
FREELIST_FILES = {
    "Objects/floatobject.c", "Objects/tupleobject.c", "Objects/listobject.c",
    "Objects/dictobject.c", "Objects/sliceobject.c", "Python/context.c",
    "Objects/genobject.c",
}
ARENA_FILES = {"Python/pyarena.c", "Parser/pegen.c", "Parser/tokenizer.c",
               "Python/ast.c", "Python/compile.c", "Python/assemble.c"}
ALLOCATOR_FILES = {"Objects/obmalloc.c", "Include/internal/pycore_obmalloc.h",
                   "Include/internal/pycore_freelist.h"}


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                          text=True).stdout


def entries(*rev_args):
    # Every revision argument passed separately. Folding "--since=X ref" into
    # one string made git see a single token, silently matched nothing, and
    # reported an empty group as a result rather than as the bug it was.
    out = git("log", "--format=%x01%h%x02%s", "-E", "-i", f"--grep={PAT}",
              *rev_args)
    rows = {}
    for rec in out.split("\x01"):
        if not rec.strip():
            continue
        h, s = (rec.split("\x02") + [""])[:2]
        m = GH.search(s)
        rows.setdefault((m.group(1) or m.group(2)) if m else h, (h, s.strip()))
    return rows


group_a = entries("v3.13.7..3.13")
main_since = entries("--since=2024-10-07", "main")
b313 = entries("--since=2024-10-07", "3.13")
group_b = {k: v for k, v in main_since.items() if k not in b313}

live = dict(group_b)
live.update(group_a)

result = {"noise": [], "concurrency": [], "allocator-internal": [],
          "freelist": [], "arena": [], "pymalloc": [], "unclear": []}

for key, (h, subject) in live.items():
    files = [f for f in git("show", "--stat=400", "--format=", "--name-only", h).split("\n") if f]
    code = [f for f in files if f.endswith((".c", ".h"))]
    if not code:
        result["noise"].append(subject); continue
    if CONC.search(subject):
        result["concurrency"].append(subject); continue
    if any(f in ALLOCATOR_FILES for f in code):
        result["allocator-internal"].append(subject); continue
    if any(f in FREELIST_FILES for f in code):
        result["freelist"].append(subject); continue
    if any(f in ARENA_FILES for f in code):
        result["arena"].append(subject); continue
    if any(f.startswith(("Modules/", "Objects/", "Python/")) for f in code):
        result["pymalloc"].append(subject); continue
    result["unclear"].append(subject)

print(f"live in 3.13.7: {len(live)}  "
      f"(A: backported into 3.13 after v3.13.7 = {len(group_a)}, "
      f"B: never backported to 3.13 = {len(group_b)})\n")
order = [("noise", "no C code"), ("concurrency", "needs true concurrency"),
         ("allocator-internal", "the allocator itself, not a consumer"),
         ("arena", "PyArena -- no port"),
         ("freelist", "type free list -- no port"),
         ("pymalloc", "PYMALLOC -- covered by the port"),
         ("unclear", "unclear")]
for k, label in order:
    print(f"{len(result[k]):>3}  {label}")
print("\n--- covered by the port ---")
for s in sorted(result["pymalloc"]):
    print("   ", s[:100])
print("\n--- free-list layer (no port) ---")
for s in sorted(result["freelist"]):
    print("   ", s[:100])
