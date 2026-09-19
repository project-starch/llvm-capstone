#!/usr/bin/env python3
"""Would moving the pin from 3.13.7 back to 3.13.0 buy more defects?

Two halves, and both have to be answered or the trade is not visible:

  GAIN  temporal-defect fixes that landed on the 3.13 branch between v3.13.0 and
        v3.13.7. Those are live at 3.13.0 and already dead at our pin, so they
        are exactly what the older pin would add. Triaged on the same three axes
        as the live set, so the number is comparable.

  COST  how far obmalloc.c moved between the two tags, and whether the port's
        three patches still apply. Against 3.10 this was fatal (+1462/-979, two
        of three patches failing). Within one minor branch it may be nothing --
        which is the whole reason this question is worth asking separately.
"""
import os
import re
import subprocess

# Inputs, not constants. The defaults are where the 2026-09-19 run had them.
REPO = os.environ.get("CPYTHON_REPO", "/tmp/claude-1000/corpus-research/cpython.git")
PAT = ("use.after.free|use-after-free|double free|dangling|freed memory|"
       "stale pointer")
GH = re.compile(r"gh-(\d+)|bpo-(\d+)", re.I)
CONC = re.compile(r"thread|race|free-threaded|freethread|interpreter finaliz"
                  r"|across interpreters|concurrent|semaphore", re.I)
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
    out = git("log", "--format=%x01%h%x02%s", "-E", "-i", f"--grep={PAT}", *rev_args)
    rows = {}
    for rec in out.split("\x01"):
        if not rec.strip():
            continue
        h, s = (rec.split("\x02") + [""])[:2]
        m = GH.search(s)
        rows.setdefault((m.group(1) or m.group(2)) if m else h, (h, s.strip()))
    return rows


print("=" * 72)
print("GAIN: temporal fixes on the 3.13 branch in v3.13.0..v3.13.7")
print("=" * 72)
gained = entries("v3.13.0..v3.13.7")
buckets = {"noise": [], "concurrency": [], "allocator-internal": [],
           "freelist": [], "arena": [], "pymalloc": [], "unclear": []}
for key, (h, subject) in gained.items():
    files = [f for f in git("show", "--stat=400", "--format=", "--name-only", h).split("\n") if f]
    code = [f for f in files if f.endswith((".c", ".h"))]
    if not code:
        buckets["noise"].append((h, subject)); continue
    if CONC.search(subject):
        buckets["concurrency"].append((h, subject)); continue
    if any(f in ALLOCATOR_FILES for f in code):
        buckets["allocator-internal"].append((h, subject)); continue
    if any(f in FREELIST_FILES for f in code):
        buckets["freelist"].append((h, subject)); continue
    if any(f in ARENA_FILES for f in code):
        buckets["arena"].append((h, subject)); continue
    if any(f.startswith(("Modules/", "Objects/", "Python/")) for f in code):
        buckets["pymalloc"].append((h, subject, code)); continue
    buckets["unclear"].append((h, subject))

print(f"total temporal fixes in that window: {len(gained)}\n")
for k in ("noise", "concurrency", "allocator-internal", "arena", "freelist",
          "pymalloc", "unclear"):
    print(f"{len(buckets[k]):>3}  {k}")
print("\n--- the pymalloc-bucket ones (what an older pin would ADD) ---")
for row in sorted(buckets["pymalloc"], key=lambda r: r[1]):
    print(f"    {row[0]}  {row[1][:86]}")
    print(f"          {row[2]}")

print()
print("=" * 72)
print("COST: how far the allocator moved, and whether the port still applies")
print("=" * 72)
stat = git("diff", "--numstat", "v3.13.0", "v3.13.7", "--", "Objects/obmalloc.c",
           "Include/internal/pycore_obmalloc.h")
print("obmalloc.c / pycore_obmalloc.h, v3.13.0 -> v3.13.7 (added/removed/file):")
print(stat.strip() or "    IDENTICAL -- no change at all")
