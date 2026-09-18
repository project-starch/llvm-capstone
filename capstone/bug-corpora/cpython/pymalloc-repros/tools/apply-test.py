#!/usr/bin/env python3
"""Group B, tested instead of assumed: does the pre-fix code exist in v3.13.7?

The 3.13.7 triage split the live set in two. Group A is proven live by its own
backport. Group B -- fixed on main, never backported to 3.13 -- is NOT proven:
"never backported" usually means the code did not exist yet, which the 3.10
exercise measured at about 12% survival.

The sharp test is whether the upstream FIX still applies to a pristine v3.13.7
tree. If it does, the pre-fix code is there verbatim and the defect is live.

Two forms of the test are run and both are printed, because they can disagree
and the disagreement is informative:

  full  the whole commit, tests and NEWS included. Strict: a test file that has
        since moved fails the check without saying anything about the defect.
  code  only the .c/.h hunks. This is the one that answers the question.

CONTROL. The same test is run over group A, whose answer is known independently:
those fixes were cherry-picked onto the 3.13 branch at or just after our tag, so
they SHOULD apply. A run where group A and group B score the same has measured
the instrument, not the subject.
"""
import os
import re
import subprocess
import tempfile
import pathlib

# Both paths are inputs, not constants: point them at any CPython clone and at a
# pristine worktree of the pin being tested. The defaults are where the
# 2026-09-18 run had them.
REPO = os.environ.get("CPYTHON_REPO", "/tmp/claude-1000/corpus-research/cpython.git")
TREE = os.environ.get("CPYTHON_TREE", "/tmp/claude-1000/corpus-research/cpy313")
PATCH = pathlib.Path(tempfile.gettempdir()) / "cpy-apply-test.patch"

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


def code_files(h):
    return [f for f in git("show", "--stat=400", "--format=", "--name-only",
                           h).split("\n") if f.endswith((".c", ".h"))]


def is_pymalloc(h, subject):
    """The triage's axis-3 bucket, reproduced so the two agree by construction."""
    files = [f for f in git("show", "--stat=400", "--format=", "--name-only",
                            h).split("\n") if f]
    code = [f for f in files if f.endswith((".c", ".h"))]
    if not code or CONC.search(subject):
        return False
    if any(f in ALLOCATOR_FILES or f in FREELIST_FILES or f in ARENA_FILES
           for f in code):
        return False
    return any(f.startswith(("Modules/", "Objects/", "Python/")) for f in code)


def applies(h, only_code):
    """git apply --check, never piped: the exit status is the whole result."""
    args = ["show", h]
    if only_code:
        cf = code_files(h)
        if not cf:
            return None
        args += ["--"] + cf
    PATCH.write_text(git(*args))
    if not PATCH.read_text().strip():
        return None
    return subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=TREE,
                          capture_output=True).returncode == 0


def run(label, rows):
    live_code, dead_code, live_full = [], [], 0
    for h, subject in rows:
        c = applies(h, only_code=True)
        f = applies(h, only_code=False)
        if f:
            live_full += 1
        if c is None:
            continue
        (live_code if c else dead_code).append((subject, h))
    print(f"\n=== {label}: {len(rows)} pymalloc-bucket cases ===")
    print(f"  fix applies, C/H hunks only -> code is there, defect LIVE : "
          f"{len(live_code)}")
    print(f"  fix does not apply          -> code absent or moved       : "
          f"{len(dead_code)}")
    print(f"  (whole commit, tests + NEWS included, applies             : "
          f"{live_full})")
    return live_code, dead_code


group_a = entries("v3.13.7..3.13")
main_since = entries("--since=2024-10-07", "main")
b313 = entries("--since=2024-10-07", "3.13")
group_b = {k: v for k, v in main_since.items() if k not in b313}

a_rows = [(h, s) for h, s in group_a.values() if is_pymalloc(h, s)]
b_rows = [(h, s) for h, s in group_b.values() if is_pymalloc(h, s)]

a_live, a_dead = run("GROUP A (control: should mostly apply)", a_rows)
b_live, b_dead = run("GROUP B (the question)", b_rows)

print("\n--- GROUP B, proven live by the apply test ---")
for s, h in sorted(b_live):
    print(f"    {h}  {s[:96]}")
print("\n--- GROUP B, fix does NOT apply: not counted ---")
for s, h in sorted(b_dead):
    print(f"    {h}  {s[:96]}")
print("\n--- GROUP A control, fix does NOT apply (expected to be few) ---")
for s, h in sorted(a_dead):
    print(f"    {h}  {s[:96]}")
print(f"\nREACHABLE = group A ({len(a_rows)}) + group B proven ({len(b_live)}) "
      f"= {len(a_rows) + len(b_live)}")
