#!/usr/bin/env python3
"""Moving the pin back only gains if nothing is LOST.

A defect introduced after 3.13.0 and fixed after 3.13.7 is live at our pin and
absent from the older one. Rare, but "more bugs" is not a safe assumption until
it is checked, so run the same apply test the corpus used -- does each of the
twenty fixes still apply to a pristine v3.13.0 tree? -- over the corpus itself.

Sufficient, not necessary, exactly as before: "applies" proves the pre-fix code
is there; a failure means read the case, not that it is gone.
"""
import os
import pathlib
import subprocess
import tempfile

# Inputs, not constants. The defaults are where the 2026-09-19 run had them.
REPO = os.environ.get("CPYTHON_REPO", "/tmp/claude-1000/corpus-research/cpython.git")
TREE = os.environ.get("CPYTHON_TREE", "/tmp/claude-1000/corpus-research/cpy3130")
PATCH = pathlib.Path(tempfile.gettempdir()) / "twenty-3130.patch"

# The corpus, in driver order: (case, gh id).
CORPUS = [
    (0, "143543"), (1, "146613"), (2, "142829"), (3, "142831"), (4, "145244"),
    (5, "148660"), (6, "151295"), (7, "148395"), (8, "112127"), (9, "139210"),
    (10, "142560"), (11, "142783"), (12, "143004"), (13, "144833"),
    (14, "146011"), (15, "149449"), (16, "151403"), (17, "151416"),
    (18, "151695"), (19, "153539"),
]


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                          text=True).stdout


def code_files(h):
    return [f for f in git("show", "--format=", "--name-only", h).split("\n")
            if f.endswith((".c", ".h"))]


def pick(gid):
    out = git("log", "--format=%H%x02%s", "--all", "-i", f"--grep=gh-{gid}")
    rows = [r.split("\x02") for r in out.strip().split("\n") if r.strip()]
    rows = [r for r in rows if code_files(r[0])]
    if not rows:
        return None
    rows.sort(key=lambda r: r[1].startswith("[3.1"))
    return rows[0]


live, dead = [], []
for case, gid in CORPUS:
    row = pick(gid)
    if row is None:
        dead.append((case, gid, "no commit with C files found"))
        continue
    h, subject = row
    cf = code_files(h)
    PATCH.write_text(git("show", "--format=", h, "--", *cf))
    rc = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=TREE,
                        capture_output=True).returncode
    (live if rc == 0 else dead).append((case, gid, subject[:70]))

print(f"the corpus's twenty, apply-tested against v3.13.0\n")
print(f"  fix applies -> pre-fix code is there, defect LIVE at 3.13.0 : {len(live)}")
print(f"  fix does not apply -> needs reading, not counting           : {len(dead)}\n")
print("--- does not apply ---")
for case, gid, why in dead:
    print(f"    case {case:>2}  gh-{gid}  {why}")
