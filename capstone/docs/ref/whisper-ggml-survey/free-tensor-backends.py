#!/usr/bin/env python3
"""At v1.4.3, did ANY backend implement the free_tensor hook non-NULL?

The CPU backend sets it to NULL, and the domain runs CPU -- but "the CPU backend
does not implement it" is a narrower claim than "nobody does", and the wider one
is the one worth making if it is true.
"""
import re
import subprocess

W = "/tmp/claude-1000/corpus-research/whispercpp.git"
TAG = "v1.4.3"


def git(*a):
    return subprocess.run(["git", "-C", W, *a], capture_output=True,
                          text=True).stdout


files = [p for p in git("ls-tree", "-r", "--name-only", TAG).split()
         if p.endswith((".c", ".cpp", ".m", ".cu", ".h"))
         and not p.startswith("bindings/")]
print(f"{TAG}: every '.free_tensor =' initialiser in the tree\n")
found = False
for p in files:
    for i, l in enumerate(git("show", f"{TAG}:{p}").split("\n")):
        if re.search(r"\.free_tensor\s*=", l) or ".free_tensor    = " in l:
            found = True
            val = l.split("=", 1)[1].strip() if "=" in l else "?"
            print(f"  {p}:{i+1}  -> {val[:64]}")
if not found:
    print("  (none found -- check the spelling of the initialiser)")
