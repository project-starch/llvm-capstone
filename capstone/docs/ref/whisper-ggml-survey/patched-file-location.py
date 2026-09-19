#!/usr/bin/env python3
"""Where does the file our port patches live, release by release?

The port's three patches all target ggml/src/ggml.c. If an older pin keeps that
file somewhere else, the patches do not apply at all and the "older pin" question
is answered by the port cost before it is answered by the defect count.
"""
import re
import subprocess

W = "/tmp/claude-1000/corpus-research/whispercpp.git"


def git(*a):
    return subprocess.run(["git", "-C", W, *a], capture_output=True,
                          text=True).stdout


tags = sorted({t for t in git("tag", "-l").split()
               if re.match(r"^v\d+\.\d+\.\d+$", t)},
              key=lambda t: tuple(int(x) for x in t[1:].split(".")))
prev = None
for t in tags:
    names = [p for p in git("ls-tree", "-r", "--name-only", t).split()
             if p.rsplit("/", 1)[-1] == "ggml.c" and not p.startswith("bindings/")]
    cur = tuple(names)
    if cur != prev:
        print(f"  {t:<10} {names}")
        prev = cur
