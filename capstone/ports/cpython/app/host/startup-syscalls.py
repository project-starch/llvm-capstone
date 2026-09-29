#!/usr/bin/env python3
"""Which syscalls does CPython itself make while starting, that a domain does not serve?

Runs a NATIVE CPython 3.13.7 under `strace -f -k` (a stack per syscall), keeps
the syscalls the domain leaves unserved -- read from the domain-support.txt that
link-cpython-capstone.py writes -- and attributes each call to the first CPython
function on its stack, or to the loader/libc when there is none.

This is an APPROXIMATION of the domain, and says so in its output: the native
run uses glibc and a dynamic loader, the domain musl and a static image. Calls
the loader makes (mapping libraries) and glibc's malloc makes (brk) do not exist
in a domain. What it is good for is the calls CPython makes on purpose.

Usage:  startup-syscalls.py <native-python> <domain-support.txt> [-- python args]
        default python args: -S -I -c pass
"""

import collections
import re
import shutil
import subprocess
import sys
import tempfile

# Linux syscall names that musl's checker spells differently.
ALIAS = {"getdents64": "getdents"}
CPYTHON_FRAME = re.compile(r"^(_?Py|_Py|pymain|PyOS|os_|signal_|arena|_PyMem|_PyObject)")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    python, support = sys.argv[1], sys.argv[2]
    pyargs = sys.argv[4:] if len(sys.argv) > 3 and sys.argv[3] == "--" else ["-S", "-I", "-c", "pass"]
    if not shutil.which("strace"):
        print("ERROR: strace not installed", file=sys.stderr)
        return 2
    unserved = set()
    for line in open(support):
        m = re.match(r"\s+\S+\s+needs (.+?)\s+src/", line)
        if m:
            unserved.update(m.group(1).split())
    if not unserved:
        print(f"ERROR: no 'needs' lines in {support}; is it check-domain-support.py output?",
              file=sys.stderr)
        return 2
    with tempfile.NamedTemporaryFile("r", suffix=".strace") as trace:
        done = subprocess.run(["strace", "-f", "-k", "-o", trace.name, python, *pyargs],
                              capture_output=True, text=True)
        text = open(trace.name).read()
    blocks, cur = [], None
    for line in text.splitlines():
        m = re.match(r"\d+\s+(\w+)\(", line)
        if m:
            cur = [m.group(1), []]
            blocks.append(cur)
        elif cur and line.lstrip().startswith(">"):
            f = re.search(r"\(([^)+]+)\+", line)
            if f:
                cur[1].append(f.group(1))
    if not blocks:
        print("ERROR: strace recorded no syscalls", file=sys.stderr)
        return 2
    if not any(frames for _, frames in blocks):
        print("ERROR: strace -k gave no stacks (strace built without unwinding?)", file=sys.stderr)
        return 2

    rows = collections.Counter()
    for call, frames in blocks:
        if ALIAS.get(call, call) not in unserved:
            continue
        who = next((f for f in frames if CPYTHON_FRAME.match(f)), None)
        rows[(call, "CPython " + who if who else "loader/libc " + (frames[0] if frames else "?"))] += 1
    print(f"native run: {python} {' '.join(pyargs)} -> exit {done.returncode}, "
          f"{len(blocks)} syscalls")
    print(f"unserved in the domain (from {support}): {len(unserved)} syscalls")
    print("calls to unserved syscalls, by first CPython frame "
          "(glibc + dynamic loader: an approximation of a musl static domain):")
    for (call, origin), n in sorted(rows.items(), key=lambda r: (r[0][1].startswith("loader"), r[0][0], -r[1])):
        print(f"  {n:4d}  {call:14s} {origin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
