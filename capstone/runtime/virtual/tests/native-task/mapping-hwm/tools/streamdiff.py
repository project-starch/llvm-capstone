"""streamdiff.py <log a> <log b>: compare two TEST_LOG request streams (acquires M/U, releases C/F, remaps X, with
their sizes; the policy's own evictions E/R/H/P dropped). Prints the lengths and the first index where they differ, or
IDENTICAL. Exit 0 identical, 1 different, 2 no data. Two runs of one image under one policy that differ show the
program's own run-to-run variation, not the policy's."""
import re, sys
def stream(p):
    s = [("A" if m.group(1) in "MU" else "R" if m.group(1) in "CF" else "X", int(m.group(2)), int(m.group(3) or 0))
         for m in re.finditer(r"^@@HM ([MUCFX]) (\d+) \d+ \d+ \d+(?: (\d+))?$", open(p, errors="replace").read(), re.M)]
    if not s: print("NO DATA:", p); sys.exit(2)
    return s
a, b = stream(sys.argv[1]), stream(sys.argv[2])
print(f"lengths {len(a)} {len(b)}")
for i, (u, v) in enumerate(zip(a, b)):
    if u != v:
        print(f"DIFFER at {i}: {u} {v}; the {i} before are equal"); sys.exit(1)
if len(a) != len(b): print(f"DIFFER in length after an equal prefix of {min(len(a), len(b))}"); sys.exit(1)
print("IDENTICAL"); sys.exit(0)
