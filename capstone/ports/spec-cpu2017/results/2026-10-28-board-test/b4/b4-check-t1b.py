#!/usr/bin/env python3
"""b4-check-t1b.py <transcript> <run.py> <contract.dom> <twin result.json>: judge part 2 of B's first board boot (PREREG
Addendum 14) with run.py's OWN analysis block, lifted verbatim from <run.py> (69fb6cae3db4: the text between
`lines = [s.strip()` and `result = {`) and executed over the transcript's lines from the stage's `@@B4-STAGE t1b`
marker on. board-check.py's method (vp2-kit-2026-10-17/board-tier12), with the attributes run.py@69fb6cae3db4 reads.
The prediction (module-ids55 PREREG section 2, prediction 2): every cell equals the twin's F0 cell, except `cleanup`,
false on the board by construction (the board kernel cannot unload a module).
Exit 0: as predicted; 1: a cell differs; 2: no data. Needs CAPSTONE_LLVM_BIN (llvm-nm) for the denial-site cells."""
import json, os, re, subprocess, sys
from pathlib import Path

if len(sys.argv) != 5:
    sys.exit(__doc__)
tp, runpy, app, twin = (Path(x) for x in sys.argv[1:])
if not tp.is_file():
    print(f"NO DATA: {tp} does not exist"); sys.exit(2)
src = runpy.read_text()
i = src.index("    lines = [s.strip() for s in out.decode(errors='replace').splitlines()]")
j = src.index("    result = {'status'", i)
block = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i:j].splitlines()[1:])
text = tp.read_text(errors="replace").replace("\r", "")
k = text.find("@@B4-STAGE t1b")
if k < 0:
    print(f"NO DATA: no @@B4-STAGE t1b marker in {tp}"); sys.exit(2)
lines = [s.strip() for s in text[k:].splitlines()]
if not any(s.startswith("VM_EXIT:") for s in lines):
    print(f"NO DATA: no VM_EXIT line in {tp}"); sys.exit(2)


class A:
    pass


a = A()
a.skip_recycling = False; a.legacy_application = None; a.perl = None; a.perl_smoke = None; a.php = None
a.exact_bounds = False; a.application = app
ns = {"lines": lines, "a": a, "re": re, "os": os, "subprocess": subprocess, "Path": Path}
exec("tests = {}\n" + block, ns)
tests = ns["tests"]
ref = json.loads(twin.read_text())["tests"]
want = dict(ref); want["cleanup"] = False
off = sorted(n for n in set(tests) | set(want) if tests.get(n) != want.get(n))
for n in sorted(set(tests) | set(want)):
    print(f"  {'OK ' if n not in off else 'OFF'} {n:28} board {tests.get(n)} twin {ref.get(n)}" + ("   (cleanup: false by construction)" if n == "cleanup" else ""))
print(f"B4-T1B {sum(bool(v) for v in tests.values())} of {len(tests)} true; twin F0 {sum(ref.values())} of {len(ref)}; "
      + ("AS PREDICTED (every cell equals F0's, cleanup false)" if not off else f"OFF THE PREDICTION in {len(off)} cells: {off}"))
sys.exit(1 if off else 0)
