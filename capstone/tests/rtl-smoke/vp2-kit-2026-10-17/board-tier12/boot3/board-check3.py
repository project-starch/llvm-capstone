#!/usr/bin/env python3
"""board-check.py <transcript> <run.py> <application contract.dom> [--rung NAME]: judge a vm-suite transcript with
run.py's OWN analysis block, lifted verbatim from <run.py> (the text between `lines = [s.strip()` and `result = {`) and
executed over the transcript's lines. With --rung, only the lines of that rung's segment (between `### RUNG <n>/<m>
<NAME>` and `### RUNG <n> <NAME> END`) are read, so a control's output cannot leak in. Prints every test and the
pass count; exit 0 only if every test in EXPECT_FALSE is false and every other test is true; 2 on no data.
EXPECT_FALSE: tests pre-registered false on the board by construction (env BOARD_EXPECT_FALSE, comma-separated)."""
import os, re, sys, subprocess, argparse
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('transcript'); p.add_argument('runpy'); p.add_argument('application')
p.add_argument('--rung'); a0 = p.parse_args()
src = Path(a0.runpy).read_text()
i = src.index("    lines = [s.strip() for s in out.decode(errors='replace').splitlines()]")
j = src.index("    result = {'status'", i)
block = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i:j].splitlines()[1:])
text = Path(a0.transcript).read_text(errors='replace').replace('\r', '')
if a0.rung:
    # Anchored at line start, END with a NUMERIC rc: the driver's echoed command line carries both markers literally
    # ('### RUNG n/m NAME ###' ... "### RUNG n NAME END rc=$rc ###"), and an unanchored match took that echo, which
    # holds no VM_EXIT line (found by a board-format dry run, 2026-10-17).
    ms = list(re.finditer(r"^### RUNG \d+/\d+ " + re.escape(a0.rung) + r" ###$(.*?)^### RUNG \d+ " + re.escape(a0.rung)
                          + r" END rc=-?\d+ ###$", text, re.S | re.M))
    if len(ms) == 0 and os.environ.get('BOARD_TO_END') == '1':
        # vp2 boot 3, option (b): the parallel pair runs last and the kernel panics in it, so vm-suite has no END line.
        # Read from the rung's start marker to the end of the transcript; exactly one start marker is required.
        st = list(re.finditer(r"^### RUNG \d+/\d+ " + re.escape(a0.rung) + r" ###$", text, re.M))
        if len(st) != 1: print(f"NO DATA: {len(st)} start markers for rung {a0.rung} in {a0.transcript} (need exactly 1)"); sys.exit(2)
        text = text[st[0].end():]; print("(BOARD_TO_END: no END marker; read to the end of the transcript)")
    elif len(ms) != 1: print(f"NO DATA: {len(ms)} segments for rung {a0.rung} in {a0.transcript} (need exactly 1)"); sys.exit(2)
    else: text = ms[0].group(1)
lines = [s.strip() for s in text.splitlines()]
if not any(s.startswith('VM_EXIT:') for s in lines): print(f"NO DATA: no VM_EXIT line in {a0.transcript}"); sys.exit(2)
class A: pass
a = A(); a.skip_recycling = False; a.legacy_application = None; a.perl = None; a.perl_smoke = None
a.application = Path(a0.application)
ns = {'lines': lines, 'a': a, 're': re, 'os': os, 'subprocess': subprocess, 'Path': Path, 'tests': {}}
exec("tests = {}\n" + block, ns)
tests = ns['tests']
expect_false = set(x for x in os.environ.get('BOARD_EXPECT_FALSE', '').split(',') if x)
bad = [k for k, v in tests.items() if v != (k not in expect_false)]
for k, v in tests.items(): print(f"  {k:28} {v}" + ("   (pre-registered false on the board)" if k in expect_false else ""))
print(f"{sum(tests.values())} of {len(tests)} true; {len(bad)} off the pre-registration {bad}")
sys.exit(1 if bad else 0)
