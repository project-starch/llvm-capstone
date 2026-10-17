#!/usr/bin/env python3
"""Which of contract.dom's modes create a domain thread? Measured per mode body.

My first classification named `threads` and `heap-threads` from the mode NAMES and from run.py's
expected exit codes. That missed `vm`, whose vm_contract() creates a thread to exercise an RX
mapping -- so the one-slot expectation was wrong by one step. This greps each mode's body.

Attribution rule: a mode "needs a second live context" if its body, or a function it calls that is
defined in the same file, creates a thread. Resolved one level of call depth, which is enough here
(`vm` -> vm_contract) and is stated rather than assumed.
"""
import re, pathlib, sys, collections

# Relative to this file, never absolute: HERE is <repo>/capstone/tests/rtl-smoke/<folder>, so
# parents[2] is <repo>/capstone. An absolute path would also carry the operator's account name
# into a committed file, which the name rule forbids.
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[2] / "runtime" / "virtual" / "contract.c"
CREATE = re.compile(r"capstone_virtual_thread_create|pthread_create|thrd_create")

def bodies(text):
    """name -> source text, for every top-level function."""
    out, starts = {}, []
    for m in re.finditer(r"^(?:static\s+)?[A-Za-z_][\w \*]*\b(\w+)\s*\([^;{]*\)\s*\{", text, re.M):
        starts.append((m.start(), m.group(1)))
    starts.append((len(text), None))
    for (a, name), (b, _) in zip(starts, starts[1:]):
        if name: out[name] = text[a:b]
    return out

def main():
    if not SRC.exists():
        print(f"NO DATA: {SRC} missing", file=sys.stderr); return 2
    t = SRC.read_text()
    fns = bodies(t)
    # the mode dispatch: every !strcmp(mode, "X")
    modes = re.findall(r'!strcmp\(mode,\s*"([^"]+)"\)', t)
    if not modes:
        print("NO DATA: no mode dispatch found", file=sys.stderr); return 2
    # map each mode to the text that runs for it: the if-body, plus any same-file callee
    main_fn = fns.get("main") or t
    needs, rows = [], []
    for mode in modes:
        i = main_fn.find(f'!strcmp(mode, "{mode}")')
        seg = main_fn[i:]
        j = re.search(r'!strcmp\(mode,\s*"', seg[20:])
        seg = seg[:j.start() + 20] if j else seg[:2500]
        direct = len(CREATE.findall(seg))
        # one level of callee resolution, named explicitly
        callees = [c for c in re.findall(r"\b(\w+)\s*\(", seg) if c in fns and c != mode]
        viacall = {c: len(CREATE.findall(fns[c])) for c in set(callees) if CREATE.search(fns[c])}
        tot = direct + sum(viacall.values())
        rows.append((mode, direct, viacall, tot))
        if tot: needs.append(mode)
    print(f"  {'mode':<16} {'direct':>6} {'via callee':>24} {'total':>5}")
    for mode, direct, via, tot in rows:
        print(f"  {mode:<16} {direct:>6} {str(via) if via else '-':>24} {tot:>5}"
              f"{'   <-- needs a 2nd live context' if tot else ''}")
    print(f"\nMODES NEEDING A SECOND LIVE CONTEXT: {len(needs)} -> {', '.join(needs)}")
    # positive control: the scan must find the one everybody agrees on
    if "threads" not in needs:
        print("CONTROL FAILED: `threads` not detected, so the zeros mean nothing", file=sys.stderr)
        return 2
    print("control: `threads` detected, so the scan fires")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
