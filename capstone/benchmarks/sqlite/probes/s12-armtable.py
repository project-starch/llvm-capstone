#!/usr/bin/env python3
"""Tabulate the S-12 three-arm ladder from the board logs, one classifier for all arms.

WHY ONE CLASSIFIER. The single most expensive habit in this investigation has been reading arms
that were scored by different rules -- different slots, different images, different greps -- and
comparing the numbers as if they were commensurable. This reads every arm the same way, prints the
per-draw evidence rather than only the tally, and refuses to score a draw whose image it cannot
identify.

A draw counts only if:
  * the domain ENTERED (`SQ: G/enter`), otherwise it is an entry stall and says nothing;
  * the per-run sha256 of the arm's own .dom matches what the arm is supposed to be.
Anything else is VOID and is reported as VOID, never folded into the denominator.
"""
import argparse, os, re, sys, subprocess

ROOT = "/home/alexey/dev/llvm-capstone"
VERDICT = f"{ROOT}/capstone/benchmarks/sqlite/probes/s12-verdict.py"

ARMS = [
    # label,       log glob prefix,  .dom name,    expected sha16,      what it holds
    ("ANCHOR", "anchor", "sqbase.dom", "69fe70b767e76b2b", "a4=0, null store uses a4"),
    ("CTRL",   "ctrl",   "sqctl.dom",  "145beaef6a426abe", "a4=0, null store uses t0"),
    ("SENT",   "sent",   "sqli.dom",   "d0245dae79df868c", "a4=0x5a5, null store uses t0"),
]


def read(path, dom, sha):
    try:
        raw = open(path, errors="replace").read().replace("\r", "")
    except OSError:
        return None
    entered = "SQ: G/enter" in raw
    m = re.findall(rf"verifying {re.escape(dom)}\s+sha256=([0-9a-f]{{16}})", raw)
    got = m[-1] if m else None
    wedged = "NO RETURN within" in raw
    returned = "Every domain returned" in raw
    slt = re.findall(r"SLT-SUMMARY [^\n]*completed=1", raw)
    fields = {}
    try:
        out = subprocess.run(["python3", VERDICT, path], capture_output=True, text=True,
                             timeout=120).stdout
        for k in ("mepc", "tval", "DBAS", "VA"):
            mm = re.search(rf"^\s*{k}\s+(\S+)", out, re.M)
            if mm:
                fields[k] = mm.group(1)
    except Exception:
        pass
    return dict(entered=entered, sha=got, wedged=wedged, returned=returned,
                completed=bool(slt), **fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/capstone")
    a = ap.parse_args()
    any_void = False
    for label, pref, dom, sha, shape in ARMS:
        logs = sorted(f for f in os.listdir(a.dir)
                      if re.fullmatch(rf"{pref}-\d+\.log", f))
        print(f"\n=== {label:6s}  {shape}")
        if not logs:
            print("    no draws")
            continue
        wedges = valid = 0
        for f in logs:
            r = read(os.path.join(a.dir, f), dom, sha)
            if r is None:
                continue
            bad = []
            if not r["entered"]:
                bad.append("no G/enter (entry stall)")
            if r["sha"] is None:
                bad.append(f"no per-run sha for {dom}")
            elif r["sha"] != sha:
                bad.append(f"sha {r['sha']} != expected {sha}")
            if bad:
                any_void = True
                print(f"    {f:16s} VOID: {'; '.join(bad)}")
                continue
            valid += 1
            if r["wedged"]:
                wedges += 1
                print(f"    {f:16s} WEDGE   mepc={r.get('mepc','?')} tval={r.get('tval','?')} "
                      f"VA={r.get('VA','?')}")
            else:
                print(f"    {f:16s} clean   completed={r['completed']}")
        print(f"    -> {wedges} wedges / {valid} valid draws")
    if any_void:
        print("\nSome draws are VOID. They are excluded from the denominators above; a rate "
              "computed over them would be wrong in the direction that flatters a cure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
