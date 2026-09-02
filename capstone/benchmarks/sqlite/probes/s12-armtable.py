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
    # The TIGHT arm exists because ANCHOR vs CTRL changes five instructions and therefore cannot
    # attribute. This one changes two: [28] sw a4 -> movc t0, zero and [33] stc a4 -> stc t0, with
    # [26] [27] [30] [32] left on a4. The only property that moves is which register supplies the
    # null the store writes.
    ("TIGHT",  "tight",  "sqtight.dom", "a6e853e25888958c", "a4=0, null store uses t0 (2-instr)"),
]


def read(path, dom, sha):
    try:
        raw = open(path, errors="replace").read().replace("\r", "")
    except OSError:
        return None
    # SCOPE THE GUEST-DERIVED FACTS TO THIS RUN'S OWN TEST STAGE. The console replays a few
    # hundred KB of the PREVIOUS boot on connect, so `SQ: G/enter` and the SLT-SUMMARY of the
    # previous draw are both present in a log whose own domain never ran. Read over the whole
    # file, an entry-stalled draw inherits the previous draw's markers and scores clean -- which
    # is the same replay hazard that produced today's retraction, one level up.
    #
    # The driver's own lines (staging, sha verification, the wedge/return summary) are NOT
    # replayed, so those stay whole-file.
    _cut = raw.rfind("--> TEST")
    guest = raw[_cut:] if _cut != -1 else ""
    entered = "SQ: G/enter" in guest
    m = re.findall(rf"verifying {re.escape(dom)}\s+sha256=([0-9a-f]{{16}})", raw)
    got = m[-1] if m else None
    # THE CONTROL SLOT MUST HAVE PASSED. A boot whose known-good control fails carries no verdict
    # about anything -- it separates "this image failed" from "the board, firmware or boot failed",
    # and the control fails often enough that this is not a formality. Checked here rather than by
    # hand, because a check performed by hand is a check that will eventually be skipped.
    ctl_ok = bool(re.search(r"sqslt\.dom:--slt \S*dd1_one\S*\s+returned in \d+s", raw))
    wedged = "NO RETURN within" in raw
    returned = "Every domain returned" in raw
    slt = re.findall(r"SLT-SUMMARY [^\n]*completed=1", guest)
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
                completed=bool(slt), ctl_ok=ctl_ok, **fields)


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
            if not r["ctl_ok"]:
                bad.append("control slot did not pass -- boot carries no verdict")
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
