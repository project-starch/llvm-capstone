#!/usr/bin/env python3
"""s26-judge.py BOARD_TXT ARM (var|cur|mir): S-26's pair, read from the run's console. Exit 0 as predicted, 1 not, 2 no data.
Predictions (pre-registered 2026-10-19, before the boots; the RTL lane's simulation on bf3db3175: a context entering with
quantum 0 retires nothing and escapes kind 1, bit-identical for budgets 40 to 1,500; 2,000 completes):
- QNT0, 0x7C3 as boot leaves it: var 001E8480 (2,000,000); cur and mir 00000000.
- QNT1 (mir only, the census handler's read-back): 000003E8 (1,000) at every b0-stats.
- vm-contract: var returns, retval 0, rev7's identity branch (VIRTUAL_APPLICATION_OK, then stale denied with 25).
  cur and mir SPIN: retval 930 (VM_SPIN at the 150 s bound), with resume_satp_refreshes STRICTLY GROWING over
  t30 < t90 < t150 (the candidate's escape/resume loop). A spin with FLAT counters is the other reading (escaped and
  never resumed), reported as such.
- mir counts only if b0-hello returned 0 (b0-hello's process-ABI STEP ran with 2,000,000): otherwise VOID."""
import re, sys
t = open(sys.argv[1], errors="replace").read().replace("\r", ""); arm = sys.argv[2]
def rung(r):
    m = re.findall(rf"^RESULT {r} retval=(-?\d+)", t, re.M); return m[-1] if m else None
out, bad = [], 0
def say(ok, msg):
    global bad; out.append(("OK   " if ok else "FAIL ") + msg); bad += not ok
# QNT0 is printed in cap_env_init, after this boot's OpenSBI banner and before Linux starts; board.txt holds only the
# rungs, so it is read from board-raw.txt beside it (fixed 2026-10-19 after the var arm, whose raw console carried
# QNT0:001E8480 at offset 37935, after its banner at 35610, while board.txt had none; the prediction is unchanged).
# board-raw.txt replays earlier boots first, so only the text after this boot's (the LAST) "OpenSBI v" banner is read:
# a replayed boot's QNT0 can never be taken as this one's.
import os
rawp = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "board-raw.txt")
raw = open(rawp, errors="replace").read().replace("\r", "") if os.path.exists(rawp) else ""
b = raw.rfind("OpenSBI v")
q0 = re.findall(r"QNT0:([0-9A-F]{8})", raw[b:]) if b >= 0 else []
q1 = re.findall(r"QNT1:([0-9A-F]{8})", t)
if len(q0) != 1: print(f"NO DATA: {len(q0)} QNT0 lines in this boot's segment of board-raw.txt (want 1)"); sys.exit(2)
say(q0 == [{"var": "001E8480"}.get(arm, "00000000")], f"QNT0 {q0}")
if arm == "mir":
    say(len(q1) >= 2 and set(q1) == {"000003E8"}, f"QNT1 {q1} (want 000003E8 at each b0-stats, at least 2)")
    if rung("b0-hello") != "0": print("\n".join(out)); print(f"VOID: b0-hello read {rung('b0-hello')}, not 0"); sys.exit(2)
else:
    say(not q1, f"no QNT1 line in {arm} {q1}")
vc = rung("vm-contract")
if vc is None: print("\n".join(out)); print("NO DATA: no vm-contract RESULT (wedge or cut)"); sys.exit(2)
ref = {w: int(v) for v, w in re.findall(r"^@@PARAM resume_satp_refreshes=(\d+):(t30|t90|t150)$", t, re.M)}
if arm == "var":
    say(vc == "0", f"vm-contract retval {vc} (want 0, identity branch)")
    say("VIRTUAL_APPLICATION_OK" in t and re.search(r"domain fault cause=25 ", t) is not None, "identity branch lines present")
else:
    say(vc == "930" and "VM_SPIN:ok at the 150 s bound" in t, f"vm-contract retval {vc} (want 930, VM_SPIN)")
    if vc != "930":
        out.append(f"NOTE {arm} did not spin (retval {vc}): the candidate's prediction for this arm FAILS")
    elif len(ref) == 3:
        grow = ref["t30"] < ref["t90"] < ref["t150"]
        out.append(("OK   " if grow else "NOTE ") + f"resume_satp_refreshes t30/t90/t150 = {ref['t30']}/{ref['t90']}/{ref['t150']}: "
                   + ("strictly growing (escape/resume loop)" if grow else "NOT growing (escaped, not resumed)"))
        bad += not grow
    else:
        out.append(f"NO DATA: refreshes readings {ref}"); print("\n".join(out)); sys.exit(2)
print("\n".join(out)); print("JUDGE PASS" if not bad else f"JUDGE FAIL ({bad})"); sys.exit(1 if bad else 0)
