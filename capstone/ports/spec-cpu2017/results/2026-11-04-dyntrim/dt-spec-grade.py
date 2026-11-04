#!/usr/bin/env python3
"""dt-spec-grade.py: grade SPEC test on stage A (c9c7952bd, board 2, kit bkit19-perf-dt-b2) against
results/2026-11-04-dyntrim/PREREG.md: each arm of out-dt-b2 against the same stage on C.5a (board 2, out-c5a-b2: the same
images), with bitstream 3 (board 1, spec/board/out) reported. A stage not finished prints PENDING. Exit 0 if every graded
check holds, 1 otherwise, 2 if no data. DA1: control, rc, specdiff, outputs = twin, fresh, instret within 0.1 % of C.5a's.
DA2: on mcf, xz and gcc (LDC+STC at 2-3.4 % of cycles at one cycle each) stage A takes at least 0.2 % fewer cycles than
C.5a, and no more than 5 % fewer; nab, imagick and lbm are reported (their expected drop is under 1 %)."""
import json, os, sys
B = os.path.join(os.environ["CAPSTONE_WORK"], "spec", "board")
DT = os.environ.get("DT_OUT", os.path.join(B, "out-dt-b2")); C5 = os.environ.get("C5_OUT", os.path.join(B, "out-c5a-b2"))  # same board as stage A (board 2)
# bench: (stage name, arms run, lc-model's predicted Full/Spatial excess)
ST = {"gcc": ("502.gcc_r-test-capstone-lifetimes", 0.0), "imagick": ("538.imagick_r-test", 0.0),
      "nab": ("544.nab_r-test", 0.0), "lbm": ("519.lbm_r-test", 0.0),
      "xz": ("557.xz_r-test-capstone-lifetimes", 0.0), "mcf": ("505.mcf_r-test-capstone-lifetimes", 0.0075)}
FULL = {"gcc": "502.gcc_r", "imagick": "538.imagick_r", "nab": "544.nab_r", "lbm": "519.lbm_r", "xz": "557.xz_r", "mcf": "505.mcf_r"}
BAND = 0.01      # board 1 against board 2, and day against day: unmeasured, allowed 1 % either way
S3 = {"nab", "imagick", "xz", "mcf", "gcc"}   # Full_b3 / Spatial >= 1.018 today

def load(d, arm):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p): return None
    r = json.load(open(p))
    if not r.get("done") or arm not in r["arms"]: return None
    a = r["arms"][arm]; runs = [c for c in a["commands"] if c["tag"] == "run"]
    why = []
    if any(c["rc"] != 0 for c in a["commands"]): why.append("rc!=0")
    if not all(v["verdict"] == "PASS" for v in a["validation"]): why.append("specdiff")
    if r.get("vs_twin", {}).get(arm): why.append("twin diff")
    if not all(c.get("bench_line_fresh") for c in a["commands"]): why.append("not fresh")
    ctl = os.path.join(d, "control.result.json")
    if not os.path.exists(ctl) or json.load(open(ctl)) != {"kind": "exit", "value": 0}: why.append("control")
    return sum(c["cycles"] for c in runs), sum(c["instret"] for c in runs), why

fail = 0; seen = 0
for b, (st, exc) in ST.items():
    arms = ["capstone-lifetimes"] + (["capstone-spatial"] if b in ("lbm", "nab", "imagick") else [])
    for arm in arms:
        lab = {"capstone-lifetimes": "Full", "capstone-spatial": "MapLife"}[arm]
        dt = load(os.path.join(DT, st), arm); c5 = load(os.path.join(C5, st), arm); b3 = load(os.path.join(B, "out", st), arm)
        if c5 is None: print(f"{b:8s} {lab:8s} NO C.5a REFERENCE (yet)"); continue
        if dt is None: print(f"{b:8s} {lab:8s} PENDING"); continue
        seen += 1
        d1 = not dt[2] and abs(dt[1] / c5[1] - 1) <= 0.001
        r = dt[0] / c5[0]
        d2 = (0.95 <= r <= 0.998) if b in ("mcf", "xz", "gcc") else None
        if not d1 or d2 is False: fail = 1
        print(f"{b:8s} {lab:8s} DA1 {'holds' if d1 else 'FAILS ' + ','.join(dt[2]) + f' instret {dt[1] / c5[1]:.5f}'}"
              f" | stageA/C.5a cycles {r:.4f} DA2 {'rep' if d2 is None else ('holds' if d2 else 'FAILS')}"
              f" | stageA/bs3 {dt[0] / b3[0]:.4f}" if b3 else "")
if not seen: print(f"NO DATA: no finished stage-A stage under {DT}"); sys.exit(2)
sys.exit(fail)
