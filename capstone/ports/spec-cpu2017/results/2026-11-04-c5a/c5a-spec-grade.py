#!/usr/bin/env python3
"""c5a-spec-grade.py: grade SPEC test on the C.5a bitstream (board 1, kit bkit17-perf-c5a-b1) against
results/2026-11-04-c5a/PREREG.md. The C.5a run is $CAPSTONE_WORK/spec/board/out-c5a/<stage>; the references are the
3 Nov bitstream 3 run on board 1 (spec/board/out/<stage>) and the ObjBounds Spatial run on board 2
(spec/board/out-objb/<bench>-test-capstone-spatial). The same images run in all three.
A stage not yet finished prints PENDING, never a number. Exit 0 if every graded check holds, 1 otherwise,
2 if no C.5a stage has finished."""
import json, os, sys

B = os.path.join(os.environ["CAPSTONE_WORK"], "spec", "board")
C5 = os.environ.get("C5_OUT", os.path.join(B, "out-c5a"))
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
    sp = load(os.path.join(B, "out-objb", f"{FULL[b]}-test-capstone-spatial"), "capstone-spatial")
    arms = ["capstone-lifetimes"] + (["capstone-spatial"] if b in ("lbm", "nab", "imagick") else [])
    for arm in arms:
        lab = {"capstone-lifetimes": "Full", "capstone-spatial": "MapLife"}[arm]
        c5 = load(os.path.join(C5, st), arm); b3 = load(os.path.join(B, "out", st), arm)
        if b3 is None or sp is None: print(f"{b:8s} {lab:8s} NO REFERENCE"); fail = 1; continue
        if c5 is None: print(f"{b:8s} {lab:8s} PENDING"); continue
        seen += 1
        s1 = not c5[2] and abs(c5[1] / b3[1] - 1) <= 0.001
        r_sp = c5[0] / sp[0]; r_b3 = c5[0] / b3[0]
        pe = exc if arm == "capstone-lifetimes" else 0.0
        s2 = 1 + pe - BAND <= r_sp <= 1 + pe + BAND
        s3 = (r_b3 < 1) if (arm == "capstone-lifetimes" and b in S3) else None
        if not s1 or not s2 or s3 is False: fail = 1
        print(f"{b:8s} {lab:8s} S1 {'holds' if s1 else 'FAILS ' + ','.join(c5[2]) + f' instret {c5[1] / b3[1]:.5f}'}"
              f" | C5a/Spatial {r_sp:.4f} (band {1 + pe - BAND:.3f}-{1 + pe + BAND:.3f}) S2 {'holds' if s2 else 'FAILS'}"
              f" | C5a/bs3 {r_b3:.4f} (bs3/Spatial {b3[0] / sp[0]:.4f}) S3 {'rep' if s3 is None else ('holds' if s3 else 'FAILS')}")
if not seen: print(f"NO DATA: no finished C.5a stage under {C5}"); sys.exit(2)
sys.exit(fail)
