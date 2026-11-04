#!/usr/bin/env python3
"""all-grade.py: grade the all-optimisations run on board 1 (results/2026-11-04-all/PREREG.md): the compiler
evaluation's devmk images (dev compiler + movc flag) on C.5a with the fault-ahead module (kit bkit20-all-b1), out-all-b1.
References, all board 1: the 3 Nov baseline (spec/board/out, bitstream 3, 3 Nov images), the same devmk images on
bitstream 3 with the stock module (out-hfl/<stage>; for mcf only the dev step exists, out-hfl/...-dev), and Plain test
(out-hfl/<bench>-test-plain-rv64). Run time = the run commands' elapsed seconds. A stage not finished prints PENDING.
Exit 0 if every graded check holds, 1 otherwise, 2 if no data.
AL1: control, rc, specdiff, outputs = twin, fresh; instret within 0.1 % of the devmk-on-bitstream-3 run (same images).
AL2: run time below the devmk-on-bitstream-3 run's (mcf: below the dev-step run's), every arm.
Reported: run time and cycles against the 3 Nov baseline, and run time against Plain's."""
import json, os, sys

B = os.path.join(os.environ["CAPSTONE_WORK"], "spec", "board")
ALL = os.environ.get("ALL_OUT", os.path.join(B, "out-all-b1"))
ST = [("gcc", "502.gcc_r-test-capstone-lifetimes-devmk", "502.gcc_r-test-capstone-lifetimes", "502.gcc_r"),
      ("imagick", "538.imagick_r-test-devmk", "538.imagick_r-test", "538.imagick_r"),
      ("nab", "544.nab_r-test-devmk", "544.nab_r-test", "544.nab_r"),
      ("lbm", "519.lbm_r-test-devmk", "519.lbm_r-test", "519.lbm_r"),
      ("xz", "557.xz_r-test-capstone-lifetimes-devmk", "557.xz_r-test-capstone-lifetimes", "557.xz_r"),
      ("mcf", "505.mcf_r-test-capstone-lifetimes-devmk", "505.mcf_r-test-capstone-lifetimes", "505.mcf_r")]

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
    if not all(c.get("bench_line_fresh", True) for c in a["commands"]): why.append("not fresh")
    ctl = os.path.join(d, "control.result.json")
    if os.path.exists(ctl) and json.load(open(ctl)) != {"kind": "exit", "value": 0}: why.append("control")
    return dict(cyc=sum((c["cycles"] or 0) for c in runs), ins=sum((c["instret"] or 0) for c in runs),
                sec=sum(c["seconds"] for c in runs), why=why)

fail = 0; seen = 0
for b, st, base, full in ST:
    ref_dir = os.path.join(B, "out-hfl", st if b != "mcf" else st.replace("-devmk", "-dev"))
    for arm in ("capstone-spatial", "capstone-lifetimes"):
        x = load(os.path.join(ALL, st), arm); bl = load(os.path.join(B, "out", base), arm); ref = load(ref_dir, arm)
        if bl is None: continue
        lab = {"capstone-spatial": "MapLife", "capstone-lifetimes": "Full"}[arm]
        if x is None: print(f"{b:8s} {lab:8s} PENDING"); continue
        if ref is None: print(f"{b:8s} {lab:8s} NO REFERENCE {ref_dir}"); fail = 1; continue
        seen += 1
        pl = load(os.path.join(B, "out-hfl", f"{full}-test-plain-rv64"), "plain-rv64")
        a1 = not x["why"] and (b == "mcf" or abs(x["ins"] / ref["ins"] - 1) <= 0.001)
        a2 = x["sec"] < ref["sec"]
        if not a1 or not a2: fail = 1
        print(f"{b:8s} {lab:8s} AL1 {'holds' if a1 else 'FAILS ' + ','.join(x['why'])} | run {x['sec']:.0f} s vs "
              f"{'dev' if b == 'mcf' else 'devmk'}-on-bs3 {ref['sec']:.0f} s AL2 {'holds' if a2 else 'FAILS'} | vs 3 Nov: run "
              f"{x['sec'] / bl['sec']:.3f} cycles {x['cyc'] / bl['cyc']:.3f}"
              + (f" | / Plain {x['sec'] / pl['sec']:.2f}x (3 Nov {bl['sec'] / pl['sec']:.2f}x)" if pl else ""))
if not seen: print(f"NO DATA: no finished stage under {ALL}"); sys.exit(2)
sys.exit(fail)
