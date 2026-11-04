#!/usr/bin/env python3
"""eval-grade.py: grade the compiler-evaluation stages (results/2026-11-03-eval/PREREG.md and its Addendum 1).
Reads each finished dev/devmk stage and its 3 Nov baseline, prints E1/E2/E3/E4/E5 per benchmark and arm.
A stage that has not finished prints PENDING, never a number. Exit 1 if any graded check fails."""
import json, os, sys

B = os.path.join(os.environ["CAPSTONE_WORK"], "spec", "board")  # $CAPSTONE_WORK: the lane's work root
HZ = 25e6
# twin instret ratios from the PREREG (dev/baseline, devmk/dev)
TWIN = {"lbm": (0.8521, 0.9996), "nab": (0.9256, 0.9952), "imagick": (0.9279, 0.9696),
        "xz": (0.9966, 0.9904), "mcf": (0.8168, 0.9379), "gcc": (0.9475, 0.9678)}
FULL = {"lbm": "519.lbm_r", "nab": "544.nab_r", "imagick": "538.imagick_r", "xz": "557.xz_r",
        "mcf": "505.mcf_r", "gcc": "502.gcc_r"}
E3 = {"lbm", "nab", "imagick", "mcf", "gcc"}
E4 = {"mcf", "gcc", "imagick"}

def stages(board, b):
    n = FULL[b]
    if board == 1:
        if b in ("lbm", "nab", "imagick"):
            base = f"{B}/out/{n}-test"; dev = f"{B}/out-hfl/{n}-test-dev"; mk = f"{B}/out-hfl/{n}-test-devmk"
            return [("MapLife", "capstone-spatial", base, dev, mk), ("Full", "capstone-lifetimes", base, dev, mk)]
        s = f"{n}-test-capstone-lifetimes"
        return [("Full", "capstone-lifetimes", f"{B}/out/{s}", f"{B}/out-hfl/{s}-dev", f"{B}/out-hfl/{s}-devmk")]
    s = f"{n}-test-capstone-spatial"
    return [("Spatial", "capstone-spatial", f"{B}/out-objb/{s}", f"{B}/out-objb/{s}-dev", f"{B}/out-objb/{s}-devmk")]

def load(d, arm):
    """(cycles, instret, seconds, ok, why) summed over run commands; None if not done."""
    p = os.path.join(d, "result.json")
    if not os.path.exists(p): return None
    r = json.load(open(p))
    if not r.get("done"): return None
    a = r["arms"].get(arm)
    if a is None: return None
    runs = [c for c in a["commands"] if c["tag"] == "run"]
    why = []
    if any(c["rc"] != 0 for c in a["commands"]): why.append("rc!=0")
    if not all(v["verdict"] == "PASS" for v in a["validation"]): why.append("specdiff")
    if r.get("vs_twin", {}).get(arm): why.append("twin diff")
    if not all(c.get("bench_line_fresh") for c in a["commands"]): why.append("not fresh")
    ctl = os.path.join(d, "control.result.json")
    if os.path.exists(ctl):
        c = json.load(open(ctl))
        if c.get("kind") != "exit" or c.get("value") != 0: why.append("control")
    return (sum(c["cycles"] for c in runs), sum(c["instret"] for c in runs), sum(c["seconds"] for c in runs), not why, why)

fail = 0
for board in (1, 2):
    print(f"== board {board}")
    for b in FULL:
        for lab, arm, base, dev, mk in stages(board, b):
            X = [load(d, arm) for d in (base, dev, mk)]
            if X[0] is None: print(f"{b:8s} {lab:8s} NO BASELINE at {base}"); fail = 1; continue
            if X[1] is None: print(f"{b:8s} {lab:8s} PENDING"); continue
            out = [f"{b:8s} {lab:8s}"]
            for i, (prev, cur, tw, step) in enumerate(((X[0], X[1], TWIN[b][0], "dev/base"), (X[1], X[2], TWIN[b][1], "mk/dev"))):
                if cur is None: out.append(f"| {step} PENDING"); continue
                e1 = "E1 ok" if cur[3] else "E1 FAIL " + ",".join(cur[4])
                ri = cur[1] / prev[1]; rc = cur[0] / prev[0]
                e2 = abs(ri / tw - 1) <= 0.003
                out_prev = prev[2] - prev[0] / HZ; out_cur = cur[2] - cur[0] / HZ
                d_in = cur[0] / HZ - prev[0] / HZ; d_out = out_cur - out_prev
                e5 = abs(d_out) < abs(d_in)
                pred = (b in E3) if i == 0 else (b in E4)
                e34 = ("holds" if rc < 1 else "FAILS") if pred else "rep"
                if not cur[3] or not e2 or (pred and rc >= 1) or not e5: fail = 1
                out.append(f"| {step}: instret {ri:.4f} (twin {tw:.4f}, E2 {'ok' if e2 else 'FAIL'}) cycles {rc:.4f} "
                           f"E{3 if i == 0 else 4} {e34}; in {d_in:+.1f}s out {d_out:+.1f}s E5 {'ok' if e5 else 'FAIL'}; {e1}")
                if i == 1 and b == "mcf":
                    out.append(f"E4b {'holds' if 1 - rc >= 1 - ri else 'FAILS'}")
            print(" ".join(out))
sys.exit(fail)
