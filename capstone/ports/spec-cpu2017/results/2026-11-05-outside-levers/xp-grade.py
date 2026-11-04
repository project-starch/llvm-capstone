#!/usr/bin/env python3
"""xp-grade.py: grade the outside-time levers on board 2 (results/2026-11-05-outside-levers/PREREG.md).
Runs: $CAPSTONE_WORK/spec/board/out-xp-<kit>/<stage>; reference: out-fa-b1 (board 1, bitstream 3, fault-ahead 64).
For each Temporal arm, from result.json (rc, specdiff, twin, control, seconds, cycles) and guest/board.txt (@@PARAM,
QNT0, CAPSTONE_VM_STATS launch_ns/steps/faults/pages, 'capstone_vm: teardown pages= ns= sweep=' lines).
A run not finished prints PENDING. Exit 0 if every graded check holds, 1 otherwise, 2 if no data."""
import json, os, re, sys
B = os.path.join(os.environ["CAPSTONE_WORK"], "spec", "board"); REF = os.path.join(B, "out-fa-b1")
ROOT = os.environ.get("XP_ROOT", B)
ARM = "capstone-lifetimes"
GCC, IMG, NAB, LBM = ("502.gcc_r-test-capstone-lifetimes", "538.imagick_r-test", "544.nab_r-test", "519.lbm_r-test")
WANT = {"ctl": {"fault_ahead_max": "64"}, "eg": {"fault_ahead_max": "64"}, "sweep": {"fault_ahead_max": "64", "retire_exit_sweep": "3"},
        "nosweep": {"fault_ahead_max": "64", "retire_exit_sweep": "2"}, "root": {"fault_ahead_max": "64", "image_rootfs": "Y"},
        "q8": {"fault_ahead_max": "64"},
        "allmod": {"fault_ahead_max": "64", "retire_exit_sweep": "3", "image_rootfs": "Y", "step_budget_ns": "80000000"}}

def read(d):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p): return None
    r = json.load(open(p))
    if not r.get("done") or ARM not in r["arms"]: return None
    a = r["arms"][ARM]; runs = [c for c in a["commands"] if c["tag"] == "run"]; why = []
    if any(c["rc"] != 0 for c in a["commands"]): why.append("rc")
    if not all(v["verdict"] == "PASS" for v in a["validation"]): why.append("specdiff")
    if r.get("vs_twin", {}).get(ARM): why.append("twin")
    ctl = os.path.join(d, "control.result.json")
    if not os.path.exists(ctl) or json.load(open(ctl)) != {"kind": "exit", "value": 0}: why.append("control")
    T = open(os.path.join(d, "guest", "board.txt"), errors="replace").read()
    arm_part = T.split(f"@@SPEC-ARM-BEGIN {ARM}", 1)[-1]
    st = [dict(re.findall(r"(\w+)=(\d+)", m)) for m in re.findall(r"CAPSTONE_VM_STATS ([^\n]*)", arm_part)]
    td = [tuple(map(int, m)) for m in re.findall(r"capstone_vm: teardown pages=(\d+) ns=(\d+) sweep=(\d+) exiting=(\d+)", T)]  # whole transcript: console lines precede their arm echo
    return dict(sec=sum(c["seconds"] for c in runs), cyc=sum(c["cycles"] or 0 for c in runs), why=why,
                params=dict(re.findall(r"^@@PARAM (\w+)=(\S+)$", T, re.M)), qnt=re.findall(r"QNT0:([0-9A-F]+)", open(os.path.join(d, "guest", "driver.log"), errors="replace").read())[-1:],  # last: the console replays the previous boot first
                launch=sum(int(s.get("launch_ns", 0)) for s in st) / 1e9, steps=sum(int(s.get("steps", 0)) for s in st),
                faults=sum(int(s.get("faults", 0)) for s in st), td=td, peak=max([int(s.get("peak", 0)) for s in st] or [0]))

def nsp(x):  # teardown ns per page over the arm's passes with at least 100 pages
    t = [(p, n) for p, n, _, _ in x["td"] if p >= 100]
    return sum(n for _, n in t) / max(1, sum(p for p, _ in t)) if t else None

fail = 0; seen = 0; res = {}
for kit, st in (("ctl", GCC), ("eg", GCC), ("sweep", GCC), ("nosweep", GCC), ("root", GCC), ("root", IMG), ("q8", NAB), ("allmod", LBM)):
    x = read(os.path.join(ROOT, f"out-xp-{kit}", st)); ref = read(os.path.join(REF, st))
    lab = f"{kit:8s} {st[:16]:16s}"
    if x is None: print(f"{lab} PENDING"); continue
    if ref is None: print(f"{lab} NO REFERENCE"); fail = 1; continue
    seen += 1; res[(kit, st)] = x
    c0 = not x["why"] and all(x["params"].get(k) == v for k, v in WANT[kit].items()) and \
        not (set(x["params"]) - set(WANT[kit]))
    q = x["qnt"]; qok = (q and all(v == ("007A1200" if kit == "q8" else "001E8480") for v in q))
    ok = c0 and qok; ln = f"{lab} X0 {'holds' if ok else 'FAILS ' + ','.join(x['why']) + str(x['params']) + str(q)}"
    ln += f" | run {x['sec']:.1f} s vs ref {ref['sec']:.1f} ({x['sec'] / ref['sec']:.3f}) | launch {x['launch']:.2f} vs {ref['launch']:.2f} s"
    ln += f" | steps {x['steps']} vs {ref['steps']} faults {x['faults']} vs {ref['faults']}"
    n = nsp(x); ln += f" | teardown {n / 1e6:.3f} ms/page {x['td']}" if n else f" | teardown {x['td']}"
    if not ok: fail = 1
    print(ln)
def chk(name, cond, text):
    global fail
    if cond is None: print(f"{name} PENDING"); return
    if not cond: fail = 1
    print(f"{name} {'holds' if cond else 'FAILS'}: {text}")
g0, g1, g2, g3 = (res.get((k, GCC)) for k in ("ctl", "sweep", "nosweep", "root"))
n0 = nsp(g0) if g0 else None
chk("X1 ctl teardown 0.4-1.2 ms/page", None if not g0 else bool(n0) and 0.4e6 <= n0 <= 1.2e6, f"{(n0 or 0) / 1e6:.3f} ms/page (no teardown line reads 0)")
chk("X2 sweep <= 0.35 x ctl", None if not (g1 and g0) else bool(n0 and nsp(g1)) and nsp(g1) <= 0.35 * n0, f"{(nsp(g1) or 0) / max(n0 or 1, 1):.3f} x" if g1 and n0 else "")
chk("X3 nosweep <= 0.15 x ctl (report)", None if not (g2 and g0) else bool(n0 and nsp(g2)) and nsp(g2) <= 0.15 * n0, f"{(nsp(g2) or 0) / max(n0 or 1, 1):.3f} x" if g2 and n0 else "")
chk("X4 rootfs gcc launch <= 5 s", None if not g3 else g3["launch"] <= 5, f"{g3['launch']:.2f} s (ctl {g0['launch']:.2f})" if g3 and g0 else "")
i3 = res.get(("root", IMG))
chk("X4b rootfs imagick launch <= 1.5 s (first command)", None if not i3 else i3["launch"] <= 1.5 + 1.4, f"{i3['launch']:.2f} s over both commands" if i3 else "")
q8 = res.get(("q8", NAB)); qref = read(os.path.join(REF, NAB))
if q8 and qref:
    out = q8["sec"] - q8["cyc"] / 25e6; oref = qref["sec"] - qref["cyc"] / 25e6
    v = "H-exit" if out <= 160 else ("H-time" if out >= 185 else "partial")
    print(f"X5 q8 nab outside {out:.1f} s vs ref {oref:.1f} s, steps {q8['steps']} vs {qref['steps']}: {v} (report)")
else: print("X5 PENDING")
l6 = res.get(("allmod", LBM)); lref = read(os.path.join(REF, LBM))
chk("X6 allmod lbm run <= 0.985 x ref", None if not l6 else l6["sec"] <= 0.985 * lref["sec"], f"{l6['sec'] / lref['sec']:.4f}" if l6 else "")
chk("X6b allmod lbm faults <= 2,500", None if not l6 else l6["faults"] <= 2500, f"{l6['faults']}" if l6 else "")
ge = res.get(("eg", GCC))
chk("X7 eg gcc: peak <= 2,700 pages, launch >= 3 s and run >= 5 s below ctl", None if not (ge and g0) else
    ge["peak"] <= 2700 and ge["launch"] <= g0["launch"] - 3 and ge["sec"] <= g0["sec"] - 5,
    f"peak {ge['peak']} vs {g0['peak']}, launch {ge['launch']:.2f} vs {g0['launch']:.2f} s, run {ge['sec']:.1f} vs {g0['sec']:.1f} s" if ge and g0 else "")
if not seen: print("NO DATA: no finished run under out-xp-*"); sys.exit(2)
sys.exit(fail)
