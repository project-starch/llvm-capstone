#!/usr/bin/env python3
"""un-grade.py: gcc train on the unified stack, board 3 (results/2026-11-06-unify/PREREG.md), against c3f (out-ga2-b3).
T0 tick probe gaps/s in 90-110; T1 its gap share. U0 rc 0, specdiff PASS, control passes, @@BP reads back.
U1 run <= 4,253.3 s. Exit 0 if every check holds, 1 otherwise, 2 if no data."""
import json, os, re, sys
W = os.environ["CAPSTONE_WORK"]; B = os.path.join(W, "spec", "board")
RUN = os.environ.get("UN_RUN", f"{B}/out-un-b3/502.gcc_r-train01-capstone-lifetimes-unify-rgz")
REF = os.environ.get("UN_REF", f"{B}/out-ga2-b3/502.gcc_r-train01-capstone-lifetimes-devmk-c2ac7")
fail = 0
def chk(ok):
    global fail; fail |= not ok; return "holds" if ok else "FAILS"
def load(d):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p): return None
    r = json.load(open(p)); a = r["arms"]["capstone-lifetimes"]
    S = open(os.path.join(d, "spec.txt"), errors="replace").read()
    runs = [c for c in a["commands"] if c["tag"] == "run"]
    c = runs[0]; st = dict(re.findall(r"(\w+)=(\d+)", (re.findall(r"CAPSTONE_VM_STATS ([^\n]*)", S) or [""])[-1]))
    ex = re.findall(r"capstone_vm: exits steps=(\d+) kcyc=(\d+) rcyc=(\d+) ucyc=(\d+) ocyc=(\d+) ev=([\d,]+)", S)
    ctl = os.path.join(d, "control.result.json")
    return dict(r=r, a=a, S=S, sec=sum(x["seconds"] for x in runs), cyc=c["cycles"] or 0, ins=c["instret"] or 0, st=st,
                ex=ex[-1] if ex else None, ctl=json.load(open(ctl)) if os.path.exists(ctl) else None)
u = load(RUN); f = load(REF)
if not u: print("NO DATA", RUN); sys.exit(2)
S = u["S"]
tp = re.search(r"RESULT tickprobe retval=0 total=(\d+) gapcyc=(\d+) gaps=(\d+) meangap=(\d+) maxgap=(\d+) share=([\d.]+)", S)
if tp:
    gps = int(tp[3]) / 30
    print(f"tick probe: {tp[3]} gaps in 30 s ({gps:.1f}/s), mean gap {int(tp[4])} cycles, max {tp[5]}, share {float(tp[6]) * 100:.2f} %")
    print(f"   T0 {chk(90 <= gps <= 110)} (gaps/s in 90-110)")
    s = float(tp[6])
    print("   T1: " + ("s >= 4 %: Plain pays a comparable per-tick cost" if s >= 0.04 else "s <= 2 %: k-r is a Capstone excess" if s <= 0.02 else "2-4 %: unresolved"))
else: print("tick probe MISSING"); fail = 1
bp = re.findall(r"^@@BP (.*)$", S, re.M)
why = []
if any(c["rc"] != 0 for c in u["a"]["commands"]): why.append("rc")
if not all(v["verdict"] == "PASS" for v in u["a"]["validation"]): why.append("specdiff")
if u["ctl"] != {"kind": "exit", "value": 0}: why.append("control")
want = "step_budget_ns=10000000000 resume_quantum=Y retire_exit_sweep=2 image_rootfs=Y fault_ahead_max=64 arena_pin=Y"
if [b.strip() for b in bp] != [want]: why.append(f"@@BP {bp}")
print(f"U0 {chk(not why)}" + (f" ({', '.join(why)})" if why else ""))
for n, x in (("unify", u), ("c3f", f)):
    if not x: print(f"{n}: reference MISSING"); continue
    e = x["ex"]
    print(f"{n:6s} run {x['sec']:.1f} s, inside {x['cyc'] / 25e6:.1f} s ({x['ins'] / 1e9:.3f} G, IPC {x['ins'] / max(x['cyc'], 1):.3f}), "
          f"outside {x['sec'] - x['cyc'] / 25e6:.1f} s; steps {x['st'].get('steps')}, faults {x['st'].get('faults')}, peak {x['st'].get('peak')}"
          + (f"; exits ev {e[5]}" if e else ""))
print(f"U1 {chk(u['sec'] <= 4253.3)} (run {u['sec']:.1f} s <= 4,253.3); against Plain 3,515.5 s: {u['sec'] / 3515.5:.3f}x")
sys.exit(fail)
