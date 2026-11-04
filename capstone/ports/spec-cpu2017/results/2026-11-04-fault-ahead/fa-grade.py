#!/usr/bin/env python3
"""fa-grade.py: grade adaptive fault-ahead on board 2 (results/2026-11-04-fault-ahead/PREREG.md). Each stage of
out-fa-b2 (on) and out-fa-b2-off (off) is read against its ObjBounds baseline in out-objb: same images, same bitstream,
same board. Readings per run command are summed: real (up1 - up0), inside (in-context cycles / 25 MHz), outside
(real - inside), instret, faults and steps (the transcript's CAPSTONE_VM_STATS), and the sysfs read-back of the module
parameter. A stage not yet finished prints PENDING. Exit 0 if every graded check holds, 1 otherwise, 2 if no data."""
import json, os, re, sys

B = os.path.join(os.environ["CAPSTONE_WORK"], "spec", "board")
ON = os.environ.get("FA_ON", os.path.join(B, "out-fa-b2")); OFF = os.environ.get("FA_OFF", os.path.join(B, "out-fa-b2-off"))
STAGES = ["538.imagick_r-test-capstone-spatial", "502.gcc_r-test-capstone-spatial", "544.nab_r-test-capstone-spatial",
          "519.lbm_r-test-capstone-spatial", "557.xz_r-test-capstone-spatial", "505.mcf_r-test-capstone-spatial"]
# F4: lbm at least 10x fewer faults (its faults stream: the collaborator's QEMU reading 104,725 -> 1,660); every other
# stage no more faults than its baseline. F5: outside time lower on the three fault-heavy stages.
TEN = {"519.lbm_r-test-capstone-spatial"}
FAULTS10 = {"519.lbm_r-test-capstone-spatial", "505.mcf_r-test-capstone-spatial", "557.xz_r-test-capstone-spatial"}

def read(d):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p): return None
    r = json.load(open(p))
    if not r.get("done"): return None
    a = r["arms"]["capstone-spatial"]; why = []
    if any(c["rc"] != 0 for c in a["commands"]): why.append("rc!=0")
    if not all(v["verdict"] == "PASS" for v in a["validation"]): why.append("specdiff")
    if r.get("vs_twin", {}).get("capstone-spatial"): why.append("twin diff")
    ctl = os.path.join(d, "control.result.json")
    if not os.path.exists(ctl) or json.load(open(ctl)) != {"kind": "exit", "value": 0}: why.append("control")
    T = open(os.path.join(d, "guest", "board.txt"), errors="replace").read()
    real = ins = ir = f = st = 0; n = 0
    for m in re.finditer(r"@@SPEC-END \d+ run \S+ \d+ rc=\d+ up0=([\d.]+) up1=([\d.]+).*?steps=(\d+) faults=(\d+).*?:: bench cycles=(\d+) instret=(\d+)", T):
        u0, u1, s, fa, cy, it = m.groups(); n += 1
        real += float(u1) - float(u0); ins += int(cy) / 25e6; ir += int(it); f += int(fa); st += int(s)
    if not n: why.append("no run lines")
    par = re.findall(r"^@@PARAM (\S+)", T, re.M)
    return dict(real=real, ins=ins, out=real - ins, instret=ir, faults=f, steps=st, why=why, params=par)

fail = 0; seen = 0
for arm, root in (("off", OFF), ("on", ON)):
    for s in STAGES if arm == "on" else STAGES[:1]:
        b = read(os.path.join(B, "out-objb", s)); x = read(os.path.join(root, s))
        if b is None: print(f"{arm:3s} {s[:13]:13s} NO BASELINE"); fail = 1; continue
        if x is None: print(f"{arm:3s} {s[:13]:13s} PENDING"); continue
        seen += 1
        f1 = not x["why"] and abs(x["instret"] / b["instret"] - 1) <= 0.001
        want = ["fault_ahead_max=64"] if arm == "on" else []
        f1 = f1 and x["params"] == want
        if arm == "off":
            f2 = x["faults"] == b["faults"]   # same fault path: the same faults
            f3 = abs(x["real"] / b["real"] - 1) <= 0.05
            v = f"F1 {'holds' if f1 else 'FAILS ' + ','.join(x['why']) + str(x['params'])} | faults {x['faults']} vs {b['faults']} F2 {'holds' if f2 else 'FAILS'} | real {x['real']:.1f} vs {b['real']:.1f} s F3 {'holds' if f3 else 'FAILS'}"
            ok = f1 and f2 and f3
        else:
            f4 = (x["faults"] * 10 <= b["faults"]) if s in TEN else (x["faults"] <= b["faults"])
            f5 = x["out"] < b["out"] if s in FAULTS10 else None
            f6 = abs(x["ins"] / b["ins"] - 1) <= 0.03
            v = (f"F1 {'holds' if f1 else 'FAILS ' + ','.join(x['why']) + str(x['params'])} | faults {x['faults']} vs {b['faults']} F4 {'holds' if f4 else 'FAILS'}"
                 f" | steps {x['steps']} vs {b['steps']} | outside {x['out']:.1f} vs {b['out']:.1f} s F5 {'rep' if f5 is None else ('holds' if f5 else 'FAILS')}"
                 f" | inside {x['ins'] / b['ins']:.4f} F6 {'holds' if f6 else 'FAILS'} | real {x['real'] / b['real']:.4f}")
            ok = f1 and f4 and f6 and f5 is not False
        if not ok: fail = 1
        print(f"{arm:3s} {s[:13]:13s} {v}")
if not seen: print("NO DATA: no finished fault-ahead stage"); sys.exit(2)
sys.exit(fail)
