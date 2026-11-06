#!/usr/bin/env python3
"""thp-grade.py: mcf test, Temporal on 2 MiB THP, and Plain on the THP kernel (results/2026-11-06-thp/PREREG.md).
Runs: $THP_ROOT (default $CAPSTONE_WORK/spec/board/out-thp-b3)/<stage>. Reference (4 KiB pages, all2 configuration, same
kit family): $THP_REF (default out-gtm-b1). Exit 0 if every graded check holds, 1 otherwise, 2 if no data.
H0 rc 0, specdiff PASS, control passes; @@THP reads [madvise] and thp=Y; thp_fault_alloc > 500.
H1 exits <= 15,000 and faults <= 5,000.   H2 outside <= 1,500 s.   H3 run <= 10,900 s.
H4 Plain on the THP kernel within +-1.5 % of 9,094.9 s."""
import json, os, re, sys
W = os.environ["CAPSTONE_WORK"]; B = os.path.join(W, "spec", "board")
ROOT = os.environ.get("THP_ROOT", f"{B}/out-thp-b3")
REF = os.environ.get("THP_REF", f"{B}/out-gtm-b1/505.mcf_r-test-capstone-lifetimes-devmk")
EV = ["L1I miss", "L1D miss", "DTLB miss", "mispredict", "issue stall", "scoreboard full"]
EXRE = r"capstone_vm: exits steps=(\d+) kcyc=(\d+) rcyc=(\d+) ucyc=(\d+) ocyc=(\d+) ev=([\d,]+)"
HPRE = r"capstone_vm: hpm hpm3=(\d+) hpm4=(\d+) hpm5=(\d+) hpm6=(\d+) hpm7=(\d+) hpm8=(\d+)"
fail = 0; seen = 0
def chk(ok):
    global fail
    fail |= not ok; return "holds" if ok else "FAILS"
def load(d):
    r = json.load(open(os.path.join(d, "result.json")))
    T = open(os.path.join(d, "guest", "board.txt"), errors="replace").read()
    return r, T
def temporal(d, label):
    r, T = load(d); a = r["arms"]["capstone-lifetimes"]
    run = [c for c in a["commands"] if c["tag"] == "run"][0]
    sec, cyc, ins = run["seconds"], run["cycles"], run["instret"]
    # the program's context is the LAST exits/hpm line of the boot (the control's comes first)
    ex = re.findall(EXRE, T)[-1]; hp = list(map(int, re.findall(HPRE, T)[-1]))
    kinds = list(map(int, ex[5].split(","))); n = sum(kinds)
    st = dict(re.findall(r"(\w+)=(\d+)", (re.findall(r"CAPSTONE_VM_STATS ([^\n]*)", T) or [""])[-1]))
    out = sec - cyc / 25e6
    print(f"== {label}: run {sec:.1f} s, inside {cyc / 25e6:.1f} s ({ins / 1e9:.3f} G, IPC {ins / cyc:.3f}), outside {out:.1f} s")
    print(f"   exits {n} by kind {kinds}, CV_STEPs {ex[0]}; faults {st.get('faults')}, peak pages {st.get('peak')}")
    print(f"   per exit ms: CV_STEP {int(ex[1]) / n / 25e3:.2f} (resolve {int(ex[2]) / n / 25e3:.2f}), between {int(ex[3]) / n / 25e3:.2f},"
          f" other {int(ex[4]) / n / 25e3:.2f}; outside per exit {out * 1e3 / n:.2f}")
    print("   in-context per 1k instructions: " + ", ".join(f"{e} {1e3 * v / ins:.2f}" for e, v in zip(EV, hp)))
    return dict(r=r, T=T, a=a, sec=sec, cyc=cyc, ins=ins, n=n, faults=int(st.get("faults", 10**9)), out=out, hp=hp, d=d)
st = os.path.join(ROOT, "505.mcf_r-test-capstone-lifetimes-devmk-thp")
if os.path.exists(os.path.join(st, "result.json")):
    seen += 1; t = temporal(st, "Temporal THP")
    why = []
    if any(c["rc"] != 0 for c in t["a"]["commands"]): why.append("rc")
    if not t["a"]["validation"] or not all(v["verdict"] == "PASS" for v in t["a"]["validation"]): why.append("specdiff")
    if json.load(open(os.path.join(st, "control.result.json"))) != {"kind": "exit", "value": 0}: why.append("control")
    th = re.findall(r"^@@THP (.*)$", t["T"], re.M); va = re.findall(r"thp_fault_alloc:(\d+)", t["T"])
    if not th or "[madvise]" not in th[-1] or "thp=Y" not in th[-1]: why.append(f"THP {th}")
    if not va or int(va[-1]) <= 500: why.append(f"thp_fault_alloc {va}")
    print(f"   H0 {chk(not why)} {why or ''} (@@THP {th[-1] if th else None}; thp_fault_alloc {va[-1] if va else None})")
    print(f"   H1 {chk(t['n'] <= 15000 and t['faults'] <= 5000)} (exits {t['n']} <= 15,000; faults {t['faults']} <= 5,000)")
    print(f"   H2 {chk(t['out'] <= 1500)} (outside {t['out']:.1f} s <= 1,500)")
    print(f"   H3 {chk(t['sec'] <= 10900)} (run {t['sec']:.1f} s <= 10,900); against Plain 9,094.9 s: {t['sec'] / 9094.9:.3f}x")
    f = temporal(REF, "reference, 4 KiB pages")
    print(f"   THP - reference: run {t['sec'] - f['sec']:+.1f} s, inside {(t['cyc'] - f['cyc']) / 25e6:+.1f} s, outside {t['out'] - f['out']:+.1f} s,"
          f" instructions {t['ins'] / f['ins']:.4f}x, DTLB misses {t['hp'][2] / f['hp'][2]:.3f}x, L1D misses {t['hp'][1] / f['hp'][1]:.3f}x")
else: print("Temporal THP PENDING")
pd = os.path.join(ROOT, "505.mcf_r-test-plain-rv64-hpm-thpk")
if os.path.exists(os.path.join(pd, "result.json")):
    seen += 1; r, T = load(pd); a = r["arms"]["plain-rv64"]
    sec = sum(c["seconds"] for c in a["commands"] if c["tag"] == "run")
    ok = all(c["rc"] == 0 for c in a["commands"]) and a["validation"] and all(v["verdict"] == "PASS" for v in a["validation"])
    print(f"== Plain on the THP kernel: run {sec:.1f} s; H4 {chk(ok and abs(sec / 9094.9 - 1) <= 0.015)} ({sec / 9094.9:.4f}x of 9,094.9 s)")
else: print("Plain THP PENDING")
if not seen: print(f"NO DATA under {ROOT}"); sys.exit(2)
sys.exit(fail)
