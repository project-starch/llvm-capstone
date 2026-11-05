#!/usr/bin/env python3
"""mx-grade.py: mcf test, the outside-lever bundle and Plain with every counter (results/2026-11-05-mcfx/PREREG.md).
Runs: $MX_ROOTS (default $CAPSTONE_WORK/spec/board/out-mx-b1 (bundle) and out-mx-b2 (Plain))/<stage>. Reference for the
bundle: $MX_REF (default out-gtm-b1: board 1, the same kit with its own parameters). Exit 0 if every graded check holds,
1 otherwise, 2 if no data.
M0  rc 0, specdiff PASS, control passes, bench line fresh (Temporal); the four kit params read back; QNT0 007A1200;
    Temporal: @@MX-SET reads step_budget_ns=10000000000 resume_quantum=Y retire_exit_sweep=2, @@MX-RESTORED the kit's.
M1  counters programmed: in-context hpm3-7 in (0, cycles) (Temporal), machine-wide hpm3-7 deltas > 0 (both).
M2  Plain run 8,900-9,270 s (the 3 Nov Plain 9,082.86 s, +-2 %).
M3  bundle run <= 0.995 x the reference (11,573.79 s), and CV_STEPs <= 15,000 (reference 37,266).
Reported: as gt-grade.py, plus the bundle against the reference: run, CV_STEPs, inside cycles."""
import json, os, re, sys
W = os.environ["CAPSTONE_WORK"]; B = os.path.join(W, "spec", "board")
ROOTS = os.environ.get("MX_ROOTS", f"{B}/out-mx-b1:{B}/out-mx-b2").split(":")
REF = os.environ.get("MX_REF", f"{B}/out-gtm-b1/505.mcf_r-test-capstone-lifetimes-devmk")
ARMS = [("capstone-lifetimes", "505.mcf_r-test-capstone-lifetimes-devmk-x"), ("plain-rv64", "505.mcf_r-test-plain-rv64-hpm")]
WANT = {"fault_ahead_max": "64", "retire_exit_sweep": "3", "image_rootfs": "Y", "step_budget_ns": "80000000"}
EV = ["L1I miss", "L1D miss", "DTLB miss", "mispredict", "issue stall", "scoreboard full"]
fail = 0; seen = 0
def chk(ok):
    global fail
    fail |= not ok; return "holds" if ok else "FAILS"
def find(st):
    for r in ROOTS:
        p = os.path.join(r, st, "result.json")
        if os.path.exists(p): return os.path.join(r, st)
res = {}
for arm, st in ARMS:
    d = find(st)
    if not d: print(f"{arm:20s} PENDING"); continue
    r = json.load(open(os.path.join(d, "result.json")))
    a = r["arms"].get(arm)
    if not r.get("done") or not a: print(f"{arm:20s} NO RESULT (done={r.get('done')})"); fail = 1; continue
    seen += 1
    runs = [c for c in a["commands"] if c["tag"] == "run"]
    S = open(os.path.join(d, "spec.txt"), errors="replace").read()
    T = open(os.path.join(d, "guest", "board.txt"), errors="replace").read() if os.path.exists(os.path.join(d, "guest", "board.txt")) else S
    raw = os.path.join(d, "guest", "board-raw.txt"); raw = open(raw, errors="replace").read() if os.path.exists(raw) else ""
    why = []
    if any(c["rc"] != 0 for c in a["commands"]): why.append("rc")
    if not all(v["verdict"] == "PASS" for v in a["validation"]): why.append("specdiff")
    ctl = os.path.join(d, "control.result.json")
    if not os.path.exists(ctl) or json.load(open(ctl)) != {"kind": "exit", "value": 0}: why.append("control")
    if arm != "plain-rv64" and not all(c.get("bench_line_fresh", True) for c in a["commands"]): why.append("not fresh")
    params = dict(re.findall(r"^@@PARAM (\w+)=(\S+)$", T, re.M))
    qnt = re.findall(r"QNT0:([0-9A-F]{8})", raw[raw.rfind("OpenSBI"):])[-1:] if "OpenSBI" in raw else []
    if arm != "plain-rv64":
        ms = re.findall(r"@@MX-SET (step_budget_ns=\S+ resume_quantum=\S+ retire_exit_sweep=\S+)", T); mr = re.findall(r"@@MX-RESTORED (.*)", T)
        if ms != ["step_budget_ns=10000000000 resume_quantum=Y retire_exit_sweep=2"]: why.append(f"MX-SET {ms}")
        if [x.strip() for x in mr] != ["step_budget_ns=80000000 resume_quantum=N retire_exit_sweep=3"]: why.append(f"MX-RESTORED {mr}")
    g0 = not why and params == WANT and qnt == ["007A1200"]
    sec = sum(c["seconds"] for c in runs)
    print(f"== {arm}: run {sec:.1f} s | M0 {chk(g0)}" + ("" if g0 else f" ({','.join(why)} params={params} qnt={qnt})"))
    s0 = re.search(r"@@GT-SNAP0 0 hpm=([\d ]+) vm=(\S*) st=(\S*) mem=(\S*)", S); s1 = re.search(r"@@GT-SNAP1 0 hpm=([\d ]+) vm=(\S*) st=(\S*) mem=(\S*)", S)
    mw = None
    if s0 and s1:
        h0 = list(map(int, s0[1].split())); h1 = list(map(int, s1[1].split())); mw = [y - x for x, y in zip(h0, h1)]
        kv = lambda s: dict(x.split(":") for x in s.strip(",").split(",") if ":" in x)
        v0, v1 = kv(s0[2]), kv(s1[2]); t0, t1 = kv(s0[3]), kv(s1[3])
        cyc, ins = mw[0], mw[1]
        print(f"   machine-wide: {cyc / 25e6:.1f} s of cycles, {ins / 1e9:.3f} G instructions, IPC {ins / max(cyc, 1):.3f}")
        print("   machine-wide per 1k instructions: " + ", ".join(f"{n} {1e3 * v / max(ins, 1):.2f}" for n, v in zip(EV, mw[2:])))
        print(f"   vmstat: " + ", ".join(f"{k} {int(v1[k]) - int(v0[k])}" for k in v0 if k in v1 and k != "nr_free_pages")
              + f" | /proc/stat: " + ", ".join(f"{k} {int(t1[k]) - int(t0[k])}" for k in t0 if k in t1))
    else: print("   machine-wide snapshot MISSING")
    res[arm] = dict(sec=sec, mw=mw)
    if arm == "plain-rv64":
        ru = re.search(r"@@GT-RU 0 RUSAGE \S+ maxrss_kb=(\d+) minflt=(\d+) majflt=(\d+) utime=([\d.]+) stime=([\d.]+)", S)
        print(f"   rusage: maxrss {ru[1]} kB, minflt {ru[2]}, majflt {ru[3]}, user {ru[4]} s, sys {ru[5]} s" if ru else "   rusage MISSING")
        if mw: g1 = all(v > 0 for v in mw[2:7]); print(f"   M1 {chk(g1)} (machine-wide hpm3-7 > 0)")
        g2 = 8900 <= sec <= 9270; print(f"   M2 {chk(g2)} (8,900-9,270 s)")
        continue
    c = runs[0]; cyc = c["cycles"] or 0; ins = c["instret"] or 0
    st = dict(re.findall(r"(\w+)=(\d+)", (re.findall(r"CAPSTONE_VM_STATS ([^\n]*)", S) or [""])[-1]))
    hp = re.findall(r"capstone_vm: hpm hpm3=(\d+) hpm4=(\d+) hpm5=(\d+) hpm6=(\d+) hpm7=(\d+) hpm8=(\d+)", S)
    ex = re.findall(r"capstone_vm: exits steps=(\d+) kcyc=(\d+) rcyc=(\d+) ucyc=(\d+) ocyc=(\d+) ev=([\d,]+)", S)
    td = re.findall(r"capstone_vm: teardown pages=(\d+) ns=(\d+) sweep=(\d+)", S)
    print(f"   inside {cyc / 25e6:.1f} s ({ins / 1e9:.3f} G instructions, IPC {ins / max(cyc, 1):.3f}), outside {sec - cyc / 25e6:.1f} s;"
          f" launch {int(st.get('launch_ns', 0)) / 1e9:.2f} s; faults {st.get('faults')}, peak pages {st.get('peak')}, steps {st.get('steps')}")
    if hp:
        h = [sum(int(x[i]) for x in hp) for i in range(6)]
        print("   in-context per 1k instructions: " + ", ".join(f"{n} {1e3 * v / max(ins, 1):.2f}" for n, v in zip(EV, h)))
        g1 = all(0 < v < cyc for v in h[:5]) and bool(mw) and all(v > 0 for v in mw[2:7]); print(f"   M1 {chk(g1)} (in-context hpm3-7 in (0, cycles), machine-wide > 0)")
    else: print("   in-context hpm line MISSING"); g1 = chk(False)
    if ex:
        e = [sum(int(x[i]) for x in ex) for i in range(5)]; k = [sum(int(x[5].split(",")[j]) for x in ex) for j in range(8)]; n = max(sum(k), 1)
        print(f"   exits {n} (by kind {k}), CV_STEPs {e[0]}; per exit ms: CV_STEP {e[1] / n / 25e3:.3f} (resolve {e[2] / n / 25e3:.3f}),"
              f" between {e[3] / n / 25e3:.3f}, other ioctls {e[4] / n / 25e3:.3f}; outside per exit {(sec - cyc / 25e6) * 1e3 / n:.2f} ms")
    if td: print(f"   teardown {sum(int(x[0]) for x in td)} pages, {sum(int(x[1]) for x in td) / 1e9:.2f} s, sweep {td[0][2]}")
    rr = json.load(open(os.path.join(REF, "result.json")))["arms"]["capstone-lifetimes"]["commands"]; rsec = sum(c["seconds"] for c in rr if c["tag"] == "run")
    RS = open(os.path.join(REF, "guest", "board.txt"), errors="replace").read()
    rex = re.findall(r"capstone_vm: exits steps=(\d+) kcyc=(\d+) rcyc=(\d+) ucyc=(\d+) ocyc=(\d+) ev=([\d,]+)", RS)
    rsteps = max((int(x[0]) for x in rex), default=0); steps = max((int(x[0]) for x in ex), default=10**9)
    rcyc = rr[0]["cycles"] or 1
    print(f"   reference {rsec:.1f} s, CV_STEPs {rsteps}; bundle - reference {sec - rsec:+.1f} s ({sec / rsec:.4f}x); inside cycles {cyc / rcyc:.4f}x")
    g2 = sec <= 0.995 * rsec and steps <= 15000; print(f"   M3 {chk(g2)} (run <= 0.995 x reference, CV_STEPs {steps} <= 15,000)")
if len(res) == 2:
    t, p = res["capstone-lifetimes"], res["plain-rv64"]
    print(f"Capstone / Plain run time {t['sec'] / p['sec']:.3f}x" + (f"; machine-wide instructions {t['mw'][1] / p['mw'][1]:.3f}x, cycles {t['mw'][0] / p['mw'][0]:.3f}x" if t["mw"] and p["mw"] else ""))
if not seen: print(f"NO DATA: no finished stage under {ROOTS}"); sys.exit(2)
sys.exit(fail)
