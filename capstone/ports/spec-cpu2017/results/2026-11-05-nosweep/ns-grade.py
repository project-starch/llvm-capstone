#!/usr/bin/env python3
"""ns-grade.py: grade all2 with no exit sweep (results/2026-11-05-nosweep/PREREG.md). Runs: $NS_ROOTS (default
$CAPSTONE_WORK/spec/board/out-ns-b1 and out-ns-b2)/<stage>. Reference: out-all2-b1/<stage> (board 1, same bitstream C.5a,
images, module, launcher and firmware; only retire_exit_sweep differs, 3 there and 2 here). Stages on board 2 compare
across boards. Run time = the run commands' elapsed seconds. A stage not finished prints PENDING.
Exit 0 if every graded check holds, 1 otherwise, 2 if no data.
Z0  rc 0, specdiff PASS, outputs = twin, control passes, bench line fresh; the four parameters read back
    (retire_exit_sweep=2); QNT0 007A1200 (after the boot's own OpenSBI banner); instret within 0.1 % of the reference;
    every teardown line of the arm reads sweep=2 (spec.txt's dmesg lines, placed by time inside the arm's commands).
Z1  teardown <= 0.25 ms per page, every arm with >= 10,000 pages torn down (reference about 0.50).
Z2  xz: run >= 300 s below the reference (predicted about -528 s).
Z3  lbm: run >= 15 s below (about -34 s); mcf: run >= 25 s below (about -56 s).
Z4  gcc, imagick, nab: run within 2 % of the reference (predicted about 0).
Reported: share of the gap to Plain closed (baseline 28-29 Oct, Plain test), beside the reference's."""
import json, os, re, sys

W = os.environ["CAPSTONE_WORK"]; B = os.path.join(W, "spec", "board")
ROOTS = os.environ.get("NS_ROOTS", f"{B}/out-ns-b1:{B}/out-ns-b2").split(":")
REF = os.environ.get("NS_REF", os.path.join(B, "out-all2-b1"))
ST = [("gcc", "502.gcc_r-test-capstone-lifetimes-devmk"), ("imagick", "538.imagick_r-test-devmk"), ("nab", "544.nab_r-test-devmk"),
      ("lbm", "519.lbm_r-test-devmk"), ("xz", "557.xz_r-test-capstone-lifetimes-devmk"), ("mcf", "505.mcf_r-test-capstone-lifetimes-devmk")]
WANT = {"fault_ahead_max": "64", "retire_exit_sweep": "2", "image_rootfs": "Y", "step_budget_ns": "80000000"}
P = {"gcc": 8.11, "imagick": 20.07, "nab": 867.17, "lbm": 1895.21, "xz": 8065.87, "mcf": 9082.86}
BL = {"gcc": 55.33, "imagick": 33.53, "nab": 1180.83, "lbm": 2108.9, "xz": 12181.99, "mcf": 16208.92}

def read(d, arm, kit=False):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p): return None
    r = json.load(open(p))
    if not r.get("done") or arm not in r["arms"]: return None
    a = r["arms"][arm]; runs = [c for c in a["commands"] if c["tag"] == "run"]; why = []
    if any(c["rc"] != 0 for c in a["commands"]): why.append("rc")
    if not all(v["verdict"] == "PASS" for v in a["validation"]): why.append("specdiff")
    if r.get("vs_twin", {}).get(arm): why.append("twin")
    if not all(c.get("bench_line_fresh", True) for c in a["commands"]): why.append("not fresh")
    ctl = os.path.join(d, "control.result.json")
    if not os.path.exists(ctl) or json.load(open(ctl)) != {"kind": "exit", "value": 0}: why.append("control")
    x = dict(sec=sum(c["seconds"] for c in runs), ins=sum(c["instret"] or 0 for c in runs), why=why)
    T = os.path.join(d, "guest", "board.txt")
    if os.path.exists(T):
        T = open(T, errors="replace").read()
        part = T.split(f"@@SPEC-ARM-BEGIN {arm}", 1)[-1].split("@@SPEC-ARM-BEGIN", 1)[0]
        # teardown lines are dmesg, printed before the step's own output: give each to the arm whose command it ends
        # (dmesg time inside that command's [up0, up1 + 5 s] window)
        S = open(os.path.join(d, "spec.txt"), errors="replace").read() if os.path.exists(os.path.join(d, "spec.txt")) else ""
        win = [(float(u0), float(u1)) for a_, u0, u1 in re.findall(r"^@@SPEC-END \d+ \w+ (\S+) \d+ rc=\d+ up0=([\d.]+) up1=([\d.]+)", S, re.M) if a_ == arm]
        td = [(int(p_), int(n_), int(s_)) for ts, p_, n_, s_ in re.findall(r"\[\s*([\d.]+)\] capstone_vm: teardown pages=(\d+) ns=(\d+) sweep=(\d+)", S)
              if any(u0 <= float(ts) <= u1 + 5 for u0, u1 in win)]
        x.update(params=dict(re.findall(r"^@@PARAM (\w+)=(\S+)$", T, re.M)), td=td,
                 tdpages=sum(t[0] for t in td), tdns=sum(t[1] for t in td))
    if kit:
        raw = os.path.join(d, "guest", "board-raw.txt")
        t = open(raw, errors="replace").read() if os.path.exists(raw) else ""
        x["qnt"] = re.findall(r"QNT0:([0-9A-F]{8})", t[t.rfind("OpenSBI"):])[-1:] if "OpenSBI" in t else []
    return x

fail = 0; seen = 0; Y = {}
def chk(k, ok):
    global fail
    Y.setdefault(k, []).append(ok)
    if not ok: fail = 1
    return "holds" if ok else "FAILS"
for b, st in ST:
    for arm in ("capstone-spatial", "capstone-lifetimes"):
        ref = read(os.path.join(REF, st), arm)
        if ref is None: continue
        lab = f"{b:8s} {'MapLife' if arm == 'capstone-spatial' else 'Full':8s}"
        x = next((v for v in (read(os.path.join(r, st), arm, kit=True) for r in ROOTS) if v), None)
        if x is None: print(f"{lab} PENDING"); continue
        seen += 1
        z0 = bool(not x["why"] and x.get("params") == WANT and x["qnt"] == ["007A1200"] and ref["ins"] > 0
              and abs(x["ins"] / ref["ins"] - 1) <= 0.001 and x.get("td") and all(t[2] == 2 for t in x["td"]))
        ln = f"{lab} Z0 {chk('Z0', z0)}"
        if not z0: ln += f" ({','.join(x['why'])} params={x.get('params')} qnt={x['qnt']} ins={x['ins'] / max(ref['ins'], 1):.5f} sweeps={sorted({t[2] for t in x.get('td', [])})})"
        mpp = x["tdns"] / 1e6 / max(x["tdpages"], 1); rpp = ref.get("tdns", 0) / 1e6 / max(ref.get("tdpages", 1), 1)
        ln += f" | teardown {mpp:.3f} vs {rpp:.3f} ms/page over {x['tdpages']} pages"
        if x["tdpages"] >= 10000: ln += f" Z1 {chk('Z1', mpp <= 0.25)}"
        d = ref["sec"] - x["sec"]
        ln += f" | run {x['sec']:.1f} vs {ref['sec']:.1f} s ({-d:+.1f})"
        if b == "xz": ln += f" Z2 {chk('Z2', d >= 300)}"
        if b == "lbm": ln += f" Z3 {chk('Z3', d >= 15)}"
        if b == "mcf": ln += f" Z3 {chk('Z3', d >= 25)}"
        if b in ("gcc", "imagick", "nab"): ln += f" Z4 {chk('Z4', abs(d) <= 0.02 * ref['sec'])}"
        if arm == "capstone-lifetimes":
            g = BL[b] - P[b]; ln += f" | gap closed {100 * (BL[b] - x['sec']) / g:.0f} % (reference {100 * (BL[b] - ref['sec']) / g:.0f} %)"
        print(ln)
if not seen: print(f"NO DATA: no finished stage under {ROOTS}"); sys.exit(2)
print("SUMMARY " + " ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(Y.items())))
sys.exit(fail)
