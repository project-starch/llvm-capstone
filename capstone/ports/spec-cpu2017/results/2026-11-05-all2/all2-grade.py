#!/usr/bin/env python3
"""all2-grade.py: grade the all-together run plus the five outside-time switches on board 1
(results/2026-11-05-all2/PREREG.md). Runs: $A2_ROOT (default $CAPSTONE_WORK/spec/board/out-all2-b1)/<stage>.
Reference, same board, bitstream (C.5a) and images: out-all-b1/<stage>. Also reported: the 3 Nov baseline (spec/board/out)
and Plain test (out-hfl/<bench>-test-plain-rv64). Run time = the run commands' elapsed seconds; inside = cycles / 25 MHz.
--canary: gcc's Y0 only. A stage not finished prints PENDING. Exit 0 if every graded check holds, 1 otherwise, 2 if no data.
Y0  rc 0, specdiff PASS, outputs = twin, control passes, bench line fresh; every kit parameter read back; QNT0 007A1200
    (reassembled console, after the boot's own OpenSBI banner); instret within 0.1 % of the reference arm.
Y1  run time below the reference arm's, every arm.
Y2  gcc: launch <= 4 s and run <= 30 s (reference 16.83 s and 41.1 s).
Y3  imagick: launch <= 4.7 s over both commands, run >= 2 s lower, each arm.
Y4  nab: steps <= 4,500 and run >= 25 s lower, each arm.
Y5  lbm: steps <= 8,000 and run >= 40 s lower, each arm.
Y6  xz, mcf: run <= 0.98 x the reference.
Y7  inside cycles 0.97-1.005 x the reference, every arm."""
import json, os, re, sys

W = os.environ["CAPSTONE_WORK"]; B = os.path.join(W, "spec", "board")
ROOT = os.environ.get("A2_ROOT", os.path.join(B, "out-all2-b1")); REF = os.environ.get("A2_REF", os.path.join(B, "out-all-b1"))
CANARY = "--canary" in sys.argv[1:]
ST = [("gcc", "502.gcc_r-test-capstone-lifetimes-devmk", "502.gcc_r-test-capstone-lifetimes", "502.gcc_r"),
      ("imagick", "538.imagick_r-test-devmk", "538.imagick_r-test", "538.imagick_r"),
      ("nab", "544.nab_r-test-devmk", "544.nab_r-test", "544.nab_r"),
      ("lbm", "519.lbm_r-test-devmk", "519.lbm_r-test", "519.lbm_r"),
      ("xz", "557.xz_r-test-capstone-lifetimes-devmk", "557.xz_r-test-capstone-lifetimes", "557.xz_r"),
      ("mcf", "505.mcf_r-test-capstone-lifetimes-devmk", "505.mcf_r-test-capstone-lifetimes", "505.mcf_r")]
WANT = {"fault_ahead_max": "64", "retire_exit_sweep": "3", "image_rootfs": "Y", "step_budget_ns": "80000000"}

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
    x = dict(sec=sum(c["seconds"] for c in runs), cyc=sum(c["cycles"] or 0 for c in runs), ins=sum(c["instret"] or 0 for c in runs), why=why)
    T = os.path.join(d, "guest", "board.txt")
    if os.path.exists(T):
        T = open(T, errors="replace").read()
        part = T.split(f"@@SPEC-ARM-BEGIN {arm}", 1)[-1].split("@@SPEC-ARM-BEGIN", 1)[0]
        st = [dict(re.findall(r"(\w+)=(\d+)", m)) for m in re.findall(r"CAPSTONE_VM_STATS ([^\n]*)", part)]
        x.update(launch=sum(int(s.get("launch_ns", 0)) for s in st) / 1e9, steps=sum(int(s.get("steps", 0)) for s in st), nst=len(st),
                 params=dict(re.findall(r"^@@PARAM (\w+)=(\S+)$", T, re.M)))
    if kit:
        raw = os.path.join(d, "guest", "board-raw.txt")
        t = open(raw, errors="replace").read() if os.path.exists(raw) else ""
        # this boot's own QNT0: after the last OpenSBI banner, never a line the previous boot's replay carries (audit)
        x["qnt"] = re.findall(r"QNT0:([0-9A-F]{8})", t[t.rfind("OpenSBI"):])[-1:] if "OpenSBI" in t else []
    return x

fail = 0; seen = 0; Y = {}
def chk(k, ok):
    global fail
    Y.setdefault(k, []).append(ok)
    if not ok: fail = 1
    return "holds" if ok else "FAILS"
for b, st, base, full in ST:
    if CANARY and b != "gcc": continue
    for arm in ("capstone-spatial", "capstone-lifetimes"):
        ref = read(os.path.join(REF, st), arm)
        if ref is None: continue
        lab = f"{b:8s} {'MapLife' if arm == 'capstone-spatial' else 'Full':8s}"
        x = read(os.path.join(ROOT, st), arm, kit=True)
        if x is None: print(f"{lab} PENDING"); continue
        seen += 1
        y0 = (not x["why"] and x.get("params") == WANT and x["qnt"] == ["007A1200"] and x.get("nst", 0) > 0
              and ref["ins"] > 0 and abs(x["ins"] / ref["ins"] - 1) <= 0.001)
        ln = f"{lab} Y0 {chk('Y0', y0)}"
        if not y0: ln += f" ({','.join(x['why'])} params={x.get('params')} qnt={x['qnt']} ins={x['ins'] / max(ref['ins'], 1):.5f})"
        if CANARY: print(ln + f" | run {x['sec']:.1f} vs {ref['sec']:.1f} s | launch {x.get('launch', 0):.2f} vs {ref.get('launch', 0):.2f} s"); continue
        ln += f" | Y1 {chk('Y1', x['sec'] < ref['sec'])} run {x['sec']:.1f} vs {ref['sec']:.1f} s ({x['sec'] / ref['sec']:.4f})"
        d = ref["sec"] - x["sec"]
        if b == "gcc": ln += f" | Y2 {chk('Y2', x['launch'] <= 4 and x['sec'] <= 30)} launch {x['launch']:.2f} vs {ref['launch']:.2f} s"
        if b == "imagick": ln += f" | Y3 {chk('Y3', x['launch'] <= 4.7 and d >= 2)} launch {x['launch']:.2f} vs {ref['launch']:.2f} s, -{d:.1f} s"
        if b == "nab": ln += f" | Y4 {chk('Y4', x['steps'] <= 4500 and d >= 25)} steps {x['steps']} vs {ref['steps']}, -{d:.1f} s"
        if b == "lbm": ln += f" | Y5 {chk('Y5', x['steps'] <= 8000 and d >= 40)} steps {x['steps']} vs {ref['steps']}, -{d:.1f} s"
        if b in ("xz", "mcf"): ln += f" | Y6 {chk('Y6', x['sec'] <= 0.98 * ref['sec'])} steps {x['steps']} vs {ref['steps']}"
        ci = x["cyc"] / ref["cyc"]; ln += f" | Y7 {chk('Y7', 0.97 <= ci <= 1.005)} inside {ci:.4f}"
        ln += f" | outside {x['sec'] - x['cyc'] / 25e6:.1f} vs {ref['sec'] - ref['cyc'] / 25e6:.1f} s"
        bl = read(os.path.join(B, "out", base), arm); pl = read(os.path.join(B, "out-hfl", f"{full}-test-plain-rv64"), "plain-rv64")
        if bl: ln += f" | vs 3 Nov: run {x['sec'] / bl['sec']:.3f}"
        if pl: ln += f" | / Plain {x['sec'] / pl['sec']:.2f}x (all-b1 {ref['sec'] / pl['sec']:.2f}x" + (f", 3 Nov {bl['sec'] / pl['sec']:.2f}x)" if bl else ")")
        print(ln)
if not seen: print(f"NO DATA: no finished stage under {ROOT}"); sys.exit(2)
print("SUMMARY " + " ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(Y.items())))
sys.exit(fail)
