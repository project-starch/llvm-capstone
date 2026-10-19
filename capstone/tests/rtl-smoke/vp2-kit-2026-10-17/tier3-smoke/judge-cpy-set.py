#!/usr/bin/env python3
"""judge-cpy-set.py BUNDLE IMAGE [PREDICTION]: the cpython allocator-boundary silicon gate and verdict for one arm.
PREDICTION defaults to dev's results/2026-10-19-vp2-trimmed-pinned/prediction.json; IMAGE is its key (lt0 or lt1).
1. The pin gate, first (the runtime lane's ruling, 2026-10-19): the bundle's inputs.json `guest` must carry
   kernel_cmdline containing the token norandmaps, and hash_randomization 0, pythonhashseed "0",
   hash_abc 1453079729188098211 and guest_env ["PYTHONHASHSEED=0"] exactly. A mismatch VOIDS the boot (exit 3).
2. Every case's signature, the record's fields (verdict, attribution, fault cause/offset/symbol, notes, infra;
   absolute pcs excluded), judged against the case's set:
   - inside the set: agreement;
   - outside, on a varying case: "outside the sampled set";
   - outside, on a stable case: DIVERGENCE (a finding to check; "stable" means not seen to vary in five QEMU runs).
   Every image sha256 must equal the record's image; the cases must be the record's, exactly.
Exit 0 when all agree; 1 on any outside-set or divergence; 2 on no data; 3 on a voided pin.

AMENDED 2026-10-19 evening to the grading rule on dev (18a7cf883de3, the lt0 README beside the prediction table):
- a STABLE cell must match its set's full signature (verdict and evidence site), as before;
- a VARYING cell is graded on its VERDICT alone, inside its recorded verdict set; the site is reported, not graded; a
  varying verdict outside its set is a MOVED cell (the re-record rule applies);
- lt0's board image is the stage 20bcfdd1073976e0, accepted as lt0 by a RECORDED identity only: every ALLOC section and
  the loaded image (objcopy -O binary, 47dcef5f380b3217) byte-equal to 0d322044b9118713; the lt1 image's loaded bytes
  differ (942096f9f1e82dbc), the control. Exactly that pair is mapped; any other image is refused as before."""
import json, sys
from pathlib import Path
DEFAULT = Path("/tmp/capstone/land-board/capstone/bug-corpora/cpython/allocator-boundary-repros/results/2026-10-19-vp2-trimmed-pinned/prediction.json")
PIN = {"hash_randomization": 0, "pythonhashseed": "0", "hash_abc": 1453079729188098211, "guest_env": ["PYTHONHASHSEED=0"]}
IMG = {"lt0": "0d322044b9118713", "lt1": "1034e6d3715e5355"}
ALIAS = {"lt0": "20bcfdd1073976e0"}   # the recorded loaded-identity pair, and only it
def sig(r):
    o = r.get("observation", {}); f = o.get("fault") or {}
    return {"verdict": r.get("verdict"), "attribution": o.get("attribution"), "cause": f.get("cause"), "offset": f.get("offset"),
            "symbol": f.get("symbol"), "notes": o.get("notes"), "infra": o.get("infra")}
def main():
    b, img = Path(sys.argv[1]), sys.argv[2]
    pred = json.loads(Path(sys.argv[3] if len(sys.argv) > 3 else DEFAULT).read_text())["images"][img]
    try:
        inp = json.loads((b / "inputs.json").read_text())
        rows = [json.loads(l) for l in (b / "verdicts.jsonl").read_text().splitlines() if l.strip()]
    except Exception as e:
        print(f"NO DATA: {e}"); return 2
    g = inp.get("guest") or {}
    bad = [k for k, v in PIN.items() if g.get(k) != v]
    if "norandmaps" not in str(g.get("kernel_cmdline", "")).split(): bad.append("kernel_cmdline")
    if bad:
        print(f"VOID: the guest's pin differs in {bad}: {json.dumps(g)}"); return 3
    print(f"pin OK: {json.dumps(g)}")
    cases = pred["cases"]; got = {r["observation"]["case"]: r for r in rows}
    if set(got) != set(cases):
        print(f"NO DATA: cases differ from the record: missing {sorted(set(cases) - set(got))}, extra {sorted(set(got) - set(cases))}"); return 2
    n = {"agree": 0, "outside": 0, "diverge": 0}
    for c in sorted(cases):
        r = got[c]; s = sig(r)
        ish = str(r["observation"].get("image_sha256", ""))
        if not (ish.startswith(IMG[img]) or (img in ALIAS and ish.startswith(ALIAS[img]))):
            print(f"NO DATA: {c} image {r['observation'].get('image_sha256', '')[:16]} is not the record's {IMG[img]}"); return 2
        if cases[c]["stable"]:
            inside = s in cases[c]["signatures"]
        else:
            inside = s["verdict"] in cases[c]["verdicts"]
            if inside and s not in cases[c]["signatures"]:
                print(f"  varying {c}: {s['verdict']} inside its verdict set; site {s['cause']}/{s['symbol']}+{s['offset']} "
                      f"not among the sampled sites (reported, not graded)")
        k = "agree" if inside else ("diverge" if cases[c]["stable"] else "outside")
        n[k] += 1
        if not inside:
            print(f"  {'DIVERGENCE (stable case)' if k == 'diverge' else 'MOVED (varying verdict outside its set)'} {c}: {s['verdict']} "
                  f"{s['cause']}/{s['symbol']}+{s['offset']} notes={s['notes']!r} (set: {cases[c]['verdicts']})")
    print(f"{img}: {n['agree']} of {len(cases)} inside their sets; {n['outside']} moved (varying verdict outside its set); "
          f"{n['diverge']} divergence(s) on stable cases")
    return 0 if n["agree"] == len(cases) else 1
if __name__ == "__main__":
    sys.exit(main())
