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
Exit 0 when all agree; 1 on any outside-set or divergence; 2 on no data; 3 on a voided pin."""
import json, sys
from pathlib import Path
DEFAULT = Path("/tmp/capstone/land-board/capstone/bug-corpora/cpython/allocator-boundary-repros/results/2026-10-19-vp2-trimmed-pinned/prediction.json")
PIN = {"hash_randomization": 0, "pythonhashseed": "0", "hash_abc": 1453079729188098211, "guest_env": ["PYTHONHASHSEED=0"]}
IMG = {"lt0": "0d322044b9118713", "lt1": "1034e6d3715e5355"}
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
        if not str(r["observation"].get("image_sha256", "")).startswith(IMG[img]):
            print(f"NO DATA: {c} image {r['observation'].get('image_sha256', '')[:16]} is not the record's {IMG[img]}"); return 2
        inside = s in cases[c]["signatures"]
        k = "agree" if inside else ("diverge" if cases[c]["stable"] else "outside")
        n[k] += 1
        if not inside:
            print(f"  {'DIVERGENCE (stable case)' if k == 'diverge' else 'outside the sampled set'} {c}: {s['verdict']} "
                  f"{s['cause']}/{s['symbol']}+{s['offset']} notes={s['notes']!r} (set: {cases[c]['verdicts']})")
    print(f"{img}: {n['agree']} of {len(cases)} inside their sets; {n['outside']} outside the sampled set (varying cases); "
          f"{n['diverge']} divergence(s) on stable cases")
    return 0 if n["agree"] == len(cases) else 1
if __name__ == "__main__":
    sys.exit(main())
