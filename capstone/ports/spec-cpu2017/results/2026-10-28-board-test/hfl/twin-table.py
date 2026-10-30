#!/usr/bin/env python3
"""twin-table.py RUN_DIR REF_DIR [RUN_DIR REF_DIR ...]: per arm, the run commands' instret of a twin run of the hfl
images against the campaign twin's run of the same stage, plus each run's rc, specdiff and output verdicts.
A run without result.json, or an arm with no instret, is printed as NO DATA and makes the exit status 2."""
import json
import sys
from pathlib import Path


def load(d):
    p = Path(d) / "result.json"
    return json.loads(p.read_text()) if p.is_file() else None


def main():
    a = sys.argv[1:]
    rc = 0
    print(f"{'run':44s} {'arm':19s} {'cmds':>5s} {'rc!=0':>5s} {'specdiff':>8s} {'instret new':>16s} {'instret ref':>16s} "
          f"{'new/ref':>8s} {'nodes new/ref':>17s}  outputs")
    for run, ref in zip(a[::2], a[1::2]):
        r, f = load(run), load(ref)
        if r is None or f is None:
            print(f"{Path(run).name:44s} NO DATA ({'run' if r is None else 'reference'} has no result.json)")
            rc = 2
            continue
        for arm, d in r["arms"].items():
            cs = [c for c in d["commands"] if c["tag"] == "run"]
            fr = [c for c in f["arms"].get(arm, {}).get("commands", []) if c["tag"] == "run"]
            n, m = sum(c["instret"] or 0 for c in cs), sum(c["instret"] or 0 for c in fr)
            if not n or not m:
                print(f"{Path(run).name:44s} {arm:19s} NO DATA (instret new {n}, reference {m})")
                rc = 2
                continue
            v = d.get("validation", [])
            sd = f"{sum(x['verdict'] == 'PASS' for x in v)}/{len(v)}"
            bad = sum(c["rc"] != 0 for c in d["commands"])
            nodes = f"{sum(c['nodes'] or 0 for c in cs)}/{sum(c['nodes'] or 0 for c in fr)}"
            tw = r.get("vs_twin", {}).get(arm)
            xa = [x for k, x in r.get("cross_arm", {}).items()]
            outs = ("vs-ref identical" if tw == [] else f"vs-ref DIFFER {tw[:4]}" if tw else "vs-ref n/a")
            if xa:
                outs += "; cross-arm " + ("identical" if all(x == [] for x in xa) else "DIFFER")
            print(f"{Path(run).name:44s} {arm:19s} {len(d['commands']):5d} {bad:5d} {sd:>8s} {n:16,d} {m:16,d} "
                  f"{n / m:8.5f} {nodes:>17s}  {outs}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
