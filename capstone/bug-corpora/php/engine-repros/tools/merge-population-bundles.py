#!/usr/bin/env python3
"""Merge run-realphp.py's per-population bundles into one bundle per arm.

    merge-population-bundles.py <run dir with asan-blind/ and temporal-asan-visible/> <results/<stamp>>

run-realphp.py boots once per arm per POPULATION (asan-blind, temporal-asan-visible) and
writes results/<population>/bundle-<arm>/. corpus.json's verdict_bundles names ONE bundle per
arm, which derive-verdicts.py keys every case.json by. This writes that bundle: the two
populations' verdicts.jsonl concatenated (their case sets are disjoint by construction, and
that is checked), inputs.json carried over from one population with the other's recorded
beside it and both sources named by sha256, the tally recomputed. Nothing is re-judged here;
derive-verdicts.py does that from the merged rows.

The stock configuration (php-real-stock) is the judge's control, not a case arm; it is merged
the same way so the record is complete, but corpus.json does not name it.
"""
import hashlib, json, sys
from pathlib import Path

POPULATIONS = ("asan-blind", "temporal-asan-visible")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    arms = sorted({p.name[len("bundle-"):] for pop in POPULATIONS for p in (run / pop).glob("bundle-*")})
    for arm in arms:
        rows, seen, sources, inputs = [], set(), {}, None
        for pop in POPULATIONS:
            b = run / pop / f"bundle-{arm}"
            if not b.is_dir():
                sys.exit(f"{arm}: no bundle for population {pop}")
            lines = (b / "verdicts.jsonl").read_text().splitlines()
            for line in lines:
                case = json.loads(line)["observation"]["case"]
                if case in seen:
                    sys.exit(f"{arm}: {case} appears in both populations")
                seen.add(case)
            rows += lines
            inp = json.loads((b / "inputs.json").read_text())
            sources[pop] = {"verdicts.jsonl": sha(b / "verdicts.jsonl"), "inputs.json": sha(b / "inputs.json"),
                            "cases": inp["cases"], "tally": inp["tally"]}
            if inputs is None:
                inputs = inp
            else:
                for key in ("platform", "configuration", "arm", "corpus", "build", "contract"):
                    if inputs.get(key) != inp.get(key):
                        sys.exit(f"{arm}: the two populations' bundles differ in {key}")
        tally = {}
        for line in rows:
            d = json.loads(line)
            key = d["verdict"] if d["verdict"] != "NO-READING" else f"NO-READING:{d['reason']}"
            tally[key] = tally.get(key, 0) + 1
        inputs = dict(inputs, cases=len(rows), tally=dict(sorted(tally.items())),
                      population="asan-blind + temporal-asan-visible, one boot per population; see populations",
                      populations=sources)
        d = out / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "verdicts.jsonl").write_text("\n".join(rows) + "\n")
        (d / "inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
        print(f"{arm}: {len(rows)} cases, {tally}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
