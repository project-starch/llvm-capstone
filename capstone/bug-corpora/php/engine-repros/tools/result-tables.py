#!/usr/bin/env python3
"""The tables a results record's README carries, computed from its bundles and nothing else.
    result-tables.py <results dir>          # bundle-<arm>/ (default scoring), sensitivity/bundle-<arm>/
Per scoring and arm: CAUGHT, MISSED and every NO-READING reason, over all 65 cases and over the
CWE-416 cases (the CWE from each case's oracle.json), then the CWE-416 catches split by native
population. Every case must have a row in every arm's bundle, or the run exits 1 and prints
which are missing: a table with a hole in it is not printed as if it were complete. The stock
configuration is the judge's control and is tabulated like the arms so the record is whole.
"""
import collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent
ARMS = ("php-real-stock", "virtual-malloc", "virtual-nested-pools", "virtual-nested-exact")
LABEL = {"php-real-stock": "stock", "virtual-malloc": "system allocator",
         "virtual-nested-pools": "nested", "virtual-nested-exact": "nested exact"}


def cases():
    out = {}
    for d in sorted(CORPUS.iterdir()):
        if (d / "oracle.json").is_file():
            o = json.loads((d / "oracle.json").read_text())
            out[d.name] = (o["cwe"], o["population"])
    return out


def rows(bundle):
    return [json.loads(l) for l in (bundle / "verdicts.jsonl").read_text().splitlines() if l.strip()]


def tables(scoring_dir, known, title):
    missing, lines = [], [f"### {title}", ""]
    tallies = {}
    for arm in ARMS:
        b = scoring_dir / f"bundle-{arm}"
        if not (b / "verdicts.jsonl").is_file():
            missing.append(f"{b}: no bundle")
            continue
        seen = {}
        for r in rows(b):
            c = r["observation"]["case"]
            if c in seen:
                missing.append(f"{arm}: two rows for {c}; a bundle states each cell once")
            seen[c] = r
        for c in known:
            if c not in seen:
                missing.append(f"{arm}: no row for {c}")
        tallies[arm] = seen
    if missing:
        return lines, missing
    def tally(arm, pick):
        t = collections.Counter()
        for c, r in tallies[arm].items():
            if pick(c):
                t[r["verdict"] if r["verdict"] != "NO-READING" else f"NO-READING/{r.get('reason')}"] += 1
        return t
    for name, pick in (("all 65 cases", lambda c: True),
                       ("the CWE-416 cases", lambda c: known[c][0] == "CWE-416")):
        n = sum(1 for c in known if pick(c))
        lines += [f"**{name}** ({n}):", "", "| arm | CAUGHT | MISSED | NO-READING |", "|---|--:|--:|---|"]
        for arm in ARMS:
            t = tally(arm, pick)
            nr = ", ".join(f"{k.split('/', 1)[1]} {v}" for k, v in sorted(t.items()) if k.startswith("NO-READING/"))
            lines.append(f"| {LABEL[arm]} | {t['CAUGHT']} | {t['MISSED']} | {nr or '0'} |")
        lines.append("")
    lines += ["**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):", ""]
    for arm in ARMS:
        parts = []
        for pop in ("temporal-asan-visible", "asan-blind"):
            sel = [c for c in known if known[c][0] == "CWE-416" and known[c][1] == pop]
            caught = sum(1 for c in sel if tallies[arm][c]["verdict"] == "CAUGHT")
            parts.append(f"{caught} of {len(sel)}")
        lines.append(f"- {LABEL[arm]}: {' / '.join(parts)}")
    lines.append("")
    return lines, []


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"{root}: no such results directory", file=sys.stderr)
        return 1
    known = cases()
    out, problems = [], []
    for sub, title in (("", "Default scoring (native crashers out of the denominator)"),
                       ("sensitivity", "Sensitivity scoring (every case scored by fault site)")):
        d = root / sub if sub else root
        if sub and not d.is_dir():
            continue  # a record without a sensitivity re-scoring says so in its README
        lines, missing = tables(d, known, title)
        out += lines
        problems += missing
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
