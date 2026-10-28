#!/usr/bin/env python3
"""The tables a results record's README carries, computed from its bundles and nothing else.
    result-tables.py <results dir>          # bundle-<arm>/ (default scoring), sensitivity/bundle-<arm>/
Per scoring and arm: CAUGHT, MISSED and every NO-READING reason, over every case the record
measured, over its temporal cases (CWE-416/415/590) and over its CWE-416 cases (the CWE from
each case's oracle.json), then the CWE-416 catches split by native population. Every case one
arm measured must have a row in every arm's bundle, or the run exits 1 and prints which are
missing: a table with a hole in it is not printed as if it were complete. The stock
configuration is the judge's control and is tabulated like the arms so the record is whole.
"""
import collections, json, sys
from pathlib import Path

TEMPORAL = ("CWE-416", "CWE-415", "CWE-590")
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
        for c in seen:
            if c not in known:
                missing.append(f"{arm}: a row for {c}, which is not a case of this corpus")
        tallies[arm] = seen
    if missing:
        return lines, missing
    # THE RECORD'S OWN CASE SET: every case any arm measured, and every arm must have every one
    # of them. A record taken before the corpus grew (run7 to run15: 65 cases; the merge added
    # five, 2026-10-28) is complete over the cases it measured, and says how many; a hole
    # between arms is still refused.
    measured = set().union(*(set(s) for s in tallies.values()))
    for arm in ARMS:
        for c in sorted(measured - set(tallies[arm])):
            missing.append(f"{arm}: no row for {c}")
    if missing:
        return lines, missing
    # Was this record judged with the `chain` attribution class at all? A bundle judged before
    # the class existed (run7, run10, run14) carries its chain catches as `function`, and a
    # column reading 0 there would be false; it reads "not classified" instead. The judge marks
    # its scoring since the class exists; a record without the mark and without a chain row is
    # unclassified (fifth review, 2026-10-28).
    classified = any("chain" in str(json.loads((scoring_dir / f"bundle-{arm}" / "inputs.json").read_text())
                                    .get("scoring", {}).get("attribution", ""))
                     or any(r["observation"].get("attribution") == "chain" for r in tallies[arm].values())
                     for arm in ARMS)
    def tally(arm, pick):
        t = collections.Counter()
        for c, r in tallies[arm].items():
            if pick(c):
                t[r["verdict"] if r["verdict"] != "NO-READING" else f"NO-READING/{r.get('reason')}"] += 1
                # a catch attributed by the call path alone (verdicts.py ATTRIBUTIONS "chain")
                # is counted inside CAUGHT and shown apart: it is the weakest attribution
                if r["verdict"] == "CAUGHT" and r["observation"].get("attribution") == "chain":
                    t["CAUGHT/chain"] += 1
        return t
    present = sorted(measured)
    for name, pick in ((f"all {len(present)} cases", lambda c: True),
                       ("the temporal cases (CWE-416, CWE-415, CWE-590)", lambda c: known[c][0] in TEMPORAL),
                       ("the CWE-416 cases", lambda c: known[c][0] == "CWE-416")):
        n = sum(1 for c in present if pick(c))
        lines += [f"**{name}** ({n}):", "", "| arm | CAUGHT | of which by chain | MISSED | NO-READING |", "|---|--:|--:|--:|---|"]
        for arm in ARMS:
            t = tally(arm, pick)
            nr = ", ".join(f"{k.split('/', 1)[1]} {v}" for k, v in sorted(t.items()) if k.startswith("NO-READING/"))
            chain = t["CAUGHT/chain"] if classified else "not classified"
            lines.append(f"| {LABEL[arm]} | {t['CAUGHT']} | {chain} | {t['MISSED']} | {nr or '0'} |")
        lines.append("")
    if not classified:
        lines += ["(\"not classified\": this record was judged before the `chain` attribution class existed, "
                  "2026-10-28; its chain catches are carried as `function` and are not re-labelled)", ""]
    lines += ["**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):", ""]
    for arm in ARMS:
        parts = []
        for pop in ("temporal-asan-visible", "asan-blind"):
            sel = [c for c in present if known[c][0] == "CWE-416" and known[c][1] == pop]
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
    print("\n".join(out).rstrip("\n"))   # one newline at the end, so a README built from this has no trailing blank line
    return 0


if __name__ == "__main__":
    sys.exit(main())
