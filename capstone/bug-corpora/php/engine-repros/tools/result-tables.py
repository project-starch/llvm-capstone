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
LABEL = {"php-real-stock": "stock", "virtual-malloc": "system allocator",
         "virtual-nested-pools": "nested", "virtual-nested-exact": "nested exact",
         "cheribsd-revocation": "CheriBSD revocation"}
# ORDER, for the tables; a record is tabulated over the arms IT HOLDS, not over this list.
ARM_ORDER = ("php-real-stock", "virtual-malloc", "virtual-nested-pools", "virtual-nested-exact",
             "cheribsd-revocation")


def arms_of(scoring_dir):
    """The arms this record measured, and the refusals a half-written record earns.

    A HARDCODED ARM LIST IS THE SUBSET FAILURE. This list was the four virtual arms, so the
    CheriBSD record added on 2026-10-30 could not be tabulated at all, and adding a fifth name to
    the tuple would instead have made every earlier record fail for lacking it. Neither is a
    reading. So the arm set is the record's own, and the three ways that could go quiet are each
    refused rather than tabulated: a bundle directory with no verdicts.jsonl is HALF WRITTEN, not
    absent; an arm this corpus does not declare is a typo, not a new arm; and a record with no
    bundle at all is not an empty table. What a record does NOT hold is named in the output, so a
    short table reads as short instead of looking complete.
    """
    found, problems = [], []
    for b in sorted(scoring_dir.glob("bundle-*")):
        if not b.is_dir():
            continue
        arm = b.name[len("bundle-"):]
        if arm not in LABEL:
            problems.append(f"{b}: {arm!r} is not an arm this corpus declares (known: {sorted(LABEL)})")
        elif not (b / "verdicts.jsonl").is_file():
            problems.append(f"{b}: a bundle directory with no verdicts.jsonl -- half written, not absent")
        else:
            found.append(arm)
    if not found and not problems:
        problems.append(f"{scoring_dir}: no bundle-<arm>/ at all; a record with no bundles is not an empty table")
    return [a for a in ARM_ORDER if a in found], problems


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
    arms, missing = arms_of(scoring_dir)
    if missing:
        return lines, missing
    absent = [LABEL[a] for a in ARM_ORDER if a not in arms]
    lines += [f"Arms in this record: {', '.join(LABEL[a] for a in arms)}."
              + (f" Not measured here: {', '.join(absent)}." if absent else ""), ""]
    tallies = {}
    for arm in arms:
        b = scoring_dir / f"bundle-{arm}"
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
    for arm in arms:
        for c in sorted(measured - set(tallies[arm])):
            missing.append(f"{arm}: no row for {c}")
    if missing:
        return lines, missing
    # Was this record judged with the `chain` attribution class at all? A bundle judged before
    # the class existed (run7, run10, run14) carries its chain catches as `function`, and a
    # column reading 0 there would be false; it reads "not classified" instead. The judge marks
    # its scoring since the class exists; a record without the mark and without a chain row is
    # unclassified (fifth review, 2026-10-28).
    # inputs.scoring is an OBJECT. A bundle whose `scoring` is prose is refused by name rather than
    # raising AttributeError here: on 2026-10-30 the CheriBSD runner wrote a sentence, and a reader
    # of someone else's corpus gets a traceback instead of a sentence saying which file is wrong.
    def attribution_of(arm):
        s = json.loads((scoring_dir / f"bundle-{arm}" / "inputs.json").read_text()).get("scoring", {})
        if not isinstance(s, dict):
            missing.append(f"bundle-{arm}/inputs.json: scoring is {type(s).__name__}, not an object "
                           f"with an `attribution` key; this tool cannot read it")
            return ""
        return str(s.get("attribution", ""))
    classified = any("chain" in attribution_of(arm)
                     or any(r["observation"].get("attribution") == "chain" for r in tallies[arm].values())
                     for arm in arms)
    if missing:
        return lines, missing
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
        for arm in arms:
            t = tally(arm, pick)
            nr = ", ".join(f"{k.split('/', 1)[1]} {v}" for k, v in sorted(t.items()) if k.startswith("NO-READING/"))
            chain = t["CAUGHT/chain"] if classified else "not classified"
            lines.append(f"| {LABEL[arm]} | {t['CAUGHT']} | {chain} | {t['MISSED']} | {nr or '0'} |")
        lines.append("")
    if not classified:
        # DO NOT ASSERT A REASON THE BUNDLE DOES NOT CARRY. This used to say the record was judged
        # before the `chain` class existed, which is false for a record whose arm does not
        # attribute by call path at all -- the CheriBSD arm attributes by a paired boot, and the
        # sentence appeared in its generated README as a statement about a date (2026-10-30). The
        # two causes are not distinguishable from a bundle: run7 predates the class and its
        # attribution string is non-empty too. So the fact is stated and the reason is left open.
        lines += ["(\"not classified\": no catch in this record carries the `chain` attribution "
                  "class. Either the record was judged before the class existed (2026-10-28), in "
                  "which case its chain catches are carried as `function` and are not re-labelled, "
                  "or its arms do not attribute by call path at all. The bundles do not say which, "
                  "so neither is claimed here)", ""]
    lines += ["**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):", ""]
    for arm in arms:
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
