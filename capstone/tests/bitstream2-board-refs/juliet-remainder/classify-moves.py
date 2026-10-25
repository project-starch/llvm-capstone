#!/usr/bin/env python3
"""classify-moves.py REF_TSV NEW_TSV LABEL: Juliet spatial verdicts against the 2026-10-19 exact-bounds record, every
moved case sorted into the classes pre-registered for the remainder (juliet-rem), the rest listed as UNEXPECTED.
  wide-snprintf  NOT-TRIGGERED:wide-snprintf -> MISSED: judge.py relabels that family only from the case's own twin
                 reading, and these runs have no twin phase (the sample's tree-kit run moved 18 this way)
  cwe170         a CWE-126/127 CWE170 case moving either way: whether its over-read runs on depends on whether a NUL
                 follows the copy (the sample moved 3 this way, both directions)
  intra-reason   an intra-object case whose NO-READING reason changes, NO-READING both sides
Exit 0 when every move is in a class, 1 when any is UNEXPECTED, 2 when a table is missing or empty."""
import collections, csv, sys


def load(p):
    try:
        return {r["case"]: r for r in csv.DictReader(open(p), delimiter="\t")}
    except FileNotFoundError:
        return {}


ref, new = load(sys.argv[1]), load(sys.argv[2])
if not new:
    print(f"  {sys.argv[3]}: NO DATA (no new table)"); sys.exit(2)
missing = [c for c in new if c not in ref]
if missing:
    print(f"  {sys.argv[3]}: NO DATA ({len(missing)} cases absent from the record)"); sys.exit(2)
cls, unexpected = collections.Counter(), []
for c, n in new.items():
    a, b = ref[c]["verdict"], n["verdict"]
    if a == b:
        continue
    if a.startswith("NOT-TRIGGERED:wide-snprintf") and b == "MISSED":
        cls["wide-snprintf"] += 1
    elif "CWE170" in c:
        cls["cwe170"] += 1
    elif "NO-READING" in a and "NO-READING" in b and n.get("intra_object") in ("1", "True", "true"):
        cls["intra-reason"] += 1
    else:
        unexpected.append((c, a, b))
tally = collections.Counter(n["verdict"].split(":")[0] for n in new.values())
print(f"  {sys.argv[3]}: cases {len(new)}, verdicts {dict(tally)}, moved {sum(cls.values()) + len(unexpected)} "
      f"(classes {dict(cls)}), UNEXPECTED {len(unexpected)}" + "".join(f"\n    {c}: {a} -> {b}" for c, a, b in unexpected[:40]))
sys.exit(1 if unexpected else 0)
