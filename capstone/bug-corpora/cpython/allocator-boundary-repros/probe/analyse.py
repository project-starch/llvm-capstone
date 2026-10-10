#!/usr/bin/env python3
"""Cross the arms into one matrix.

sysalloc-bounds is the BASELINE, not a control to be subtracted: per-object
bounds are what applications get since PR #170, so a catch is a case that
sysalloc-bounds does NOT fault on and a protected arm does. sysalloc-none says
what the defect does with no heap protection at all; a fault there is not a
catch, because tag integrity is in the hardware and cannot be switched off.

The pair this corpus exists for is sysalloc-sublet against sublet-pymalloc:
the first issues and revokes only the outer allocation, the second pymalloc's
own blocks too, so the difference is what sublets the nested allocator.

(Review note, 2026-10-10.) The paragraph above describes the earlier five-arm
design. The corpus now runs the three arms corpus.json lists: spatial is
sysalloc-bounds, sublet is level0 plus patch 0014 (not sysalloc-sublet), and
cheribsd-revocation. results/20261006/matrix.tsv was NOT written by this script.
It has a different shape (one row per case and arm, with a delivered column),
and the verdicts.tsv files this script reads are not committed.
probe/counts.py derives the published table from that matrix.
"""
import argparse, csv, json, pathlib, sys

# Read from the corpus rather than written here: a hardcoded list went stale
# the moment the corpus changed its arms, and a stale list makes this script
# refuse perfectly good result directories.
_CORPUS = pathlib.Path(__file__).resolve().parent.parent / "corpus.json"
ARMS = json.loads(_CORPUS.read_text())["required_arms"]

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("results", type=pathlib.Path, nargs="+",
               help="one run directory per arm (each with verdicts.tsv and run.meta)")
p.add_argument("--out", type=pathlib.Path, required=True)
a = p.parse_args()

rows, meta = {}, {}
for d in a.results:
    v = d / "verdicts.tsv"
    if not v.is_file():
        sys.exit("%s has no verdicts.tsv" % d)
    m = dict(l.rstrip("\n").split("\t", 1)
             for l in (d / "run.meta").read_text().splitlines() if "\t" in l)
    arm = m.get("arm") or "?"
    if arm not in ARMS:
        sys.exit("%s: arm %r is not one of %s" % (d, arm, ARMS))
    if arm in meta:
        sys.exit("two run directories claim arm %r; a matrix cannot have both" % arm)
    meta[arm] = m
    for r in csv.DictReader(v.open(), delimiter="\t"):
        rows.setdefault(r["case"], {})[arm] = r

present = [x for x in ARMS if x in meta]
a.out.mkdir(parents=True, exist_ok=True)
with (a.out / "matrix.tsv").open("w") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["case"] + present)
    for case in sorted(rows):
        w.writerow([case] + [(rows[case].get(arm, {}).get("last") or "NOROW")
                             for arm in present])
json.dump({"arms": {k: meta[k] for k in present},
           "cases": len(rows),
           "note": ("Each cell is the trigger's own last lines, not a verdict derived "
                    "from the exit status. Judging a case by the batch's status has "
                    "produced wrong rows in this lane before.")},
          (a.out / "inputs.json").open("w"), indent=1, sort_keys=True)
print("arms present: %s" % ", ".join(present))
print("cases: %d" % len(rows))
missing = [(c, arm) for c in rows for arm in present if arm not in rows[c]]
if missing:
    print("MISSING rows: %d (printed in matrix.tsv as NOROW)" % len(missing))
print("wrote %s/matrix.tsv and inputs.json" % a.out)
