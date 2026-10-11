#!/usr/bin/env python3
"""The per-cell table, derived from results/20261006/matrix.tsv and nothing else.

Every count in this corpus's prose had been typed by hand, and by the time the
corpus was reviewed four files gave four different sets of numbers. This prints
the two tables the READMEs carry -- the per-cell table of
results/20261006/README.md and the per-arm summary of README.md -- and --check
fails if either README's copy differs, so the prose cannot drift from the
matrix again.

A cell is `detected / delivered`. A row counts as delivered when its
`delivered` column is `yes`, and as detected only when the fault is the arm's
mechanism: on the Capstone arms a capability cause (5, 7 bounds; 24, 25, 26
UNEXP_OP_TYPE, INVALID_CAP, UNEXP_CAP_TYPE -- capstone-qemu
target/riscv/cpu_bits.h); on CheriBSD any SIGPROT si_code. Case 32 on the base
arm is DETECTED with cause 2, an illegal-instruction trap, and is therefore
delivered but not detected -- the rule results/20261006/README.md states for it.

    python3 probe/counts.py           print both tables
    python3 probe/counts.py --check   exit 1 if either README's copy differs
"""
import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
MATRIX = HERE / "results/20261006/matrix.tsv"
CELL_README = HERE / "results/20261006/README.md"
SUMMARY_README = HERE / "README.md"
ARMS = ["spatial", "sublet", "cheribsd-revocation"]
CELLS = [("spatial", "nested"), ("spatial", "non-nested"),
         ("temporal", "nested"), ("temporal", "non-nested")]
CAPABILITY_CAUSES = {"5", "7", "24", "25", "26"}


def detected(row):
    if row["delivered"] != "yes" or row["verdict"] != "DETECTED":
        return False
    if row["arm"].startswith("cheribsd"):
        return row["detail"].startswith("si_code=")
    return row["detail"] in CAPABILITY_CAUSES


def tables():
    if not MATRIX.is_file():
        sys.exit(f"counts: no matrix at {MATRIX} -- nothing to count, not a zero")
    rows = list(csv.DictReader(MATRIX.open(), delimiter="\t"))
    if not rows:
        sys.exit(f"counts: {MATRIX} has no rows -- nothing to count, not a zero")
    unknown = {r["arm"] for r in rows} - set(ARMS)
    if unknown:
        sys.exit(f"counts: arms {sorted(unknown)} in the matrix are not counted here")
    cases = {r["case"]: (r["class"], r["side"]) for r in rows}
    out = ["| | | corpus | " + " | ".join(ARMS) + " |",
           "|---|---|---:|" + "---:|" * len(ARMS)]
    totals = {a: [0, 0] for a in ARMS}
    for cls, side in CELLS:
        corpus = sum(1 for v in cases.values() if v == (cls, side))
        cells = []
        for arm in ARMS:
            mine = [r for r in rows if r["arm"] == arm and (r["class"], r["side"]) == (cls, side)]
            d = sum(detected(r) for r in mine)
            n = sum(r["delivered"] == "yes" for r in mine)
            totals[arm][0] += d
            totals[arm][1] += n
            cells.append(f"**{d}/{n}**" if arm == "sublet" else f"{d}/{n}")
        out.append(f"| {cls} | {side} | {corpus} | " + " | ".join(cells) + " |")
    out.append(f"| **total** | | **{len(cases)}** | "
               + " | ".join(f"**{d}/{n}**" for d, n in totals.values()) + " |")
    summary = ["| arm | detected / scored | of the detections, the mechanism |",
               "|---|---:|---:|"]
    for arm, (d, n) in totals.items():
        cell = f"**{d} / {n}**" if arm == "sublet" else f"{d} / {n}"
        summary.append(f"| `{arm}` | {cell} | {d} |")
    return "\n".join(out), "\n".join(summary)


def main():
    cell, summary = tables()
    if sys.argv[1:] == ["--check"]:
        stale = [str(p.relative_to(HERE)) for p, text in ((CELL_README, cell), (SUMMARY_README, summary))
                 if not p.is_file() or text not in p.read_text()]
        if stale:
            print("counts: STALE -- these do not carry the table matrix.tsv gives:",
                  ", ".join(stale))
            print(cell + "\n\n" + summary)
            return 1
        print("counts: OK -- both READMEs carry the matrix's tables")
        return 0
    if sys.argv[1:]:
        sys.exit("usage: counts.py [--check]")
    print(cell + "\n\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
