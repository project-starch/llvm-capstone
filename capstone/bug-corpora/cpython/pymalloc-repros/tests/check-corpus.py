#!/usr/bin/env python3
"""Enforce the corpus contract in SCHEMA.md.

A documented convention that nothing checks is a convention that drifts: the
umbrella README of bug-corpora still promises a per-case run.sh that no corpus
has, and points at an index directory that does not exist. This checker exists
so that cannot happen here.

It reports EVERY violation rather than stopping at the first, and exits
non-zero if there is any. "No cases found" is an error, not a pass.
"""

import json
from pathlib import Path
import re
import sys

CORPUS = Path(__file__).resolve().parent.parent

REQUIRED = [
    "case",
    "upstream_fix",
    "title",
    "consumer",
    "live_in_pin",
    "live_proof",
    "object",
    "lifetime_ender",
    "shape",
    "allocator_layer",
    "fidelity",
    "arms",
    "status",
]
OPTIONAL = {
    "distinguishing",
    "sibling_issue",
    "size_class",
    "size_note",
    "layer_note",
    "note",
}
# Arm -> the keys that arm's oracle must carry. An arm may instead declare
# itself unwritten with {"status": "not written"}.
ARM_ORACLES = {
    "spatial": {"oracle"},
    "sublet": {"oracle", "cause"},
    "poisoncap-spatial": {"oracle", "mode"},
    "poisoncap-protected": {"oracle", "mode", "signal", "si_code"},
}
REQUIRED_ARMS = {"spatial", "sublet", "poisoncap-spatial", "poisoncap-protected"}


WORDS = {"eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def check_shape_table(count, problems):
    """The README's shape table carries the corpus's headline ratio.

    Two things are checked, because both have been wrong: that the table
    partitions the cases -- every case in exactly one shape, none missing, none
    twice -- and that the count the prose claims equals the number of rows. The
    prose said "nine shapes" over a ten-row table until this checker was
    written, in the README twice and in defects.c once.
    """
    text = (CORPUS / "README.md").read_text()
    block = re.search(r"\| shape \| cases \|.*?\n\n", text, re.S)
    if not block:
        problems.append("README.md: the shape table is missing or reshaped")
        return
    rows = [l for l in block.group(0).splitlines()[2:] if l.startswith("|")]
    seen = {}
    for row in rows:
        cells = row.split("|")
        label = cells[1].strip()
        for number in (int(n) for n in re.findall(r"\d+", cells[2])):
            seen.setdefault(number, []).append(label)
    for number, labels in sorted(seen.items()):
        if len(labels) > 1:
            problems.append(
                f"README shape table: case {number} appears in "
                f"{len(labels)} shapes: {', '.join(labels)}"
            )
    missing = set(range(count)) - set(seen)
    if missing:
        problems.append(
            f"README shape table: cases {sorted(missing)} appear in " f"no shape"
        )
    extra = set(seen) - set(range(count))
    if extra:
        problems.append(
            f"README shape table: cases {sorted(extra)} are not in " f"the corpus"
        )
    for claim in re.findall(r"[Tt]wenty reports, (\w+) shapes", text) + re.findall(
        r"The (\w+) shapes, and what each", text
    ):
        if WORDS.get(claim.lower()) != len(rows):
            problems.append(
                f"README claims {claim!r} shapes but the table has " f"{len(rows)} rows"
            )


def main():
    problems = []
    files = sorted(CORPUS.glob("[0-9][0-9]_*/case.json"))
    if not files:
        print("ERROR: no case.json found under", CORPUS, file=sys.stderr)
        return 2
    check_shape_table(len(files), problems)

    numbers = {}
    for path in files:
        where = path.parent.name
        try:
            case = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{where}: case.json is not valid JSON: {exc}")
            continue

        for key in REQUIRED:
            if key not in case:
                problems.append(f"{where}: missing required field {key!r}")
        for key in case:
            if key not in REQUIRED and key not in OPTIONAL:
                problems.append(
                    f"{where}: unknown field {key!r} -- add it to "
                    f"SCHEMA.md and to this checker, or drop it"
                )

        # The directory name is metadata and is checked as such: NN_<fix>_<slug>,
        # where NN is the case number a run selects. Sorting the tree then puts
        # the corpus in run order, and --cases N is findable by eye.
        fix, number = case.get("upstream_fix"), case.get("case")
        if isinstance(number, int) and fix:
            prefix = f"{number:02d}_{fix}_"
            if not where.startswith(prefix):
                problems.append(f"{where}: directory should start with {prefix!r}")

        if case.get("live_in_pin") and not str(case.get("live_proof", "")).strip():
            problems.append(f"{where}: live_in_pin is set with no live_proof")

        if not str(case.get("shape", "")).strip():
            problems.append(f"{where}: shape is empty")

        arms = case.get("arms", {})
        if isinstance(arms, dict):
            for name in REQUIRED_ARMS - set(arms):
                problems.append(f"{where}: arm {name!r} is not declared")
            for name, arm in arms.items():
                if not isinstance(arm, dict):
                    problems.append(f"{where}: arm {name!r} is not an object")
                    continue
                if arm.get("status") == "not written":
                    continue
                if name not in ARM_ORACLES:
                    problems.append(f"{where}: arm {name!r} is not in SCHEMA.md")
                    continue
                for key in ARM_ORACLES[name] - set(arm):
                    problems.append(f"{where}: arm {name!r} lacks {key!r}")

        number = case.get("case")
        if isinstance(number, int):
            numbers.setdefault(number, []).append(where)

    duplicates = {n: w for n, w in numbers.items() if len(w) > 1}
    for number, where in sorted(duplicates.items()):
        problems.append(f"case number {number} claimed by {', '.join(where)}")
    expected = set(range(len(files)))
    missing = expected - set(numbers)
    extra = set(numbers) - expected
    if missing:
        problems.append(
            f"case numbers are not dense: {sorted(missing)} missing "
            f"from 0..{len(files) - 1}"
        )
    if extra:
        problems.append(f"case numbers outside 0..{len(files) - 1}: {sorted(extra)}")

    # A case directory without a PROVENANCE.md is a claim without its source,
    # and without a case.c it is a claim with no sequence. The case.c must
    # declare the number its directory carries, so a copied file that kept the
    # wrong PYC_CASE cannot pass.
    for path in files:
        where = path.parent
        if not (where / "PROVENANCE.md").is_file():
            problems.append(f"{where.name}: no PROVENANCE.md")
        source = where / "case.c"
        if not source.is_file():
            problems.append(f"{where.name}: no case.c")
            continue
        number = json.loads(path.read_text()).get("case")
        declared = re.search(r"PYC_CASE\((\d+)\)", source.read_text())
        if not declared:
            problems.append(f"{where.name}: case.c declares no PYC_CASE")
        elif int(declared.group(1)) != number:
            problems.append(
                f"{where.name}: case.c says PYC_CASE({declared.group(1)}), "
                f"case.json says {number}"
            )

    for problem in problems:
        print("FAIL " + problem)
    print(f"\n{len(files)} cases checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
