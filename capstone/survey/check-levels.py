#!/usr/bin/env python3
"""Check the survey level map against this repository and against the manuscript.

The map claims one level per ported allocator. Two things can drift: a level can
name a port that is not here, and the manuscript's allocator list can gain or
lose an entry without the map following. Both are checked, the second only when
the manuscript's path is given, so the check still runs in a clone that has no
copy of the paper beside it.

    ./check-levels.py
    ./check-levels.py --manuscript ../../../nested-allocators-paper
    ./check-levels.py --self-test
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MAP = HERE / "levels.json"


def load(path=MAP):
    return json.loads(Path(path).read_text())


def manuscript_allocators(root):
    """The (program, allocator) pairs the manuscript's own list derives from."""
    source = (Path(root) / "macros/evaluation-plan.tex").read_text()
    block = source.split(r"\newcommand{\evalAllocators}[1]{%", 1)[1]
    block = block.split("\n}", 1)[0]
    pairs = re.findall(r"#1\{([^}]+)\}\{([^}]+)\}", block)
    return {(program, allocator.replace(r"\_", "_")) for program, allocator in pairs}


def check(doc, repo=REPO, manuscript=None):
    errors, notes = [], []

    if doc.get("schema") != 1:
        errors.append("unknown schema version")

    levels = doc.get("levels", [])
    if not levels:
        errors.append("no levels")

    ids = [level["id"] for level in levels]
    names = [level["level"] for level in levels]
    for label, values in (("id", ids), ("level name", names)):
        duplicates = {v for v in values if values.count(v) > 1}
        for value in sorted(duplicates):
            errors.append(f"duplicate {label}: {value}")

    workloads = doc.get("workloads", {})
    known_levels = set(names) | {"libc"}
    for level in levels:
        where = level["id"]
        if level["workload"] not in workloads:
            errors.append(f"{where}: undeclared workload {level['workload']}")
        if level["below"] not in known_levels:
            errors.append(f"{where}: level 0 names no known level: {level['below']}")
        if level["below"] == level["level"]:
            errors.append(f"{where}: a level cannot sit beneath itself")
        if not level["upstream"].get("sources"):
            errors.append(f"{where}: no upstream source file named")
        if level.get("confirmed") not in {"port-patch", "port-document", "to-confirm", "measured"}:
            errors.append(f"{where}: seam confidence must be stated")

        recorded = level.get("recorded")
        if recorded and not (HERE / recorded / "manifest.json").is_file():
            errors.append(f"{where}: recorded bundle has no manifest: {recorded}")
        if level.get("confirmed") == "measured" and not recorded:
            errors.append(f"{where}: a measured seam must name its bundle")

        port = level["port"]
        present = (repo / port["path"]).exists()
        if port.get("status") == "merged":
            if not present:
                errors.append(f"{where}: port claimed merged but absent: {port['path']}")
            evidence = level["seam"].get("evidence")
            if evidence and not (repo / evidence).exists():
                errors.append(f"{where}: seam evidence missing: {evidence}")
        elif port.get("status") == "pending":
            if not port.get("lands_with"):
                errors.append(f"{where}: a pending port must name what lands it")
            if present:
                notes.append(f"{where}: port marked pending is already present, "
                             f"mark it merged: {port['path']}")
            else:
                notes.append(f"{where}: waiting on {port['lands_with']}")
        else:
            errors.append(f"{where}: port status must be merged or pending")

    # The nesting the map exists to expose: a level whose level 0 is another
    # custom allocator rather than libc. Losing these silently would turn the
    # survey back into one row per program.
    nested = [level["id"] for level in levels if level["below"] != "libc"]
    if not nested:
        errors.append("no level sits on another custom level, which the map is for")

    if manuscript:
        wanted = manuscript_allocators(manuscript)
        have = {(level["program"], level["allocator"]) for level in levels}
        for pair in sorted(wanted - have):
            errors.append(f"manuscript allocator has no survey level: {pair[0]} {pair[1]}")
        for pair in sorted(have - wanted):
            errors.append(f"survey level is not a manuscript allocator: {pair[0]} {pair[1]}")

    return errors, notes


def self_test():
    """Every check must fail on a map that breaks it, or it is not a check."""
    base = load()
    cases = []

    doc = json.loads(json.dumps(base))
    doc["levels"][1]["id"] = doc["levels"][0]["id"]
    cases.append(("duplicate id", doc))

    doc = json.loads(json.dumps(base))
    doc["levels"][0]["workload"] = "no-such-workload"
    cases.append(("undeclared workload", doc))

    doc = json.loads(json.dumps(base))
    doc["levels"][0]["below"] = "no-such-level"
    cases.append(("unknown level 0", doc))

    doc = json.loads(json.dumps(base))
    for level in doc["levels"]:
        level["below"] = "libc"
    cases.append(("no nesting left", doc))

    doc = json.loads(json.dumps(base))
    doc["levels"][0]["port"] = {"path": "capstone/ports/sqlite", "status": "pending"}
    cases.append(("pending without a landing PR", doc))

    doc = json.loads(json.dumps(base))
    doc["levels"][0]["seam"]["evidence"] = "capstone/ports/sqlite/no-such-file.md"
    cases.append(("missing seam evidence", doc))

    failures = []
    for label, broken in cases:
        errors, _ = check(broken)
        if not errors:
            failures.append(f"a broken map passed: {label}")
    for line in failures:
        print(line)
    print(f"self-test: {len(cases) - len(failures)} of {len(cases)} checks fire")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", help="path to the nested-allocators-paper checkout")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    doc = load()
    errors, notes = check(doc, manuscript=args.manuscript)
    for note in notes:
        print(f"note: {note}")
    for error in errors:
        print(f"error: {error}")
    if errors:
        return 1

    levels = doc["levels"]
    programs = {level["program"] for level in levels}
    pending = [l for l in levels if l["port"].get("status") == "pending"]
    nested = [l for l in levels if l["below"] != "libc"]
    print(f"{len(levels)} levels over {len(programs)} programs, "
          f"{len(nested)} of them on another custom level")
    print(f"{len(levels) - len(pending)} levels have their port here, "
          f"{len(pending)} wait on a pull request")
    done = [l for l in levels if l.get("recorded")]
    print(f"{len(done)} of {len(levels)} levels are recorded"
          + (": " + ", ".join(l["id"] for l in done) if done else ""))
    if args.manuscript:
        print("every manuscript allocator has exactly one survey level")
    return 0


if __name__ == "__main__":
    sys.exit(main())
