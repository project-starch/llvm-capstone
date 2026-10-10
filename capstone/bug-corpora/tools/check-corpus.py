#!/usr/bin/env python3
"""Enforce the corpus contract in ../SCHEMA.md, for every declared corpus.

Until 2026-09-28 this checker existed three times -- once in the pymalloc
corpus, and twice as a copy differing only in a macro name -- so five of the
eight corpora had no checker at all, and the counts in the umbrella README and
in `docs/ref/paper-bug-inventory.md` had drifted from the tree without anything
saying so. This is one checker for all of them, driven by each corpus's own
`corpus.json` declaration, which is also what `build-index.py` reads.

    check-corpus.py                 check every corpus, exit 1 on any violation
    check-corpus.py --self-test     prove the checker can FAIL, then check
    check-corpus.py <corpus> ...    check only the named corpora

The self-test exists because a gate that has never rejected anything is not a
passing gate, it is an unproven one.

Every path a declaration names is relative to the repository root, so a
declaration reads the same from anywhere and the index can quote it verbatim.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

TOOLS = Path(__file__).resolve().parent
CORPORA = TOOLS.parent
REPO = CORPORA.parents[1]
SEARCH = ["capstone/bug-corpora", "xlang"]

DECL_REQUIRED = {"program", "boundary", "title", "cases", "case_schema", "status",
                 "live_in_pin_recorded"}
DECL_OPTIONAL = {"upstream", "expect_live_in_pin", "live_in_pin_note", "case_macro",
                 "case_glob", "case_exclude", "case_number_base", "case_table",
                 "case_dir_column", "case_doc", "required_arms", "arm_keys",
                 "expect_provenance", "runners", "checker", "inventory", "evidence",
                 "advisories", "related", "note", "shape_table", "shape_prose"}
SCHEMAS = {"case-json", "sqlite-row", "script-trigger", "xlang-row"}
STATUSES = {"planned", "built", "measured", "triaged"}
PATH_FIELDS = ("runners", "checker", "inventory", "evidence", "related")

# The required fields differ by case schema, because the two schemas record
# different things: a case-json case is a reduction with a shape and an
# allocator layer, while a sqlite-row case is a binding defect carrying the
# provenance ledger's own columns. Neither set is a subset of the other, so
# stating one and calling the rest optional would let a real omission pass.
CASE_REQUIRED = {
    "case-json": ["case", "upstream_fix", "title", "consumer", "object", "lifetime_ender",
                  "shape", "allocator_layer", "fidelity", "arms", "status"],
    "sqlite-row": ["case", "upstream_fix", "title", "consumer", "class", "verdict",
                   "primitive", "arms", "status"],
    # A defect whose trigger is an interpreter script rather than a C
    # reduction -- SQL against a running server, a .rb against mruby, a .t
    # against perl. There is no case.c and no allocator to name: the case IS
    # its script, and what distinguishes one from another is the defect class
    # and how faithful the script is to the upstream report. Each case names
    # its own file in `trigger`, so one corpus may mix .t and .pl.
    "script-trigger": ["case", "upstream_fix", "title", "consumer", "class",
                       "trigger", "fidelity", "arms", "status"],
}
CASE_OPTIONAL = {"live_in_pin", "live_proof", "live_note", "distinguishing",
                 "sibling_issue", "size_class", "size_note", "layer_note", "note",
                 "advisory", "shape", "object", "capstone_column", "comparison",
                 "taxonomy_class", "allocator_layer", "lifetime_ender",
                 "allocator_consumed", "channel", "harness_limit",
                 "oracle_is_recording", "nested", "nested_why",
                 "citation_constraint"}
# `citation_constraint` records that a case's upstream commit cannot be quoted
# freely -- in practice that its SUBJECT names a person, so the fix may be cited
# by HASH AND PATH ONLY. This tree's naming rule is absolute and applies to
# committed files, so the constraint belongs in the case rather than in someone's
# memory: the first instance (memcached d5d9ff0) sat undispositioned in a triage
# doc precisely because the reason it was awkward was not recorded as a field.
# `nested` is a BOOLEAN, and it exists because the inventory's headline nesting
# share was being computed from `allocator_layer` PROSE. On 2026-10-06 a script
# doing that put 11 of 25 spatial cases into an "unclassified" bucket and
# reported 44%; the correct figure is 60%, so taking that number would have
# published one wrong by 16 points -- the unanswered probe was reading as "not
# nested". `nested_why` carries the one-sentence reason. The axis is WHO
# ALLOCATED THE OBJECT: an inner allocator's sub-allocation is nested, a direct
# malloc is not. It is NOT which bound the access crosses; collapsing those two
# produced a retraction on 2026-10-05.
# `harness_limit` says this case cannot execute on the arms at all (it needs a
# postmaster, a transaction block, something the harness does not provide), and
# `oracle_is_recording` says its directive was written from its own run and so
# necessarily fires. Both exist so a runner can act on them: the same facts are
# in `fidelity` as prose, and a scorer matching prose matches wording, which
# drifts. A case carrying either must be excluded from that arm's denominator
# rather than given a verdict.
# `trigger` is deliberately NOT here. A field meaningful in exactly one schema
# belongs in that schema's required list; putting it in the global optional set
# would stop the unknown-field check from catching a stray `trigger` on a
# case-json or sqlite-row case, which is the typo that check exists to find.
# Arm -> the keys that arm's oracle must carry. An arm may instead declare
# itself unwritten with {"status": "not written"}.
ARM_ORACLES = {
    "spatial": {"oracle"},
    "sublet": {"oracle"},
    "poisoncap-spatial": {"oracle", "mode"},
    "poisoncap-protected": {"oracle", "mode", "signal", "si_code"},
    "cheribsd-revocation": {"oracle"},
    # The three system-allocator arms of docs/ref/runtime-terms-glossary.md
    # section 6. They are one image each of the same source, differing only in
    # the heap it links: the first-fit heap without per-object bounds, the same
    # heap as applications get it today, and the Sublet heap. A port that also
    # sublets its own allocator names that arm for the port (sublet-gc,
    # sublet-svheads) -- the name says which nested allocator was sublet, because
    # a program has more than one and only the named one is protected.
    "sysalloc-none": {"oracle"},
    "sysalloc-bounds": {"oracle"},
    "sysalloc-sublet": {"oracle"},
    "sublet-gc": {"oracle"},
    "sublet-svheads": {"oracle"},
    # Bounds narrower than the allocation, the remedies for a crossing that stays inside one:
    # struct-field bounds from each compiler, and the carved corpus's narrowing at the carve.
    "capstone-subobject": {"oracle"},
    "cheribsd-subobject": {"oracle"},
    "capstone-carve-bounds": {"oracle"},
    "cheribsd-carve-bounds": {"oracle"},
    # tshark's chunk allocator over the Sublet heap (the wireshark plain-heap corpus).
    "sublet-chunks": {"oracle"},
    # The three columns asked for per bug (docs/ref/spatial-vs-temporal-three-programs.md section 0):
    # Sublet ONLY as the system allocator under the program's stock nested allocator; the whole
    # program's Sublet configuration (heap + its nested allocator's port) on a plain case; and the
    # Sublet port of a carving routine (regions split from a linear block, each its own alias).
    "sublet-malloc": {"oracle"},
    "sublet-full": {"oracle"},
    "sublet-carve": {"oracle"},
    "sublet-pymalloc": {"oracle"},
    "native-detect": set(),
    "native-fix-differential": set(),
    "backing": set(),
    "host-asan": {"oracle"},
    "capstone-domain": {"oracle"},
}
WORDS = {"eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "twenty": 20}


def declarations(only=None):
    found = []
    for base in SEARCH:
        for path in sorted((REPO / base).rglob("corpus.json")):
            name = str(path.parent.relative_to(REPO))
            if only and not any(o in name for o in only):
                continue
            found.append(path)
    return found


def read_tsv(path):
    with path.open() as handle:
        rows = [r for r in csv.reader(handle, delimiter="\t")
                if r and not r[0].startswith("#")]
    header, body = rows[0], rows[1:]
    return [dict(zip(header, r)) for r in body]


def case_dirs(corpus, decl, problems):
    """The case directories this corpus declares, in run order."""
    schema = decl.get("case_schema")
    if schema == "xlang-row":
        table = decl.get("case_table")
        column = decl.get("case_dir_column")
        if not table and not column:
            # A corpus small enough to have no row table names its cases by glob.
            if "case_glob" not in decl:
                problems.append("xlang-row corpus declares neither case_table nor case_glob")
                return []
            doc = decl.get("case_doc", "README.md")
            dirs = [d for d in sorted(corpus.glob(decl["case_glob"])) if d.is_dir()]
            for target in dirs:
                if not (target / doc).is_file():
                    problems.append(f"{target.name}: no {doc}")
            return dirs
        if not table or not column:
            problems.append("case_table and case_dir_column go together")
            return []
        tsv = REPO / table
        if not tsv.is_file():
            problems.append(f"case_table {table} does not exist")
            return []
        rows = read_tsv(tsv)
        if len(rows) != decl["cases"]:
            problems.append(f"case_table has {len(rows)} rows, cases says {decl['cases']}")
        dirs = []
        for row in rows:
            name = row.get(column, "")
            target = corpus / name
            if not target.is_dir():
                problems.append(f"case_table names {name!r}, which is not a directory")
                continue
            if not (target / decl.get("case_doc", "README.md")).is_file():
                problems.append(f"{name}: no {decl.get('case_doc', 'README.md')}")
            dirs.append(target)
        return dirs
    default = "row[0-9]*" if schema == "sqlite-row" else "[0-9][0-9]_*"
    exclude = set(decl.get("case_exclude", []))
    return [d for d in sorted(corpus.glob(decl.get("case_glob", default)))
            if d.is_dir() and d.name not in exclude]


def check_shape_table(corpus, decl, count, problems):
    """The README's shape table carries the corpus's headline ratio.

    Two things are checked, because both have been wrong: that the table
    partitions the cases -- every case in exactly one shape, none missing, none
    twice -- and that any count the prose claims equals the number of rows.
    """
    readme = corpus / "README.md"
    if not readme.is_file():
        problems.append("README.md is missing, so its shape table cannot be checked")
        return
    text = readme.read_text()
    block = re.search(r"\| shape \| cases \|.*?\n\n", text, re.S)
    if not block:
        problems.append("README.md: the shape table is missing or reshaped")
        return
    rows = [l for l in block.group(0).splitlines()[2:] if l.startswith("|")]
    seen = {}
    for row in rows:
        cells = row.split("|")
        for number in (int(n) for n in re.findall(r"\d+", cells[2])):
            seen.setdefault(number, []).append(cells[1].strip())
    for number, labels in sorted(seen.items()):
        if len(labels) > 1:
            problems.append(f"README shape table: case {number} appears in "
                            f"{len(labels)} shapes: {', '.join(labels)}")
    base = decl.get("case_number_base", 0)
    expected = set(range(base, base + count))
    if expected - set(seen):
        problems.append(f"README shape table: cases {sorted(expected - set(seen))} "
                        f"appear in no shape")
    if set(seen) - expected:
        problems.append(f"README shape table: cases {sorted(set(seen) - expected)} "
                        f"are not in the corpus")
    for pattern in decl.get("shape_prose", []):
        for claim in re.findall(pattern, text):
            if WORDS.get(str(claim).lower()) != len(rows):
                problems.append(f"README claims {claim!r} shapes but the table has "
                                f"{len(rows)} rows")


def check_cases(corpus, decl, dirs, problems):
    """The per-case rules, for the two schemas that carry a case.json."""
    base = decl.get("case_number_base", 0)
    recorded = decl["live_in_pin_recorded"]
    required_arms = set(decl.get("required_arms", []))
    macro = decl.get("case_macro")
    # case-json fixes the name; script-trigger takes it from each case, so a
    # corpus may mix .t and .pl, or .sql and anything else, without splitting.
    source_name = "case.c" if decl["case_schema"] == "case-json" else None
    live = {"true": 0, "false": 0, "not_asserted": 0}
    arm_keys = dict(ARM_ORACLES)
    for name, keys in decl.get("arm_keys", {}).items():
        arm_keys[name] = set(keys)
    provenance = 0
    numbers = {}

    for path in dirs:
        where = path.name
        manifest = path / "case.json"
        if not manifest.is_file():
            problems.append(f"{where}: no case.json")
            continue
        try:
            case = json.loads(manifest.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{where}: case.json is not valid JSON: {exc}")
            continue

        required = CASE_REQUIRED[decl["case_schema"]]
        for key in required:
            if key not in case:
                problems.append(f"{where}: missing required field {key!r}")
        for key in case:
            if key not in required and key not in CASE_OPTIONAL:
                problems.append(f"{where}: unknown field {key!r} -- add it to "
                                f"SCHEMA.md and to this checker, or drop it")

        number, fix = case.get("case"), case.get("upstream_fix")
        if isinstance(number, int) and fix:
            prefix = f"row{number}_" if decl["case_schema"] == "sqlite-row" \
                else f"{number:02d}_{fix}_"   # case-json and script-trigger share this
            if not where.startswith(prefix):
                problems.append(f"{where}: directory should start with {prefix!r}")
        if isinstance(number, int):
            numbers.setdefault(number, []).append(where)

        if "shape" in required and not str(case.get("shape", "")).strip():
            problems.append(f"{where}: shape is empty")

        # live_in_pin is the field the index reports per version, so its
        # absence must be a corpus-wide decision rather than a per-case
        # oversight: a corpus either records it for every case or for none.
        if recorded:
            value = case.get("live_in_pin", "MISSING")
            if value == "MISSING":
                problems.append(f"{where}: live_in_pin is not recorded, but the "
                                f"corpus declares live_in_pin_recorded")
            elif value is True or value is False:
                live["true" if value else "false"] += 1
                if not str(case.get("live_proof", "")).strip():
                    problems.append(f"{where}: live_in_pin is stated with no live_proof")
            elif value is None:
                # Recorded and deliberately not asserted: the corpus pins an
                # allocator rather than a shipped application, and says so.
                live["not_asserted"] += 1
                if not (str(case.get("live_proof", "")).strip()
                        or str(case.get("live_note", "")).strip()):
                    problems.append(f"{where}: live_in_pin is null with neither "
                                    f"live_proof nor live_note saying why")
            else:
                problems.append(f"{where}: live_in_pin is {value!r}, not true, false or null")
        elif "live_in_pin" in case:
            problems.append(f"{where}: carries live_in_pin, but the corpus declares "
                            f"live_in_pin_recorded false -- flip the declaration")

        arms = case.get("arms", {})
        if not isinstance(arms, dict):
            problems.append(f"{where}: arms is not an object")
        else:
            for name in sorted(required_arms - set(arms)):
                problems.append(f"{where}: arm {name!r} is not declared")
            for name, arm in arms.items():
                if not isinstance(arm, dict):
                    problems.append(f"{where}: arm {name!r} is not an object")
                    continue
                # Name before status. An unwritten arm still has to be a real
                # arm: short-circuiting on "not written" first let a misspelled
                # name through silently, which is the one case where nobody
                # would ever notice, because an unwritten arm produces no
                # output to look wrong.
                if name not in arm_keys:
                    problems.append(f"{where}: arm {name!r} is not in SCHEMA.md")
                    continue
                if arm.get("status") == "not written":
                    continue
                for key in sorted(arm_keys[name] - set(arm)):
                    problems.append(f"{where}: arm {name!r} lacks {key!r}")

        if (path / "PROVENANCE.md").is_file():
            provenance += 1
        elif "expect_provenance" not in decl:
            problems.append(f"{where}: no PROVENANCE.md")
        if decl["case_schema"] == "script-trigger":
            named = str(case.get("trigger", "")).strip()
            # Shape before existence. A name like "../x.t" is both malformed
            # and absent, and reporting the absence sends the reader looking
            # for a missing file instead of at the field they mistyped. Once
            # the order is right, elif is correct: a malformed name has no
            # meaningful existence to report, so the second message is noise.
            if named and ("/" in named or named.startswith(".")):
                problems.append(f"{where}: trigger {named!r} must be a plain filename "
                                f"inside the case directory")
            elif named and not (path / named).is_file():
                problems.append(f"{where}: trigger names {named!r}, which is not here")
        if source_name:
            source = path / source_name
            if not source.is_file():
                problems.append(f"{where}: no {source_name}")
            elif macro:
                declared = re.search(rf"{macro}_CASE\((\d+)\)", source.read_text())
                if not declared:
                    problems.append(f"{where}: {source_name} declares no {macro}_CASE")
                elif int(declared.group(1)) != number:
                    problems.append(f"{where}: {source_name} says {macro}_CASE"
                                    f"({declared.group(1)}), case.json says {number}")

    for number, where in sorted((n, w) for n, w in numbers.items() if len(w) > 1):
        problems.append(f"case number {number} claimed by {', '.join(where)}")
    expected = set(range(base, base + len(dirs)))
    if expected - set(numbers):
        problems.append(f"case numbers are not dense: {sorted(expected - set(numbers))} "
                        f"missing from {base}..{base + len(dirs) - 1}")
    if set(numbers) - expected:
        problems.append(f"case numbers outside {base}..{base + len(dirs) - 1}: "
                        f"{sorted(set(numbers) - expected)}")

    expect_provenance = decl.get("expect_provenance")
    if expect_provenance is not None and expect_provenance != provenance:
        problems.append(f"expect_provenance says {expect_provenance}, "
                        f"{provenance} of {len(dirs)} cases have a PROVENANCE.md")

    if recorded:
        expect = decl.get("expect_live_in_pin")
        if expect is None:
            problems.append("live_in_pin_recorded is set but expect_live_in_pin is absent")
        else:
            counted = {k: v for k, v in live.items() if v}
            declared = {k: v for k, v in expect.items() if v}
            if counted != declared:
                problems.append(f"expect_live_in_pin says {declared}, the cases say "
                                f"{counted}")


def check_one(manifest):
    """Check the corpus that `manifest` declares. Returns a list of problems."""
    corpus = manifest.parent
    where = corpus.relative_to(REPO)
    problems = []
    try:
        decl = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        return [f"{where}: corpus.json is not valid JSON: {exc}"]

    for key in sorted(DECL_REQUIRED - set(decl)):
        problems.append(f"missing required declaration {key!r}")
    for key in sorted(set(decl) - DECL_REQUIRED - DECL_OPTIONAL):
        problems.append(f"unknown declaration {key!r} -- add it to SCHEMA.md and to "
                        f"this checker, or drop it")
    if problems:
        return [f"{where}: {p}" for p in problems]

    if decl["case_schema"] not in SCHEMAS:
        problems.append(f"case_schema {decl['case_schema']!r} is not one of "
                        f"{sorted(SCHEMAS)}")
    if decl["status"] not in STATUSES:
        problems.append(f"status {decl['status']!r} is not one of {sorted(STATUSES)}")
    if not isinstance(decl["cases"], int) or decl["cases"] < 0:
        problems.append("cases is not a non-negative integer")
    if not isinstance(decl["live_in_pin_recorded"], bool):
        problems.append("live_in_pin_recorded is not a boolean")
    if not decl["live_in_pin_recorded"] and not decl.get("live_in_pin_note"):
        problems.append("live_in_pin_recorded is false with no live_in_pin_note saying why")
    if decl["cases"] == 0 and decl["status"] not in {"planned", "triaged"}:
        problems.append(f"no cases, but status is {decl['status']!r}")
    if problems:
        return [f"{where}: {p}" for p in problems]

    for field in PATH_FIELDS:
        value = decl.get(field)
        for item in ([value] if isinstance(value, str) else value or []):
            if not (REPO / item).exists():
                problems.append(f"{field} names {item}, which does not exist")

    dirs = case_dirs(corpus, decl, problems)
    if len(dirs) != decl["cases"]:
        problems.append(f"cases says {decl['cases']}, the tree has {len(dirs)}")
    if decl["cases"] and decl["case_schema"] != "xlang-row":
        check_cases(corpus, decl, dirs, problems)
    if decl.get("shape_table"):
        check_shape_table(corpus, decl, len(dirs), problems)

    return [f"{where}: {p}" for p in problems]


def self_test():
    """Prove the checker rejects corruption. Without this its silence is unproven.

    Returns (accepted, attempted). The caller reports the attempted count rather
    than a written-out number: a hard-coded "five" keeps printing five when a
    sixth corruption is added, and keeps printing five when one stops running.
    The number has to come from the list that ran, or it is decoration.
    """
    victim = CORPORA / "cpython/pymalloc-repros"
    accepted = []
    corruptions = (
        ("a missing required field", lambda c: c.pop("consumer")),
        ("a case number the directory does not carry", lambda c: c.update(case=42)),
        ("an arm with neither oracle nor status", lambda c: c["arms"].update(spatial={})),
        ("live_in_pin without its proof", lambda c: c.update(live_proof="")),
        # An unwritten arm produces no output, so a misspelled one is the single
        # corruption nobody would ever spot by reading a run. It has to be the
        # gate that spots it.
        ("a misspelled arm hidden behind 'not written'",
         lambda c: c["arms"].update(spatail={"status": "not written"})),
    )
    for label, mutate in corruptions:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "corpus"
            shutil.copytree(victim, copy, ignore=shutil.ignore_patterns("results", ".git"))
            target = sorted(copy.glob("[0-9][0-9]_*"))[1]
            case = json.loads((target / "case.json").read_text())
            mutate(case)
            (target / "case.json").write_text(json.dumps(case, indent=2))
            problems = []
            decl = json.loads((copy / "corpus.json").read_text())
            check_cases(copy, decl, sorted(copy.glob("[0-9][0-9]_*")), problems)
            if not problems:
                accepted.append(label)
    return accepted, len(corruptions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="*", help="substring of a corpus path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        accepted, attempted = self_test()
        if accepted:
            print("CHECKER BROKEN: it accepted " + "; ".join(accepted), file=sys.stderr)
            return 2
        if not attempted:
            print("CHECKER BROKEN: the self-test ran no corruptions at all",
                  file=sys.stderr)
            return 2
        print(f"self-test: the checker rejected all {attempted} corruptions")

    manifests = declarations(args.corpus)
    if not manifests:
        print("ERROR: no corpus.json found", file=sys.stderr)
        return 2

    problems, cases = [], 0
    for manifest in manifests:
        problems += check_one(manifest)
        try:
            cases += json.loads(manifest.read_text()).get("cases", 0)
        except json.JSONDecodeError:
            pass
    for line in problems:
        print("FAIL " + line, file=sys.stderr)
    print(f"check-corpus: {'BLOCKED' if problems else 'CLEAN'} -- "
          f"{len(manifests)} corpora, {cases} declared cases, "
          f"{len(problems)} problem{'' if len(problems) == 1 else 's'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
