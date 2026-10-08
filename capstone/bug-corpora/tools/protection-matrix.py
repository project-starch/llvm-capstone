#!/usr/bin/env python3
"""One verdict per bug per protection arm, over every corpus in this tree.

WHY THIS EXISTS. The tree carries sixteen mechanism-level arm names and
seventeen result-table shapes, every one of them earned by the vehicle that
produced it. None of that lets a reader answer the question the corpora are
for: *for this bug, which defence reports?* This tool answers it for all of
them at once, in one table, with three columns -- the arms `arms.json`
defines -- and every cell carrying the bundle it was read from.

HOW A CELL IS SOURCED. Nothing here is typed by hand. Each corpus declares, in
its `corpus.json`, a `protection` block naming for each arm the results bundle
of record, the reader that understands that bundle's shape, and which of the
bundle's own arm names feeds the cell. This file holds one reader per legacy
shape; a reader returns `caught`/`missed` and the bundle's own words for the
detail, so a cell can always be traced back to the run that produced it.

A cell that has no measurement says `not-run` and carries the reason, because
"cannot be measured here", "waiting on a port" and "nobody got to it" weigh
differently in a denominator and a bare blank hides which one it is.

WHAT IT WRITES. `PROTECTION.md`, the table, and `protection.json`, the same
data for a script. Both are generated: edit the corpora, not them.

    tools/protection-matrix.py              # regenerate both
    tools/protection-matrix.py --check      # fail if they are stale
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent
ARMS = ("cheribsd", "capstone-sysalloc", "capstone-sublet")

# ---------------------------------------------------------------------------
# Readers. One per result-table shape that exists in the tree.
#
# Every reader takes the bundle directory, the arm's spec from `corpus.json`,
# and returns {case key: (caught, verdict, detail)} where `caught` is True,
# False or None. The key is whatever the bundle uses to name a case; the spec's
# `key` field says which of the corpus's own identifiers that is, and the
# caller joins on it.
# ---------------------------------------------------------------------------
READERS = {}


def reader(name):
    def register(fn):
        READERS[name] = fn
        return fn
    return register


def rows(bundle, filename="matrix.tsv"):
    with (bundle / filename).open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def strip_suffix(case):
    """`00_x_y:m2:buggy` -> `00_x_y`. The virtual runner brackets one case as
    several rows -- a fixed arm, a defect arm, one per mode -- and names them by
    suffixing the case directory."""
    return case.split(":", 1)[0]


@reader("virtual")
def read_virtual(bundle, spec):
    """`case arm verdict cause attributed evidence`, written by
    tools/run-virtual-corpus.py. The canonical shape: one row per bracketed
    case, the arm column naming the plan's arm and `control` naming its
    control rows."""
    out = {}
    want = spec["arm"]
    modes = spec.get("mode")
    for row in rows(bundle):
        if row["arm"] != want:
            continue
        case = row["case"]
        if modes and f":{modes}:" not in case and not case.endswith(f":{modes}"):
            continue
        key = strip_suffix(case)
        verdict = row["verdict"]
        if verdict == "detected":
            cell = (True, "caught", f"cause {row['cause']}"
                    + ("" if row["attributed"] in ("--", "") else
                       f", at the case's own probe" if row["attributed"] == "yes"
                       else ", attribution not established"))
        elif verdict == "silent":
            cell = (False, "missed", "ran to the end, status 0")
        else:
            cell = (None, "not-run", f"the run did not measure it: {verdict} -- "
                    f"{row['evidence']}")
        # A case bracketed once per mode may appear twice; the first defect row wins
        # and a later one must agree, which the caller checks.
        out.setdefault(key, cell)
    return out


@reader("arm-verdict")
def read_arm_verdict(bundle, spec):
    """`case arm verdict evidence` -- the postgres roll-ups, already per-arm."""
    out = {}
    for row in rows(bundle):
        if row["arm"] != spec["arm"]:
            continue
        verdict = row["verdict"]
        caught = {"detected": True, "silent": False}.get(verdict)
        out[strip_suffix(row["case"])] = (
            caught, "caught" if caught else "missed" if caught is False else "not-run",
            row["evidence"][:160])
    return out


@reader("sqlite-axis")
def read_sqlite_axis(bundle, spec):
    """`case directory axis arm verdict oracle` -- the engine corpus's roll-up."""
    out = {}
    for row in rows(bundle):
        if row["arm"] != spec["arm"]:
            continue
        caught = {"detected": True, "not-detected": False}.get(row["verdict"])
        out[row["directory"]] = (
            caught, "caught" if caught else "missed" if caught is False else "not-run",
            row["oracle"][:160])
    return out


@reader("qemu-paired")
def read_qemu_paired(bundle, spec):
    """`case fix name shape mode expected passed cause pc ... completed`.

    The bare-metal paired-arm shape. `mode` names the arm; `expected` is what
    the case pre-registered and `passed` whether the run met it. A cell is a
    catch when the run FAULTED, which is `expected=fault and passed=True`, or
    `expected=complete and passed=False` with a cause -- the second is how a
    refuted prediction reads, and it is still a catch.
    """
    out = {}
    column = spec.get("mode_column", "mode")
    for row in rows(bundle):
        if row[column] != spec["arm"]:
            continue
        runs = spec.get("run")
        if runs and "run" in row and row["run"] not in (
                [runs] if isinstance(runs, str) else runs):
            continue
        faulted = bool(row.get("cause")) and row.get("cause") not in ("", "0")
        passed = row["passed"] == "True"
        expected_fault = row["expected"].startswith("fault")
        caught = faulted or (expected_fault and passed)
        detail = (f"cause {row['cause']} at {row['pc']}"
                  if faulted else f"{row['expected']}, passed={row['passed']}")
        if expected_fault and not passed and not faulted:
            caught, detail = None, f"pre-registered {row['expected']}, run did not meet it"
        cell = (caught, "caught" if caught else
                "missed" if caught is False else "not-run", detail)
        # Several runs of one build may score the same case: a retry exists
        # because the first did not reach it, so a catch is the measurement.
        if out.get(row["case"], (None,))[0] is not True:
            out[row["case"]] = cell
    return out


@reader("revocation-rows")
def read_revocation_rows(bundle, spec):
    """`case fix name revocation expected passed exit completed ...` -- the APR
    runners' CheriBSD shape, which runs each case with revocation on and off and
    puts the setting in a column. Only the `on` rows are this arm: `off` is that
    bundle's own control, and `control_*` is the run's positive control, which
    the bundle's README reads and this table does not.
    """
    out = {}
    for row in rows(bundle):
        if row["revocation"] != spec.get("revocation", "on"):
            continue
        completed = row.get("completed") == "1"
        exit_code = row.get("exit", "0")
        if completed and exit_code == "0":
            out[row["case"]] = (False, "missed", "ran to the end, status 0")
        elif exit_code not in ("", "0"):
            out[row["case"]] = (True, "caught", f"exit {exit_code}, {row['expected']}")
        else:
            out[row["case"]] = (None, "not-run",
                                f"neither completion nor a fault: exit {exit_code}")
    return out


@reader("mmgr-columns")
def read_mmgr_columns(bundle, spec):
    """`case fix name allocator spatial sublet cause probe_pc fault_pc` -- the
    arms are columns, each holding `PASS:complete` or `PASS:fault`."""
    out = {}
    for row in rows(bundle):
        value = row[spec["arm"]]
        caught = value.endswith("fault")
        out[row["case"]] = (caught, "caught" if caught else "missed",
                            f"{value}, cause {row['cause']} at {row['fault_pc']}"
                            if caught else value)
    return out


@reader("column-per-arm")
def read_column_per_arm(bundle, spec):
    """`case <arm> <arm> <arm>` -- Perl's tables, one column per arm, the cell
    holding the verdict in words."""
    out = {}
    for row in rows(bundle):
        value = row[spec["arm"]].strip()
        caught = value.startswith(("fault", "SIGPROT"))
        verdict = "caught" if caught else "missed"
        if "panic" in value or "oracle fail" in value:
            verdict = "wrong-answer"
        out[row["case"]] = (caught, verdict, value)
    return out


@reader("verdict-word")
def read_verdict_word(bundle, spec):
    """`case arm verdict ...` where the verdict is a word rather than
    detected/silent: mruby's runs say SIGPROT, completes, fault, exit1,
    watchdog, abort, timeout."""
    caught_words = {"SIGPROT", "fault"}
    missed_words = {"completes", "completed", "clean", "exit1"}
    out = {}
    for row in rows(bundle):
        if row["arm"] != spec["arm"]:
            continue
        verdict = row["verdict"]
        if verdict in caught_words:
            cell = (True, "caught", verdict + (f", cause {row['cause']}"
                                               if row.get("cause", "none") not in ("none", "", None) else ""))
        elif verdict in missed_words:
            cell = (False, "missed", verdict)
        else:
            cell = (None, "not-run", f"the run did not measure it: {verdict}")
        if row["case"] in out and out[row["case"]][0] != cell[0]:
            cell = (None, "not-run", f"the bundle has two disagreeing rows for this case")
        out[row["case"]] = cell
    return out


SUPERVISE_BLOCK = re.compile(r'^### (?P<heading>.*)$', re.M)


@reader("supervise-blocks")
def read_supervise_blocks(bundle, spec):
    """`result-lines.txt`, the CheriBSD runners' own record: `### <heading>`
    opens a block and the lines under it are the guest's.
    `SUPERVISE fault signal=34` is the catch; `VERDICT DEFECT-REPRODUCED` with
    `SUPERVISE exit status=0` is the measured silence -- the defect ran and the
    mechanism stayed quiet.

    Which case a block belongs to is read out of the block, not out of the
    heading, because the headings are prose in some bundles and tags in others:
    `case_from` is a regex whose first group is the case number, applied to the
    block's body. A block whose heading names the fixed arm is skipped, since a
    fixed arm has no defect to catch.
    """
    text = (bundle / "result-lines.txt").read_text()
    starts = [(m.start(), m.group("heading")) for m in SUPERVISE_BLOCK.finditer(text)]
    skip = re.compile(spec.get("fixed_heading", r'-fixed\b'))
    case_from = re.compile(spec.get("case_from",
                                    r'allocator-tests/[a-z0-9]+-(\d+)-buggy|'
                                    r'case=(\d+) arm=buggy'))
    out = {}
    for (start, heading), (end, _) in zip(starts, starts[1:] + [(len(text), None)]):
        if skip.search(heading):
            continue
        body = text[start:end]
        m = case_from.search(body)
        if not m:
            continue
        case = next(g for g in m.groups() if g)
        case = spec.get("remap", {}).get(case, case)
        fault = re.search(r'SUPERVISE fault signal=(\d+) code=(\d+) addr=(\S+)', body)
        if fault:
            out[case] = (True, "caught",
                         f"SIGPROT({fault.group(1)}) si_code {fault.group(2)} "
                         f"at {fault.group(3)}")
        elif re.search(spec.get("completed",
                                r'VERDICT DEFECT-REPRODUCED|\bstatus=0\b'), body):
            out[case] = (False, "missed", "ran to the end, status 0")
        else:
            out[case] = (None, "not-run", "the block records no verdict line")
    return out


@reader("case-lines")
def read_case_lines(bundle, spec):
    """`result-lines.txt` in the suite runners' shape: one line per case,
    `OK   case=N  mode=0 revocation=on <fix> exit=0 completed=1` for a silence
    and `FAIL case=N ... exit=162 FAULT signal=34 code=1 pc=...` for a catch."""
    out = {}
    for line in (bundle / "result-lines.txt").read_text().splitlines():
        m = re.match(r'(OK|FAIL)\s+case=(\d+)\s+(.*)$', line)
        if not m:
            continue
        case, rest = m.group(2), m.group(3)
        fault = re.search(r'FAULT signal=(\d+) code=(\d+) pc=(\S+)', rest)
        if fault:
            out[case] = (True, "caught",
                         f"SIGPROT({fault.group(1)}) si_code {fault.group(2)} at {fault.group(3)}")
        elif "exit=0" in rest:
            out[case] = (False, "missed", "ran to the end, status 0")
    return out


@reader("pass-lines")
def read_pass_lines(bundle, spec):
    """`result-lines.txt` where the record is the suite's own `PASS <nn>-<slug>`
    lines against an arm whose oracle is "completes". A PASS there is a measured
    silence; a FAIL on such an arm means the case did not complete, which the
    spec has to interpret, so this reader refuses to guess and says not-run."""
    out = {}
    for line in (bundle / "result-lines.txt").read_text().splitlines():
        m = re.match(r'(PASS|FAIL)\s+(\d+)-\S+-mode' + str(spec.get("mode", 0)) + r'\s*$', line)
        if not m:
            continue
        out[m.group(2)] = ((False, "missed", "the suite's completion oracle passed")
                           if m.group(1) == "PASS" else
                           (None, "not-run", "the completion oracle failed; "
                            "what the arm did needs the run's own reading"))
    return out


# ---------------------------------------------------------------------------
# Joining a reader's keys to the corpus's cases
# ---------------------------------------------------------------------------
def slug(key):
    """A case's identity without its position.

    `07_129371553c_fts3_destroy_oom` and `06_129371553c_fts3_destroy_oom` are the
    same case: the number says where it sits in the run order, and deleting a
    case ahead of it moves that number. An archived bundle keeps the number it
    was run under, so a cell joined on the number breaks the moment the corpus is
    renumbered -- which is how 17 cells silently became "not run" once. Joining
    on what follows the number does not break, because the fix id and the slug
    are the case.
    """
    return key.split("_", 1)[1] if key[:2].isdigit() and "_" in key else key


def numeric(key):
    """`00` and `0` are the same case: one bundle zero-pads its tags and another
    does not, and nothing is gained by making the declaration say which."""
    return str(int(key)) if key.isdigit() else key


def case_identifiers(case_dir, case):
    """Every name a bundle might use for this case."""
    return {
        "dir": case_dir.name,
        "slug": slug(case_dir.name),
        "number": str(case["case"]),
        "number0": f"{case['case']:02d}",
        "fix": case.get("upstream_fix", ""),
    }


def load_cases(directory, declaration):
    """The case directories of one group, in run order."""
    cases = []
    for d in sorted(directory.glob(declaration.get("case_glob", "[0-9][0-9]_*"))):
        manifest = d / "case.json"
        if manifest.exists():
            cases.append((d, json.loads(manifest.read_text())))
    return cases


def entry_for(block, arm, group):
    """The protection entry that scores this group's cases under this arm.

    An arm holds one entry per group, because the groups of one application were
    measured by different runs and a cell must name the run that scored its own
    case."""
    entries = block.get(arm)
    if isinstance(entries, dict):
        entries = [entries]
    for spec in entries or []:
        if spec.get("group", "") == group:
            return spec
    return None


def build():
    """One row per case, grouped by application. ONE CORPUS PER APPLICATION: the
    groups inside it are what used to be separate corpora, and they are kept
    because the boundary a case crosses, its upstream pin and the run that
    measured it are all per group."""
    corpora = []
    problems = []
    for manifest in sorted(CORPORA.glob("*/corpus.json")):
        corpus = manifest.parent
        decl = json.loads(manifest.read_text())
        name = corpus.name
        block = decl.get("protection")
        if not block:
            problems.append(f"{name}: no protection block in corpus.json")
            continue
        out_cases = []
        for group, group_decl in sorted(decl.get("groups", {}).items()):
            # A group that runs on no arm here owes no bundles: its own `ignore`
            # is the statement, and every one of its cases carries it.
            group_ignore = group_decl.get("ignore")
            read = {}
            for arm in ARMS:
                spec = entry_for(block, arm, group)
                if spec is None:
                    if not group_ignore:
                        problems.append(f"{name}/{group}: protection has no "
                                        f"{arm} entry")
                    continue
                if "bundle" not in spec:
                    continue
                names = spec["bundle"]
                table = {}
                for one in [names] if isinstance(names, str) else names:
                    bundle = corpus / group / one
                    if not bundle.is_dir():
                        problems.append(f"{name}/{group}/{arm}: bundle {one} "
                                        f"does not exist")
                        continue
                    normalise = slug if spec.get("key") == "slug" else numeric
                    try:
                        table.update({normalise(k): v for k, v in
                                      READERS[spec["reader"]](bundle, spec).items()})
                    except KeyError as exc:
                        problems.append(f"{name}/{group}/{arm}: unknown reader "
                                        f"{spec['reader']!r} ({exc})")
                    except Exception as exc:               # noqa: BLE001
                        problems.append(f"{name}/{group}/{arm}: reader "
                                        f"{spec['reader']} failed on {one}: {exc!r}")
                read[arm] = table
            for case_dir, case in load_cases(corpus / group, group_decl):
                ids = case_identifiers(case_dir, case)
                ignore = case.get("ignore") or group_ignore
                cells = {}
                for arm in ARMS:
                    spec = entry_for(block, arm, group) or {}
                    if ignore and (ignore.get("kind") != "arm"
                                   or arm in (ignore.get("arms") or [])):
                        cells[arm] = {"verdict": "ignored", "caught": None,
                                      "kind": ignore.get("kind"),
                                      "why": ignore.get("why", "")}
                        continue
                    if "coincides_with" in spec:
                        cells[arm] = {"verdict": "coincides",
                                      "with": spec["coincides_with"],
                                      "why": spec.get("why", "")}
                        continue
                    if "not_run" in spec:
                        cells[arm] = {"verdict": "not-run", "caught": None,
                                      "why": spec["not_run"]}
                        continue
                    table = read.get(arm)
                    if table is None:
                        cells[arm] = {"verdict": "not-run", "caught": None,
                                      "why": "the bundle could not be read"}
                        continue
                    key = ids.get(spec.get("key", "dir"), "")
                    key = key if spec.get("key") == "slug" else numeric(key)
                    if key not in table:
                        cells[arm] = {"verdict": "not-run", "caught": None,
                                      "why": spec.get("missing_reason",
                                                      "the bundle of record has no "
                                                      "row for this case")}
                        continue
                    caught, verdict, detail = table[key]
                    # A group whose cases do not prove they ran cannot report a
                    # silence: a completion with no evidence that the defective
                    # path was taken is not a measured miss. The declaration
                    # says so per group and the cell becomes not-run with that
                    # reason, which is reversible the moment a proof exists.
                    if verdict == "missed" and spec.get("silence_unproven"):
                        cells[arm] = {"verdict": "not-run", "caught": None,
                                      "why": spec["silence_unproven"],
                                      "group": group}
                        continue
                    # And the mirror of it. A detection is only a detection of
                    # THIS defect if the same image is quiet without it. Where a
                    # control run shows the fault fires on the upstream-fixed
                    # sequence too, the cell is not a catch and not a silence
                    # either: it is unmeasured, with the control named.
                    void = spec.get("detection_unproven")
                    if verdict == "caught" and void and (
                            void.get("cases") is None
                            or case["case"] in void["cases"]):
                        cells[arm] = {"verdict": "not-run", "caught": None,
                                      "why": void["why"], "group": group,
                                      "control_bundle": void["bundle"]}
                        continue
                    # A miss leaves one question open -- was the object in the
                    # platform's quarantine and merely unswept, or never there --
                    # and a group that has measured it says so here, so the cell
                    # carries the answer instead of the reader having to know it.
                    # A group may declare several dispositions, each for its own
                    # cases, because one group's silences can have more than one
                    # cause. The one that names this case wins; one that names none
                    # covers whatever is left.
                    declared = spec.get("disposition")
                    declared = (declared if isinstance(declared, list)
                                else [declared] if declared else [])
                    disposition = None
                    for one in declared:
                        if one.get("cases") is None or case["case"] in one["cases"]:
                            disposition = one
                            break
                    if verdict != "missed":
                        disposition = None
                    # IN THE QUARANTINE COUNTS AS CAUGHT -- the user's rule, on
                    # purpose generous to the arm: the mechanism did receive the
                    # object and did quarantine it, and only the batching of its
                    # own sweep let the stale read through. Crediting it is the
                    # fair reading of what the mechanism saw, and the cell keeps
                    # the disposition so the evidence stays visible.
                    if disposition and disposition["finding"] == "quarantined-unswept":
                        verdict, caught = "caught", True
                    cells[arm] = {"verdict": verdict, "caught": caught,
                                  "disposition": disposition,
                                  "detail": detail, "group": group,
                                  "bundle": spec["bundle"] if isinstance(
                                      spec["bundle"], str) else ", ".join(spec["bundle"]),
                                  "vehicle": spec["vehicle"],
                                  "local_arm": spec.get("arm", spec["reader"])}
                boundary_class = group_decl.get("allocator_boundary")
                if boundary_class == "mixed":
                    boundary_class = case.get("allocator_boundary")
                out_cases.append({"case": case_dir.name, "group": group,
                                  "allocator_boundary": boundary_class,
                                  "number": case["case"],
                                  "upstream_fix": case.get("upstream_fix", ""),
                                  "title": case["title"],
                                  "boundary": group_decl.get("boundary", ""),
                                  "nested": case.get("nested"),
                                  "ignored": bool(ignore),
                                  "arms": cells})
        corpora.append({"corpus": name, "program": decl["program"],
                        "title": decl["title"], "cases": out_cases,
                        "groups": {g: {"boundary": d.get("boundary", ""),
                                       "cases": d.get("cases", 0),
                                       "ignore": d.get("ignore")}
                                   for g, d in sorted(decl.get("groups", {}).items())},
                        "protection": block})
    return corpora, problems


def resolve(cell, cells):
    """A coincides cell reports its target's verdict, which is the honest reading:
    where a corpus has no nested allocator, protecting one is the same binary and
    the same measurement, not an absent arm."""
    while cell.get("verdict") == "coincides":
        cell = cells[cell["with"]]
    return cell


MARK = {"caught": "**C**", "missed": "·", "wrong-answer": "w", "not-run": "—",
        "ignored": "∅"}


MEASURED = ("caught", "missed", "wrong-answer")
BOUNDARY_ORDER = ("system", "nested", "interior")
BOUNDARY_TITLE = {
    "system": "the system allocator handed the object out directly -- the boundary "
              "all three arms protect",
    "nested": "the program's own allocator carved the object out of a block it holds "
              "-- only `capstone-sublet` protects it",
    "interior": "the crossing is inside ONE allocation and no allocator vested the "
                "crossed region -- no arm in this study claims it",
}


FINDING_MEANS = {
    "quarantined-unswept": ("the object WAS in the quarantine and no sweep cleared "
                            "it", "a synchronous sweep would have caught these"),
    "never-freed": ("the object never reached the system allocator, so it never "
                    "entered the quarantine",
                    "no sweep policy reaches these; only protecting the nested "
                    "allocator does"),
    "not-temporal": ("nothing was freed at all; the crossing is inside a live "
                     "allocation", "revocation is not the mechanism in play"),
}


def disposition_ledger(corpora):
    """What the arm's silences are made of, where a run has measured it.

    The arm stays what CheriBSD ships -- revocation on, asynchronous, batched. Nothing
    here re-runs it under another policy. What the dispositions establish is where
    each silence came from: an object the mechanism held in its quarantine and lost to
    its own batching is a different fact than an object it never received, and the
    first is credited as a catch while the second is not.
    """
    ledger = {arm: {} for arm in ARMS}
    for corpus in corpora:
        for case in corpus["cases"]:
            if case["ignored"]:
                continue
            for arm in ARMS:
                cell = resolve(case["arms"][arm], case["arms"])
                if not (cell["verdict"] == "missed" or cell.get("disposition")):
                    continue
                found = (cell.get("disposition") or {}).get("finding", "open")
                ledger[arm][found] = ledger[arm].get(found, 0) + 1
    if not any(set(v) - {"open"} for v in ledger.values()):
        return []
    lines = ["## What the silences are made of", "",
             disposition_ledger.__doc__.split("\n\n")[1].replace("\n    ", " ").strip(),
             "",
             "The count in each heading is cells on which the arm's mechanism printed "
             "nothing. It is NOT the miss count: the `quarantined-unswept` ones are "
             "credited as catches by the rule above, so they appear here as an account "
             "of the silence and in the caught column as the verdict.", ""]
    for arm in ARMS:
        found = ledger[arm]
        total = sum(found.values())
        # Only an arm whose silences have been taken apart gets a section. The
        # synchronous arms have no quarantine and nothing to dispose of, so a row
        # of "not yet measured" for them would invent a question.
        if not (set(found) - {"open"}):
            continue
        lines += [f"### `{arm}`: {total} cells where the mechanism reported "
                  f"nothing", "",
                  "| disposition | cells | what it means | what would change it |",
                  "|---|---:|---|---|"]
        for finding, (means, changes) in FINDING_MEANS.items():
            if finding in found:
                lines.append(f"| `{finding}` | **{found[finding]}** | {means} "
                             f"| {changes} |")
        if found.get("open"):
            lines.append(f"| not yet measured | {found['open']} | the run does not "
                         "say whether the object reached the mechanism | the probe, "
                         "`ports/common/host/cheribsd/quarantine-probe.c` |")
        lines.append("")
    return lines


def intersection(corpora):
    """The arms compared only where all three have a verdict, split by who allocated
    the object.

    Two corrections to the headline table, and both of them cut the same way. A
    cell that was never run is not evidence, so the arms are compared on the cases
    where all three were measured. And on a case whose object came out of the
    program's own allocator, two of the three arms are not protecting that object at
    all -- counting those together with the malloc-boundary cases reads as a
    weakness of the mechanism when it is a statement about what each arm covers.
    """
    buckets, credit = {}, {}
    for corpus in corpora:
        for case in corpus["cases"]:
            if case["ignored"]:
                continue
            cells = {a: resolve(case["arms"][a], case["arms"]) for a in ARMS}
            if any(cells[a]["verdict"] not in MEASURED for a in ARMS):
                continue
            where = buckets.setdefault(case["allocator_boundary"],
                                       {a: {v: 0 for v in MARK} for a in ARMS})
            here = credit.setdefault(case["allocator_boundary"],
                                     {a: 0 for a in ARMS})
            for arm in ARMS:
                where[arm][cells[arm]["verdict"]] += 1
                if cells[arm]["verdict"] == "caught" and cells[arm].get("disposition"):
                    here[arm] += 1
    lines = ["## The arms compared where all three were measured, split by who "
             "allocated the object", "", intersection.__doc__.split("\n\n")[1].replace(
                 "\n    ", " ").strip(), ""]
    for cls in BOUNDARY_ORDER:
        if cls not in buckets:
            continue
        counts, credited = buckets[cls], credit[cls]
        n = sum(counts[ARMS[0]].values())
        lines += [f"### `{cls}`: {n} case" + ("" if n == 1 else "s"), "",
                  BOUNDARY_TITLE[cls], "",
                  "| arm | caught | of those, by quarantine | missed "
                  "| share caught |",
                  "|---|---:|---:|---:|---:|"]
        for arm in ARMS:
            c = counts[arm]
            missed = c["missed"] + c["wrong-answer"]
            lines.append(f"| `{arm}` | **{c['caught']}** | {credited[arm]} | "
                         f"{missed} | {c['caught'] / n:.0%} |")
        lines.append("")
    total = {a: {v: 0 for v in MARK} for a in ARMS}
    credited = {a: sum(c[a] for c in credit.values()) for a in ARMS}
    for counts in buckets.values():
        for arm in ARMS:
            for v, k in counts[arm].items():
                total[arm][v] += k
    n = sum(total[ARMS[0]].values())
    lines += [f"### All {n} together", "",
              "Kept for continuity with the per-application tables above. Read the "
              "split first: this row's mixture of boundaries is a property of which "
              "corpora happen to be fully measured, not of the arms.", "",
              "| arm | caught | of those, by quarantine | missed "
              "| share caught |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        c = total[arm]
        lines.append(f"| `{arm}` | **{c['caught']}** | {credited[arm]} | "
                     f"{c['missed'] + c['wrong-answer']} | {c['caught'] / n:.0%} |")
    return lines + [""]


def render(corpora):
    lines = [
        "# Does it catch it? Every bug in this tree, against three protection arms",
        "",
        "Generated by `tools/protection-matrix.py` from each application's",
        "`protection` declaration -- do not edit. `arms.json` defines the three arms and",
        "the verdict vocabulary; every cell below was read out of the results bundle its",
        "corpus names, and the JSON beside this file carries the bundle and the detail",
        "per cell.",
        "",
        "ONE CORPUS PER APPLICATION. The groups inside an application are what used to be",
        "separate corpora: each crosses one boundary, pins one upstream version and was",
        "measured by its own run, which is why a cell names the bundle of its own group.",
        "",
        "| | |",
        "|---|---|",
        "| **C** | caught: the arm's mechanism reported |",
        "| · | missed: the sequence ran to the end and the mechanism said nothing |",
        "| **C**<sup>q</sup> | caught by QUARANTINE MEMBERSHIP, not by a reported "
        "fault: a run measured the object in the arm's quarantine, and only the "
        "batching of its own sweep let the stale access through |",
        "| ·<sup>q</sup> | missed, and WHY is measured -- the object never reached "
        "the mechanism, or nothing was freed at all. The cell's JSON says which |",
        "| w | the program's own oracle failed, no mechanism fired |",
        "| — | not run; the per-case reason is in the JSON |",
        "| ∅ | ignored: the case stays as material and leaves every denominator |",
        "",
    ]
    totals = {arm: {v: 0 for v in MARK} for arm in ARMS}
    totals_disposed = {arm: 0 for arm in ARMS}
    totals_credited = {arm: 0 for arm in ARMS}
    lines += ["## Totals, ignored cases excluded", "",
              "IN THE QUARANTINE COUNTS AS CAUGHT. Where a run has measured that the "
              "object was in `cheribsd`'s revocation quarantine and only the batching "
              "of its own sweep let the stale access through, the cell is a CATCH, "
              "scored to the arm and marked <code>C<sup>q</sup></code>: the mechanism "
              "received the object and held it, so crediting it is the fair reading of "
              "what it saw. A silence whose object never reached the mechanism, or "
              "where nothing was freed at all, stays a MISS -- `of those, disposition "
              "measured` counts the misses a run has explained that way, and the rest "
              "are open questions.", "",
              "| arm | caught | of those, by quarantine | missed | of those, "
              "disposition measured | not run | of | ignored |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    body = []
    for corpus in corpora:
        counts = {arm: {v: 0 for v in MARK} for arm in ARMS}
        disposed = {arm: 0 for arm in ARMS}
        credited = {arm: 0 for arm in ARMS}
        rows_out = []
        for case in corpus["cases"]:
            cells = []
            for arm in ARMS:
                cell = resolve(case["arms"][arm], case["arms"])
                counts[arm][cell["verdict"]] += 1
                totals[arm][cell["verdict"]] += 1
                mark = MARK[cell["verdict"]]
                if case["arms"][arm].get("verdict") == "coincides":
                    mark += "<sup>=</sup>"
                elif cell.get("vehicle") == "capstone-baremetal":
                    mark += "<sup>b</sup>"
                if cell.get("disposition"):
                    disposed[arm] += 1
                    totals_disposed[arm] += 1
                    mark += "<sup>q</sup>"
                    if cell["verdict"] == "caught":
                        credited[arm] += 1
                        totals_credited[arm] += 1
                cells.append(mark)
            rows_out.append(f"| `{case['group']}` | {case['number']} | "
                            f"`{case['upstream_fix']}` | {case['title'][:76]} | "
                            + " | ".join(cells) + " |")
        n = len(corpus["cases"])
        declared = sum(g["cases"] for g in corpus["groups"].values())
        head = f"### {corpus['corpus']} -- {n} cases"
        if declared != n:
            head += (f" with a directory, of {declared} declared")
        body += [head, "", corpus["title"], ""]
        if len(corpus["groups"]) > 1:
            body += ["| group | cases | the boundary its cases cross |",
                     "|---|---:|---|"]
            for g, info in corpus["groups"].items():
                mark = (f" **∅ ignored**: {info['ignore']['why']}"
                        if info["ignore"] else "")
                body.append(f"| `{g}` | {info['cases']} | {info['boundary']}{mark} |")
            body += [""]
        body += ["| group | case | upstream | what the defect is | CheriBSD "
                 "| capstone-sysalloc | capstone-sublet |",
                 "|---|---:|---|---|:---:|:---:|:---:|"] + rows_out + [""]
        for arm in ARMS:
            c = counts[arm]
            measured = n - c["ignored"]
            body.append(f"- `{arm}`: **{c['caught']}** caught"
                        + (f" ({credited[arm]} by quarantine)"
                           if credited[arm] else "")
                        + f", {c['missed']} missed"
                        + (f", {c['wrong-answer']} wrong answer" if c["wrong-answer"] else "")
                        + (f", {c['not-run']} not run" if c["not-run"] else "")
                        + f", of {measured}"
                        + (f"; {c['ignored']} ignored" if c["ignored"] else "")
                        + (f". {disposed[arm] - credited[arm]} of the misses "
                           + ("has" if disposed[arm] - credited[arm] == 1 else "have")
                           + " a measured disposition"
                           if disposed[arm] - credited[arm] else "")
                        + ".")
        body.append("")
    for arm in ARMS:
        c = totals[arm]
        n = sum(c.values()) - c["ignored"]
        lines.append(f"| `{arm}` | {c['caught']} | {totals_credited[arm]} | "
                     f"{c['missed'] + c['wrong-answer']} | "
                     f"{totals_disposed[arm] - totals_credited[arm]} | "
                     f"{c['not-run']} | {n} | {c['ignored']} |")
    lines += [""] + disposition_ledger(corpora) + intersection(corpora) + [
              "A `=` marks a cell that coincides with the arm to its left because the "
              "group has no nested allocator to protect; a `b` marks one measured on the "
              "freestanding vehicle, which is NOT paired with the cell to its left.",
              ""] + body
    return "\n".join(lines) + "\n"


def selected(case, cells, specs):
    """Does this case match every --focus narrowing?

    A verdict spec widens within its own kind -- `--focus caught --focus missed`
    is either -- and narrows across kinds: a program spec and a verdict spec both
    have to hold. That is the only sensible reading of "the caught FFmpeg ones".
    """
    verdicts, places, armed = set(), set(), {}
    for spec in specs:
        if "@" in spec:
            verdict, arm = spec.split("@", 1)
            armed.setdefault(arm, set()).add(verdict)
        elif spec in MARK or spec in ("parked", "disposed", "undisposed"):
            verdicts.add(spec)
        else:
            places.add(spec)
    # One cell can answer to more than one spec: a miss is also `disposed` or
    # `undisposed`, which is how "show me the misses nobody has explained yet"
    # is asked for.
    kinds = {}
    for a in ARMS:
        tags = {cells[a]["verdict"]}
        if cells[a]["verdict"] in ("missed", "caught"):
            tags.add("disposed" if cells[a].get("disposition") else "undisposed")
        kinds[a] = tags
    parked = (case["arms"][ARMS[0]].get("kind") == "parked")
    if verdicts:
        have = set().union(*kinds.values()) | ({"parked"} if parked else set())
        if not (verdicts & have):
            return False
    for arm, want in armed.items():
        if arm not in ARMS or not (want & kinds.get(arm, set())):
            return False
    if places:
        here = {case["corpus"], f"{case['corpus']}/{case['group']}", case["group"]}
        if not (places & here):
            return False
    return True


def focus(corpora, specs):
    """One line per selected case. Deliberately not written to a file: a view
    that overwrote the record would make the record a view."""
    rows, counts = [], {arm: 0 for arm in ARMS}
    total = 0
    for corpus in corpora:
        for case in corpus["cases"]:
            case = dict(case, corpus=corpus["corpus"])
            cells = {a: resolve(case["arms"][a], case["arms"]) for a in ARMS}
            if not selected(case, cells, specs):
                continue
            total += 1
            for arm in ARMS:
                if cells[arm].get("caught"):
                    counts[arm] += 1
            rows.append((f"{corpus['corpus']}/{case['group']}", case["number"],
                         case["upstream_fix"], case["title"],
                         [(cells[a]["verdict"], bool(cells[a].get("disposition")))
                          for a in ARMS]))
    width = max((len(r[0]) for r in rows), default=20)
    short = {"caught": "CAUGHT", "missed": "·",
             "not-run": "—", "ignored": "∅", "wrong-answer": "wrong"}
    # A miss with a measured disposition prints as ·q, so a view of the misses
    # never hides which of them are still open questions.
    short_disposed = dict(short, missed="·q", caught="CAUGHT-q")
    print(f"{'group':{width}s} case  {'upstream':12s} "
          + "  ".join(f"{a.split('-')[-1]:>9s}" for a in ARMS) + "  what it is")
    for where, number, fix, title, verdicts in rows:
        print(f"{where:{width}s} {number:4d}  {fix:12s} "
              + "  ".join(f"{(short_disposed if d else short).get(v, v):>9s}"
                          for v, d in verdicts)
              + f"  {title[:70]}")
    print(f"\n{total} cases selected by {' '.join(specs)}: "
          + ", ".join(f"{a} {counts[a]}" for a in ARMS))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the generated files are stale")
    ap.add_argument("--focus", action="append", default=[], metavar="SPEC",
                    help="PRINT a subset to stdout and write nothing. A view, not a "
                         "declaration: PROTECTION.md stays the whole record. SPEC is "
                         "caught | missed | not-run | ignored | parked | disposed "
                         "| undisposed, a program, a program/group, or "
                         "caught@<arm> / undisposed@cheribsd. Several --focus narrow "
                         "together; repeat a verdict to widen it.")
    args = ap.parse_args()
    corpora, problems = build()
    if args.focus:
        for problem in problems:
            print(f"PROBLEM {problem}", file=sys.stderr)
        focus(corpora, args.focus)
        return 0
    md = render(corpora)
    data = json.dumps({"arms": ARMS, "corpora": corpora}, indent=1) + "\n"
    md_path, json_path = CORPORA / "PROTECTION.md", CORPORA / "protection.json"
    if args.check:
        stale = [p.name for p, new in ((md_path, md), (json_path, data))
                 if not p.exists() or p.read_text() != new]
        for p in problems:
            print(f"PROBLEM {p}")
        if stale:
            print(f"STALE {', '.join(stale)}: run tools/protection-matrix.py")
        return 1 if stale or problems else 0
    md_path.write_text(md)
    json_path.write_text(data)
    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    counted = {arm: sum(1 for c in corpora for case in c["cases"]
                        if resolve(case["arms"][arm], case["arms"])["verdict"] == "caught")
               for arm in ARMS}
    total = sum(len(c["cases"]) for c in corpora)
    print(f"{total} cases, {len(corpora)} corpora -> PROTECTION.md, protection.json")
    for arm in ARMS:
        print(f"  {arm:20s} {counted[arm]:4d} caught")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
