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
def numeric(key):
    """`00` and `0` are the same case: one bundle zero-pads its tags and another
    does not, and nothing is gained by making the declaration say which."""
    return str(int(key)) if key.isdigit() else key


def case_identifiers(case_dir, case):
    """Every name a bundle might use for this case."""
    return {
        "dir": case_dir.name,
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
                    try:
                        table.update({numeric(k): v for k, v in
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
                    key = numeric(ids.get(spec.get("key", "dir"), ""))
                    if key not in table:
                        cells[arm] = {"verdict": "not-run", "caught": None,
                                      "why": spec.get("missing_reason",
                                                      "the bundle of record has no "
                                                      "row for this case")}
                        continue
                    caught, verdict, detail = table[key]
                    cells[arm] = {"verdict": verdict, "caught": caught,
                                  "detail": detail, "group": group,
                                  "bundle": spec["bundle"] if isinstance(
                                      spec["bundle"], str) else ", ".join(spec["bundle"]),
                                  "vehicle": spec["vehicle"],
                                  "local_arm": spec.get("arm", spec["reader"])}
                out_cases.append({"case": case_dir.name, "group": group,
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
        "| w | the program's own oracle failed, no mechanism fired |",
        "| — | not run; the per-case reason is in the JSON |",
        "| ∅ | ignored: the case stays as material and leaves every denominator |",
        "",
    ]
    totals = {arm: {v: 0 for v in MARK} for arm in ARMS}
    lines += ["## Totals, ignored cases excluded", "",
              "| arm | caught | missed | not run | of | ignored |",
              "|---|---:|---:|---:|---:|---:|"]
    body = []
    for corpus in corpora:
        counts = {arm: {v: 0 for v in MARK} for arm in ARMS}
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
            body.append(f"- `{arm}`: **{c['caught']}** caught, {c['missed']} missed"
                        + (f", {c['wrong-answer']} wrong answer" if c["wrong-answer"] else "")
                        + (f", {c['not-run']} not run" if c["not-run"] else "")
                        + f", of {measured}"
                        + (f"; {c['ignored']} ignored" if c["ignored"] else "") + ".")
        body.append("")
    for arm in ARMS:
        c = totals[arm]
        n = sum(c.values()) - c["ignored"]
        lines.append(f"| `{arm}` | {c['caught']} | {c['missed'] + c['wrong-answer']} | "
                     f"{c['not-run']} | {n} | {c['ignored']} |")
    lines += ["",
              "A `=` marks a cell that coincides with the arm to its left because the "
              "group has no nested allocator to protect; a `b` marks one measured on the "
              "freestanding vehicle, which is NOT paired with the cell to its left.",
              ""] + body
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the generated files are stale")
    args = ap.parse_args()
    corpora, problems = build()
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
