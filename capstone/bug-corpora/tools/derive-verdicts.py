#!/usr/bin/env python3
"""Write, or check, every verdict a corpus states, from its result bundles and nothing else.

    derive-verdicts.py [--check] [corpus dir ...]      default: every corpus that declares verdict_bundles

A corpus opts in by naming, in corpus.json, the bundle that is current for each arm:

    "verdict_bundles": {"spatial": "results/2026-10-10-qemu/spatial", ...}

Each bundle is a tools/verdicts.py bundle (verdicts.jsonl + inputs.json). From it this tool writes

  * each case.json's arms[arm]: verdict (CAUGHT / MISSED / NO-READING), verdict_reason for a
    NO-READING, verdict_note (the judge's evidence) and verdict_from (the bundle) -- the fields
    catch-tables.py reads;
  * results/verdicts.tsv, the corpus's whole table, one row per case and arm.

Neither is edited by hand any more: a hand-merged matrix and verdicts copied into case.json were
four copies of one result, and they drifted. --check recomputes both and fails on any difference,
and it RE-JUDGES every stored observation against the current tools/arms.json and judge, so a
change to either cannot leave an old verdict standing silently.

Exit: 0 written or current, 1 --check found drift, 2 a declaration or bundle is unusable.
"""
import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CORPORA = TOOLS.parent
REPO = CORPORA.parents[1]
sys.path.insert(0, str(TOOLS))
import verdicts as v  # noqa: E402

FIELDS = ("verdict", "verdict_reason", "verdict_note", "verdict_from")
# Outcome fields the runners of 2026-10-06 wrote by hand beside an arm's oracle. A derived verdict
# replaces them; leaving them would put two answers in one arm.
LEGACY = ("evidence", "run")


def indent_of(text):
    for i in (1, 2, 4):
        if json.dumps(json.loads(text), indent=i, ensure_ascii=False) + "\n" == text:
            return i, False
        if json.dumps(json.loads(text), indent=i) + "\n" == text:
            return i, True
    return 2, False


def derive(corpus):
    """{case dir name: {arm: fields}}, tsv text, problems."""
    decl = json.loads((corpus / "corpus.json").read_text())
    bundles = decl.get("verdict_bundles", {})
    configs = decl.get("arm_configurations", {})
    arms = v.load_arms()
    problems, cells, lines = [], {}, []
    for arm, rel in sorted(bundles.items()):
        path = corpus / rel
        if arm not in configs:
            problems.append(f"verdict_bundles: {arm} has no arm_configurations entry")
            continue
        if not (path / "verdicts.jsonl").is_file():
            problems.append(f"verdict_bundles: {rel} has no verdicts.jsonl")
            continue
        inputs = json.loads((path / "inputs.json").read_text())
        if inputs.get("contract") != "verdicts-v1" or inputs.get("arm") != arm:
            problems.append(f"{rel}: not a verdicts-v1 bundle for arm {arm}")
            continue
        if inputs.get("configuration") != configs[arm]:
            problems.append(f"{rel}: measured {inputs.get('configuration')}, but corpus.json says "
                            f"{arm} is {configs[arm]}")
            continue
        spec = arms[configs[arm]]
        for obs, verdict, reason, evidence in v.read_bundle(path):
            again = v.judge(obs, spec)
            if again != (verdict, reason, evidence):
                problems.append(f"{rel}: {obs.case} was judged {verdict}/{reason} when recorded and "
                                f"is {again[0]}/{again[1]} under the current judge and arms.json")
            cell = {"verdict": verdict, "verdict_note": evidence, "verdict_from": rel}
            if reason:
                cell["verdict_reason"] = reason
            if obs.case in cells.get(arm, {}):
                problems.append(f"{rel}: {obs.case} appears twice")
            cells.setdefault(arm, {})[obs.case] = cell
            lines.append("\t".join([obs.case, arm, configs[arm], verdict, reason or "",
                                    " ".join(evidence.split()), rel]))
    by_case = {}
    for arm, rows in cells.items():
        for case, cell in rows.items():
            by_case.setdefault(case, {})[arm] = cell
    tsv = "case\tarm\tconfiguration\tverdict\treason\tevidence\tbundle\n" + "".join(
        line + "\n" for line in sorted(lines))
    return by_case, tsv, problems


def apply(corpus, by_case, write):
    """Bring each case.json in line with by_case. Returns the paths that differ."""
    differ = []
    dirs = {d.name: d for d in corpus.glob("[0-9][0-9]_*") if (d / "case.json").is_file()}
    for name in sorted(set(by_case) - set(dirs)):
        differ.append(f"{name}: in a bundle but not a case directory")
    for name, d in sorted(dirs.items()):
        path = d / "case.json"
        text = path.read_text()
        case = json.loads(text)
        before = json.dumps(case, sort_keys=True)
        for arm, cell in by_case.get(name, {}).items():
            entry = case["arms"].setdefault(arm, {})
            for key in FIELDS + LEGACY:
                entry.pop(key, None)
            entry.update(cell)
        if json.dumps(case, sort_keys=True) != before:
            differ.append(str(path.relative_to(REPO)))
            if write:
                ind, ascii_only = indent_of(text)
                path.write_text(json.dumps(case, indent=ind, ensure_ascii=ascii_only) + "\n")
    return differ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("corpora", nargs="*", type=Path)
    a = p.parse_args()
    corpora = [c.resolve() for c in a.corpora] or sorted(
        m.parent for m in CORPORA.glob("*/*/corpus.json")
        if "verdict_bundles" in json.loads(m.read_text()))
    if not corpora:
        print("derive-verdicts: no corpus declares verdict_bundles", file=sys.stderr)
        return 2
    status = 0
    for corpus in corpora:
        where = corpus.relative_to(REPO)
        by_case, tsv, problems = derive(corpus)
        if problems:
            for line in problems:
                print(f"FAIL {where}: {line}", file=sys.stderr)
            status = max(status, 2 if not by_case else 1)
            continue
        differ = apply(corpus, by_case, write=not a.check)
        table = corpus / "results/verdicts.tsv"
        if not table.is_file() or table.read_text() != tsv:
            differ.append(str(table.relative_to(REPO)))
            if not a.check:
                table.write_text(tsv)
        if a.check and differ:
            for path in differ:
                print(f"STALE {where}: {path}", file=sys.stderr)
            status = max(status, 1)
        else:
            print(f"{where}: {'current' if a.check or not differ else 'wrote ' + str(len(differ))}"
                  f" ({sum(len(x) for x in by_case.values())} cells)")
    return status


if __name__ == "__main__":
    sys.exit(main())
