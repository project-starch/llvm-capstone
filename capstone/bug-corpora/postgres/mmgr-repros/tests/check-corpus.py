#!/usr/bin/env python3
"""Enforce the corpus contract on this corpus.

The contract is ../../cpython/pymalloc-repros/SCHEMA.md and that file says the
checker is the authority; this is that checker for the PostgreSQL corpus.

    check-corpus.py                 check, exit 1 on any violation
    check-corpus.py --self-test     prove the checker can FAIL, then check

The self-test exists because a gate that has never rejected anything is not a
passing gate, it is an unproven one.
"""

import argparse
import copy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {"case", "upstream_fix", "title", "consumer", "live_in_pin", "live_proof",
            "object", "lifetime_ender", "shape", "allocator_layer", "fidelity",
            "arms", "status"}
OPTIONAL = {"distinguishing", "sibling_issue", "size_class", "size_note", "layer_note",
            "note"}
ARMS = {"spatial", "sublet", "poisoncap-spatial", "poisoncap-protected", "native-detect"}


def shapes(readme):
    """Only the shape table. A header cell is not a shape, and neither is a row
    of some other table that happens to start with a lowercase letter."""
    block = re.search(r"^## Shapes$(.*?)^## ", readme, re.M | re.S)
    if not block:
        return set()
    rows = re.findall(r"^\| (.+?) \| .+? \|$", block.group(1), re.M)
    return {r for r in rows if r not in ("shape",) and not set(r) <= set("-: ")}


def check(root):
    problems = []
    readme = (root / "README.md").read_text()
    allowed = shapes(readme)
    if not allowed:
        return ["README.md has no ## Shapes table; shape cannot be validated"]
    dirs = sorted(d for d in root.glob("[0-9][0-9]_*") if d.is_dir())
    if not dirs:
        return ["no NN_* case directories: a corpus that holds nothing must not pass"]
    numbers = []
    for d in dirs:
        try:
            case = json.loads((d / "case.json").read_text())
        except FileNotFoundError:
            problems.append(f"{d.name}: no case.json")
            continue
        except json.JSONDecodeError as exc:
            problems.append(f"{d.name}: case.json is not JSON ({exc})")
            continue
        missing = REQUIRED - set(case)
        if missing:
            problems.append(f"{d.name}: case.json lacks {', '.join(sorted(missing))}")
        unknown = set(case) - REQUIRED - OPTIONAL
        if unknown:
            problems.append(f"{d.name}: case.json has unknown {', '.join(sorted(unknown))}")
        if "case" not in case or "upstream_fix" not in case:
            continue
        numbers.append(case["case"])
        if not d.name.startswith(f"{case['case']:02d}_{case['upstream_fix']}_"):
            problems.append(f"{d.name}: directory name disagrees with case/upstream_fix")
        if case.get("shape") not in allowed:
            problems.append(f"{d.name}: shape {case.get('shape')!r} is not in the README table")
        for name, arm in case.get("arms", {}).items():
            if name not in ARMS:
                problems.append(f"{d.name}: unknown arm {name!r}")
            elif ("oracle" in arm) == ("status" in arm):
                problems.append(
                    f"{d.name}: arm {name} must declare an oracle OR say why it is not written")
        source = d / "case.c"
        if not source.is_file():
            problems.append(f"{d.name}: no case.c")
            continue
        declared = re.search(r"PG_CASE\((\d+)\)", source.read_text())
        if not declared:
            problems.append(f"{d.name}: case.c declares no PG_CASE")
        elif int(declared.group(1)) != case["case"]:
            problems.append(
                f"{d.name}: case.c says PG_CASE({declared.group(1)}), directory says {case['case']}")
    if sorted(numbers) != list(range(len(numbers))):
        problems.append(f"case numbers are not dense from zero: {sorted(numbers)}")
    return problems


def self_test(root):
    """Four corruptions the checker MUST reject. If any is accepted, the checker
    is broken and its silence on the real corpus means nothing."""
    import tempfile, shutil
    failures = []
    for label, mutate in (
        ("a shape that is not in the README", lambda c: c.update(shape="not a real shape")),
        ("a missing required field", lambda c: c.pop("consumer")),
        ("an arm with neither oracle nor status", lambda c: c["arms"].update(spatial={})),
        ("a case number the directory does not carry", lambda c: c.update(case=42)),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            copy_root = Path(tmp) / "corpus"
            shutil.copytree(root, copy_root, ignore=shutil.ignore_patterns("results", ".git"))
            target = sorted(copy_root.glob("[0-9][0-9]_*"))[1]
            case = json.loads((target / "case.json").read_text())
            mutate(case)
            (target / "case.json").write_text(json.dumps(case, indent=2))
            if not check(copy_root):
                failures.append(label)
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        accepted = self_test(ROOT)
        if accepted:
            print("CHECKER BROKEN: it accepted " + "; ".join(accepted), file=sys.stderr)
            return 2
        print("self-test: the checker rejected all four corruptions")
    problems = check(ROOT)
    for line in problems:
        print(line, file=sys.stderr)
    print(f"check-corpus: {'BLOCKED' if problems else 'CLEAN'}"
          f" ({len(problems)} problem{'' if len(problems) == 1 else 's'})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
