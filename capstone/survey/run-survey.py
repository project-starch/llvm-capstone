#!/usr/bin/env python3
"""Drive one survey level: build the pinned upstream with the instrument, run its
workload, and write a report tree the manuscript's reader can consume unchanged.

What is shared sits here: resolving the instrument, the output layout, the
repetition loop and the provenance record. What differs per allocator sits in
`recipes/<level-id>.sh`, which this driver calls once per repetition with the
environment below. A level with no recipe is refused by name rather than
silently skipped, because a campaign that quietly records fewer levels than it
claims is worse than one that stops.

    ./run-survey.py --plan
    ./run-survey.py --plan --level postgres-slab
    ./run-survey.py --level postgres-slab --instrument ../../../nested-allocators-paper/experiments/a1/hook

The recipe receives:

    A1_HOOK_DIR     the instrument directory, holding a1hook.h and a1core.inc
    A1_LEVEL        the level name the wiring must compile in as A1_L1_NAME
    A1_LEVEL_BELOW  the level 0 name, "libc" or another custom allocator
    A1_WIRING       the wiring file for this level, if the branch carries one
    SURVEY_OUT      the directory this repetition must write its reports into
    SURVEY_REP      the repetition number, from 1
    SURVEY_REPO     this repository's root

and must leave one `rep.<pid>` file per instrumented process in SURVEY_OUT.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MAP = HERE / "levels.json"
RECIPES = HERE / "recipes"
RESULTS = HERE / "results"
DEFAULT_REPETITIONS = 3


def load():
    return json.loads(MAP.read_text())


def level_by_id(doc, level_id):
    for level in doc["levels"]:
        if level["id"] == level_id:
            return level
    raise SystemExit(f"no such level: {level_id}\n"
                     f"known: {', '.join(l['id'] for l in doc['levels'])}")


def find_instrument(given):
    """The instrument is the manuscript's, never a copy kept here."""
    candidates = [Path(given)] if given else [
        REPO.parent / "nested-allocators-paper/experiments/a1/hook",
        REPO.parents[1] / "nested-allocators-paper/experiments/a1/hook",
    ]
    for path in candidates:
        if (path / "a1hook.h").is_file() and (path / "a1core.inc").is_file():
            return path.resolve()
    raise SystemExit(
        "the a1 instrument was not found. Pass --instrument <paper>/experiments/a1/hook.\n"
        "It is not vendored here on purpose: two copies of a counting rule are two rules.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readiness(level, instrument=None):
    """What stands between this level and a recording."""
    missing = []
    port = level["port"]
    if not (REPO / port["path"]).exists():
        missing.append(f"port not here, lands with {port.get('lands_with', 'an unnamed change')}")
    if not (RECIPES / f"{level['id']}.sh").is_file():
        missing.append(f"no recipe: recipes/{level['id']}.sh")
    wiring = HERE / "wiring" / f"{level['id']}.inc"
    if not wiring.is_file():
        missing.append(f"no wiring: wiring/{level['id']}.inc")
    if instrument is None:
        missing.append("instrument not resolved")
    return missing


def plan(doc, only=None, instrument=None):
    levels = [l for l in doc["levels"] if only is None or l["id"] == only]
    ready = 0
    print(f"{'level':<26} {'program':<12} {'below':<16} {'workload':<24} state")
    print("-" * 104)
    for level in levels:
        missing = readiness(level, instrument)
        state = "ready" if not missing else "; ".join(missing)
        if not missing:
            ready += 1
        print(f"{level['id']:<26} {level['program']:<12} {level['below']:<16} "
              f"{level['workload']:<24} {state}")
    print("-" * 104)
    print(f"{ready} of {len(levels)} levels can record now")
    if ready < len(levels):
        print("A level is recorded when its port is here, its wiring names the seam, "
              "and its recipe builds and runs the workload.")
    return 0


def record(doc, level, instrument, repetitions, out_root):
    missing = readiness(level, instrument)
    blocking = [m for m in missing if not m.startswith("instrument")]
    if blocking:
        for item in blocking:
            print(f"cannot record {level['id']}: {item}")
        return 1

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = Path(out_root) / f"{stamp}-{level['id']}"
    raw = base / "raw" / level["level"]
    raw.mkdir(parents=True, exist_ok=True)

    wiring = HERE / "wiring" / f"{level['id']}.inc"
    recipe = RECIPES / f"{level['id']}.sh"
    env = dict(os.environ)
    env.update({
        "A1_HOOK_DIR": str(instrument),
        "A1_LEVEL": level["level"],
        "A1_LEVEL_BELOW": level["below"],
        "A1_WIRING": str(wiring),
        "SURVEY_REPO": str(REPO),
    })

    produced = []
    for rep in range(1, repetitions + 1):
        out = raw / f"{level['workload']}.hooked.rep{rep}.reports"
        out.mkdir(parents=True, exist_ok=True)
        env["SURVEY_OUT"] = str(out)
        env["SURVEY_REP"] = str(rep)
        print(f"-- {level['id']} repetition {rep} of {repetitions}")
        result = subprocess.run(["bash", str(recipe)], env=env)
        if result.returncode != 0:
            print(f"recipe failed for repetition {rep}, exit {result.returncode}")
            return result.returncode
        reports = sorted(out.glob("rep.*"))
        if not reports:
            print(f"repetition {rep} produced no rep.* report in {out}")
            return 1
        produced.append({"repetition": rep,
                         "reports": [{"path": str(p.relative_to(base)),
                                      "sha256": digest(p)} for p in reports]})

    manifest = {
        "study": "W3",
        "level": level["id"],
        "program": level["program"],
        "allocator": level["allocator"],
        "level_name": level["level"],
        "below": level["below"],
        "upstream": level["upstream"],
        "workload": level["workload"],
        "workload_command": doc["workloads"][level["workload"]]["command"],
        "seam": level["seam"],
        "seam_confidence": level["confirmed"],
        "instrument": {
            "path": str(instrument),
            "a1hook.h": digest(instrument / "a1hook.h"),
            "a1core.inc": digest(instrument / "a1core.inc"),
        },
        "wiring_sha256": digest(wiring),
        "recipe_sha256": digest(recipe),
        "repetitions": produced,
        "recorded": stamp,
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {base}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", help="level id from levels.json")
    parser.add_argument("--plan", action="store_true", help="report what can record and what blocks the rest")
    parser.add_argument("--instrument", help="path to the manuscript's experiments/a1/hook")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()

    doc = load()
    try:
        instrument = find_instrument(args.instrument)
    except SystemExit as exit_error:
        if not args.plan:
            raise
        print(f"note: {exit_error}".replace("\n", "\n      "))
        instrument = None

    if args.plan:
        return plan(doc, args.level, instrument)
    if not args.level:
        parser.error("give --level, or --plan to see the campaign")
    return record(doc, level_by_id(doc, args.level), instrument,
                  args.repetitions, args.out)


if __name__ == "__main__":
    sys.exit(main())
