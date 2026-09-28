#!/usr/bin/env python3
"""Check every port.json: what a port component is, and where its version pin lives.

A port's upstream version was readable only by reading its build script, which
is why a question as ordinary as "which release is the mruby port pinned to"
took a grep through shell variables, and why `experiments/study/catalog.json`
could carry PostgreSQL 17.0 while the ported backend was 17.5 without anything
noticing. `port.json` states the version once, machine-readably, and names the
line that actually decides it; this checker reads that line and compares.

    check-ports.py                  check every port.json, exit 1 on any violation
    check-ports.py --self-test      prove the checker can FAIL, then check

The declaration never becomes a second source of truth: `pin_source.grep` must
appear verbatim in `pin_source.file`, and the declared version or commit must
appear inside that same text. Bump the build script and this check fails until
the declaration follows.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[2]
PORTS = REPO / "capstone/ports"
RENAMES = PORTS / "renames.json"

REQUIRED = {"program", "role", "title", "upstream", "targets"}
OPTIONAL = {"workload", "evidence", "corpora", "related", "note", "status"}
ROLES = {"full-application", "allocator-component", "platform-build", "domain-libc",
         "census"}
TARGETS = {"capstone-domain", "cheribsd-purecap", "linux-guest", "silicon", "native"}
PATH_FIELDS = ("evidence", "corpora", "related")


def check_port(decl, where):
    problems = []
    for key in sorted(REQUIRED - set(decl)):
        problems.append(f"{where}: missing required field {key!r}")
    for key in sorted(set(decl) - REQUIRED - OPTIONAL):
        problems.append(f"{where}: unknown field {key!r} -- add it to SCHEMA.md and to "
                        f"this checker, or drop it")
    if problems:
        return problems

    if decl["role"] not in ROLES:
        problems.append(f"{where}: role {decl['role']!r} is not one of {sorted(ROLES)}")
    for target in decl["targets"]:
        if target not in TARGETS:
            problems.append(f"{where}: target {target!r} is not one of {sorted(TARGETS)}")
    if not decl["targets"]:
        problems.append(f"{where}: targets is empty")

    for field in PATH_FIELDS:
        for item in decl.get(field, []):
            if not (REPO / item).exists():
                problems.append(f"{where}: {field} names {item}, which does not exist")

    upstream = decl["upstream"]
    if not isinstance(upstream, dict):
        return problems + [f"{where}: upstream is not an object"]
    for key in ("version", "pin_source"):
        if key not in upstream:
            problems.append(f"{where}: upstream has no {key!r}")
    if problems:
        return problems

    pins = upstream["pin_source"]
    pins = pins if isinstance(pins, list) else [pins]
    versions = upstream["version"]
    versions = versions if isinstance(versions, list) else [versions]
    commit = upstream.get("commit")

    for pin in pins:
        source = REPO / pin.get("file", "")
        if not source.is_file():
            problems.append(f"{where}: pin_source file {pin.get('file')!r} does not exist")
            continue
        text = source.read_text()
        needle = pin.get("grep", "")
        if needle not in text:
            problems.append(f"{where}: pin_source grep {needle!r} is not in "
                            f"{pin['file']} -- the pin moved, or the declaration is stale")
            continue
        for key in sorted(set(pin) - {"file", "grep", "version"}):
            problems.append(f"{where}: pin_source has unknown key {key!r}")
        # The pinned identity has to be IN the line the declaration points at,
        # otherwise the pointer proves nothing about the version it claims. A pin
        # may name its own version, for a component pinning more than one release
        # or encoding the release differently (SQLite's numeric SQLITE_VERSION).
        wanted = [pin["version"]] if "version" in pin else versions
        if not any(v in needle for v in wanted) and not (commit and commit[:7] in needle):
            problems.append(f"{where}: pin_source grep {needle!r} contains neither the "
                            f"declared version {wanted} nor the commit")
    return problems


def check_renames(ledger=None):
    """The path-history ledger, held to the tree and to the evidence it names.

    A rename is only finished when the old path is gone, the new one carries a
    declaration, and every archived bundle the ledger says still quotes the old
    path really does -- otherwise the ledger is a claim about provenance that
    provenance does not support.
    """
    problems = []
    if ledger is None:
        if not RENAMES.is_file():
            return [f"{RENAMES.relative_to(REPO)}: missing"]
        try:
            ledger = json.loads(RENAMES.read_text())
        except json.JSONDecodeError as exc:
            return [f"renames.json is not valid JSON: {exc}"]
    for key in ("schema_version", "note", "renames", "deferred"):
        if key not in ledger:
            problems.append(f"renames.json has no {key!r}")
    if problems:
        return problems
    for entry in ledger["renames"]:
        for key in ("from", "to", "date", "reason", "quoted_by"):
            if key not in entry:
                problems.append(f"renames.json: an entry has no {key!r}")
                break
        else:
            old, new = entry["from"], entry["to"]
            if (REPO / old).exists():
                problems.append(f"renames.json: {old} still exists, so it was not renamed")
            if not (REPO / new / "port.json").is_file():
                problems.append(f"renames.json: {new} has no port.json")
            for quoter in entry["quoted_by"]:
                target = REPO / quoter
                if not target.is_file():
                    problems.append(f"renames.json: quoted_by names {quoter}, which is gone")
                elif old.split("capstone/ports/")[-1] not in target.read_text():
                    problems.append(f"renames.json: {quoter} no longer quotes {old} -- either "
                                    f"the record was rewritten or the entry is stale")
    for entry in ledger["deferred"]:
        for key in ("from", "to", "reason"):
            if key not in entry:
                problems.append(f"renames.json: a deferred entry has no {key!r}")
                break
        else:
            if not (REPO / entry["from"]).exists():
                problems.append(f"renames.json: deferred {entry['from']} does not exist; it was "
                                f"moved without the ledger following")
    return problems


def self_test():
    """Prove the checker rejects corruption, on a copy of a real declaration."""
    victim = PORTS / "cpython/app/port.json"
    good = json.loads(victim.read_text())
    accepted = []
    for label, mutate in (
        ("a missing required field", lambda d: d.pop("role")),
        ("a role that is not in the contract", lambda d: d.update(role="something")),
        ("a version the pinned line does not contain",
         lambda d: d["upstream"].update(version="9.9.9")),
        ("a pin_source line that is not in the file",
         lambda d: d["upstream"]["pin_source"].update(grep="not in any file")),
        ("evidence that does not exist", lambda d: d.update(evidence=["capstone/nope"])),
    ):
        broken = copy.deepcopy(good)
        mutate(broken)
        if not check_port(broken, "self-test"):
            accepted.append(label)

    # The ledger is a claim about provenance, so it gets its own controls.
    ledger = json.loads(RENAMES.read_text())
    for label, mutate in (
        ("a rename whose old path is claimed to be gone but is not",
         lambda l: l["renames"][0].update(**{"from": "capstone/ports/cpython/pymalloc"})),
        ("a rename to a component with no declaration",
         lambda l: l["renames"][0].update(to="capstone/ports/cpython")),
        ("a quoted_by file that does not quote the old path",
         lambda l: l["renames"][0]["quoted_by"].append(
             "capstone/ports/whisper/ggml-context/port.json")),
        ("a deferred rename whose path is already gone",
         lambda l: l["deferred"][0].update(**{"from": "capstone/ports/perl/gone"})),
    ):
        broken = copy.deepcopy(ledger)
        mutate(broken)
        if not check_renames(broken):
            accepted.append(label)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        accepted = self_test()
        if accepted:
            print("CHECKER BROKEN: it accepted " + "; ".join(accepted), file=sys.stderr)
            return 2
        print("self-test: the checker rejected all nine corruptions")

    manifests = sorted(PORTS.rglob("port.json"))
    if not manifests:
        print("ERROR: no port.json found under", PORTS, file=sys.stderr)
        return 2

    problems, roles = check_renames(), {}
    for manifest in manifests:
        where = str(manifest.parent.relative_to(REPO))
        try:
            decl = json.loads(manifest.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{where}: port.json is not valid JSON: {exc}")
            continue
        problems += check_port(decl, where)
        roles[decl.get("role", "?")] = roles.get(decl.get("role", "?"), 0) + 1

    for line in problems:
        print("FAIL " + line, file=sys.stderr)
    summary = ", ".join(f"{n} {r}" for r, n in sorted(roles.items()))
    ledger = json.loads(RENAMES.read_text()) if RENAMES.is_file() else {"renames": [],
                                                                        "deferred": []}
    print(f"check-ports: {'BLOCKED' if problems else 'CLEAN'} -- {len(manifests)} "
          f"components ({summary}), {len(ledger['renames'])} renames and "
          f"{len(ledger['deferred'])} deferred in the ledger, {len(problems)} "
          f"problem{'' if len(problems) == 1 else 's'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
