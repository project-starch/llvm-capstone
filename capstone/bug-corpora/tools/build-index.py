#!/usr/bin/env python3
"""Generate INDEX.md and index.json: every piece of bug material, in one table.

Bug material lives in four places -- the corpora here, the cross-language corpora
in `xlang/`, our own silicon defects in `capstone/tests/fpga-repros/`, and our own
compiler and runtime defects in `docs/ref/ISSUES.md` -- and until now nothing
listed all four. The hand-written counts that tried drifted: the umbrella README
said memcached had two cases when it had five and did not mention the Wireshark
corpus at all, and `docs/ref/paper-bug-inventory.md` covers 30 of the 107 cases.

So this index is generated and the numbers in it are never typed. It reads each
corpus's `corpus.json`, each port component's `port.json`, and counts what is on
disk for the two sorts that carry no declaration, because a README's own first
heading is a better source than a manifest copying it.

    build-index.py            write INDEX.md and index.json
    build-index.py --check    fail if either file is out of date (for a gate)

Run `check-corpus.py` and `check-ports.py` first: this tool trusts the
declarations, and those two are what make them true.
"""

import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
CORPORA = TOOLS.parent
REPO = CORPORA.parents[1]
INDEX_MD = CORPORA / "INDEX.md"
INDEX_JSON = CORPORA / "index.json"
PORTS_MD = REPO / "capstone/ports/INDEX.md"
RENAMES = REPO / "capstone/ports/renames.json"
FPGA = REPO / "capstone/tests/fpga-repros"
ISSUES = REPO / "capstone/docs/ref/ISSUES.md"
ISSUES_ARCHIVE = REPO / "capstone/docs/ref/ISSUES-ARCHIVE.md"
ISSUE_HEADING = re.compile(r"^#+ *(?:C|S|R|I|M|Q)-[0-9]+")


def read_json(path):
    return json.loads(path.read_text())


def corpora():
    found = []
    for base in ("capstone/bug-corpora", "xlang"):
        for path in sorted((REPO / base).rglob("corpus.json")):
            decl = read_json(path)
            decl["where"] = str(path.parent.relative_to(REPO))
            found.append(decl)
    return found


def ports():
    found = []
    for path in sorted((REPO / "capstone/ports").rglob("port.json")):
        decl = read_json(path)
        decl["where"] = str(path.parent.relative_to(REPO))
        found.append(decl)
    return found


def fpga_defects():
    """Our own silicon defects. Each folder IS the report, so the folder is the source."""
    out = []
    if not FPGA.is_dir():
        return out
    for folder in sorted(d for d in FPGA.iterdir() if d.is_dir() and d.name != "ARCHIVED"):
        readme = folder / "00-README.md"
        title = ""
        if readme.is_file():
            for line in readme.read_text().splitlines():
                if line.startswith("#"):
                    title = line.lstrip("# ").strip()
                    break
        out.append({"id": folder.name, "title": title,
                    "has_sha256sums": (folder / "SHA256SUMS").is_file()})
    return out


def issue_counts():
    def count(path):
        if not path.is_file():
            return None
        return sum(1 for line in path.read_text().splitlines() if ISSUE_HEADING.match(line))
    return {"open": count(ISSUES), "resolved": count(ISSUES_ARCHIVE)}


def fixtures():
    return sorted(str(p.parent.relative_to(REPO / "capstone/ports"))
                  for p in (REPO / "capstone/ports").rglob("security-tests")
                  if p.is_dir())


def live_cell(decl):
    if not decl.get("live_in_pin_recorded"):
        return "not recorded"
    expect = decl.get("expect_live_in_pin", {})
    order = ("true", "false", "not_asserted")
    labels = {"true": "live", "false": "fixed before the pin", "not_asserted": "not asserted"}
    parts = [f"{expect[k]} {labels[k]}" for k in order if expect.get(k)]
    return ", ".join(parts) or "none"


def version_of(decl):
    upstream = decl.get("upstream") or {}
    version = upstream.get("version")
    if isinstance(version, list):
        return ", ".join(version)
    return version or "per row"


def gaps(all_corpora, all_ports):
    """Every gap the declarations imply, stated rather than left to be noticed."""
    out = []
    for decl in all_corpora:
        if decl["status"] == "planned":
            out.append(f"`{decl['where']}` is declared and empty: {decl['cases']} cases, "
                       f"status planned.")
        if decl["status"] == "triaged" and not decl["cases"]:
            out.append(f"`{decl['where']}` is triage only -- nothing built.")
        if decl.get("expect_provenance") is not None:
            out.append(f"`{decl['where']}`: {decl['expect_provenance']} of {decl['cases']} "
                       f"cases have a PROVENANCE.md.")
        if not decl.get("live_in_pin_recorded"):
            out.append(f"`{decl['where']}` does not record live_in_pin: "
                       f"{decl['live_in_pin_note']}")
        elif decl.get("expect_live_in_pin", {}).get("not_asserted"):
            count = decl["expect_live_in_pin"]["not_asserted"]
            out.append(f"`{decl['where']}`: liveness deliberately not asserted for "
                       f"{count} case{'' if count == 1 else 's'}.")
        if decl["cases"] and not decl.get("inventory"):
            out.append(f"`{decl['where']}` has no triage inventory under docs/ref/.")
        if decl["cases"] and not decl.get("evidence"):
            out.append(f"`{decl['where']}` commits no result bundle of its own.")
    linked = {c for port in all_ports for c in port.get("corpora", [])}
    for port in all_ports:
        if port["role"] == "full-application" and not port.get("corpora"):
            out.append(f"`{port['where']}` is a complete application with no corpus.")
    for decl in all_corpora:
        if decl["where"].startswith("capstone/bug-corpora") and decl["where"] not in linked:
            out.append(f"`{decl['where']}` is not referenced by any port.json.")
    return out


def render(data):
    lines = []
    add = lines.append
    add("# Bug material index")
    add("")
    add("**Generated by `tools/build-index.py`. Do not edit by hand** -- every number here "
        "is read off a declaration or off the tree, because the hand-written counts this "
        "replaces had all drifted.")
    add("")
    add("    python3 capstone/bug-corpora/tools/check-corpus.py --self-test")
    add("    python3 capstone/bug-corpora/tools/check-ports.py --self-test")
    add("    python3 capstone/bug-corpora/tools/build-index.py")
    add("")
    add("## The four sorts of bug material")
    add("")
    add("| sort | where | count | what it is |")
    add("|---|---|---:|---|")
    add(f"| third-party defects, as cases | `capstone/bug-corpora/` | {data['cases_here']} | "
        f"one directory per case, `case.json` + `PROVENANCE.md`, a runner per corpus |")
    add(f"| the same, cross-language | `xlang/` | {data['cases_xlang']} | distilled C shims "
        f"with their own row tables and measured columns |")
    add(f"| our own silicon defects | `capstone/tests/fpga-repros/` | "
        f"{len(data['fpga'])} | one self-contained report per defect, the folder is the report |")
    add(f"| our own compiler and runtime defects | `docs/ref/ISSUES.md` | "
        f"{data['issues']['open']} open, {data['issues']['resolved']} resolved | the registry, "
        f"not reproduced cases |")
    add("")
    add("Protection fixtures under `capstone/ports/*/security-tests/` are **not** bug "
        "material and are counted nowhere above: they are this project's own oracles. "
        f"{len(data['fixtures'])} components have them: "
        + ", ".join(f"`{f}`" for f in data['fixtures']) + ".")
    add("")
    add("## Every corpus")
    add("")
    add("| corpus | program | pinned version | cases | liveness in that pin | advisories | "
        "schema | status |")
    add("|---|---|---|---:|---|---:|---|---|")
    for decl in data["corpora"]:
        add(f"| [`{decl['where']}`]({data['rel'][decl['where']]}) | {decl['program']} | "
            f"{version_of(decl)} | {decl['cases']} | {live_cell(decl)} | "
            f"{len(decl.get('advisories', []))} | {decl['case_schema']} | {decl['status']} |")
    add("")
    add(f"**{data['cases_total']} cases in {len(data['corpora'])} corpora**, of which "
        f"{data['live_true']} are recorded live in the version their corpus pins, "
        f"{data['live_false']} were fixed upstream before it, and {data['live_not']} carry an "
        f"explicit decision not to assert liveness. "
        f"{data['advisories_total']} advisories are cited across all corpora.")
    add("")
    add("## Every port component, and what bug material it has")
    add("")
    add("| component | role | pinned version | targets | corpora | cases |")
    add("|---|---|---|---|---|---:|")
    for port in data["ports"]:
        names = port.get("corpora", [])
        cases = sum(data["cases_by_corpus"].get(name, 0) for name in names)
        add(f"| `{port['where']}` | {port['role']} | {version_of(port)} | "
            f"{', '.join(port['targets'])} | "
            f"{', '.join(f'`{n.split(chr(47))[-1]}`' for n in names) or '--'} | "
            f"{cases if names else 0} |")
    add("")
    add("## Gaps, as the declarations state them")
    add("")
    for gap in data["gaps"]:
        add(f"- {gap}")
    add("")
    add("## Our own silicon defects")
    add("")
    add(f"{len(data['fpga'])} folders under `capstone/tests/fpga-repros/`, plus `ARCHIVED/`. "
        f"{sum(1 for d in data['fpga'] if d['has_sha256sums'])} carry a `SHA256SUMS`, which is "
        f"what lets a board result be cited by image hash rather than by label.")
    add("")
    for defect in data["fpga"]:
        add(f"- `{defect['id']}` -- {defect['title'] or 'no heading in 00-README.md'}")
    add("")
    return "\n".join(lines) + "\n"


ROLE_ORDER = ["full-application", "allocator-component", "platform-build", "domain-libc",
              "census"]
ROLE_BLURB = {
    "full-application": "the whole program runs, in a domain or on CheriBSD purecap",
    "allocator-component": "one allocator, driven or replayed; not the application",
    "platform-build": "the same release built for another platform, without an adapter",
    "domain-libc": "the libc the domain images link against",
    "census": "what compiles, and what stops the rest",
}


def render_ports(data):
    """The port tree's own generated surface: what each component is, and its pin."""
    lines = []
    add = lines.append
    add("# Port components")
    add("")
    add("**Generated by `../bug-corpora/tools/build-index.py`. Do not edit by hand** -- every "
        "row comes from a component's own `port.json`, whose version pin the checker verifies "
        "against the build recipe that decides it.")
    add("")
    add("    python3 capstone/bug-corpora/tools/check-ports.py --self-test")
    add("    python3 capstone/bug-corpora/tools/build-index.py")
    add("")
    add("**A component is named for what it is:** `app/` for the complete application, "
        "`<boundary>/` for one allocator, `cheribsd/` for the same release built for another "
        "platform. `role` states it as well, so nothing depends on reading the path.")
    add("")
    deferred = data["renames"]["deferred"]
    if deferred:
        add(f"{len(deferred)} components do not follow that rule yet. Each is listed under "
            f"**Path history** below with the reason, because in every case the move would "
            f"cost more than the name is worth today.")
        add("")
    for role in ROLE_ORDER:
        rows = [p for p in data["ports"] if p["role"] == role]
        if not rows:
            continue
        add(f"## {role} -- {ROLE_BLURB[role]}")
        add("")
        add("| component | pinned version | runs on | workload | corpora |")
        add("|---|---|---|---|---|")
        for port in rows:
            names = port.get("corpora", [])
            add(f"| `{port['where'].split('capstone/ports/')[-1]}` | {version_of(port)} | "
                f"{', '.join(port['targets'])} | {port.get('workload', '--')} | "
                f"{', '.join(f'`{n.split(chr(47))[-1]}`' for n in names) or '--'} |")
        add("")
    add("## Path history")
    add("")
    add("A component's directory has been renamed where the old name did not say what it is. "
        "**Archived result bundles keep quoting the old path on purpose** -- a "
        "`build-manifest.json` records which directory a measured binary was built from, so it "
        "is evidence and is never rewritten -- and `docs/history/` is append-only for the same "
        "reason. Resolve an old path here; `renames.json` is the machine-readable form, and "
        "check-ports.py verifies that each old path is gone, each new one carries a "
        "declaration, and each file below really still quotes the old name.")
    add("")
    add("| old path | new path | when | archived files still quoting it |")
    add("|---|---|---|---:|")
    for entry in data["renames"]["renames"]:
        add(f"| `{entry['from']}` | `{entry['to']}` | {entry['date']} | "
            f"{len(entry['quoted_by'])} |")
    add("")
    if deferred:
        add("Deferred, with the reason:")
        add("")
        for entry in deferred:
            add(f"- `{entry['from']}` &rarr; `{entry['to']}` -- {entry['reason']}")
        add("")
    add("Which defects each corpus holds, and whether they are live in the version above, is in "
        "the [bug-material index](../bug-corpora/INDEX.md).")
    add("")
    return "\n".join(lines)


def collect():
    all_corpora, all_ports = corpora(), ports()
    cases_by_corpus = {d["where"]: d["cases"] for d in all_corpora}
    here = sum(v for k, v in cases_by_corpus.items() if k.startswith("capstone/bug-corpora"))
    data = {
        "corpora": all_corpora,
        "ports": all_ports,
        "fpga": fpga_defects(),
        "issues": issue_counts(),
        "fixtures": fixtures(),
        "cases_by_corpus": cases_by_corpus,
        "cases_here": here,
        "cases_xlang": sum(cases_by_corpus.values()) - here,
        "cases_total": sum(cases_by_corpus.values()),
        "live_true": sum(d.get("expect_live_in_pin", {}).get("true", 0) for d in all_corpora),
        "live_false": sum(d.get("expect_live_in_pin", {}).get("false", 0) for d in all_corpora),
        "live_not": sum(d.get("expect_live_in_pin", {}).get("not_asserted", 0)
                        for d in all_corpora),
        "advisories_total": sum(len(d.get("advisories", [])) for d in all_corpora),
        "renames": read_json(RENAMES) if RENAMES.is_file() else {"renames": [], "deferred": []},
    }
    data["gaps"] = gaps(all_corpora, all_ports)
    # Links are written relative to this file, so they work on the forge and on disk.
    data["rel"] = {}
    for decl in all_corpora:
        target = REPO / decl["where"]
        try:
            data["rel"][decl["where"]] = str(target.relative_to(CORPORA))
        except ValueError:
            data["rel"][decl["where"]] = "../../" + decl["where"]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the written files differ from what this would write")
    args = parser.parse_args()

    data = collect()
    if not data["corpora"]:
        print("ERROR: no corpus.json found", file=sys.stderr)
        return 2
    markdown = render(data)
    ports_markdown = render_ports(data)
    payload = json.dumps({k: v for k, v in data.items() if k != "rel"}, indent=2) + "\n"

    if args.check:
        stale = [str(p.relative_to(REPO)) for p, want in ((INDEX_MD, markdown),
                                                          (INDEX_JSON, payload),
                                                          (PORTS_MD, ports_markdown))
                 if not p.is_file() or p.read_text() != want]
        if stale:
            print("build-index: STALE -- " + ", ".join(stale) +
                  " differ from the declarations; re-run build-index.py", file=sys.stderr)
            return 1
        print("build-index: current")
        return 0

    INDEX_MD.write_text(markdown)
    INDEX_JSON.write_text(payload)
    PORTS_MD.write_text(ports_markdown)
    print(f"wrote {INDEX_MD.relative_to(REPO)}, {INDEX_JSON.relative_to(REPO)} and "
          f"{PORTS_MD.relative_to(REPO)}: "
          f"{len(data['corpora'])} corpora, {data['cases_total']} cases, "
          f"{len(data['ports'])} port components, {len(data['fpga'])} silicon defects, "
          f"{len(data['gaps'])} gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
