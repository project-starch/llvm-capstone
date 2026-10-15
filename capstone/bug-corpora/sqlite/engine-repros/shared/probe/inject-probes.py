#!/usr/bin/env python3
"""Generate a probe amalgamation: one SQLite 3.22.0 amalgamation plus reachability probes.

    inject-probes.py --src <amalgamation dir> --out <new dir> [--sites FILE] [--only ID,...]

--src holds sqlite3.c, sqlite3.h and sqlite3ext.h. For the virtual Capstone arms that is the
amalgamation the recorded runs were built from (sha256 7fc92e19..., the output of
ports/sqlite/app/prepare-sources.sh); for CheriBSD it is the adapted sqlite3-cheri.c
(ports/sqlite/cheribsd/adapt-sqlite-cheri.py), copied in as sqlite3.c. The output directory gets
sqlite3.c with the probes and the two headers unchanged, so it can be passed to
build-virtual.py --amalgamation, or, renamed sqlite3-probe.c, to build-corpus-cheri.sh with PROBE=1.

The probe is lb_probe_decl.h and lb_probe_impl.h, beside this script, unchanged from the
CheriBSD arm's probe: memsys5 alloc/free hooks keep a shadow map of requested size and freed state,
and LB_HIT at a defect site reports kind=UAF or kind=OOB when the pointer it is given is in a freed
block or past the requested bytes. That is a judgement by ADDRESS: an over-read that lands in a
freed neighbour reads as UAF, one that lands in a live block reads as nothing. Where the class of
the defect is the question, a site can instead test the object itself (see site 33 in
probe-sites.tsv). Armed at run time by LB_SITE=<id>; LB_ARMED is printed at start and LB_SUMMARY
at exit, so a log without LB_ARMED is a build without the probe, not a site never reached.

Sites are located by function, then by anchor text inside that function's body, never by line
number: the amalgamations the arms build from are offset from one another, and a line number that
drifts still points at a line, just the wrong one. Every anchor, hook and marker must match
exactly once or nothing is written.

Scope: unpatched memsys5 (virtual-malloc, CheriBSD). The Sublet patch rewrites the memsys5
functions the hooks sit in, so build-virtual.py --sublet refuses this source (patch -F0); that is
intended. Blind spots of the address judgement: a pointer into a lookaside slot or a page-cache
bulk slot resolves to the one memsys5 block those pools were carved from, so it reports nothing.
"""
import argparse
import hashlib
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent

# The CheriBSD arm's hooks, unchanged.
HOOKS = [
    ("p = memsys5MallocUnsafe(nBytes);", "after", "    lb_on_alloc((void*)p, nBytes);"),
    ("  memsys5FreeUnsafe(pPrior);", "before", "  lb_on_free(pPrior);"),
    ("  if( nBytes<=nOld ){", "after", "    lb_on_alloc(pPrior, nBytes);"),
]
MEM5_MARKER = "#define mem5 GLOBAL(struct Mem5Global, mem5)"


def read_sites(path):
    """TSV: site_id, case, function definition line (prefix), anchor, kind, payload, [target].
    kind 'before' inserts payload as its own line above the anchor line; kind 'sub' replaces
    the anchor text inside its line with payload. target is the file the site belongs to and
    defaults to the amalgamation; a defect in an extension names that extension's file, and
    --extra says where to read and write it."""
    sites = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) not in (6, 7) or cols[4] not in ("before", "sub"):
            sys.exit(f"{path}:{n}: expected 6 or 7 tab-separated columns with kind before|sub")
        sites.append(dict(id=int(cols[0]), case=cols[1], func=cols[2], anchor=cols[3],
                          kind=cols[4], payload=cols[5],
                          target=(cols[6] if len(cols) == 7 and cols[6] else "sqlite3.c")))
    return sites


def place(lines, sites, fail):
    """Resolve every site's anchor, then edit bottom-up so earlier edits do not move later ones."""
    edits = []
    for s in sites:
        span, err = function_body(lines, s["func"])
        if err:
            fail.append(f"site {s['id']}: {err}")
            continue
        hits = [k for k in range(span[0], span[1]) if s["anchor"] in lines[k]]
        if len(hits) != 1:
            fail.append(f"site {s['id']}: anchor {s['anchor']!r} matched {len(hits)} lines in {s['func']!r}")
            continue
        if s["kind"] == "sub" and lines[hits[0]].count(s["anchor"]) != 1:
            fail.append(f"site {s['id']}: anchor {s['anchor']!r} is not unique on its line")
            continue
        edits.append((hits[0], s))
    for k, s in sorted(edits, key=lambda e: -e[0]):
        if s["kind"] == "before":
            lines.insert(k, s["payload"])
        else:
            lines[k] = lines[k].replace(s["anchor"], s["payload"])


def function_body(lines, func):
    """(start, end) line indexes of the one function whose definition line starts with FUNC;
    the body ends at the first line that is exactly '}' after it."""
    starts = [k for k, l in enumerate(lines) if l.startswith(func)]
    if len(starts) != 1:
        return None, f"definition {func!r} matched {len(starts)} lines, expected 1"
    for k in range(starts[0] + 1, len(lines)):
        if lines[k] == "}":
            return (starts[0], k), None
    return None, f"definition {func!r} has no closing brace"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--sites", type=pathlib.Path, default=HERE / "probe-sites.tsv")
    p.add_argument("--only", help="comma-separated site ids (default: every site in the file)")
    p.add_argument("--extra", action="append", default=[], metavar="NAME=IN:OUT",
                   help="A file other than the amalgamation that carries sites, for a defect in an "
                        "extension: NAME is the site's target column, IN the source to read and OUT "
                        "the copy to write. The copy gets the probe declarations prepended; the "
                        "implementation stays in the amalgamation, so both are built together. "
                        "Repeatable.")
    a = p.parse_args()

    src_c = a.src / "sqlite3.c"
    lines = src_c.read_text().split("\n")
    sites = read_sites(a.sites)
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        sites = [s for s in sites if s["id"] in keep]
        if {s["id"] for s in sites} != keep:
            sys.exit(f"--only names site ids not in {a.sites}: {sorted(keep - {s['id'] for s in sites})}")
    if not sites:
        sys.exit("no sites selected")
    fail = []
    extra = {}
    for spec in a.extra:
        name, _, paths = spec.partition("=")
        src, _, dst = paths.partition(":")
        if not (name and src and dst):
            sys.exit(f"--extra wants NAME=IN:OUT, got {spec!r}")
        extra[name] = (pathlib.Path(src), pathlib.Path(dst))
    named = {s["target"] for s in sites if s["target"] != "sqlite3.c"}
    if named - set(extra):
        sys.exit(f"sites target {sorted(named - set(extra))}, which no --extra names")

    # 1. Sites in the amalgamation.
    place(lines, [s for s in sites if s["target"] == "sqlite3.c"], fail)

    # 2. Allocator hooks, by unique anchor text.
    for anchor, where, payload in HOOKS:
        hits = [k for k, l in enumerate(lines) if anchor in l]
        if len(hits) != 1:
            fail.append(f"hook {anchor!r} matched {len(hits)} lines, expected 1")
            continue
        lines.insert(hits[0] + 1 if where == "after" else hits[0], payload)

    # 3. The probe implementation, immediately after memsys5's globals.
    hits = [k for k, l in enumerate(lines) if l.strip() == MEM5_MARKER]
    if len(hits) != 1:
        fail.append(f"mem5 marker matched {len(hits)} lines, expected 1")
    else:
        lines.insert(hits[0] + 1, "\n" + (HERE / "lb_probe_impl.h").read_text() + "\n")

    if fail:
        print(f"REFUSING TO WRITE -- {len(fail)} problem(s):", file=sys.stderr)
        for f in fail:
            print("  " + f, file=sys.stderr)
        return 1

    # 4. Any other file that carries sites, written beside the amalgamation copy.
    written = []
    for name, (src, dst) in extra.items():
        mine = [s for s in sites if s["target"] == name]
        if not mine:
            continue
        extra_lines = src.read_text().split("\n")
        place(extra_lines, mine, fail)
        written.append((name, src, dst, extra_lines))

    a.out.mkdir(parents=True, exist_ok=False)
    for name, src, dst, extra_lines in written:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((HERE / "lb_probe_decl.h").read_text() + "\n" + "\n".join(extra_lines))
    out_c = a.out / "sqlite3.c"
    out_c.write_text((HERE / "lb_probe_decl.h").read_text() + "\n" + "\n".join(lines))
    for name in ("sqlite3.h", "sqlite3ext.h"):
        shutil.copy2(a.src / name, a.out / name)
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    print(f"source  {src_c}  sha256 {sha(src_c)}")
    print(f"wrote   {out_c}  sha256 {sha(out_c)}")
    for s in sorted(sites, key=lambda s: s["id"]):
        print(f"site {s['id']:>3}  case {s['case']}  in {s['func'].split('(')[0].split()[-1]}  "
              f"{s['kind']} {s['anchor']!r}")
    for name, src, dst, _ in written:
        print(f"wrote   {dst}  sha256 {sha(dst)}  (from {src})")
    print(f"hooks {len(HOOKS)}; arm one site at run time with LB_SITE=<id>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
