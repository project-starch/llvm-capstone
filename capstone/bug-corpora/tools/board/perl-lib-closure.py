#!/usr/bin/env python3
"""perl-lib-closure.py --lib LIB --out FILE REQUIRES.json...: the perl library files the release-differential cases
can load, for a board boot whose payload cannot carry the whole lib (5.36.3's lib is 7.5 MB as tar.gz, 26.5 MB of it
raw unicore/, against ~8.4 MB of headroom with a 4.8 MB image).

The closure is the union of:
- every name perl-lib-requires.py measured in the guest (the core's implicit loads included), over every REQUIRES
  file given (both arms, in practice);
- the static closure of the corpus harness and every trigger: literal `use`/`no`/`require` module names, `use base`/
  `use parent` lists and literal `require`/`do` file names;
- expanded transitively through each closure file's own literal names.
A name is kept when it is a file under LIB. One that is not (the harness's shim.pl, a case's own path, a module the
lib does not ship) is reported and dropped, since the full library cannot supply it either.

It over-approximates the static loads and sees only the dynamic loads the measured runs reached. A run on it is
therefore valid only after a QEMU A/B, the closure against the full lib, shows every reading equal.
Writes FILE (one path per line, relative to LIB, sorted) and FILE.json (sizes, sources and which names came from
where). Exit 2 when a REQUIRES file has no runs: no data is an error, not an empty closure.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[2] / "perl/release-differential"
USE = re.compile(r'^\s*(?:use|no|require)\s+([A-Za-z_]\w*(?:::\w+)*)\b', re.M)
FILE = re.compile(r'\b(?:require|do)\s+["\']([^"\'$@]+\.p[lm])["\']')
BASE = re.compile(r'^\s*use\s+(?:base|parent)\s+(?:-norequire\s*,\s*)?'
                  r'(?:qw\s*[(\[{/]([^)\]}/]*)[)\]}/]|["\']([\w:]+)["\'])', re.M)


def names(text):
    out = {m.group(1).replace("::", "/") + ".pm" for m in USE.finditer(text)}
    out |= {m.group(1) for m in FILE.finditer(text)}
    for m in BASE.finditer(text):
        out |= {n.replace("::", "/") + ".pm" for n in (m.group(1) or m.group(2) or "").split()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", type=Path, required=True, help="the build's src/perl-5.36.3/lib")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("requires", type=Path, nargs="+", help="perl-lib-requires.py's OUT/requires.json")
    a = p.parse_args()
    seeds = [CORPUS / "harness/shim.pl"] + sorted(CORPUS.glob("[0-9][0-9]_*/trigger.pl"))
    origin, dropped = {}, {}
    todo = [(n, "static") for s in seeds for n in sorted(names(s.read_text(errors="replace")))]
    for r in a.requires:
        runs = json.loads(r.read_text()).get("runs") or {}
        if not runs:
            print(f"perl-lib-closure: {r} has no runs (no data is an error)", file=sys.stderr)
            return 2
        todo += [(n, f"measured:{r.parent.parent.name}:{case}") for case, run in sorted(runs.items())
                 for n in run["requires"]]
    while todo:
        n, why = todo.pop(0)
        if n in origin:
            continue
        if not (a.lib / n).is_file():
            dropped.setdefault(n, why)
            continue
        origin[n] = why
        todo += [(m, f"via:{n}") for m in sorted(names((a.lib / n).read_text(errors="replace")))]
    files = sorted(origin)
    a.out.write_text("".join(f + "\n" for f in files))
    raw = sum((a.lib / f).stat().st_size for f in files)
    gz = sum(len(gzip.compress((a.lib / f).read_bytes(), 9)) for f in files)
    measured = sorted({n for n, w in origin.items() if w.startswith("measured")})
    (a.out.parent / (a.out.name + ".json")).write_text(json.dumps({
        "closure_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(), "files": len(files), "raw_bytes": raw,
        "gzip_bytes_per_file_sum": gz, "requires": [{"path": re.sub(r"^/mnt/hdd/[^/]+/", "<mnt-hdd>/", str(r.resolve())),
                      "sha256": hashlib.sha256(r.read_bytes()).hexdigest()} for r in a.requires], "tool": hashlib.sha256(
            HERE.read_bytes()).hexdigest(), "origin": origin, "dropped": dropped, "measured_first": measured},
        indent=1) + "\n")
    print(f"closure {len(files)} files, {raw} B raw, ~{gz} B gzipped; {len(measured)} first seen by measurement; "
          f"dropped (not in lib) {sorted(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
