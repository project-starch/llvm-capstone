#!/usr/bin/env python3
"""What a port cost BESIDE the allocator, file by file, and refuse a file nobody classified.

    port-ledger.py <port-dir> [--skeleton]

WHY A SECOND TOOL. port-effort.py counts the allocator patch, which is the number the port-effort
claim is about. It is not the number a reader asks for next. Getting an allocator to run in a
domain also costs a shim for the headers it cannot have, a level below it can carve from, and a
harness to drive it, and none of that is in the patch. I1 asks for those counted SEPARATELY, and
separately is the point: a port that needed six hundred lines of harness did not change the
allocator six hundred times.

THE BUCKETS, and they are the experiment's words:

    allocator   the classified patch. Counted by port-effort.py against the .classes file and
                NOT recounted here, so the two tools cannot disagree about it.
    compat      what replaces upstream's own headers and libc. It exists because the target is
                freestanding and would not exist on the program's own platform.
    glue        the runtime the port needs in a domain: the level below it carves from, the
                entry the monitor calls, the stubs a freestanding build leaves undefined.
    test        drivers, guests and probes. The harness that asks the questions, which is not
                the port and must not be counted as if it were.

THE RULE. Every file under the port directory carries a line in ledger.manifest:

    <path relative to the port dir>   <bucket>   <a note saying why>

A file with no line is an error and the tool exits 2 naming it. A line whose file is gone is the
same error from the other side. --skeleton prints a manifest with every file UNCLASSIFIED, which
is where a new port starts.

Code, comment and blank are three columns, for the reason port-effort.py gives: one port counted
comment and blank together and the two figures then disagreed by exactly the blank lines.
"""
import sys, pathlib, re

BUCKETS = ("allocator", "compat", "glue", "test")
COUNTED = (".c", ".h")
PINNED = (".patch",)   # named, hashed, never recounted


def classify_lines(path):
    code = comment = blank = 0
    in_block = False
    for raw in path.read_text(errors="replace").splitlines():
        s = raw.strip()
        if not s:
            blank += 1
            continue
        if in_block:
            comment += 1
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("/*"):
            comment += 1
            if "*/" not in s[2:]:
                in_block = True
            continue
        if s.startswith("//"):
            comment += 1
            continue
        code += 1
    return code, comment, blank


def files_of(port):
    return sorted(p for p in port.rglob("*")
                  if p.is_file() and p.suffix in COUNTED)


def pins_of(listed, port):
    """The classified patches, named and hashed. I1 asks for the revisions pinned, and a hash of
    the patch is the pin that survives a rebase: the commit that carried it can be rewritten, the
    bytes it counted cannot."""
    import hashlib
    out = []
    for path, bucket in sorted(listed.items()):
        if bucket == "allocator":
            f = port / path
            h = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
            out.append((path, h))
    return out


def main(argv):
    if not argv:
        sys.exit(__doc__)
    port = pathlib.Path(argv[0]).resolve()
    if not port.is_dir():
        sys.exit("port-ledger: %s is not a directory" % port)
    present = {str(p.relative_to(port)) for p in files_of(port)}
    patches = {str(p.relative_to(port)) for p in port.rglob("*")
               if p.is_file() and p.suffix in PINNED}

    if "--skeleton" in argv:
        print("# %s: what the adaptation cost beside the allocator, for I1." % port.name)
        print("# One line per file: <path>  <bucket>  <why>. Buckets: %s." % ", ".join(BUCKETS))
        for f in sorted(present):
            print("%-52s UNCLASSIFIED   " % f)
        return 0

    manifest = port / "ledger.manifest"
    if not manifest.exists():
        sys.exit("port-ledger: no %s. Start with --skeleton." % manifest)

    listed, bad = {}, []
    for n, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            bad.append("line %d: %s" % (n, line.strip()))
            continue
        path, bucket = parts[0], parts[1]
        if bucket not in BUCKETS:
            bad.append("line %d: bucket %r is not one of %s" % (n, bucket, ", ".join(BUCKETS)))
            continue
        listed[path] = bucket

    missing = sorted(present - set(listed))
    gone = sorted(set(listed) - present - patches)
    if bad or missing or gone:
        for b in bad:
            print("port-ledger: %s" % b, file=sys.stderr)
        for m in missing:
            print("port-ledger: %s has no line in ledger.manifest" % m, file=sys.stderr)
        for g in gone:
            print("port-ledger: ledger.manifest names %s, which is not there" % g, file=sys.stderr)
        return 2

    tot = {b: [0, 0, 0, 0] for b in BUCKETS}
    for path, bucket in listed.items():
        if bucket == "allocator":
            continue                      # pinned below, never counted here
        c, m, bl = classify_lines(port / path)
        t = tot[bucket]
        t[0] += c; t[1] += m; t[2] += bl; t[3] += 1

    print("%-12s %6s %6s %7s %6s" % ("bucket", "files", "code", "comment", "blank"))
    for b in BUCKETS:
        if tot[b][3]:
            print("%-12s %6d %6d %7d %6d" % (b, tot[b][3], tot[b][0], tot[b][1], tot[b][2]))
    beside = sum(tot[b][0] for b in ("compat", "glue", "test"))
    print("%-12s %6s %6d %7s %6s" % ("beside", "", beside, "", ""))
    pins = pins_of(listed, port)
    if pins:
        print("\nthe classified patch, pinned by its own bytes:")
        for path, h in pins:
            print("  %-44s %s" % (path, h[:32] if h else "MISSING"))
    print("\nThe allocator patch is counted by port-effort.py against its .classes file and not")
    print("here, so the two tools cannot disagree about the number the claim is about.")
    return 0


sys.exit(main(sys.argv[1:]))
