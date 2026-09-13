#!/usr/bin/env python3
"""Count what a Sublet port cost, one hunk at a time, and refuse to count a hunk nobody classified.

WHY A TOOL AND NOT A TALLY. The port-effort experiment says the counting rule has to stand before
the first ported line and must never be adjusted at the result, because a number produced after
the fact cannot be checked. Two ports were finished before any rule existed. What rescues them is
that each keeps its Sublet change as a standalone patch, so the hunks are still exactly there --
but only a tool that refuses an unclassified hunk can make the count mean the same thing for a
port already done and a port not yet started.

THE TWO FACTORS ARE THE EXPERIMENT'S, NOT THIS TOOL'S. A7 fixes them: a changed line has a CLASS
(hierarchy annotation, metadata layout, workaround) and it sits at a LEVEL (the allocator's own
implementation, or application code above its interface). Both are reported because both are a
hypothesis. H4 says metadata layout carries most of the lines, because one rule drives them:
metadata may no longer live in freed memory, since freed memory is revoked. H1 says the
application level is zero, and a line there is the finding, not a rounding error.

    hierarchy    the block arriving linear under a handle the level below keeps, the carve, the
                 hand-out, the give back, the senior revoke that kills a pool
    metadata     free-list links, control bytes and handles moving out of memory that is handed
                 away, into a table beside the pool
    workaround   a line present only because a compiler or an RTL defect demands it, which is a
                 class of its own so that it never flatters or inflates the other two

    allocator    inside the allocator's own implementation
    application  above its interface

THE RULE. A patch file is the unit. Every hunk in it carries a class and a level, given in a
sibling file named <patch>.classes, one line per hunk:

    <file>:<new_start>   <class>   <level>   <a note saying why>

A hunk with no line is an error, not a zero: the tool exits 2 and names it. A class or level
outside the vocabulary is an error for the same reason. Added and removed lines are counted
separately and never netted, because a port that deletes forty lines and writes forty is not a
port of zero.

A HUNK THAT DOES TWO THINGS takes the class of the change that drove it, and its note names the
other. That tie-break is part of the rule and is written here rather than decided per port,
because deciding it at the patch is how a share becomes whatever the author needed it to be.

CODE, COMMENT AND BLANK ARE THREE COLUMNS, not two, and the third is why. One port counted code
and comment apart and the other did not, and when both were counted here the code figures agreed
to the line while the comment figures did not: 261 against 268 added, 79 against 123 removed. The
whole gap was blank lines, 7 and 44 of them, which that port had counted as comment. Neither
choice is wrong and picking one silently would have made two ports incomparable for a second time,
so a blank line gets a column of its own and nobody has to choose.

A line counts as comment when what it starts with, after leading space, is `*`, `//` or `/*`. That
is a prefix test and it is stated here rather than implied: it cannot see a block comment whose
continuation lines begin with something else, so a codebase that does not use the `**` house style
needs the test revisited rather than the number trusted. Both allocators counted so far use it.
"""
import argparse
import pathlib
import re
import sys
from collections import defaultdict

CLASSES = ("hierarchy", "metadata", "workaround")
LEVELS = ("allocator", "application")
REFUSED = 2   # every refusal, so a caller can tell "the input is incomplete" from "it crashed"

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
NEWFILE_RE = re.compile(r"^\+\+\+ (?:b/)?(\S+)")


def refuse(message):
    print(message, file=sys.stderr)
    sys.exit(REFUSED)


def is_comment(text):
    stripped = text.strip()
    return stripped.startswith(("*", "//", "/*"))


def hunks(patch_path):
    """Every hunk as (id, {added,removed} x {code,comment}), in file order."""
    out, current, newfile = [], None, None
    counts = None
    for line in patch_path.read_text(errors="replace").splitlines():
        m = NEWFILE_RE.match(line)
        if m:
            newfile = m.group(1)
            continue
        m = HUNK_RE.match(line)
        if m:
            if current:
                out.append((current, counts))
            if newfile is None:
                refuse(f"{patch_path}: a hunk before any +++ line, so it has no file")
            current = f"{newfile}:{m.group(1)}"
            counts = {"added_code": 0, "added_comment": 0, "added_blank": 0,
                      "removed_code": 0, "removed_comment": 0, "removed_blank": 0}
            continue
        if current is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:]
        elif line.startswith("-") and not line.startswith("---"):
            body = line[1:]
        else:
            continue
        kind = "blank" if not body.strip() else ("comment" if is_comment(body) else "code")
        counts[("added_" if line[0] == "+" else "removed_") + kind] += 1
    if current:
        out.append((current, counts))
    return out


def classes_for(patch_path):
    path = patch_path.with_name(patch_path.name + ".classes")
    if not path.exists():
        refuse(f"no {path}: every hunk needs a class and a level before it can be counted")
    table = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 3:
            refuse(f"{path}:{lineno}: want '<file>:<line> <class> <level> <note>'")
        hunk_id, cls, level = parts[0], parts[1], parts[2]
        if cls not in CLASSES:
            refuse(f"{path}:{lineno}: class {cls!r} is not one of {', '.join(CLASSES)}")
        if level not in LEVELS:
            refuse(f"{path}:{lineno}: level {level!r} is not one of {', '.join(LEVELS)}")
        table[hunk_id] = (cls, level)
    return path, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patches", nargs="+")
    ap.add_argument("--label", default="", help="a name for the allocator, for the heading")
    args = ap.parse_args()

    total = defaultdict(lambda: {"added_code": 0, "added_comment": 0, "added_blank": 0,
                                 "removed_code": 0, "removed_comment": 0, "removed_blank": 0})
    unclassified = []
    for p in (pathlib.Path(x) for x in args.patches):
        found = hunks(p)
        classes_path, table = classes_for(p)
        for hunk_id, counts in found:
            entry = table.get(hunk_id)
            if entry is None:
                unclassified.append((hunk_id, p, classes_path))
                continue
            for k, v in counts.items():
                total[entry][k] += v
        stale = set(table) - {h for h, _ in found}
        for hunk_id in sorted(stale):
            print(f"warning: {classes_path} classifies {hunk_id}, which the patch no longer has",
                  file=sys.stderr)

    if unclassified:
        print("REFUSED: these hunks carry no class, so nothing was counted:", file=sys.stderr)
        for hunk_id, p, classes_path in unclassified:
            print(f"  {hunk_id}   from {p}, wanted in {classes_path}", file=sys.stderr)
        return REFUSED

    if args.label:
        print(f"== {args.label}")
    cols = ("added_code", "added_comment", "added_blank",
            "removed_code", "removed_comment", "removed_blank")
    print(f"{'class':<11} {'level':<12}" + "".join(f"{c:>9}" for c in
          ("+code", "+comment", "+blank", "-code", "-comment", "-blank")))
    sums = {k: 0 for k in cols}
    metadata_code = hierarchy_code = 0
    for cls in CLASSES:
        for level in LEVELS:
            c = total[(cls, level)]
            if any(c.values()):
                print(f"{cls:<11} {level:<12}" + "".join(f"{c[k]:>9}" for k in cols))
            for k in sums:
                sums[k] += c[k]
            if cls == "metadata":
                metadata_code += c["added_code"]
            if cls == "hierarchy":
                hierarchy_code += c["added_code"]
    print(f"{'TOTAL':<24}" + "".join(f"{sums[k]:>9}" for k in cols))
    net = sums["added_code"] - sums["removed_code"]
    print(f"code lines net: {net:+d}" + ("   the port leaves the allocator smaller"
                                         if net < 0 else ""))
    app = {k: 0 for k in sums}
    for cls in CLASSES:
        for k in app:
            app[k] += total[(cls, "application")][k]
    print(f"application code, which the claim says is zero: "
          f"+{app['added_code']}/-{app['removed_code']}")
    if sums["added_code"]:
        share = 100 * metadata_code // sums["added_code"]
        verdict = ("the largest class, as the hypothesis says"
                   if metadata_code > hierarchy_code
                   else "NOT the largest: hierarchy annotation is "
                        f"{100 * hierarchy_code // sums['added_code']}%, which refutes it")
        print(f"metadata layout is {share}% of the added code, {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
