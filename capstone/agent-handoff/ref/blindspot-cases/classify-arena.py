#!/usr/bin/env python3
"""Assign each bug row the SQLite arena its object lives in.

The mapping is not guesswork: it comes from counting which allocator family each
source file calls, in sqlite-src-3530400. Files that allocate through the
connection allocator (sqlite3Db*) are lookaside-eligible; files that call
sqlite3_malloc directly get ordinary CHERI coverage. See sqlite-arenas.md.

A row is classified from whatever file or component name it carries. Rows whose
component names no file stay UNKNOWN on purpose -- an unclassified row is honest,
a wrongly classified one is a wrong verdict.
"""
import csv, re, sys

# file -> arena, from the allocator survey (Db-calls vs direct-heap calls)
LOOKASIDE_FILES = """alter attach build delete expr insert prepare select tokenize
                     trigger update util vdbeaux vdbemem where window json""".split()
HEAP_FILES = "func main fts3 fts4 fts5 rtree geopoly session rbu zipfile appendvfs expert carray".split()

# component wording -> arena, for rows that name no file
WORD_LOOKASIDE = ["window function", "query flattener", "parse tree", "select processing",
                  "schema parsing", "expression codegen", "generated column", "alter table",
                  "sql query processing", "record decoding", "intersect", "trigger"]
WORD_HEAP = ["fts3", "fts4", "fts5", "rtree", "geopoly", "session", "rbu", "zipfile",
             "appendvfs", "expert", "carray", "json1 extension"]
WORD_NA = ["shell/cli", "shell", "cli", "c api", "sqlite3_db_config", "command-line"]

def classify(row):
    """-> (arena, why) ; arena UNKNOWN when nothing in the row identifies a site."""
    hay = " ".join([row.get("component", ""), row.get("affected_function", ""),
                    row.get("notes", "")]).lower()
    for f in HEAP_FILES:                      # extensions win: ext/fts3/fts3.c also says "core"
        if re.search(rf"\b(ext/)?{f}[\w/]*\.c\b", hay) or re.search(rf"\b{f}\b", hay):
            return "direct-malloc", f"{f} allocates via sqlite3_malloc"
    if re.search(r"\bpcache1?\.c\b|page cache|pcache", hay):
        return "pcache-bulk", "page cache carves pages from one bulk malloc"
    if re.search(r"\bbtree\.c\b|balance_nonroot", hay):
        return "btree-scratch", "balance_nonroot cuts three regions from one block"
    for f in LOOKASIDE_FILES:
        if re.search(rf"\b{f}\.c\b", hay):
            return "lookaside", f"{f}.c allocates via the connection allocator"
    for w in WORD_NA:
        if w in hay:
            return "n/a", "not library heap memory"
    for w in WORD_HEAP:
        if w in hay:
            return "direct-malloc", f"{w} allocates via sqlite3_malloc"
    for w in WORD_LOOKASIDE:
        if w in hay:
            return "lookaside", f"{w} is core parse/query machinery"
    if re.search(r"\bcore\b", hay):
        return "lookaside?", "core, but no file named -- needs tracing"
    return "UNKNOWN", ""


EXPECT = {
    "lookaside":     "BLINDSPOT-candidate",
    "pcache-bulk":   "BLINDSPOT-candidate",
    "btree-scratch": "BLINDSPOT-spatial-only",
    "direct-malloc": "CAUGHT-candidate",
    "lookaside?":    "ARENA-DECIDES",
    "n/a":           "N/A-not-heap",
    "UNKNOWN":       "UNKNOWN",
}

def main(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = list(rows[0].keys())
    n = 0
    for r in rows:
        # never overwrite a verdict established by reading the source for that row
        if r.get("verification", "").startswith(("arena-verified", "mechanism-verified",
                                                 "sqlite.org-assessed")):
            continue
        if r.get("alloc_arena") not in ("", "UNKNOWN", None):
            continue
        arena, why = classify(r)
        if arena == "UNKNOWN":
            continue
        r["alloc_arena"] = arena
        if r.get("cheri_expectation") in ("", "UNKNOWN", None):
            r["cheri_expectation"] = EXPECT[arena]
        r["rationale"] = (r.get("rationale", "") + f" | {why}").strip(" |")
        r["verification"] = "arena-inferred-from-file"
        n += 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    import collections
    c = collections.Counter(r["alloc_arena"] for r in rows)
    print(f"{n} Zeilen klassifiziert; Verteilung:")
    for k, v in c.most_common():
        print(f"  {v:>3}  {k}")
    left = [r["id"] for r in rows if r["alloc_arena"] == "UNKNOWN"]
    print(f"ohne Zuordnung (bewusst): {len(left)}")


if __name__ == "__main__":
    main(sys.argv[1])
