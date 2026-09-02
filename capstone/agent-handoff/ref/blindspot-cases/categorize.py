#!/usr/bin/env python3
"""Add code_category: WHAT KIND of code each row's defect lives in.

Without this column the table reads as "396 SQLite bugs", which is not what it
is. A defect in the core library, in an optional extension that needs a compile
flag, in the command-line shell, or in demo code that ships only for the test
suite are four different claims, and only the first is unconditionally part of
"SQLite" as a library consumer links it.

Categories, most specific first:
  survey        not a bug at all -- a recorded negative result
  duplicate     a second CVE id for a bug already in the table
  not-sqlite    filed against SQLite, actually a third party's defect
  sqlite2       the SQLite 2 codebase, unrelated to the 3.x source
  cli           the sqlite3 shell -- a separate program, not the library
  demo          ext/misc demo code, built for the test suite only
  extension     ext/ code that needs a -DSQLITE_ENABLE_* flag
  c-api         only reachable by a C caller passing bad arguments
  core          src/ -- the library everyone links
"""
import csv, re, sys, collections

RULES = [
    ("survey",     re.compile(r"^component-survey", re.I), ("id",)),
    ("duplicate",  re.compile(r"\bduplicate of\b", re.I), ("notes", "sqlite_assessment")),
    ("not-sqlite", re.compile(r"not a (sqlite )?bug( in sqlite)?|third-party|"
                              r"bug in (the )?(jdbc|node|php|application)", re.I),
                   ("notes", "sqlite_assessment")),
    ("sqlite2",    re.compile(r"\bsqlite ?2\b|sqlite2 encode", re.I),
                   ("component", "notes")),
    ("cli",        re.compile(r"\bshell\b|\bcli\b|command-line|\.dump\b|\.expert\b|"
                              r"appendvfs", re.I), ("component", "notes", "affected_function")),
    ("demo",       re.compile(r"\bamatch\b|\bfileio\b|ext/misc|demo code|test suite only",
                              re.I), ("component", "notes", "affected_function")),
    ("extension",  re.compile(r"\bfts[345]?\b|\brtree\b|geopoly|\bsession\b|\brbu\b|"
                              r"zipfile|carray|stat4|requires-extension", re.I),
                   ("component", "notes", "affected_function", "trigger")),
    ("c-api",      re.compile(r"c[- ]language api|sqlite3_db_config|C-API-misuse|"
                              r"only reachable via c api", re.I),
                   ("component", "notes", "trigger")),
]


def categorize(r):
    for name, pat, fields in RULES:
        if any(pat.search(r.get(f, "") or "") for f in fields):
            return name
    return "core"


def main(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = list(rows[0].keys())
    if "code_category" not in cols:
        cols.insert(cols.index("component") + 1, "code_category")
    for r in rows:
        r["code_category"] = categorize(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    c = collections.Counter(r["code_category"] for r in rows)
    tot = len(rows)
    print(f"{tot} Zeilen:")
    for k, v in c.most_common():
        print(f"  {v:>4}  ({100*v/tot:4.1f}%)  {k}")
    real = tot - c["survey"] - c["duplicate"] - c["not-sqlite"]
    print(f"\ndistinkte SQLite-Defekte (ohne survey/duplicate/not-sqlite): {real}")
    print(f"davon Core-Bibliothek: {c['core']}")


if __name__ == "__main__":
    main(sys.argv[1])
