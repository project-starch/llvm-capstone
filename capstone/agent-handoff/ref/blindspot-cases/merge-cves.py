#!/usr/bin/env python3
"""Merge the per-source CVE research fragments into one master table.

Inputs : cve-research/*.json   (one JSON array per research stream)
         sqlite-cves.csv       (the hand-classified seed table, carries alloc_arena)
Outputs: master CSV + XLSX

Dedupe key is the normalised id. Fields merge field-by-field: a non-empty value
wins over an empty one; when two sources disagree on a non-empty field, both are
kept as "a || b" so the conflict stays visible instead of being silently picked.
"""
import csv, glob, json, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "cve-research")

COLS = ["id", "year", "component", "bug_class", "affected_function", "trigger",
        "poc", "fix_commit", "fixed_in", "alloc_arena", "cheri_expectation",
        "rationale", "sqlite_assessment", "verification", "source", "notes"]


def norm_id(i):
    i = (i or "").strip()
    m = re.match(r"(?i)^(cve)[-_ ]?(\d{4})[-_ ]?(\d{4,7})$", i)
    return f"CVE-{m.group(2)}-{m.group(3)}" if m else i


# Entries that are NOT SQLite memory-safety bugs. They keep coming back from the
# fragments on every re-merge -- six of them are CVE ids sqlite.org itself calls
# AI hallucinations -- so the filter belongs IN the pipeline, not in a one-off
# pass someone forgets to repeat.
NOTBUG = re.compile(
    r"not a bug in sqlite|ai hallucin|hallucinat|unreproducible|misinformation|"
    r"bug in (the )?(third-party|application|jdbc|node\.js|php|luxcal)|"
    r"never appeared in|not a true vulnerability|fabricat", re.I)


def is_notbug(r):
    return bool(NOTBUG.search(" ".join([r.get("sqlite_assessment", ""),
                                        r.get("notes", ""),
                                        r.get("rationale", "")])))


def merge_field(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b
    if not b or a == b:
        return a
    if b.lower() in a.lower():
        return a
    if a.lower() in b.lower():
        return b
    return f"{a} || {b}"


def main():
    rows = {}          # id -> dict
    per_source = {}

    # 1. the hand-classified seed keeps its arena/expectation/rationale columns
    seed_path = os.path.join(BASE, "sqlite-cves.csv")
    if os.path.exists(seed_path):
        for r in csv.DictReader(open(seed_path, encoding="utf-8")):
            # the seed used a "cve" column; once the master is fed back in it is
            # "id". Accept either, or the seed silently contributes ZERO rows and
            # every arena classification is lost on the next merge.
            rid = norm_id(r.get("id") or r.get("cve", ""))
            if not rid:
                continue
            rows[rid] = {c: (r.get(c) or "").strip() for c in COLS if c in r}
            rows[rid]["id"] = rid
            if r.get("affected_site"):
                rows[rid]["affected_function"] = r["affected_site"]
            rows[rid].setdefault("source", "seed")
        per_source["seed"] = len(rows)

    # 2. research fragments. ONLY the agreed per-stream outputs: the agents also
    # drop raw NVD/API dumps in this directory, and an NVD response is a list of
    # objects that happen to carry an "id" field -- merging one would silently
    # fill the table with unclassified junk.
    fragments = [p for p in sorted(glob.glob(os.path.join(RES, "?-*.json")))
                 if re.match(r"^[A-F]-", os.path.basename(p))]
    if not fragments:
        print("  WARNING: no A-F fragments found; emitting the seed alone",
              file=sys.stderr)
    for path in fragments:
        name = os.path.basename(path)
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {name}: unreadable ({e})", file=sys.stderr)
            continue
        if isinstance(data, dict):
            data = data.get("entries") or data.get("bugs") or []
        n = 0
        for e in data:
            if not isinstance(e, dict):
                continue
            rid = norm_id(e.get("id", ""))
            if not rid:
                continue
            n += 1
            tgt = rows.setdefault(rid, {"id": rid})
            for c in COLS:
                if c == "id":
                    continue
                tgt[c] = merge_field(tgt.get(c, ""), str(e.get(c, "") or ""))
        per_source[name] = n

    # 3. every row must carry an honest verification state
    for r in rows.values():
        if not r.get("verification"):
            r["verification"] = "unverified"
        if not r.get("cheri_expectation"):
            r["cheri_expectation"] = "UNKNOWN"
        if not r.get("alloc_arena"):
            r["alloc_arena"] = "UNKNOWN"

    def sort_key(r):
        m = re.match(r"CVE-(\d{4})-(\d+)", r["id"])
        return (0, -int(m.group(1)), -int(m.group(2))) if m else (1, 0, hash(r["id"]) % 997)

    keep = [r for r in rows.values() if not is_notbug(r)]
    excl = [r for r in rows.values() if is_notbug(r)]
    out = sorted(keep, key=sort_key)
    if excl:
        ex_path = os.path.join(BASE, "sqlite-bugs-excluded.csv")
        with open(ex_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            for r in sorted(excl, key=sort_key):
                w.writerow({c: r.get(c, "") for c in COLS})
        print(f"excluded as non-bugs: {len(excl)}  ->  {ex_path}")
    csv_path = os.path.join(BASE, "sqlite-bugs-master.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c, "") for c in COLS})

    # self-check: reread and compare row count + that no id is empty
    back = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert len(back) == len(out), f"CSV roundtrip {len(back)} != {len(out)}"
    assert all(r["id"] for r in back), "empty id survived the merge"

    print(f"sources: {per_source}")
    print(f"master rows: {len(out)}  ->  {csv_path}")
    blind = [r for r in out if "BLINDSPOT" in r.get("cheri_expectation", "").upper()]
    print(f"blind-spot candidates: {len(blind)}")
    return csv_path


if __name__ == "__main__":
    main()
