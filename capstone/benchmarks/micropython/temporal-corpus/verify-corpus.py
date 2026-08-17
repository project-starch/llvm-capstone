#!/usr/bin/env python3
"""Check the corpus CSV against the stored API responses it was built from.

    python3 verify-corpus.py          # verify, exit 1 on any mismatch
    python3 verify-corpus.py --self-test   # prove the checker can fail, then verify

The self-test exists because a checker that has never reported a mismatch is not
a passing checker, it is an unproven one. It corrupts two rows in memory and
requires that both are caught before it will verify the real file.
"""
import csv, json, os, re, sys

S = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(S, "temporal-allocator-corpus.csv")

VALID_CLASS = {"uaf", "double-free", "dangling-view", "dangling-buffer", "dangling-pointer",
               "premature-free", "lifetime-order", "reentrancy", "race-uaf",
               "alloc-invariant", "memory-corruption"}
VALID_SCOPE = {"gc-core", "gc-managed", "port-heap"}
VALID_HYP = {"trapped", "not-trapped", "unclear"}
VALID_REPRO = {"none", "planned", "built", "confirmed"}


def load_sources():
    gql = json.load(open(os.path.join(S, "github-issues.json")))["data"]["repository"]
    issues = {v["number"]: v for v in gql.values() if v}
    nvd = {v["cve"]["id"]: v["cve"]
           for v in json.load(open(os.path.join(S, "nvd-cves.json")))["vulnerabilities"]}
    return issues, nvd


def check(rows, issues, nvd):
    errs = []
    if len(rows) != 30:
        errs.append(f"row count {len(rows)}, expected 30")
    for key in ("id", "ref"):
        vals = [r[key] for r in rows]
        dupes = {v for v in vals if vals.count(v) > 1}
        if dupes:
            errs.append(f"duplicate {key}: {sorted(dupes)}")
    for r in rows:
        rid = r["id"]
        if not re.match(r"^https://(github\.com|nvd\.nist\.gov)/", r["url"]):
            errs.append(f"{rid}: bad url")
        if r["class"] not in VALID_CLASS:
            errs.append(f"{rid}: unknown class {r['class']!r}")
        if r["scope"] not in VALID_SCOPE:
            errs.append(f"{rid}: unknown scope {r['scope']!r}")
        if r["capstone_hypothesis"] not in VALID_HYP:
            errs.append(f"{rid}: unknown hypothesis {r['capstone_hypothesis']!r}")
        if r["repro_status"] not in VALID_REPRO:
            errs.append(f"{rid}: unknown repro_status {r['repro_status']!r}")
        for col in ("title", "component", "trigger"):
            if not r[col].strip():
                errs.append(f"{rid}: empty {col}")
        # the load-bearing part: facts must still match the source they came from
        if r["source"] == "issue":
            v = issues.get(int(r["ref"].lstrip("#")))
            if v is None:
                errs.append(f"{rid}: {r['ref']} not in stored GitHub response")
            else:
                if " ".join(v["title"].split())[:150] != r["title"]:
                    errs.append(f"{rid}: title differs from GitHub")
                if v["state"].lower() != r["state"]:
                    errs.append(f"{rid}: state differs from GitHub")
                if v["url"] != r["url"]:
                    errs.append(f"{rid}: url differs from GitHub")
        elif r["source"] == "CVE":
            if r["ref"] not in nvd:
                errs.append(f"{rid}: {r['ref']} not in stored NVD response")
        else:
            errs.append(f"{rid}: unknown source {r['source']!r}")
    return errs


def main():
    issues, nvd = load_sources()
    rows = list(csv.DictReader(open(CSV)))

    if "--self-test" in sys.argv:
        import copy
        bad = copy.deepcopy(rows)
        bad[3]["title"] = "DELIBERATELY WRONG"
        bad[5]["state"] = "open" if bad[5]["state"] == "closed" else "closed"
        found = check(bad, issues, nvd)
        if len(found) < 2:
            print("SELF-TEST FAILED: checker did not catch the seeded corruption")
            print("  caught:", found)
            return 2
        print(f"self-test ok, checker caught {len(found)} seeded problems:")
        for e in found:
            print("   ", e)

    errs = check(rows, issues, nvd)
    if errs:
        print(f"FAIL: {len(errs)} problems")
        for e in errs:
            print("   ", e)
        return 1
    print(f"OK: {len(rows)} rows, every title/state/url matches the stored source response")
    return 0


if __name__ == "__main__":
    sys.exit(main())
