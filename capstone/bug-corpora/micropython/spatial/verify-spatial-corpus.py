#!/usr/bin/env python3
"""Check the spatial CSV against the stored API response, and check ITSELF.

--self-test seeds two known-bad rows and requires the checker to catch both. A
checker that has never rejected anything is not a passing checker, it is an
unproven one; this project has published three wrong findings behind exactly that.
"""
import csv, json, os, sys

S = os.path.dirname(os.path.abspath(__file__))
VALID_DOMAIN = {"not-run", "untrapped-identical", "untrapped-no-crash", "fault-cause24", "fault-cause7"}
VALID_SPATIAL = {"yes", "no", "uncertain"}

def check(rows, issues):
    errs = []
    for r in rows:
        n = int(r["ref"].lstrip("#"))
        iss = issues.get(n)
        if not iss:
            errs.append(f"{r['id']}: #{n} not in the stored response"); continue
        if r["title"] != iss["title"]:  errs.append(f"{r['id']}: title differs from GitHub")
        if r["state"] != iss["state"]:  errs.append(f"{r['id']}: state differs from GitHub")
        if r["url"]   != iss["url"]:    errs.append(f"{r['id']}: url differs from GitHub")
        if r["is_spatial"] not in VALID_SPATIAL:
            errs.append(f"{r['id']}: unknown is_spatial {r['is_spatial']!r}")
        if r["domain_behaviour"] not in VALID_DOMAIN:
            errs.append(f"{r['id']}: unknown domain_behaviour {r['domain_behaviour']!r}")
        # A verdict with no evidence is the failure mode the temporal audit found
        # fourteen times. Refuse it here rather than discover it later.
        if r["is_spatial"] in ("yes", "no") and not r["spatial_evidence"].strip():
            errs.append(f"{r['id']}: is_spatial={r['is_spatial']} with no quoted evidence")
        # The whole point of the corpus: a gc-heap row may not claim it is trapped
        # by ordinary bounds, because the block has no bounds of its own.
        if r["scope"] == "gc-heap" and r["predicted_trap"] == "yes":
            errs.append(f"{r['id']}: gc-heap cannot predict a plain trap")
        # A row that reports a domain measurement must say what was measured.
        if r["domain_behaviour"] != "not-run" and r["repro_status"] == "none":
            errs.append(f"{r['id']}: domain result without a repro_status")
    return errs

def main():
    rows = list(csv.DictReader(open(os.path.join(S, "spatial-allocator-corpus.csv"))))
    gql = json.load(open(os.path.join(S, "github-issues.json")))["data"]["repository"]
    issues = {v["number"]: v for v in gql.values() if v}

    if "--self-test" in sys.argv:
        bad = [dict(r) for r in rows[:2]]
        bad[0]["title"] = bad[0]["title"] + " (seeded)"
        bad[1]["is_spatial"], bad[1]["spatial_evidence"] = "yes", "   "
        caught = check(bad, issues)
        if len(caught) < 2:
            print("SELF-TEST FAILED: the checker did not catch the seeded problems"); return 1
        print(f"self-test ok, checker caught {len(caught)} seeded problems:")
        for e in caught: print("   ", e)

    errs = check(rows, issues)
    if errs:
        for e in errs: print("FEHLER:", e)
        return 1
    print(f"OK: {len(rows)} rows, every title/state/url matches the stored source response")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
