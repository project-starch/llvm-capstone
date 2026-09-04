#!/usr/bin/env python3
"""Per-record comparison of an SQLLogicTest run in the Capstone domain against the
native run of the SAME .test file.

The .test file's expected values are the oracle; the native run (slt_native, the same
slt_runner.h and the same SQLite configuration) establishes which records this
configuration is expected to fail (OMIT_FLOATING_POINT and friends).  So the question
this tool answers is: does the domain fail EXACTLY the records the native build fails,
and nothing else?  A record that fails in the domain and passes natively is the only
kind of result the twins exist to find -- a miscompile or a runtime defect.

Verdicts (exit code):
  AGREE     (0)  same tally, same failure records, both runs complete
  MISMATCH  (1)  a tally field or a failure record differs
  ERROR     (2)  a side produced no well-formed summary, did not complete, ran zero
                 records, or reported failures at or above the SLT_MAX_REPORTED cap
                 (so its failure set may be truncated)

ABSENCE IS AN ERROR, NEVER A MATCH.  A domain that wedges, traps or dies at create_dom
produces no summary; comparing an empty tally against anything must not read as a pass.
The same holds for the cap: a run that reports as many failures as the runner is willing
to print has a failure set nobody can compare.

Every verdict line names the two summaries it was computed from.
"""
import argparse
import re
import sys

SUMMARY_KEYS = ("records", "stmt_pass", "stmt_fail", "query_pass", "query_fail",
                "skip_big", "oom", "skip_cond", "parse_err", "completed")
MARK = re.compile(r"(SLT-(?:SUMMARY|FAIL|PARSE|FILE)\b.*)$")


def parse_log(path):
    """Return (summary dict or None, [(kind, line, detail)], n_summaries, n_files)."""
    summaries, fails = [], []
    n_files = 0
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", "replace")
    for line in raw.replace("\r", "\n").split("\n"):
        # slt_native ends with "SLT-TOTAL SLT-SUMMARY ..." over all its files; that is
        # not a second run of this file.
        if "SLT-TOTAL" in line:
            continue
        m = MARK.search(line)
        if not m:
            continue
        text = m.group(1).strip()
        if text.startswith("SLT-SUMMARY"):
            d = {}
            for kv in text.split()[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = v
            summaries.append(d)
        elif text.startswith("SLT-FAIL") or text.startswith("SLT-PARSE"):
            kind = text.split()[0]
            lm = re.search(r"\bline=(\d+)", text)
            fails.append((kind, int(lm.group(1)) if lm else -1, text))
        elif text.startswith("SLT-FILE"):
            n_files += 1
    return (summaries[-1] if summaries else None), fails, len(summaries), n_files


def well_formed(summary, side, errors):
    if summary is None:
        errors.append(f"no {side} summary -- the run did not complete")
        return False
    missing = [k for k in SUMMARY_KEYS if k not in summary]
    if missing:
        errors.append(f"{side} summary lacks {','.join(missing)}")
        return False
    bad = [k for k in SUMMARY_KEYS if not summary[k].lstrip("-").isdigit()]
    if bad:
        errors.append(f"{side} summary has non-numeric {','.join(bad)}")
        return False
    if summary["completed"] != "1":
        errors.append(f"{side} summary says completed={summary['completed']}")
        return False
    if summary.get("open_failed", "0") != "0":
        errors.append(f"{side} could not open its database (open_failed=1)")
        return False
    if int(summary["records"]) <= 0:
        errors.append(f"{side} ran zero records")
        return False
    return True


def fail_count(summary):
    return int(summary["stmt_fail"]) + int(summary["query_fail"]) + int(summary["parse_err"])


def fmt(summary):
    if summary is None:
        return "(none)"
    return " ".join(f"{k}={summary.get(k, '?')}" for k in SUMMARY_KEYS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", required=True, help="log of slt_native on the .test file")
    ap.add_argument("--domain", required=True, help="console log of the domain run on the same file")
    ap.add_argument("--cap", type=int, default=8, help="SLT_MAX_REPORTED both runs were built with")
    ap.add_argument("--label", default="", help="free text carried into the verdict line (test, level, binary)")
    ap.add_argument("--tsv", help="append the verdict line to this file")
    a = ap.parse_args()

    errors = []
    nat, nat_fails, n_nat, nat_files = parse_log(a.native)
    dom, dom_fails, n_dom, _ = parse_log(a.domain)
    # Exactly one run per log.  The native runner announces each file with SLT-FILE and
    # prints one summary per file (the TOTAL line is skipped above); a count that does
    # not match means a format change or a concatenated log, and either must fail
    # loudly rather than shift which summary gets compared.
    if n_dom != 1:
        errors.append(f"domain log holds {n_dom} summaries, expected exactly 1 (a wedge, or a stale or replayed capture)")
    if n_nat != 1 or nat_files != 1:
        errors.append(f"native log holds {n_nat} summaries for {nat_files} SLT-FILE markers, expected 1 and 1")
    ok_n = well_formed(nat, "native", errors)
    ok_d = well_formed(dom, "domain", errors)
    if ok_n and fail_count(nat) >= a.cap:
        errors.append(f"native reported {fail_count(nat)} failures, at or above the cap {a.cap}: failure set truncated")
    if ok_d and fail_count(dom) >= a.cap:
        errors.append(f"domain reported {fail_count(dom)} failures, at or above the cap {a.cap}: failure set truncated")

    if errors:
        verdict, detail = "ERROR", "; ".join(errors)
    else:
        diffs = [f"{k}: native={nat[k]} domain={dom[k]}" for k in SUMMARY_KEYS if nat[k] != dom[k]]
        nat_set = sorted((k, l) for k, l, _ in nat_fails)
        dom_set = sorted((k, l) for k, l, _ in dom_fails)
        only_dom = sorted(set(dom_set) - set(nat_set))
        only_nat = sorted(set(nat_set) - set(dom_set))
        if only_dom:
            diffs.append("fails only in the domain: " + ", ".join(f"{k} line={l}" for k, l in only_dom))
        if only_nat:
            diffs.append("fails only natively: " + ", ".join(f"{k} line={l}" for k, l in only_nat))
        # Same record failing on both sides but with different reported detail is still
        # a disagreement about the answer.
        nat_detail = {(k, l): t for k, l, t in nat_fails}
        for k, l, t in dom_fails:
            if (k, l) in nat_detail and nat_detail[(k, l)] != t:
                diffs.append(f"{k} line={l} differs: native '{nat_detail[(k, l)]}' domain '{t}'")
        if diffs:
            verdict, detail = "MISMATCH", "; ".join(diffs)
        else:
            verdict, detail = "AGREE", f"{nat['records']} records, {fail_count(nat)} shared failures"

    line = "\t".join([a.label, verdict, detail, "native: " + fmt(nat), "domain: " + fmt(dom)])
    print(line)
    if a.tsv:
        with open(a.tsv, "a") as f:
            f.write(line + "\n")
    sys.exit({"AGREE": 0, "MISMATCH": 1, "ERROR": 2}[verdict])


if __name__ == "__main__":
    main()
