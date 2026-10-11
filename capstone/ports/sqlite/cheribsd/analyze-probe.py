#!/usr/bin/env python3
"""Read the guest probe sweep and say, per case, what changed against the
pre-probe CheriBSD tally.

The pre-probe tally classified on exit code and step markers only. Its SILENT
bucket mixed three different things, and this is the script that splits them.
It refuses to print a summary if the sweep did not finish (no DONE line) or if
any row says NOT-ARMED, because a probe that was not compiled in produces hits=0,
which is indistinguishable from "the defect site was never reached".
"""
import re, sys, pathlib, collections

# the tally produced by /tmp/tally2.py BEFORE any probe existed
PRE = {
 "33cf194218_0":"SILENT","415540ddaa_0":"SILENT","634ac14488_0":"SILENT",
 "634ac14488_0_sys":"SILENT","a783931794_0":"SILENT","a783931794_0_sys":"SILENT",
 "adfb203a7d_0":"SILENT","bfe33f80dd_0":"SILENT","bfe33f80dd_0_sys":"SILENT",
 "blobclose":"SILENT","c7def600bd_0":"SILENT","c7def600bd_0_sys":"SILENT",
 "expertrem":"SILENT","fts3destroyoom":"SILENT","fts3snipor":"SILENT",
 "fts5inplace":"SILENT","fts5rank":"SILENT","fts5structwrite":"SILENT",
 "fts5vocabeof":"SILENT","fz09_sys":"SILENT","jsoneachroot":"SILENT",
 "jsoneachstatic":"SILENT","mem5design":"SILENT","rtreestatic":"SILENT",
 "staticbind":"SILENT","wschema":"SILENT",
 "2c7a73eaea_0":"NOT-REACHED","8f5b14a5c2_0":"NOT-REACHED",
 "2c7a73eaea_0_sys":"DETECTED","33cf194218_0_sys":"DETECTED",
 "415540ddaa_0_sys":"DETECTED","8f5b14a5c2_0_sys":"DETECTED",
 "adfb203a7d_0_sys":"DETECTED","fz09":"DETECTED",
}

ROW = re.compile(r"^(\S+)\s+(\d+)\s+(\S+)\s+rc=(\S+)\s+hits=(\S+)\s+wit=(\S+)\s*(.*)$")

def main(path):
    text = pathlib.Path(path).read_text(errors="replace")
    rows, bad = [], []
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if not m: continue
        tag, site, verdict, rc, hits, wit, rest = m.groups()
        rows.append(dict(tag=tag, site=int(site), verdict=verdict, rc=rc,
                         hits=hits, wit=int(wit), detail=rest.strip()))
    if "DONE" not in text:
        bad.append("the sweep did not print DONE -- it was cut off, so the rows below are partial")
    for r in rows:
        if r["verdict"] in ("NOT-ARMED", "NO-SUMMARY", "BADCONFIG", "NOBIN"):
            bad.append("%s: %s -- this row measures the harness, not the case" % (r["tag"], r["verdict"]))
    if not rows:
        print("NO ROWS PARSED -- refusing to summarize"); return 1

    print("%-20s %-5s %-13s %-13s %s" % ("CASE","SITE","PRE-PROBE","WITH PROBE","EVIDENCE"))
    counts = collections.Counter()
    for r in sorted(rows, key=lambda r: (r["site"], r["tag"])):
        pre = PRE.get(r["tag"], "-")
        counts[(pre, r["verdict"])] += 1
        print("%-20s %-5d %-13s %-13s hits=%-7s wit=%-4d %s"
              % (r["tag"], r["site"], pre, r["verdict"], r["hits"], r["wit"], r["detail"][:46]))

    print("\n--- what the probe changed ---")
    for (pre, now), n in sorted(counts.items()):
        mark = ""
        if pre == "SILENT" and now == "NOT-REACHED":
            mark = "  <-- was being counted as 'CHERI did not catch it'"
        if pre == "SILENT" and now == "WITNESS":
            mark = "  <-- reached AND the defective access is witnessed: a real negative"
        print("  %-13s -> %-13s %3d%s" % (pre, now, n, mark))

    if bad:
        print("\n!!! DO NOT USE THESE NUMBERS UNTIL THESE ARE RESOLVED:")
        for b in bad: print("   " + b)
        return 2
    return 0

sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"))
