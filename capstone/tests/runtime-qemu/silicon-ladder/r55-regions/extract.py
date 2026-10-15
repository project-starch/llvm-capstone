#!/usr/bin/env python3
"""Extract the monitor-visible carve sequence of an R-43 a2 board run (boot-raw.txt): per test, the domain base
(DBAS), the domain size (from 'Domain requirement' + code via the module's rule is not printed, so the caller passes
tot per program), the region ids and the shared regions' BASE/ALEN. Prints one JSON object."""
import json, re, sys
s = open(sys.argv[1], "rb").read().decode("latin-1").replace("\r", "")
# keep only this boot: after the last kernel banner
i = s.rfind("Linux version")
s = s[i:] if i >= 0 else s
tests = []
for m in re.finditer(r"### TEST (\d+)/(\d+) START (\S+)", s):
    tests.append((m.start(), m.group(3)))
out = []
for k, (pos, name) in enumerate(tests):
    end = tests[k + 1][0] if k + 1 < len(tests) else len(s)
    seg = s[pos:end]
    if "END" in seg[:200] and "DBAS" not in seg:   # the echoed command line, not the run
        continue
    rec = {"test": name.split(":")[0].split("/")[-1]}
    d = re.search(r"DBAS:([0-9A-F]+)", seg)
    if not d:
        continue
    rec["dbas"] = int(d.group(1), 16)
    r = re.search(r"Domain requirement = (\d+)", seg); rec["domreq"] = int(r.group(1)) if r else None
    c = re.search(r"Loadable size = (\d+)", seg); rec["code"] = int(c.group(1)) if c else None
    ids = re.search(r"SQ: r1=(\d+)\s*\nSQ: r2=(\d+)", seg); rec["r1r2"] = [int(ids.group(1)), int(ids.group(2))] if ids else None
    shares = []
    for g in re.finditer(r"RGID:([0-9A-F]+)\s*\n(?:[A-Z0-9]{4}:[0-9A-F]+\s*\n){0,6}?BASE:([0-9A-F]+)\s*\nALEN:([0-9A-F]+)", seg):
        shares.append([int(g.group(1), 16), int(g.group(2), 16), int(g.group(3), 16)])
    rec["shares"] = shares
    rf = re.findall(r"RGN[FON]:([0-9A-F]+)", seg)
    if rf: rec["rgn"] = rf
    out.append(rec)
print(json.dumps(out))
