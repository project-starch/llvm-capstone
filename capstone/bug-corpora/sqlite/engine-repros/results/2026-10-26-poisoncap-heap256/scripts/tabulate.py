#!/usr/bin/env python3
"""Tabulate the 17-case population over every configuration of the 256 MiB comparison (run on p13)."""
import json
import re

POP = "00 03 05 06 07 10 12 14 15 16 17 19 20 21 22 23 24".split()
SUBLET = ("v6-sublet-driver", "v6-sublet-driver-rep2", "v6-sublet-block", "v6-sublet-block-rep2",
          "v6-sublet-block-keeps", "v6-sublet-block-keeps-rep2")
POISONCAP = ("heap256-la", "heap256-la-rep2", "driver-la", "driver-la-rep2")


def sublet(run):
    out, ctl = {}, {}
    for line in open(f"runs/{run}/results.jsonl"):
        r = json.loads(line)
        name = r["name"]
        if name.startswith("control"):
            if name.startswith(("control_realloc", "control_capacity")):
                ctl[name] = re.findall(r"CONTROL [a-z-]+ [a-z]\S*[^\n]*", r.get("output", ""))[:1]
            continue
        if name[:2] in POP:
            f = r.get("fault") or {}
            state = "AT" if r.get("at_site") else ("returned" if r.get("returned") else "other")
            out[name[:2]] = (state, f.get("cause"), f.get("function"))
    return out, ctl


def poisoncap(run):
    out = {}
    for line in open(f"verdicts/{run}/verdicts.md"):
        m = re.match(r"\| (\d\d)_[^|]*\| ([^|]*)\|", line)
        if m and m.group(1) in POP:
            out[m.group(1)] = m.group(2).strip()
    return out


res = {}
for run in SUBLET:
    s, ctl = sublet(run)
    res[run] = s
    missed = {c: s[c] for c in s if s[c][0] != "AT"}
    print(f"{run:28} at site {sum(1 for c in s if s[c][0] == 'AT'):2}/{len(s)}  not: {missed}")
    for k, v in ctl.items():
        print(f"    {k}: {v}")
for run in POISONCAP:
    p = poisoncap(run)
    res[run] = p
    print(f"{run:28} at site {sum(1 for c in p if p[c].startswith('fault')):2}/{len(p)}  not: "
          f"{[c for c in p if not p[c].startswith('fault')]}")
for a, b in (("v6-sublet-driver", "v6-sublet-driver-rep2"), ("v6-sublet-block", "v6-sublet-block-rep2"),
             ("v6-sublet-block-keeps", "v6-sublet-block-keeps-rep2"), ("heap256-la", "heap256-la-rep2"),
             ("driver-la", "driver-la-rep2")):
    print(f"reps equal {a}: {res[a] == res[b]}")
print("case 12:", {r: res[r].get("12") for r in res})
