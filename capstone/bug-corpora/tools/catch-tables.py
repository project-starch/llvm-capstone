#!/usr/bin/env python3
"""The two tables (temporal, spatial) per program and allocator layer, every column, for the cases of
FFmpeg, tshark and memcached; then the inside-one-allocation table. Computed from case.json only. A
cell that is not a reading is printed as NOT RUN / UNCLASSIFIED and counted as neither, and the run
exits 1 if any exists -- a table with a hole in it is not printed as if it were complete.

    catch-tables.py [bug-corpora dir] [--markdown]

The source of the per-program tables in docs/ref/spatial-vs-temporal-three-programs.md section 0.

Verdict sources, in order: an arm's explicit `verdict`; then the opening words of its oracle, after
any `MEASURED ...:` prefix -- caught: CAUGHT / REPORTED / ASan REPORTS / SIGPROT / fault; missed:
NOT CAUGHT / NO FAULT / NO ASan report / SILENT / complete / the sequence completes. An arm whose
status is predicted / not written / declined, or whose oracle begins PREDICTED, is NOT RUN."""
import collections
import json
import pathlib
import re
import sys

POS = [a for a in sys.argv[1:] if not a.startswith("--")]
BC = pathlib.Path(POS[0]) if POS else pathlib.Path(__file__).resolve().parents[1]
TARGETS = ("ffmpeg", "wireshark", "memcached")
NOTRUN = {"predicted", "not written", "declined"}
LAYER = {"pool-repros": "AVBufferPool", "plain-temporal-repros": "direct malloc", "plain-heap-repros": "direct malloc",
         "subobject-repros": "inside one struct", "plane-repros": "frame-pool plane", "carved-repros": "carved buffer",
         "wmem-repros": "wmem", "allocator-repros": "slabs.c / cache.c"}


def verdict(arm):
    if arm is None:
        return "absent"
    if arm.get("status") in NOTRUN:
        return "not run"
    if arm.get("status") == "not-applicable":
        return "n/a"
    v = arm.get("verdict")
    if v:
        return "caught" if v.upper() == "CAUGHT" else "missed"
    o = str(arm.get("oracle", "")).strip()
    if o.upper().startswith("PREDICTED"):
        return "not run"
    m = re.match(r"MEASURED[^:]*:\s*(.*)", o, re.S)
    head = (m.group(1) if m else o).lstrip("*").lstrip().lower()
    if head.startswith(("not caught", "no fault", "no asan", "silent", "complete", "the sequence completes")):
        return "missed"
    if head.startswith(("caught", "reported", "asan reports", "sigprot", "fault")):
        return "caught"
    if o.lower().startswith("no asan report"):
        return "missed"
    # Older measured ASan arms state the oracle first and the reading after it.
    if arm.get("status") == "measured" and re.search(r"asan reports heap-", o, re.I):
        return "caught"
    return "UNCLASSIFIED"


def protected(arms, plain, port):
    """The stronger arm where a port measured one, else the plain one."""
    a = arms.get(port)
    return a if (a and verdict(a) in ("caught", "missed")) else arms.get(plain)


rows, odd = [], []
for p in TARGETS:
    for cj in sorted((BC / p).glob("*/[0-9][0-9]_*/case.json")):
        d = json.loads(cj.read_text())
        corpus = cj.parts[-3]
        le = str(d.get("lifetime_ender", "")).strip().upper()
        kind = "spatial" if (le.startswith("NONE") or "SPATIAL" in le) else "temporal"
        a = d["arms"]
        layer = LAYER.get(corpus, corpus)
        if p == "wireshark" and layer == "direct malloc":
            layer = "direct g_malloc"
        r = dict(prog=p, corpus=corpus, layer=layer, case=cj.parent.name, kind=kind,
                 nested=bool(d.get("nested")),
                 asan=verdict(a.get("native-detect")),
                 cheri=verdict(a.get("cheribsd-revocation")),
                 pc0=verdict(a.get("poisoncap-spatial")),
                 pc1=verdict(a.get("poisoncap-protected")),
                 cap=verdict(a.get("spatial")),
                 sub=verdict(protected(a, "sublet", "sublet-chunks") if corpus != "pool-repros"
                             else protected(a, "sublet", "sublet-port")),
                 fcap=verdict(a.get("capstone-subobject")), fcheri=verdict(a.get("cheribsd-subobject")),
                 kcap=verdict(a.get("capstone-carve-bounds")), kcheri=verdict(a.get("cheribsd-carve-bounds")))
        for k in ("asan", "cheri", "pc0", "pc1", "cap", "sub"):
            if r[k] not in ("caught", "missed"):
                odd.append(f"{p}/{corpus}/{r['case']} {k}={r[k]}")
        rows.append(r)

COLS = (("asan", "ASan"), ("cheri", "CheriBSD"), ("pc0", "PoisonCap m0"), ("pc1", "PoisonCap m1"),
        ("cap", "Cap bounds"), ("sub", "Cap+Sublet"))


def cell(g, k):
    c = collections.Counter(r[k] for r in g)
    run = c["caught"] + c["missed"]
    extra = len(g) - run
    return f"{c['caught']}/{run}" + (f" (+{extra} ?)" if extra else "")


print(f"cases: {len(rows)}")
for kind in ("temporal", "spatial"):
    print(f"\n{kind.upper()}")
    print(f"  {'program':<10}{'layer':<20}{'axis':<8}{'n':>4} | " + " | ".join(f"{h:>12}" for _, h in COLS))
    groups = collections.OrderedDict()
    for r in rows:
        if r["kind"] == kind:
            groups.setdefault((r["prog"], r["layer"], "nested" if r["nested"] else "plain"), []).append(r)
    for (p, layer, ax), g in groups.items():
        print(f"  {p:<10}{layer:<20}{ax:<8}{len(g):>4} | " + " | ".join(f"{cell(g, k):>12}" for k, _ in COLS))
    for ax in ("nested", "plain"):
        g = [r for r in rows if r["kind"] == kind and (r["nested"] == (ax == "nested"))]
        print(f"  {'':<10}{'subtotal':<20}{ax:<8}{len(g):>4} | " + " | ".join(f"{cell(g, k):>12}" for k, _ in COLS))
    g = [r for r in rows if r["kind"] == kind]
    print(f"  {'':<10}{'TOTAL':<20}{'':<8}{len(g):>4} | " + " | ".join(f"{cell(g, k):>12}" for k, _ in COLS))

print("\nINSIDE ONE ALLOCATION (spatial cases Capstone bounds misses)")
inside = [r for r in rows if r["kind"] == "spatial" and r["cap"] == "missed"]
groups = collections.OrderedDict()
for r in inside:
    groups.setdefault((r["prog"], r["layer"]), []).append(r)
IC = (("sub", "Cap+Sublet"), ("fcap", "Cap field"), ("fcheri", "CHERI field"), ("kcap", "Cap carve"), ("kcheri", "CHERI carve"))
for (p, layer), g in groups.items():
    print(f"  {p:<10}{layer:<20}{len(g):>4} | " + " | ".join(f"{h}: {cell(g, k):>10}" for k, h in IC))
    for r in g:
        print(f"      {r['case'][:58]:<58} " + " ".join(f"{k}={r[k]}" for k, _ in IC))
if odd:
    print("\nNOT A READING:", *odd, sep="\n  ")


# ---- emitters: the same cells as markdown (docs) and as slide rows (html) ----------------------
def table_rows(kind):
    groups = collections.OrderedDict()
    for r in rows:
        if r["kind"] == kind:
            groups.setdefault((r["prog"], r["layer"], "nested" if r["nested"] else "plain"), []).append(r)
    order = {"ffmpeg": 0, "wireshark": 1, "memcached": 2}
    out = []
    for (p, layer, ax), g in sorted(groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][2] != "nested", kv[0][1])):
        out.append((p, layer, ax, len(g), [cell(g, k) for k, _ in COLS]))
    tot = [r for r in rows if r["kind"] == kind]
    nn = sum(1 for r in tot if r["nested"])
    out.append(("Total", "", f"{nn} n · {len(tot) - nn} p", len(tot), [cell(tot, k) for k, _ in COLS]))
    return out


if "--markdown" in sys.argv:
    NAME = {"ffmpeg": "FFmpeg", "wireshark": "tshark", "memcached": "memcached", "Total": "**Total**"}
    for kind in ("temporal", "spatial"):
        tr = table_rows(kind)
        print(f"\n**{kind.capitalize()} ({tr[-1][3]})**\n")
        print("| program | allocator layer | axis | n | ASan | CheriBSD | PoisonCap mode 0 | PoisonCap mode 1 | Capstone bounds | Capstone + Sublet |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for p, layer, ax, n, cells in tr:
            print(f"| {NAME[p]} | {layer} | {ax} | {n} | " + " | ".join(c.replace('/', ' / ') for c in cells) + " |")


if odd:
    sys.exit(1)
