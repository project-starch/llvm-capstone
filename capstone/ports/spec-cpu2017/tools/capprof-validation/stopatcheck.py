#!/usr/bin/env python3
"""Check stopat-run.sh's windows against counts fixed in advance (stopat.c).
usage: stopatcheck.py OUT   exit 1 unless every check passes"""
import glob, json, os, re, sys

d = sys.argv[1]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {detail}")


def window(boot, n):
    p = f"{d}/{boot}/capprof/window-{n}.tsv"
    if not os.path.exists(p):
        return None, []
    recs, foot = [], []
    for line in open(p):
        if line.startswith("#"):
            foot.append(line.strip())
            continue
        va, ln, b, att, tr = line.split("\t")
        recs.append((int(va, 16), int(ln), int(b, 16), int(att)))
    return recs, foot


MARKS = {m: (m << 20) | 0xe0e13 for m in (0x3d1, 0x3d2, 0x3e1, 0x3e2, 0x3e3, 0x3e4, 0x3e5, 0x3e6)}


def marks(recs):
    """attempts of each marker instruction, by immediate (absent = 0); None if one appears twice"""
    out = {}
    for m, w in MARKS.items():
        hits = [r for r in recs if r[2] == w and r[1] == 4]
        if len(hits) > 1:
            return None
        out[m] = hits[0][3] if hits else 0
    return out


import importlib.util
_spec = importlib.util.spec_from_file_location("stop_at", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stop-at.py"))
stop_at = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stop_at)
LOADS = {k: stop_at.elf(f"{d}/stopat-{k}")[1] for k in ("plain", "cap")}
SLIDE = {"plain": 0}


def program(recs, k):
    """the records that are the program's own instructions: its bytes at its (slid) address"""
    out = {}
    for va, ln, b, a in recs:
        link = va - SLIDE.get(k, 0)
        for v, m, flags, seg in LOADS[k]:
            if flags & 1 and v <= link < v + len(seg) and int.from_bytes(seg[link - v:link - v + ln], "little") == b:
                out[(va, b)] = a
    return out


FULL0 = {0x3d1: 300, 0x3d2: 300, 0x3e2: 300, 0x3e3: 300, 0x3e1: 1, 0x3e4: 1, 0x3e5: 0, 0x3e6: 0}
FULL1 = {0x3d1: 700, 0x3d2: 700, 0x3e2: 700, 0x3e3: 700, 0x3e1: 0, 0x3e4: 0, 0x3e5: 1, 0x3e6: 1}
STOP1 = {0x3d1: 199, 0x3d2: 199, 0x3e2: 200, 0x3e3: 199, 0x3e1: 0, 0x3e4: 0, 0x3e5: 1, 0x3e6: 0}
EXPECT = {
    "plain-stop": (FULL0, STOP1),
    "cap-stop": (FULL0, STOP1),
    "plain-start": ({0x3d1: 201, 0x3d2: 201, 0x3e2: 200, 0x3e3: 201, 0x3e1: 0, 0x3e4: 1, 0x3e5: 0, 0x3e6: 0}, STOP1),
    "cap-start": ({}, {0x3d1: 100, 0x3d2: 100, 0x3e2: 100, 0x3e3: 100, 0x3e1: 0, 0x3e4: 0, 0x3e5: 0, 0x3e6: 0}),
}

check("no invalid windows", not glob.glob(f"{d}/*/capprof/*.invalid"), glob.glob(f"{d}/*/capprof/*.invalid"))
# The full run: two windows per program, the counts stopat.c fixes.
full = {}
for k, (a, b) in (("plain", (0, 1)), ("cap", (2, 3))):
    w0, _ = window("full", a); w1, _ = window("full", b)
    check(f"full {k} windows present", w0 is not None and w1 is not None)
    if w0 is None or w1 is None:
        continue
    check(f"full {k} window 0 marks", marks(w0) == FULL0, marks(w0))
    check(f"full {k} window 1 marks", marks(w1) == FULL1, marks(w1))
    full[k] = (w0, w1)
check("full: no stopped file", not os.path.exists(f"{d}/full/capprof/stopped"))
# Program address ranges: plain at its link addresses; Capstone at the base the launcher recorded.
spec = {k: json.load(open(f"{d}/{k}-stop.json")) for k in ("plain", "cap")}
site = {k: spec[k]["sites"]["stop"] for k in spec}
rec = open(f"{d}/full/image-records.txt").read() if os.path.exists(f"{d}/full/image-records.txt") else ""
m = re.search(r"entry=0x([0-9a-f]+) code=0x([0-9a-f]+)-0x([0-9a-f]+)", rec)
check("full cap image record", m, rec.strip())
if m:
    base = int(m.group(2), 16)
    SLIDE["cap"] = base - spec["cap"]["low"]
    check("cap base is a multiple of the grant size", base % spec["cap"]["region"] == 0,
          f"base {base:#x} region {spec['cap']['region']:#x}")
    want_va = {"cap": base + site["cap"]["link"] - spec["cap"]["low"]}
else:
    want_va = {}
want_va["plain"] = site["plain"]["link"]
print(f"INFO sites: plain {site['plain']} cap {site['cap']} (cap offset {site['cap']['link'] - spec['cap']['low']:#x}, slide_align {spec['cap']['region']:#x})")

for boot, (e0, e1) in EXPECT.items():
    k = boot.split("-")[0]
    w0, f0 = window(boot, 0); w1, f1 = window(boot, 1)
    stopped = open(f"{d}/{boot}/capprof/stopped").read().strip() if os.path.exists(f"{d}/{boot}/capprof/stopped") else None
    check(f"{boot} stopped file", stopped is not None, stopped)
    check(f"{boot} windows 0 and 1, no window 2", w0 is not None and w1 is not None
          and not glob.glob(f"{d}/{boot}/capprof/window-2*"))
    if w0 is None or w1 is None or not stopped:
        continue
    kv = dict(x.split("=", 1) for x in stopped.split())
    va = want_va.get(k)
    check(f"{boot} stopped at the symbol's entry 500, window 1",
          va is not None and kv["stopped_at"] == f"{va:x}:500" and kv["window"] == "1", f"{stopped} (expected {va:#x}:500)" if va else stopped)
    check(f"{boot} footer: stopped_at, one candidate", any(l.startswith("# stopped_at=") for l in f1)
          and any(l.startswith("# stop_candidates") and len(l.split()) == 3 for l in f1), [l for l in f1 if "stop" in l])
    if boot.endswith("start"):
        check(f"{boot} footer: started_at", any(l.startswith("# started_at=") for l in f0 + f1), [l for l in f0 + f1 if "start" in l])
    if e0:
        check(f"{boot} window 0 marks", marks(w0) == e0, marks(w0))
    else:
        check(f"{boot} window 0 empty (closed before start_at)", not w0 and any("before start_at" in l for l in f0), f0)
    check(f"{boot} window 1 marks", marks(w1) == e1, marks(w1))
    if k in full:
        fp0, fp1 = program(full[k][0], k), program(full[k][1], k)
        p0, p1 = program(w0, k), program(w1, k)
        if boot.endswith("stop"):
            # Before the stop the slice IS the full run: window 0 equal to the instruction.
            diff = {x: (fp0.get(x), p0.get(x)) for x in set(fp0) | set(p0) if fp0.get(x) != p0.get(x)}
            check(f"{boot} window 0 = full window 0, every program instruction", fp0 and not diff,
                  f"{len(fp0)} instructions, differing {[(hex(a), v) for (a, _), v in list(diff.items())[:8]]}")
        # Window 1 is a prefix of the full window 1: no instruction counted more.
        over = {x: (fp1.get(x, 0), v) for x, v in p1.items() if v > fp1.get(x, 0)}
        check(f"{boot} window 1 <= full window 1, every program instruction", p1 and not over,
              f"{len(p1)} instructions, over {[(hex(a), v) for (a, _), v in list(over.items())[:8]]}")
        s = [v for (a, b), v in p1.items() if a == va]
        # Every completed entry runs the site once and stopat_f's body once; the stopping entry runs neither.
        check(f"{boot} the site's own attempts in window 1 = the body's (the stopping entry excluded)", s == [e1[0x3d1]],
              f"site attempts {s}, body {e1[0x3d1]}")
# A site that faults and is retried across a block boundary (stopat_g, `stopat fault`).
FM = {m: (m << 20) | 0xe0e13 for m in (0x3f2, 0x3f3)}
def fmarks(recs):
    return {m: sum(r[3] for r in recs if r[2] == w and r[1] == 4) for m, w in FM.items()}
for k in ("plain", "cap"):
    sp = json.load(open(f"{d}/{k}-fault.json"))
    g = sp["sites"]["stop"]
    if k == "cap" and "cap" not in SLIDE:
        check("cap fault: base known", False); continue
    gva = g["link"] - sp.get("low", 0) + (SLIDE["cap"] + spec["cap"]["low"] if k == "cap" else 0)
    att = lambda recs, va: sum(r[3] for r in recs if r[0] == va)
    w, f = window(f"{k}-faultcount", 0)
    if w is None:
        check(f"{k}-faultcount window", False); continue
    e = [l for l in f if l.startswith("# stop_entries_so_far")]
    site_full = att(w, gva)
    check(f"{k}-faultcount: 64 entries counted, not stopped", e and e[0].split()[2] == "64"
          and not os.path.exists(f"{d}/{k}-faultcount/capprof/stopped"), e)
    check(f"{k}-faultcount: the case is created (the site faulted and was retried: attempts > 64)",
          site_full > 64 and (k != "plain" or site_full == 128), f"site attempts {site_full}, its successor {att(w, gva + 4)}, marks {fmarks(w)}")
    w, f = window(f"{k}-fault", 0)
    st = open(f"{d}/{k}-fault/capprof/stopped").read().strip() if os.path.exists(f"{d}/{k}-fault/capprof/stopped") else ""
    check(f"{k}-fault: stopped at entry 32 of stopat_g in window 0", st.startswith(f"stopped_at={gva:x}:32 window=0 "), st)
    if w is not None:
        fm = fmarks(w)
        check(f"{k}-fault: 32 calls made, 31 completed", fm == {0x3f2: 32, 0x3f3: 31} and att(w, gva + 4) == 31,
              f"marks {fm}, successor {att(w, gva + 4)}, site attempts {att(w, gva)}")
if os.path.isdir(f"{d}/plain-fault-old"):
    # Negative control: the first plugin counted every arrival at a site that ends its block.
    w, _ = window("plain-fault-old", 0)
    fm = fmarks(w) if w is not None else {}
    print(f"INFO plain-fault-old (first plugin, retries counted as entries): marks {fm}, stopped "
          f"{open(f'{d}/plain-fault-old/capprof/stopped').read().strip() if os.path.exists(f'{d}/plain-fault-old/capprof/stopped') else None}")
    check("negative control: the first plugin stops early on the same case", w is not None and fm.get(0x3f2, 99) < 32, fm)
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
