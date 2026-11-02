#!/usr/bin/env python3
"""Check capprof's validation windows against counts fixed in advance (vt.c, alias.c).
usage: vtcheck.py RUNDIR   (RUNDIR = vt-run.sh's OUT); exit 1 unless every check passes"""
import glob, os, re, sys

d = sys.argv[1]
INVALID = sorted(glob.glob(f"{d}/capprof/window-*.invalid"))
def window(n):
    p = f"{d}/capprof/window-{n}.tsv"
    if not os.path.exists(p):
        bad = glob.glob(f"{d}/capprof/window-{n}.*")
        return None, bad
    recs = []
    for line in open(p):
        if line.startswith('#'):
            continue
        va, ln, b, att, tr = line.split('\t')
        recs.append((int(va, 16), int(ln), int(b, 16), int(att), int(tr)))
    return recs, None

def by_bytes(recs, b):
    return [r for r in recs if r[2] == b]
def at(recs, va):
    return [r for r in recs if r[0] == va]
def stats(n):
    try:
        t = open(f"{d}/run-{n}.err").read()
    except OSError:
        return {}
    m = re.search(r"CAPSTONE_VM_STATS (.*)", t)
    return dict(kv.split('=') for kv in m.group(1).split()) if m else {}

ok = True
def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {detail}")

LOOPS = {'A': (0x3a1e0e13, 0x5a1ece93), 'B': (0x3b1e0e13, 0x5b1ece93), 'C': (0x3c1e0e13, 0x5c1ece93)}
def loop_counts(recs, k):
    """attempts of the 4 body instructions: addi t3 (unique), xori t4 (unique), then +8, +12."""
    first = by_bytes(recs, LOOPS[k][0])
    if len(first) != 1:
        return None, f"{len(first)} records with loop {k}'s first instruction"
    va = first[0][0]
    body = [at(recs, va + off) for off in (0, 4, 8, 12)]
    if any(len(b) != 1 for b in body):
        return None, "body records missing/duplicated"
    return [b[0][3] for b in body], [b[0][4] for b in body]

def marker_counts(recs):
    s = sum(r[3] for r in by_bytes(recs, 0x7a102013)); e = sum(r[3] for r in by_bytes(recs, 0x7a202013))
    return s, e

def window_test(n, label, translated_before):
    recs, bad = window(n)
    if recs is None:
        check(f"W{n} {label}", False, f"no window file ({bad})"); return None
    a, ta = loop_counts(recs, 'A'); b, tb = loop_counts(recs, 'B')
    s, e = marker_counts(recs)
    check(f"W{n} {label} loopA body", a == [12345] * 4, f"{a}")
    check(f"W{n} {label} loopB body", b == [777] * 4, f"{b}")
    check(f"W{n} {label} markers (start in, stop out)", s >= 1 and e == 0, f"start={s} stop={e} (start > 1: retried attempts)")
    if translated_before and ta is not None and not isinstance(ta, str):
        check(f"W{n} {label} loopA translated before the window", ta == [0] * 4, f"translations in window {ta}")
    if tb is not None and not isinstance(tb, str):
        check(f"W{n} {label} loopB first translated in the window", all(t >= 1 for t in tb), f"translations {tb}")
    return a, b

check("no invalid windows", not INVALID, f"{INVALID}")
order = open(f"{d}/order.txt").read()
print(order.strip())
w0 = window_test(0, "domain", True)
w1 = window_test(1, "domain repeat", True)
check("W1 equals W0 (loop regions)", w0 == w1 and w0 is not None, f"{w0} vs {w1}")
recs, bad = window(2)
if recs is not None:
    c, _ = loop_counts(recs, 'C'); st = stats(3)
    steps = int(st.get('steps', -1)); N = 50_000_000
    excess = [x - N for x in c] if c else None
    check("W2 long loop: N <= attempts <= N + steps", c and all(0 <= x <= steps for x in excess),
          f"excess {excess}, steps {steps}, faults {st.get('faults')}")
else:
    check("W2 long", False, f"no window ({bad})")
for n, label, runno in ((3, "prefault", 4), (4, "fault", 5)):
    recs, bad = window(n)
    if recs is None:
        check(f"W{n} {label}", False, f"no window ({bad})"); continue
    li = by_bytes(recs, 0x6d100f13)
    sd = at(recs, li[0][0] + 4) if len(li) == 1 else []
    st = stats(runno)
    detail = f"li {[r[3] for r in li]} sd {[r[3] for r in sd]} module faults (whole run) {st.get('faults')}"
    if label == "prefault":
        check(f"W{n} {label}: 64 stores, no retries", len(li) == 1 and li[0][3] == 64 and sd and sd[0][3] == 64, detail)
    else:
        check(f"W{n} {label}: 64 li, sd = 64 + retries", len(li) == 1 and li[0][3] == 64 and sd and sd[0][3] >= 64, detail
              + (f" -> sd retries {sd[0][3] - 64}" if sd else ""))
window_test(5, "plain", True)
recs, bad = window(6)
if recs is not None:
    al = by_bytes(recs, 0x3e550513)
    print(f"INFO W6 alias records: {[(hex(r[0]), r[3], r[4]) for r in al]}")
    check("W6 alias: two addresses, 3000 and 1000", sorted(r[3] for r in al) == [1000, 3000], "")
recs6, bad6 = window(6)
check("W6 alias window present", recs6 is not None, f"{bad6}")
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
