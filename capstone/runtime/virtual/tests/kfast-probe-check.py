#!/usr/bin/env python3
"""kfast-probe-check.py <kfast=0 transcript> <kfast=1 transcript>

Verdict for one boot of tests/kfast-probe.c run with the module's kfast=0 and then kfast=1. Each transcript is the
guest's output for one setting: an `@@BEFORE` and an `@@AFTER` line carrying the module's kfast* parameters as
name=value words, and the probe's result lines between them.

The probe makes 45 calls that reach the module's delegated-call service. With kfast=1 and nothing in the way they
split as EXPECT below. The service asks whether the step has to stop (budget over, reschedule due) BEFORE it looks
at the call, so a stop that lands on one of the 45 moves that call from its own counter to kfast_fb_stop, and the
launcher serves it. Which call that is depends on timing, so an exact count per counter is not a property of the
module. What is one: no counter above its expected value, all 45 calls accounted for, and every missing count
matched by a stop. A forged call served by the module (served above 8) or filed under the wrong reason (any counter
above its expected value) fails; so does a forged call that went missing without a stop.

Prints EXACT when no stop landed, which is the only reading in which the module itself refused all seven forged
calls. Exit 0 on PASS, 1 on FAIL, 2 when a transcript has no counters or no result lines (no data is not a pass)."""
import re
import sys

EXPECT = {"kfast_served_read": 2, "kfast_served_write": 2, "kfast_served_lseek": 1, "kfast_served_fstat": 2,
          "kfast_served_clock": 1, "kfast_fb_clock": 1, "kfast_fb_file": 13, "kfast_fb_invalid": 7,
          "kfast_fb_nr": 13, "kfast_fb_private": 3}
DISPLACED = ("kfast_fb_stop", "kfast_fb_fault", "kfast_fb_signal", "kfast_fb_touch", "kfast_fb_transport")
FORGED = ("bad-read-offset", "bad-read-length", "bad-read-wrap", "bad-write-offset", "bad-fstat-offset",
          "bad-clock-offset", "bad-write-length")
CALLS = sum(EXPECT.values())


def nodata(why):
    print(why)
    sys.exit(2)


def read(path):
    lines = open(path, errors="replace").read().splitlines()
    res = [l for l in lines if re.match(r"^(ok|bad|priv|clock|direct)-\S+ -?\d+$", l)]
    edge = {}
    for tag in ("@@BEFORE", "@@AFTER"):
        hit = [l for l in lines if l.startswith(tag)]
        if len(hit) != 1:
            nodata(f"NO DATA: {path} has {len(hit)} {tag} lines")
        edge[tag] = {n: int(v) for n, v in re.findall(r"(\w+)=(\d+)", hit[0])}
        if "kfast_served" not in edge[tag]:
            nodata(f"NO DATA: {path}: the {tag} line carries no kfast counters")
    if not res:
        nodata(f"NO DATA: {path} has no result lines")
    delta = {n: edge["@@AFTER"][n] - edge["@@BEFORE"].get(n, 0) for n in edge["@@AFTER"]}
    return res, {n: v for n, v in delta.items() if v}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (res0, cnt0), (res1, cnt1) = read(sys.argv[1]), read(sys.argv[2])
    bad = []
    if res0 != res1:
        bad.append(f"result lines differ: {sorted(set(res0) ^ set(res1))}")
    r0 = dict(l.rsplit(" ", 1) for l in res0)
    for n in FORGED:
        if r0.get(n) != "-14":
            bad.append(f"{n} = {r0.get(n)} (expected -14)")
    if not any(v == "-9" for n, v in r0.items() if n.startswith("priv-")):
        bad.append("no private descriptor answered -9")
    if r0.get("direct-open") != "1":
        bad.append("the O_DIRECT file did not open: the O_DIRECT case was not exercised")
    moved0 = {n: v for n, v in cnt0.items() if n != "kfast_registered"}
    if moved0:
        bad.append(f"kfast=0 moved counters {moved0}")
    for n, want in EXPECT.items():
        if cnt1.get(n, 0) > want:
            bad.append(f"kfast=1 {n} {cnt1.get(n, 0)} above its expected {want}")
    served = sum(cnt1.get(n, 0) for n in EXPECT if n.startswith("kfast_served_"))
    if cnt1.get("kfast_served", 0) != served:
        bad.append(f"kfast=1 kfast_served {cnt1.get('kfast_served', 0)} is not the sum of its kinds {served}")
    displaced = sum(cnt1.get(n, 0) for n in DISPLACED)
    total = sum(cnt1.get(n, 0) for n in EXPECT) + displaced
    if total != CALLS:
        bad.append(f"kfast=1 accounts for {total} calls (expected {CALLS})")
    for n in DISPLACED[1:]:
        if cnt1.get(n, 0):
            bad.append(f"kfast=1 {n} {cnt1[n]} (expected 0: the probe faults nothing and takes no signal)")
    for n in EXPECT:
        if n.startswith("kfast_served_") and not cnt1.get(n, 0):
            bad.append(f"kfast=1 served no {n[len('kfast_served_'):]}")
    for n in ("kfast_fb_invalid", "kfast_fb_private", "kfast_fb_clock"):
        if not cnt1.get(n, 0):
            bad.append(f"kfast=1 {n} 0: that refusal was never exercised in the module")
    stops = cnt1.get("kfast_fb_stop", 0)
    print(f"kfast=0: {len(res0)} result lines, counters moved {cnt0}")
    print(f"kfast=1: {len(res1)} result lines, counters moved {cnt1}")
    print(f"kfast=1: {total} calls accounted for, {stops} taken by a stop, "
          f"forged calls refused in the module {cnt1.get('kfast_fb_invalid', 0)} of {EXPECT['kfast_fb_invalid']}")
    if bad:
        print("PROBE FAIL: " + "; ".join(bad))
        return 1
    print("PROBE PASS" + (" EXACT" if not displaced else f" (a stop took {displaced} of {CALLS} calls)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
