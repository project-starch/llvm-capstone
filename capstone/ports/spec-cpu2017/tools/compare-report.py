#!/usr/bin/env python3
"""The arms side by side, from tools/compare.sh's output: per benchmark and arm the reported seconds of
every repetition (median in bold), the QEMU CPU seconds, the launcher's counters, the ratios against the
first arm, and the repetitions whose host load broke the plan's bound. Measurements under QEMU: ratios
between arms are the result; absolute seconds are emulator time.
usage: compare-report.py OUT"""
import glob, json, os, statistics, sys

out = sys.argv[1]
plan = json.load(open(os.path.join(out, "plan.json")))
labels, bench, reps = plan["labels"], plan["benchmarks"], plan["repetitions"]
# The denominator of every ratio: the bounds-only arm when it is there, else the first arm.
base = "capstone-spatial" if "capstone-spatial" in labels else labels[0]
load = []
for line in open(os.path.join(out, "load.tsv"), errors="replace"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 2 and f[1]:
        load.append((f[0], float(f[1].split()[0])))
def results(lab, b):
    r = []
    for rep in range(1, reps + 1):
        p = os.path.join(out, f"rep{rep}", lab, b, "result.json")
        r.append(json.load(open(p)) if os.path.exists(p) else None)
    return r
def max_load_between(start, end):
    vals = [l for t, l in load if start <= t <= end]
    return max(vals) if vals else None
print(f"# {plan['size']}: {', '.join(labels)}; {reps} repetition(s); host load bound {plan['max_load']}\n")
print("| benchmark | arm | reported s per repetition | median | ratio to " + base + " | QEMU CPU s (median) | nodes / collections / faults (sum over invocations, first passing repetition) | flagged repetitions |")
print("|---|---|---|---|---|---|---|---|")
for b in bench:
    base_ok = [r["reported_seconds"] for r in results(base, b) if r and r["verdict"] == "PASS" and r["reported_seconds"] is not None]
    base_median = statistics.median(base_ok) if base_ok else None
    for lab in labels:
        rs = results(lab, b)
        secs = [r["reported_seconds"] if r and r["verdict"] == "PASS" else None for r in rs]
        ok = [s for s in secs if s is not None]
        med = statistics.median(ok) if ok else None
        ratio = f"{med / base_median:.3f}" if med and base_median else "--"
        passed = [r for r in rs if r and r["verdict"] == "PASS"]
        cpu = [r["qemu_cpu_seconds"] for r in passed if r.get("qemu_cpu_seconds") is not None]
        cpu_med = f"{statistics.median(cpu):.0f}" if cpu else "--"
        st = passed[0]["vm_stats"] if passed else []
        counters = "/".join(str(sum(s.get(k, 0) for s in st if isinstance(s.get(k, 0), int))) for k in ("nodes", "collections", "faults")) if st else "--"
        first_pass = next((i for i, r in enumerate(rs, 1) if r and r["verdict"] == "PASS"), None)
        flagged = []
        for i, r in enumerate(rs, 1):  # noqa: the run's own start/end bracket the load samples
            if not r: flagged.append(f"{i}:missing"); continue
            if r["verdict"] != "PASS": flagged.append(f"{i}:{r['verdict']}")
            ml = max_load_between(r["start"], r["end"])
            if ml is not None and ml > plan["max_load"]: flagged.append(f"{i}:load {ml:.0f}")
        cells = " ".join(("**%.0f**" % s if s == med else "%.0f" % s) if s is not None else "FAIL" for s in secs)
        print(f"| {b} | {lab} | {cells} | {med:.0f} | {ratio} | {cpu_med} | {counters} | {', '.join(flagged) or 'none'} |" if med else
              f"| {b} | {lab} | {cells} | -- | -- | {cpu_med} | {counters} | {', '.join(flagged) or 'none'} |")
print("\nReported seconds are runcpu's for the benchmark's own invocations; QEMU CPU seconds are the guest process's user+system time over the whole run, boot included. Counters come from the launcher's CAPSTONE_VM_STATS lines in the run directory's .err files.")
