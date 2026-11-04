#!/usr/bin/env python3
"""micro-table.py ARM=RUNDIR ...: attempts per delegated call by layer, from run-micro.sh runs of
micro.c. For every (mode, N, direct) present at two round counts R1 < R2 in a run,
the per-call figure is (W(R2) - W(R1)) / (R2 - R1) for each class (program, launcher, module, kernel,
kernel idle excluded and shown apart). A pair with a missing window or a failed program is an
ERROR line, never a zero. Exit 1 if any pair is missing."""
import glob, json, os, sys

CLASSES = ("program", "launcher", "module", "kernel")


def runs(rundir):
    out = {}
    for f in sorted(glob.glob(os.path.join(rundir, "invocations", "[0-9]" * 4 + ".json"))):
        d = json.load(open(f))
        a = d["argv"]
        if len(a) < 3 or a[0] not in ("pipe", "file", "pipew"):
            continue
        key = (a[0], int(a[1]), len(a) > 3 and a[3] == "direct")
        r = int(a[2])
        lq = [l for l in open(f[:-5] + ".out").read().splitlines() if l.startswith("LQ ")] \
            if os.path.exists(f[:-5] + ".out") else []
        n = d["detail"]["window"]
        if n is None or d["rc"] != 0 or not lq or " bad=0" not in lq[-1]:
            out.setdefault(key, {})[r] = None
            continue
        w = json.load(open(os.path.join(rundir, "profile", f"window-{n}.json")))
        out.setdefault(key, {})[r] = w["classes"]
    return out


def main(argv):
    if not argv:
        print("usage: micro-table.py ARM=RUNDIR ...", file=sys.stderr)
        return 2
    bad = 0
    print("| arm | mode | N | direct | program | launcher | module | kernel | busy total | idle |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for spec in argv:
        arm, rundir = spec.split("=", 1)
        rows = runs(os.path.expanduser(rundir))
        if not rows:
            print(f"ERROR {arm}: no I/O invocations in {rundir}")
            bad = 1
        for key, byr in sorted(rows.items()):
            rs = sorted(byr)
            if len(rs) < 2 or byr[rs[0]] is None or byr[rs[-1]] is None:
                print(f"ERROR {arm} {key}: rounds {rs}, a window missing or a program failed")
                bad = 1
                continue
            a, b, du = byr[rs[0]], byr[rs[-1]], rs[-1] - rs[0]
            per = {k: (b.get(k, 0) - a.get(k, 0)) / du for k in CLASSES}
            idle = (b.get("kernel idle", 0) - a.get("kernel idle", 0)) / du
            other = sum((b.get(k, 0) - a.get(k, 0)) for k in set(a) | set(b)
                        if k not in CLASSES and k != "kernel idle") / du
            total = sum(per.values()) + other
            print(f"| {arm} | {key[0]} | {key[1]} | {int(key[2])} | " +
                  " | ".join(f"{per[k]:,.0f}" for k in CLASSES) + f" | {total:,.0f} | {idle:,.0f} |")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
