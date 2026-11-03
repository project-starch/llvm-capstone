#!/usr/bin/env python3
"""micro-table.py ARM=RUNDIR ...: attempts per unit from run-micro.sh runs, split by class.

For every arm (a run-micro.sh output directory, or several joined with '+') and benchmark the
per-unit attempts, each a difference of two windows of the same arm so that the fixed part of a
window (the mapping of the region, the first fork, the parent's own exit) drops out:
  touch    (W(touch 256) - W(touch 4)) / (pages 256 MiB - pages 4 MiB)          per page
  heap     (W(heap R2 B) - W(heap R1 B)) / (allocations R2 - R1), the two
           largest round counts present or MICRO_HEAP=R1,R2                  per allocation
  fork     W(fork 0 R) / R                                                   per fork
           (W(fork K R) - W(fork 0 R)) / (K * R)                             per page (copy-on-write)
  exec     (W(exec 100) - W(exec 10)) / 90                                   per process
  compact  (W(compact C 1) - W(compact C 0)) / migrated                     per migrated page
Classes: program, launcher, module, kernel (kernel idle excluded and reported apart), other
(firmware, job helper, unattributed). A benchmark whose window is missing or whose program failed
is reported as such, never as a zero. Output: one markdown table, and JSON with --json FILE."""
import glob, json, os, re, sys

CLASSES = ("program", "launcher", "module", "kernel")


def windows(run):
    """[(argv, env, classes, out-line)] of the run's invocations that made one window."""
    res = []
    for f in sorted(glob.glob(os.path.join(run, "invocations", "[0-9]" * 4 + ".json"))):
        d = json.load(open(f))
        n = d["detail"]["window"]
        out = open(f[:-5] + ".out").read() if os.path.exists(f[:-5] + ".out") else ""
        lq = [l for l in out.splitlines() if l.startswith("LQ ")]
        if n is None or d["rc"] != 0 or not lq:
            res.append((d["argv"], d.get("env", ""), None, None)); continue
        w = json.load(open(os.path.join(run, "profile", f"window-{n}.json")))
        res.append((d["argv"], d.get("env", ""), w["classes"], lq[-1]))
    return res


def split(c):
    r = {k: c.get(k, 0) for k in CLASSES}
    r["idle"] = c.get("kernel idle", 0)
    r["other"] = sum(v for k, v in c.items() if k not in CLASSES and k != "kernel idle")
    r["busy"] = sum(r[k] for k in CLASSES) + r["other"]
    return r


def diff(a, b, units):
    return {k: (b[k] - a[k]) / units for k in a}


def kv(line):
    return dict(x.split("=", 1) for x in line.split()[3:] if "=" in x)


def table(arm, runs):
    ws = [w for r in runs for w in windows(r)]
    def find(pred):
        hits = [w for w in ws if pred(w[0], w[1])]
        return hits[0] if hits else None
    rows = {}
    t4, t256 = find(lambda a, e: a == ["touch", "4"]), find(lambda a, e: a == ["touch", "256"])
    if t4 and t256 and t4[2] and t256[2]:
        p = int(kv(t256[3])["pages"]) - int(kv(t4[3])["pages"])
        rows["touch / page"] = diff(split(t4[2]), split(t256[2]), p)
    else:
        rows["touch / page"] = "missing or failed"
    first = {}
    for w in ws:   # the first window of each round count (a list may repeat one for noise)
        if w[0][0] == "heap" and w[2] and int(w[0][1]) not in first:
            first[int(w[0][1])] = w
    heaps = sorted(first.items(), key=lambda h: h[0])
    pick = os.environ.get("MICRO_HEAP")   # "R1,R2": the round counts to difference (default: the two largest)
    if pick:
        want = [int(x) for x in pick.split(",")]
        heaps = [h for h in heaps if h[0] in want]
    if len(heaps) >= 2:
        (r1, w1), (r2, w2) = heaps[-2], heaps[-1]
        n = int(kv(w2[3])["allocations"]) - int(kv(w1[3])["allocations"])
        rows["heap / allocation"] = diff(split(w1[2]), split(w2[2]), n) if n else "two equal round counts"
    else:
        failed = [" ".join(w[0]) for w in ws if w[0][0] == "heap" and not w[2]]
        rows["heap / allocation"] = "failed: " + ", ".join(failed) if failed else "missing"
    for tag, child in (("", lambda e: "LQ_FORK_CHILD=1" not in e), (" (child stores)", lambda e: "LQ_FORK_CHILD=1" in e)):
        f0 = find(lambda a, e: a[:2] == ["fork", "0"] and child(e))
        fk = find(lambda a, e: a[0] == "fork" and a[1] != "0" and child(e))
        if not f0 and not fk:
            continue
        if f0 and fk and f0[2] and fk[2]:
            rounds = int(f0[0][2])
            rows["fork / fork" + tag] = {k: v / rounds for k, v in split(f0[2]).items()}
            rows["fork / COW page" + tag] = diff(split(f0[2]), split(fk[2]), int(kv(fk[3])["pages"]))
        else:
            rows["fork" + tag] = "missing or failed"
    e10, e100 = find(lambda a, e: a[:2] == ["exec", "10"]), find(lambda a, e: a[:2] == ["exec", "100"])
    if e10 and e100 and e10[2] and e100[2]:
        rows["exec / process"] = diff(split(e10[2]), split(e100[2]), 90)
    else:
        rows["exec / process"] = "missing or failed"
    c0, c1 = find(lambda a, e: a[0] == "compact" and a[2] == "0"), find(lambda a, e: a[0] == "compact" and a[2] == "1")
    if c0 and c1 and c0[2] and c1[2]:
        m = int(kv(c1[3])["migrated"])
        rows["compact / migrated page"] = diff(split(c0[2]), split(c1[2]), m) if m else f"0 pages migrated"
        rows["compact / migrated page"] = (rows["compact / migrated page"], f"migrated={m} captag_copies={kv(c1[3])['captag_copies']}")
    # repeat windows (noise): the first and last window of the same argv
    noise = {}
    for w in ws:
        same = [x for x in ws if x[0] == w[0] and x[1] == w[1] and x[2]]
        if len(same) >= 2 and " ".join(w[0]) not in noise:
            a, b = split(same[0][2])["busy"], split(same[-1][2])["busy"]
            noise[" ".join(w[0])] = (a, b)
    return rows, noise


def fmt(x):
    return f"{x:,.0f}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--json")]
    jpath = None
    if "--json" in sys.argv:
        jpath = sys.argv[sys.argv.index("--json") + 1]; args.remove(jpath)
    allrows = {}
    print("| arm | unit | busy | program | launcher | module | kernel | other | idle (apart) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for spec in args:
        arm, runs = spec.split("=", 1)
        rows, noise = table(arm, runs.split("+"))
        allrows[arm] = {"rows": rows, "noise": noise}
        for unit, r in rows.items():
            note = ""
            if isinstance(r, tuple):
                r, note = r
            if isinstance(r, str):
                print(f"| {arm} | {unit} | {r} | | | | | | |"); continue
            print(f"| {arm} | {unit}{' (' + note + ')' if note else ''} | {fmt(r['busy'])} | {fmt(r['program'])} | {fmt(r['launcher'])} | "
                  f"{fmt(r['module'])} | {fmt(r['kernel'])} | {fmt(r['other'])} | {fmt(r['idle'])} |")
        for k, (a, b) in noise.items():
            print(f"| {arm} | repeat `{k}` (busy, first / last) | {fmt(a)} / {fmt(b)} ({100 * (b - a) / a:+.2f} %) | | | | | | |")
    if jpath:
        json.dump(allrows, open(jpath, "w"), indent=1)


main()
