#!/usr/bin/env python3
"""perl-split-sum.py ARM=PREFIX ... [--full ARM=PROFILE_RUN_DIR]: the perlbench split's windows summed per arm.

PREFIX is the common prefix of perl-split-one.sh output directories (e.g. ~/lanes/laneN/ps/pinned-g),
one per group. A group counts only with check.json verdict PASS (its scripts' verdicts equal SPEC's,
"All tests successful.", one window, and for Capstone one image range); a missing or failed group
makes the arm's sum INCOMPLETE, printed as such. --full names a tools/profile.sh run of the whole
test input; its window of test.pl (the invocation whose argv ends in test.pl) is the reference.
--common: sum every arm over only the groups that PASS in all arms named (a like-for-like comparison when an
arm fails some groups); the groups left out are listed."""
import glob, json, os, sys

CLS = ("program", "launcher", "module", "kernel", "kernel idle")


def classes(c):
    r = {k: c.get(k, 0) for k in CLS}
    r["other"] = sum(v for k, v in c.items() if k not in CLS)
    r["total"] = sum(c.values())
    r["busy"] = r["total"] - r["kernel idle"]
    return r


def passed(prefix):
    ok = set()
    for d in glob.glob(prefix + "[0-9][0-9]"):
        chk = os.path.join(d, "check.json")
        if os.path.exists(chk) and json.load(open(chk))["verdict"] == "PASS":
            ok.add(d[-2:])
    return ok


def arm_sum(prefix, only=None):
    dirs = sorted(glob.glob(prefix + "[0-9][0-9]"))
    if only is not None:
        dirs = [d for d in dirs if d[-2:] in only]
    tot, bad = None, []
    for d in dirs:
        chk = os.path.join(d, "check.json")
        prof = os.path.join(d, "profile", "window-0.json")
        if not os.path.exists(chk) or json.load(open(chk))["verdict"] != "PASS" or not os.path.exists(prof):
            bad.append(os.path.basename(d)); continue
        c = classes(json.load(open(prof))["classes"])
        tot = c if tot is None else {k: tot[k] + c[k] for k in c}
    return tot, len(dirs), bad


def full_window(run):
    for f in sorted(glob.glob(os.path.join(run, "invocations", "[0-9]" * 4 + ".json"))):
        d = json.load(open(f))
        if d.get("route") in ("guest", "guest-plain") and d.get("argv", [""])[-1] == "test.pl":
            n = d["detail"]["window"]
            return classes(json.load(open(os.path.join(run, "profile", f"window-{n}.json")))["classes"])
    return None


args = sys.argv[1:]
common = "--common" in args
if common:
    args.remove("--common")
fulls = {}
while "--full" in args:
    i = args.index("--full"); arm, run = args[i + 1].split("=", 1); fulls[arm] = run; del args[i:i + 2]
cols = ("total", "busy", "program", "launcher", "module", "kernel", "kernel idle", "other")
print("| arm | groups PASS | " + " | ".join(cols) + " |")
print("|---|---|" + "---|" * len(cols))
only = None
if common:
    sets = [passed(spec.split("=", 1)[1]) for spec in args]
    only = set.intersection(*sets)
    print(f"common groups: {len(only)} of 32; left out: {' '.join(sorted(set(f'{i:02d}' for i in range(32)) - only))}")
for spec in args:
    arm, prefix = spec.split("=", 1)
    tot, n, bad = arm_sum(prefix, only)
    state = f"{n - len(bad)}/{n}" + (f" INCOMPLETE (missing/failed: {' '.join(bad)})" if bad or (n != 32 and not common) else "")
    if tot:
        print(f"| {arm} split sum | {state} | " + " | ".join(f"{tot[k]:,}" for k in cols) + " |")
    else:
        print(f"| {arm} split sum | {state} | " + " | ".join("-" for _ in cols) + " |")
    if arm in fulls:
        f = full_window(fulls[arm])
        if f is None:
            print(f"| {arm} full test.pl window | no window in {fulls[arm]} |"); continue
        print(f"| {arm} full test.pl window | 1 run | " + " | ".join(f"{f[k]:,}" for k in cols) + " |")
        if tot and not bad and n == 32 and not common:
            print(f"| {arm} split - full | | " + " | ".join(f"{tot[k] - f[k]:+,} ({100 * (tot[k] - f[k]) / f[k]:+.2f} %)" if f[k] else f"{tot[k] - f[k]:+,}" for k in cols) + " |")
