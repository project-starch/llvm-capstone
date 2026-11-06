#!/usr/bin/env python3
"""Pre-registered expectations for the check-then-copy race experiment.

usage: judge.py RUNDIR   (RUNDIR holds scenarios.txt and one ID.log per line)

A scenario line is ID|params|variant site page2 touch. Empty params is the
safe protocol: every check must hold. A control names the deliberately broken
variant(s) in params: the listed violation must be observed, otherwise the
control is void (the experiment could not have detected the defect).
Exit status 0 only if all safe scenarios pass and all controls fire."""
import re, sys, pathlib

def parse(path):
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("RACE "):
            d = {}
            for tok in line.split()[1:]:
                k, v = tok.split("=", 1)
                try:
                    d[k] = int(v, 16) if k.endswith("base") else int(v)
                except ValueError:
                    d[k] = v
            return d
    return None

def safe_checks(d):
    var, site, touch = d["variant"], d["site"], d["touch"]
    c = []
    def chk(name, ok): c.append((name, bool(ok)))
    chk("racer ran once", d["fired"] == 1 and d["bruns"] == 1 and d["ctlruns"] == 1)
    chk("no sentinel byte overwritten", d["zone_q"] == 0)
    chk("later valid access works", d["n2"] == 16 and d["n2_q"] == 16)
    if var == "nop":
        chk("benign interleaving completes", d["n"] == 8192 and d["cause"] == 0)
        return c
    if var == "objgc":
        chk("identities collected", d["collections"] == 1)
        chk("stale register refused (cause 24 or 25)", d["cause"] in (24, 25))
    else:
        chk("lifetime cause 25", d["cause"] == 25)
    if site in ("fpre", "fpost"):
        want = {"absent": 0, "cow": 1}[d["p2"]]
        chk("fault path taken", d["resolves"] >= 1)
        chk("page state at fault entry as prepared (0 absent, 1 zero page)", d["pte1"] == want)
        if d["p2"] == "cow":
            chk("zero page prepared by read fault", d["prep"] == 1)
        if site == "fpre" and not touch and d["p2"] == "cow":
            chk("copy-on-write still pending when the copy resolves it", d["pte2"] == 1)
    if site == "hold":
        chk("read fails before copying", d["n"] == -14 and d["done"] == 0)
    else:
        chk("short count 4096", d["n"] == 4096 and d["done"] == 4096)
    if touch and d["zone_len"]:
        chk("sentinel complete", d["zone_s"] == d["zone_len"])
    if var in ("obj", "objgc") and site != "hold" and site != "window":
        chk("first chunk delivered", d["data_q"] == 4096)
    if var == "map" and site != "hold":
        chk("munmap of held arena returns 0", d["munmap_rc"] == 0 and d["unmap_held"] == 1 and d["held"] >= 1)
        chk("new mapping succeeded", d["mmap_rc"] == 0 and d["new_base"] != 0)
        chk("VA not reused while held", d["new_base"] != d["old_base"])
    if var == "map" and site == "hold":
        chk("munmap returns 0", d["munmap_rc"] == 0)
    return c

def control_fires(params, d):
    var, site, touch = d["variant"], d["site"], d["touch"]
    ps = set(params.split())
    if ps == {"rt_nohold"}:
        return d["new_base"] == d["old_base"] and d["new_base"] != 0, "VA reused at once (no quarantine)"
    if "rt_dummy" in ps:
        return d["zone_s"] < d["zone_len"] or d["zone_q"] > 0, "dummy store changed a sentinel byte"
    if "rt_norecheck" in ps and var == "map" and "rt_nohold" not in ps:
        return d["n"] == 8192, "copy continued after the mapping was retired (n=8192)"
    return d["zone_q"] > 0, "sentinel bytes overwritten by the stale copy"

def main():
    run = pathlib.Path(sys.argv[1])
    bad = 0
    for line in (run / "scenarios.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sid, params, args = line.split("|")
        d = parse(run / f"{sid}.log")
        if d is None:
            print(f"{sid:34s} NO-RESULT   {args}")
            bad += 1
            continue
        if not params.strip():
            c = safe_checks(d)
            fails = [n for n, ok in c if not ok]
            print(f"{sid:34s} {'PASS' if not fails else 'FAIL'}        {args}  n={d['n']} cause={d['cause']} zone_q={d['zone_q']} zone_s={d['zone_s']}"
                  + ("" if not fails else "  failed: " + "; ".join(fails)))
            bad += bool(fails)
        else:
            fired, what = control_fires(params, d)
            print(f"{sid:34s} {'CONTROL-FIRES' if fired else 'CONTROL-VOID'} [{params}] {args}  n={d['n']} cause={d['cause']} zone_q={d['zone_q']} zone_s={d['zone_s']} :: {what}")
            bad += not fired
    print("OVERALL", "OK" if not bad else f"{bad} scenario(s) not as pre-registered")
    return 1 if bad else 0

sys.exit(main())
