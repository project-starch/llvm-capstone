#!/usr/bin/env python3
"""Node-id allocation policies: what reuse does to footprint, density and misses.

    summarize_ids.py [--md] name=<reports prefix> [...]

<prefix>.npl4.json is Capstone's node cache (16-byte records, four per line) with the ids
as traced; <prefix>.npl4-<policy>.json the same cache with the ids remapped by a policy
(lifo, bitmap, hybrid; idpolicy.h). <prefix>.bucket.json carries the index's variants,
each with its policy. Figures per 1000 lifetime checks; a missing policy file is skipped.
--md prints the three tables as Markdown for the README.
"""
import json
import os
import sys

SIZES = [64, 1024, 16384]
POLICIES = ["traced", "lifo", "bitmap", "hybrid", "chunk"]
ARROW = " → "


def load(prefix):
    cap = {}
    for p in POLICIES:
        f = f"{prefix}.npl4{'' if p == 'traced' else '-' + p}.json"
        if os.path.exists(f):
            cap[p] = json.load(open(f))
    return cap, json.load(open(prefix + ".bucket.json"))


def fmt(x):
    return f"{x:.3f}" if x < 10 else f"{x:.1f}"


def chain(values):
    return ARROW.join(fmt(x) for x in values)


def main():
    md = "--md" in sys.argv
    reps = []
    for a in sys.argv[1:]:
        if a == "--md":
            continue
        name, prefix = a.split("=", 1)
        cap, b = load(prefix)
        reps.append((name, cap, b))
    pols = [p for p in POLICIES if all(p in cap for _, cap, _ in reps)]
    reuse_pols = [p for p in pols if p != "traced"]

    # 1. footprint and density
    if md:
        print("| | allocations | peak live | footprint as traced | footprint with reuse | first reuse at allocation | reused (lifo) | mean density " + " / ".join(reuse_pols) + " |")
        print("|---|---:|---:|---:|---:|---:|---:|---|")
    else:
        print("node table: ids handed out (footprint) and live/footprint (density, mean over allocations)")
        print(f"{'':16}{'allocations':>12}{'peak live':>10}{'footprint':>11}{'w/ reuse':>10}{'1st reuse':>10}{'reused':>8}  density " + "/".join(reuse_pols))
    for name, cap, b in reps:
        t = cap["traced"]["ids"]
        r = {p: cap[p]["ids"] for p in reuse_pols}
        fp = {p: r[p]["footprint"] for p in reuse_pols}
        fp_txt = f"{fp[reuse_pols[0]]:,}" if len(set(fp.values())) == 1 else " / ".join(f"{fp[p]:,}" for p in reuse_pols)
        first = r[reuse_pols[0]].get("first_reuse_allocation", 0)
        first_txt = f"{first:,}" if first else "never"
        l = r.get("lifo", r[reuse_pols[0]])
        reused = 100 * l["reused"] / max(l["allocations"], 1)
        dens = " / ".join(f"{r[p]['density_mean']:.2f}" for p in reuse_pols)
        emu = f" ({t['reused']:,} reused by the emulator)" if t.get("reused") else ""
        if md:
            print(f"| {name} | {t['allocations']:,}{emu} | {t['peak_live']:,} | {t['footprint']:,} | {fp_txt} | {first_txt} | {reused:.1f} % | {dens} |")
        else:
            print(f"{name:16}{t['allocations']:>12,}{t['peak_live']:>10,}{t['footprint']:>11,}{fp_txt:>10}{first_txt:>10}{reused:>7.1f}%  {dens}{emu}")

    # 2. Capstone's node cache
    head = f"Capstone's node cache, 16-byte records (4 per line): misses per 1000 checks, {ARROW.join(pols)}"
    if md:
        print(f"\n{head}\n")
        print("| | " + " | ".join(f"{s} lines" for s in SIZES) + " |")
        print("|---|" + "---|" * len(SIZES))
    else:
        print(f"\n{head}")
    for name, cap, b in reps:
        k = (cap["traced"]["accesses"]["read"]["ldst"] + cap["traced"]["accesses"]["read"]["ldc"]) / 1000
        cells = []
        for s in SIZES:
            full = {p: [c for c in cap[p]["caches"] if c["entries"] == s and c["ways"] == "full"][0] for p in pols}
            cells.append(chain(full[p]["misses"] / k for p in pols))
        if md:
            print(f"| {name} | " + " | ".join(cells) + " |")
        else:
            print(f"{name:16}" + "  ".join(f"{c:>34}" for c in cells))

    # 3. the index
    head = f"the index: misses per 1000 checks, {ARROW.join(pols)}"
    if md:
        print(f"\n{head}\n")
        print("| | records | " + " | ".join(f"{s} lines" for s in SIZES) + " | node creation @64 |")
        print("|---|---|" + "---|" * (len(SIZES) + 1))
    else:
        print(f"\n{head}")
    for name, cap, b in reps:
        k = b["lifetime_checks"] / 1000
        for rec in ((12, 64), (4, 32)):
            vs = {v["policy"]: v for v in b["variants"] if (v["inline"], v["node_bytes"]) == rec}
            if any(p not in vs for p in pols):
                continue
            cells = [chain(vs[p]["misses"][b["sizes_lines"].index(s)] / k for p in pols) for s in SIZES]
            cre = chain(vs[p]["by_operation"]["tree"]["misses_64_lines"] / k for p in pols)
            label = "K%d, %d B" % rec
            if md:
                print(f"| {name if rec[0] == 12 else ''} | {label} | " + " | ".join(cells) + f" | {cre} |")
            else:
                print(f"{name if rec[0] == 12 else '':16}{label:10}" + "  ".join(f"{c:>34}" for c in cells) + f"  creation {cre}")
    errs = [(name, v["policy"], v["errors"]) for name, _, b in reps for v in b["variants"] if v["errors"]]
    if errs:
        print("\nERRORS:", errs)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
