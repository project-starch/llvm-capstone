#!/usr/bin/env python3
"""Figures for the reference-count study, from the same reports summarize_rc.py reads.

    plot_rc.py [--before DIR] [--models DIR] --out DIR name=<reports prefix> [...]

Writes four SVGs into DIR:
  rc-accesses.svg     metadata accesses per 1000 instructions, stacked by origin, against
                      Clover's stream and the index on the same trace
  rc-misses.svg       misses per 1000 instructions against the cache size, one panel per
                      program, one line per design
  rc-origin.svg       the count's exposed updates against the program's memory operations
  rc-reclamation.svg  allocations, ids the count used, and the live set by the revoke measure
Needs matplotlib (the venv has it).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".svg"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=110)
    plt.close(fig)

from summarize_rc import load, sibling  # noqa: E402


def kib(lines):
    return lines * 64 / 1024


def all_misses(report, k):
    """misses per 1000 instructions at every fully associative size, {KiB: misses}"""
    return {kib(c["entries"]): c["misses"] / k for c in report["caches"] if c["ways"] == "full"}


def model_file(prefix, models, suffix):
    """this run's model report, or the one of the run without the count"""
    f = prefix + suffix
    if os.path.exists(f):
        return f
    return sibling(models, prefix, suffix) if models else None


def gather(names, before, models):
    rows = []
    for a in names:
        name, prefix = a.split("=", 1)
        r = load(prefix, before, models)
        r["name"] = name
        k = r["k"]
        r["curve_literal"] = all_misses(json.load(open(prefix + ".npl4-nosweep.json")), k)
        r["curve_norc"] = all_misses(json.load(open(prefix + ".npl4-norc.json")), k)
        r["curve_elided"] = all_misses(json.load(open(prefix + ".npl4-rcelide.json")), k)
        for tag, key in (("nosweep", "curve_literal"), ("norc", "curve_norc"), ("rcelide", "curve_elided")):
            rep_ = json.load(open(f"{prefix}.npl4-{tag}.json"))
            after = rep_.get("after_last_allocation")
            if after:   # the run without its teardown
                tear = {kib(c["entries"]): c["misses"] / k for c in after["full_caches"]}
                r[key] = {s: m - tear.get(s, 0) for s, m in r[key].items()}
        r["wb_literal"] = {kib(c["entries"]): c.get("writebacks", 0) / k for c in json.load(open(prefix + ".npl4-nosweep.json"))["caches"] if c["ways"] == "full"}
        r["wb_norc"] = {kib(c["entries"]): c.get("writebacks", 0) / k for c in json.load(open(prefix + ".npl4-norc.json"))["caches"] if c["ways"] == "full"}
        f = model_file(prefix, models, ".clover.json")
        if f:
            cl = json.load(open(f))
            r["curve_clover"] = {kib(s): m / (cl["lifetime_checks"] / 1000) * r["checks"] / 1000 for s, m in zip(cl["sizes_lines"], cl["clover"]["misses"])}
        f = model_file(prefix, models, ".bucket.json")
        r["index_acc"] = 0
        if f:
            b = json.load(open(f))
            v = ([v for v in b["variants"] if (v["inline"], v["node_bytes"], v.get("policy", "traced")) == (4, 32, "bitmap")]
                 or [v for v in b["variants"] if (v["inline"], v["node_bytes"]) == (4, 32)])[0]   # older reports: no policy
            kb = b["lifetime_checks"] / 1000
            r["index_acc"] = v["accesses"] / kb * r["checks"] / 1000
            r["curve_index"] = {kib(s): m / kb * r["checks"] / 1000 for s, m in zip(b["sizes_lines"], v["misses"])}
        if before:
            f = sibling(before, prefix, ".npl4.json")
            if f:
                bb = json.load(open(f))
                kb = (bb["accesses"]["read"]["ldst"] + bb["accesses"]["read"]["ldc"]) / r["checks"]
                r["curve_before"] = all_misses(bb, kb)
        rows.append(r)
    return rows


def fig_accesses(rows, out):
    fig, ax = plt.subplots(figsize=(1.4 * len(rows) + 2, 4.6))
    x = range(len(rows))
    w = 0.27
    base = [r["checks"] + r["tree"] for r in rows]
    ax.bar([i - w for i in x], base, w, color="#888", label="Capstone without the count: checks and tree ops")
    bottoms = base[:]
    for key, color, label in (("reg", "#c0392b", "count: exposed (register writes)"),
                              ("hideable", "#e67e22", "count: hideable (loads, stores)"),
                              ("same", "#f1c40f", "pairs, literal only: a slot rewritten with its node"),
                              ("call", "#f7dc6f", "pairs: the return register at calls"),
                              ("move", "#fdf2a0", "pairs: linear moves")):
        vals = [r["ld"] + r["mem"] if key == "hideable" else r[key] for r in rows]
        ax.bar([i - w for i in x], vals, w, bottom=bottoms, color=color, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    clover = [r["clover"]["acc"] if r["clover"] else 0 for r in rows]
    index = [r["index_acc"] for r in rows]
    ax.bar(list(x), clover, w, color="#2980b9", label="Clover, the paper's baseline")
    ax.bar([i + w for i in x], index, w, color="#27ae60", label="the index (K4/32 B, slot cache 64)")
    for i, r in enumerate(rows):
        ax.text(i, clover[i], f"{clover[i]:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w, index[i], f"{index[i]:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i - w, bottoms[i], f"{bottoms[i]:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i - w, base[i] / 2, f"{base[i]:.0f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["name"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("metadata accesses per 1000 instructions")
    ax.set_title("the logical access streams on the same trace, per 1000 instructions")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, out, "rc-accesses")


def fig_misses(rows, out):
    rows = [r for r in rows if max(r["curve_literal"].values()) > 0.05]
    n = len(rows)
    cols = min(4, n)
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(3.4 * cols, 3 * rws), squeeze=False)
    for ax, r in zip(axes.flat, rows):
        for key, color, label, ls in (("curve_before", "#888", "Capstone, no count", "-"),
                                      ("curve_norc", "#555", "the count's reuse alone", "--"),
                                      ("curve_elided", "#e67e22", "with the count, pairs elided", "-"),
                                      ("curve_literal", "#c0392b", "with the count, literal", "-"),
                                      ("curve_clover", "#2980b9", "Clover, baseline", "-"),
                                      ("curve_index", "#27ae60", "the index K4/32 B", "-")):
            if key in r:
                xs = sorted(r[key])
                ax.plot(xs, [max(r[key][s], 1e-4) for s in xs], marker="o", ms=3, color=color, label=label, ls=ls)
        ax.axvline(8, color="#aaa", lw=0.8, ls=":")
        ax.text(8, ax.get_ylim()[1], " paper: 8 KB", fontsize=6, color="#777", va="top")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(r["name"], fontsize=9)
        ax.set_xlabel("metadata cache, KiB", fontsize=8)
        ax.set_ylabel("misses per 1000 instructions", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="major", alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, out, "rc-misses")


def fig_writebacks(rows, out):
    rows = [r for r in rows if r.get("wb_literal") and max(r["wb_literal"].values()) > 0.01]
    if not rows:
        return
    n = len(rows)
    cols = min(4, n)
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(3.4 * cols, 3 * rws), squeeze=False)
    for ax, r in zip(axes.flat, rows):
        for key, color, label in (("wb_norc", "#555", "without the count's writes"), ("wb_literal", "#c0392b", "with the count")):
            xs = sorted(r[key])
            ax.plot(xs, [max(r[key][s], 1e-4) for s in xs], marker="o", ms=3, color=color, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(r["name"], fontsize=9)
        ax.set_xlabel("metadata cache, KiB", fontsize=8)
        ax.set_ylabel("write-backs per 1000 instructions", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="major", alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, out, "rc-writebacks")


def fig_origin(rows, out):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for r in rows:
        ax.scatter(r["memops"], r["reg"], color="#c0392b")
        ax.annotate(r["name"], (r["memops"], r["reg"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("memory operations per 1000 instructions")
    ax.set_ylabel("exposed count updates per 1000 instructions")
    ax.set_title("exposed count updates against memory operations", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, out, "rc-origin")


def fig_reclamation(rows, out):
    fig, ax = plt.subplots(figsize=(1.3 * len(rows) + 2, 4))
    x = range(len(rows))
    w = 0.27
    ax.bar([i - w for i in x], [r["allocs"] for r in rows], w, color="#888", label="allocations")
    ax.bar(list(x), [r["footprint"] for r in rows], w, color="#c0392b", label="ids the count used")
    ax.bar([i + w for i in x], [r["live"] for r in rows], w, color="#2980b9", label="live set (nodes alive until their revoke)")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["name"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("nodes")
    ax.set_title("reclamation: the table the count needs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "rc-reclamation")


def main():
    before = out = models = None
    names = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--before":
            before = next(it)
        elif a == "--models":
            models = next(it)
        elif a == "--out":
            out = next(it)
        else:
            names.append(a)
    if not out or not names:
        print(__doc__)
        return 2
    os.makedirs(out, exist_ok=True)
    rows = gather(names, before, models)
    fig_accesses(rows, out)
    fig_misses(rows, out)
    fig_origin(rows, out)
    fig_reclamation(rows, out)
    fig_writebacks(rows, out)
    print("wrote", ", ".join(f"rc-{f}.svg" for f in ("accesses", "misses", "origin", "reclamation", "writebacks")), "to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
