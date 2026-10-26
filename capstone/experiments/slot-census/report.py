#!/usr/bin/env python3
"""Aggregate every pass in results/ into one report: REPORT.md and, when matplotlib is available, figures/.

usage: report.py [RESULTS_DIR] [OUT_DIR]   (defaults: results/ and this directory)
Each valid cell becomes one row: suite, program, dataset, the cell's peak (the maximum over its measured processes)
and which slot counts it fits. Cells are classed by what the allocator below would see:
  application  every heap object is a malloc (SPEC, Redis on libc malloc, SQLite with lookaside off, CPython with
               PYTHONMALLOC=malloc)
  hidden       the program's own allocator carves objects from blocks it mallocs (memcached's slabs,
               PostgreSQL's memory contexts, RocksDB's memtable arena, nginx's and Apache's pools, SpiderMonkey's
               garbage-collected heap in Firefox, SQLite's lookaside, CPython's pymalloc):
               the census counts the blocks, so the row is the count the allocator below sees with the program
               unmodified, not the number of objects
  stress       allocator stress tests (mimalloc-bench), reported apart
A cell that is not valid, or has no census line, is listed under "Excluded cells" with its reason, never as a zero.
The slot counts are 2^16, 2^20 and 2^24; "half" is a cell at or below half of a count."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import cells, processes  # noqa: E402

SLOTS = (16, 20, 24)


def classify(pass_name, provenance, cell):
    """(suite, program, dataset, class) for one cell"""
    if m := re.match(r"sqlite-(lookaside|nolookaside)-(\d+)$", cell):
        on = m[1] == "lookaside"
        return ("SQLite 3.22.0 speedtest1", "speedtest1, lookaside " + ("on" if on else "off"), f"size {m[2]}",
                "hidden" if on else "application")
    if m := re.match(r"ycsb-(redis|memcached)-(\d+)$", cell):
        if m[1] == "redis":
            return "Redis 7.2.16 (MALLOC=libc), YCSB", "redis-server", f"{int(m[2]):,} records", "application"
        return "memcached 1.6.45, YCSB", "memcached", f"{int(m[2]):,} records", "hidden"
    if cell in ("nginx", "httpd"):
        return ({"nginx": "nginx 1.20.2, wrk", "httpd": "Apache httpd 2.4.58, ab"}[cell], cell, "published setup",
                "hidden")
    if m := re.match(r"speedometer-([\d.]+)$", cell):
        return f"Firefox Nightly, Speedometer {m[1]}", "firefox", f"Speedometer {m[1]}", "hidden"
    if m := re.match(r"rocksdb-(\d+)$", cell):
        return "RocksDB db_bench", "db_bench", f"{int(m[1]):,} keys", "hidden"
    if m := re.match(r"parsec-(\w+)-(\w+)$", cell):
        return "PARSEC 3.0", m[2], m[1], "application"
    if m := re.match(r"memcached-(\d+)(?:-r(\d+))?$", cell):
        mix = int(m[2] or 20000)
        return "memcached 1.6.45, memtier", "memcached", f"{int(m[1]):,} items, mix {mix:,}/client", "hidden"
    if m := re.match(r"postgres-(\d+)$", cell):
        return "PostgreSQL 17.6 pgbench", "postgres", f"scale {m[1]}", "hidden"
    if m := re.match(r"py-(pymalloc|malloc)-(.+)$", cell):
        return ("pyperformance (CPython 3.13.7)", m[2], f"PYTHONMALLOC={m[1]}",
                "hidden" if m[1] == "pymalloc" else "application")
    if m := re.match(r"octane-(.+)$", cell):
        return "Octane on QuickJS 2026-06-04", m[1], "Octane", "application"
    if m := re.match(r"mb-(.+)$", cell):
        return "mimalloc-bench", m[1], "bench.sh arguments", "stress"
    if re.match(r"\d{3}\.\w+_r$", cell):
        size = re.search(r"^size (\w+)$", provenance, re.M)
        return "SPEC CPU2017 rate", cell, size[1] if size else "?", "application"
    return "unclassified", cell, pass_name, "unclassified"


def fits(peak):
    return {i: peak <= 2 ** i for i in SLOTS} | {f"half{i}": peak <= 2 ** (i - 1) for i in SLOTS}


def note(d, cell):
    """the runner's own note on a cell (input-missing, exit codes), from cells.tsv's fourth column"""
    path = os.path.join(d, "cells.tsv")
    for line in open(path) if os.path.exists(path) else []:
        f = line.rstrip("\n").split("\t")
        if f[0] == cell and len(f) > 3:
            return f[3]
    return ""


def collect(results):
    rows, excluded = [], []
    for p in sorted(os.listdir(results)):
        d = os.path.join(results, p)
        if not os.path.isdir(d):
            continue
        prov_path = os.path.join(d, "provenance.txt")
        prov = open(prov_path).read() if os.path.exists(prov_path) else ""
        if "measure instructions" in prov:   # slotins passes are figure 2's denominators, not census passes
            continue
        for cell, keep, verdict in cells(d):
            procs = processes(d, cell, keep)
            suite, program, dataset, klass = classify(p, prov, cell)
            if verdict != "valid" or not procs:
                excluded.append((p, cell, verdict if verdict != "valid" else "no census line", note(d, cell)))
                continue
            top = max(procs, key=lambda r: r["peak_live"])
            rows.append({"pass": p, "cell": cell, "suite": suite, "program": program, "dataset": dataset,
                         "class": klass, "peak": top["peak_live"], "processes": len(procs),
                         "bytes": max(r["peak_live_bytes"] for r in procs)})
    return rows, excluded


def dataset_order(dataset):
    """test < train < ref, else the number in the label"""
    if dataset in ("test", "train", "ref"):
        return ("test", "train", "ref").index(dataset), ""
    m = re.search(r"[\d,]+", dataset)
    return int(m[0].replace(",", "")) if m else 0, dataset


def mark(ok):
    return "yes" if ok else "**no**"


def collect_rates(results):
    """figure 2's rows: per cell of an instruction pass, allocations per 1,000 instructions over its measured
    processes (allocations and instructions from the same run, slotins.c)"""
    rows = []
    for p in sorted(os.listdir(results)):
        d = os.path.join(results, p)
        prov_path = os.path.join(d, "provenance.txt")
        prov = open(prov_path).read() if os.path.exists(prov_path) else ""
        if not os.path.isdir(d) or "measure instructions" not in prov:
            continue
        for cell, keep, verdict in cells(d):
            procs = processes(d, cell, keep)
            insns = sum(r["instructions"] for r in procs)
            if verdict != "valid" or not procs or insns == 0:
                continue
            suite, program, dataset, klass = classify(p, prov, cell)
            allocs = sum(r["allocs"] for r in procs)
            rows.append({"pass": p, "cell": cell, "suite": suite, "program": program, "dataset": dataset,
                         "class": klass, "peak": 1000 * allocs / insns, "allocs": allocs, "instructions": insns})
    return rows


def write_report(rows, excluded, out, pairs=()):
    lines = ["# Slot census report", "",
             "Generated by `report.py` from `results/`; every number is a cell's peak of simultaneously live heap "
             "objects in one process (the maximum over the cell's measured processes). Classes are defined in "
             "`report.py`: *application* cells count every object, *hidden* cells count the blocks a program's own "
             "allocator mallocs, *stress* cells are allocator stress tests.", ""]
    lines += ["## Across suites", "",
              "| class | cells | max peak | above 2^16 | above 2^20 | above 2^24 | above half of 2^24 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for klass in ("application", "hidden", "stress", "unclassified"):
        sel = [r for r in rows if r["class"] == klass]
        if not sel:
            continue
        over = {i: sum(r["peak"] > 2 ** i for r in sel) for i in SLOTS}
        lines.append(f"| {klass} | {len(sel)} | {max(r['peak'] for r in sel):,} | {over[16]} | {over[20]} | "
                     f"{over[24]} | {sum(r['peak'] > 2 ** 23 for r in sel)} |")
    lines += ["", "Cells above half of 2^24 (8,388,608):", ""]
    big = sorted((r for r in rows if r["peak"] > 2 ** 23), key=lambda r: -r["peak"])
    lines += [f"- {r['suite']}: {r['program']}, {r['dataset']}: {r['peak']:,} ({r['class']})" for r in big]
    lines += [] if big else ["- none"]
    for suite in sorted({r["suite"] for r in rows}):
        sel = sorted((r for r in rows if r["suite"] == suite),
                     key=lambda r: (r["program"], dataset_order(r["dataset"])))
        lines += ["", f"## {suite}", "",
                  "| program | dataset | class | peak live objects | peak live bytes "
                  "| ≤2^16 | ≤2^20 | ≤2^24 | pass |",
                  "|---|---|---|---:|---:|---|---|---|---|"]
        for r in sel:
            f = fits(r["peak"])
            lines.append(f"| {r['program']} | {r['dataset']} | {r['class']} | {r['peak']:,} | {r['bytes']:,} | "
                         f"{mark(f[16])} | {mark(f[20])} | {mark(f[24])} | {r['pass']} |")
    if pairs:   # the peak-rate figure's points, as a table (its marks carry no value labels)
        lines += ["", "## Peak against allocation rate", "",
                  "Every cell with a census peak and an instruction pass, as drawn in `figures/peak-rate.pdf`.", "",
                  "| suite | program | dataset | peak live objects | allocations per 1,000 instructions |",
                  "|---|---|---|---:|---:|"]
        lines += [f"| {p[0]} | {p[1]} | {p[2]} | {p[3]:,} | {p[4]:.3g} |"
                  for p in sorted(pairs, key=lambda p: (p[0], p[1], dataset_order(p[2])))]
    lines += ["", "## Excluded cells", ""]
    valid = {(r["cell"], r["pass"]) for r in rows}
    for p, c, why, extra in excluded:
        again = sorted(q for cc, q in valid if cc == c)
        lines.append(f"- {p} {c}: {why}" + (f" ({extra})" if extra else "")
                     + (f", valid in {', '.join(again)}" if again else ""))
    lines += [] if excluded else ["- none"]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(lines) + "\n")


# The figure: one message, one panel. Every benchmark program's peak lies below 2^24 slots and a handful lie above
# 2^20. Suites are columns: programs are dots, scripts on one interpreter or data set sizes of one program are ticks
# with a diamond at the maximum. Redis is a column of its data set sizes, the one thing that crosses 2^24. No value
# labels (the caption carries the numbers), black and grays only, so it prints and reads without colour.
COLUMNS = (   # label, input shown in the caption, unit, row filter
    ("SPEC CPU2017", "ref", "programs",
     lambda r: r["suite"] == "SPEC CPU2017 rate" and r["dataset"] == "ref"),
    ("PARSEC", None, "programs", lambda r: r["suite"] == "PARSEC 3.0"),
    ("pyperformance", "CPython 3.13, PYTHONMALLOC=malloc", "scripts",
     lambda r: r["suite"].startswith("pyperformance") and r["class"] == "application"),
    ("Octane", "QuickJS", "scripts", lambda r: r["suite"].startswith("Octane")),
    ("Firefox", "Nightly, Speedometer 2.1 and 3.1 (its largest process each), a lower bound: SpiderMonkey's heap is "
     "not counted", "runs",
     lambda r: r["suite"].startswith("Firefox")),
    ("SQLite", "speedtest1, lookaside off", "sizes",
     lambda r: r["suite"].startswith("SQLite") and r["class"] == "application"),
    ("Redis", "libc malloc, YCSB load + A + C, 10^5 to 1.6x10^7 records", "sizes",
     lambda r: r["suite"].startswith("Redis") and "YCSB" in r["suite"]),
)
INK, GRAY, LIGHT = "#000000", "#7a7a7a", "#efefef"


def figure_data(rows):
    cols = []
    for label, shown, unit, keep in COLUMNS:
        sel = [r for r in rows if keep(r)]
        if label == "PARSEC":   # each program at its largest measured input: native where intact, else simlarge
            native = {r["program"] for r in sel if r["dataset"] == "native"}
            sel = [r for r in sel if r["dataset"] == "native" or r["program"] not in native]
            fallback = sorted({r["program"] for r in sel if r["dataset"] != "native"})
            shown = "native" + (f" (simlarge for {', '.join(fallback)}: native input lost)" if fallback else "")
        best = {}   # a benchmark rerun in a second pass (the network ones run alone) counts once, at its maximum
        for r in sel:
            key = (r["program"], r["dataset"])
            if key not in best or r["peak"] > best[key]["peak"]:
                best[key] = r
        cols.append({"label": label, "shown": shown, "unit": unit,
                     "points": sorted(((r["program"], r["peak"], r["dataset"]) for r in best.values()),
                                      key=lambda t: t[1])})
    return cols


def beeswarm(values, x0, ax_in, xspan_in, d_in, ymin, ymax, height_in):
    """deterministic offsets: each point takes the free x nearest the column centre, in inches on the page"""
    from math import log10
    per_decade = height_in / (log10(ymax) - log10(ymin))
    placed, xs = [], []
    for v in values:
        y = log10(max(v, 1)) * per_decade
        for k in range(200):
            dx = (k + 1) // 2 * d_in * (1 if k % 2 else -1)
            if abs(dx) > xspan_in:
                dx = 0.0
                break
            if all((dx - px) ** 2 + (y - py) ** 2 >= d_in ** 2 for px, py in placed):
                break
        placed.append((dx, y))
        xs.append(x0 + dx / ax_in)
    return xs


def write_caption(cols, path):
    """the caption draft, every number taken from the data the figure draws (no semicolons, as the paper's style)"""
    programs = [c for c in cols if c["points"] and c["label"] != "Redis"]
    top = max((p for c in programs for p in c["points"]), key=lambda p: p[1])
    over20 = sum(p[1] > 2 ** 20 for c in programs for p in c["points"])
    marks = sum(len(c["points"]) for c in programs)
    lines = [
        "Peak number of simultaneously live heap objects in one process over a whole run, measured natively on x86 "
        "with an exact allocation tracer. Dots are benchmark programs. Ticks are scripts on one interpreter, data "
        "set sizes or benchmark runs of one program, with a diamond at the maximum. Marks at the same value overlap.",
        " ".join(f"{c['label']}: {c['shown']}, {len(c['points'])} {c['unit']}"
                 f"{'' if c['unit'] == 'programs' else ' of one program'}." for c in cols if c["points"]),
        f"Without Redis, {over20} of {marks} marks exceed 2^20 slots and none reaches 2^24. The largest is {top[0]} "
        f"({top[1]:,} objects, {100 * top[1] / 2 ** 24:.0f}% of 2^24).",
    ]
    redis = next((c for c in cols if c["label"] == "Redis" and c["points"]), None)
    if redis:
        ks = [int(re.search(r"([\d,]+) records", p[2])[1].replace(",", "")) for p in redis["points"]]
        vs = [p[1] for p in redis["points"]]
        k_mean, v_mean = sum(ks) / len(ks), sum(vs) / len(vs)
        a = sum((k - k_mean) * (v - v_mean) for k, v in zip(ks, vs)) / sum((k - k_mean) ** 2 for k in ks)
        b = v_mean - a * k_mean
        lines.append(f"Redis holds {a:.2f} live objects per record plus {b:,.0f} (a fit to the {len(ks)} sizes), so "
                     f"it passes 2^24 at about {(2 ** 24 - b) / a / 1e6:.2f} M records. At {ks[-1]:,} records it "
                     f"holds {vs[-1]:,} objects.")
    open(path, "w").write("\n\n".join(lines) + "\n")


def strip_figure(cols, ylabel, thresholds, base):
    """one panel, suites as columns: programs as dots, scripts, sizes or runs of one program as ticks with a diamond
    at the maximum; thresholds are 2^i lines (the band between them shaded), drawn when given"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import ceil, floor, log10
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    n = len(cols)
    W, H, AX_W, AX_H = 7.4, 3.1, 6.3, 2.6   # double-column width: up to seven bold suite titles side by side
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.55 / W, 0.3 / H, AX_W / W, AX_H / H))
    vals_all = [p[1] for c in cols for p in c["points"] if p[1] > 0]
    lo, hi = min(vals_all, default=1), max(vals_all, default=1)
    if thresholds:   # counts: from 1 up, every mark inside (Redis's largest data set included)
        ymin, ymax = 1, 10.0 ** max(8, ceil(log10(hi * 1.5)))
    else:            # rates: the decades the marks span
        ymin, ymax = 10.0 ** floor(log10(lo / 1.5)), 10.0 ** ceil(log10(hi * 1.5))
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.5, n - 0.5)
    if thresholds:
        a, b = thresholds
        ax.axhspan(2 ** a, 2 ** b, color=LIGHT, lw=0, zorder=0)
        ax.axhline(2 ** a, color=INK, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
        ax.axhline(2 ** b, color=INK, lw=1.3, zorder=1)
        for i in (a, b):   # named just outside the right edge, level with their lines
            ax.annotate(f"$2^{{{i}}}$ slots", (1, 2 ** i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=9)
    ax.grid(axis="y", which="major", color="#dddddd", lw=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, length=3)
    k0, k1 = round(log10(ymin)), round(log10(ymax))
    step = 3 if k1 - k0 > 8 else 2 if k1 - k0 > 4 else 1
    ks = range(k0, k1 + 1, step)
    ax.set_yticks([10.0 ** k for k in ks], [f"$10^{{{k}}}$" for k in ks])
    ax.minorticks_off()
    ax.set_ylabel(ylabel, fontsize=9)
    col_in = AX_W / n
    for x, c in enumerate(cols):
        vals = [max(p[1], ymin) for p in c["points"]]
        if not vals:
            continue
        if c["unit"] == "programs":
            xs = beeswarm(vals, x, col_in, 0.25 * col_in, 4.6 / 72, ymin, ymax, AX_H)   # a dot plus 1 pt apart
            ax.scatter(xs, vals, s=3.6 ** 2, color=INK, lw=0, zorder=3)
        else:
            ax.plot([x] * len(vals), vals, ls="none", marker="_", ms=6, mew=0.6, color=GRAY, zorder=2)
            ax.plot([x], [vals[-1]], ls="none", marker="D", ms=4.5, color=INK, mew=0, zorder=3)
    ax.set_xticks(range(n), [f"$\\bf{{{c['label'].replace(' ', '~')}}}$" for c in cols], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=4)

    def item(marker, size, color, mew, text):   # the key, laid out from the measured text widths
        da = DrawingArea(8, 8, 0, 0)
        da.add_artist(Line2D([4], [4], ls="none", marker=marker, ms=size, color=color, mew=mew))
        return HPacker(children=[da, TextArea(text, textprops=dict(fontsize=8))], align="center", pad=0, sep=2)
    key = HPacker(children=[item("o", 3.6, INK, 0, "program"), item("_", 6, GRAY, 0.6, "script, size or run"),
                            item("D", 4.5, INK, 0, "maximum")], align="center", pad=0, sep=10)
    ax.add_artist(AnchoredOffsetbox(loc="lower right", child=key, frameon=False, pad=0.3, borderpad=0.4))
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)
    # the PDF is the figure (vector), the PNG a screen preview small enough for a terminal image viewer
    fig.savefig(base + ".png", dpi=120, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_rate_caption(cols, path):
    marks = [(c["label"], p) for c in cols for p in c["points"]]
    label, top = max(marks, key=lambda m: m[1][1])
    per_gen = 2 ** SPLIT / (top[1] / 1000)
    lines = [
        "Heap allocations per 1,000 instructions in one process over a whole benchmark run, natively on x86. "
        "Allocations and instructions are counted in the same run under DynamoRIO 11.3 (slotins.c: every basic "
        "block's instructions, every successful malloc-family allocation and realloc move), so benchmarks that run "
        "for a fixed time are paired with their own work. Marks as in the peak figure.",
        " ".join(f"{c['label']}: {c['shown']}, {len(c['points'])} {c['unit']}"
                 f"{'' if c['unit'] == 'programs' else ' of one program'}." for c in cols if c["points"]),
        f"The highest is {top[0]} ({label}) at {top[1]:.2f} allocations per 1,000 instructions. At that rate one "
        f"generation of 2^{SPLIT} slots lasts {per_gen / 1e9:.1f} billion instructions, and each generation bit doubles "
        f"that.",
    ]
    open(path, "w").write("\n\n".join(lines) + "\n")


# The third figure: each cell with both measurements is one point, its census peak (x) against the allocations it
# would make in a year (y), the two inputs of the lifetime-id sizing: with i slot bits and g generation bits a slot
# retires after 2^g uses, so a process with peak L survives A allocations while A <= (2^i - L) * 2^g. A split's
# capacity is a curve: flat at 2^(i+g) while L << 2^i, falling to a wall at L = 2^i. Divided by Y years it is the
# same curve lower by log2(Y): the gray bands between Y = 1, 10, 100 and 1,000 years show, for the shaded split,
# how long until a process at that point runs out of slots (darker is sooner). The instruction rate that turns
# allocations per instruction into allocations per year is an assumption, stated in the caption (IPS).
# Suites are colour and shape: five hues without blue or yellow, each at least 3:1 on white, the set with the widest
# all-pairs separation validate_palette.js found among the Okabe-Ito and reference hues (normal vision 16.4,
# colour-blind 7.6, so every suite also has its own marker), and black for Redis, the one suite that crosses 2^24.
# A program's data set sizes are joined by a line in its suite's colour.
SUITE_STYLE = {   # column label of COLUMNS: colour, marker
    "SPEC CPU2017": ("#d55e00", "o"), "PARSEC": ("#009e73", "s"), "pyperformance": ("#7b2d8e", "^"),
    "Octane": ("#cc79a7", "v"), "SQLite": ("#8c2d04", "D"), "Redis": (INK, "*"),
}
ID_BITS = 55                   # slot + generation bits of the lifetime id
SPLIT = 25                     # the drawn split: slot bits (generation bits = ID_BITS - SPLIT)
OTHER_SPLITS = ()              # drawn as one-year lines (28 would hold Redis at 5 M records)
SCATTER_LEAVE_OUT = ("Redis",)   # left out of this figure for now (2026-10-26, to be looked at again)
IPS = 1e10                     # instructions per second per process, an upper bound for RISC-V cores (see README)
YEAR = 365.25 * 86400
BANDS = ()                     # years of the gray bands, e.g. (1, 10, 100, 1000); none: one line, the one-year limit
TICK_STEP = 8                  # both log2 axes tick at the same powers of two (every TICK_STEP-th)
# A square panel with unequal scales (review 2026-10-26: equal log2 scaling would only give the 45-degree direction,
# allocations per live object, a meaning the figure does not use, at 1.8x the height or with x run out to 2^58)
EQUAL_SCALE = False            # True: one power of two as long on y as on x (the figure's height follows)
AXES_WIDTH = 2.85              # inches (and the height of the square panel)
X_TOP = 2.0 ** 28              # x axis end: a little past the wall (None: from the data)
BAND_GRAYS = ("#c4c4c4", "#d6d6d6", "#e4e4e4", "#f0f0f0")   # < 1, 1-10, 10-100, 100-1,000 years; white beyond


def scatter_pairs(cols, rcols):
    """(label, program, dataset, peak, rate) for every cell with a census peak and an allocation rate"""
    rate = {(c["label"], p[0], p[2]): p[1] for c in rcols for p in c["points"]}
    return [(c["label"], p[0], p[2], p[1], rate[(c["label"], p[0], p[2])])
            for c in cols for p in c["points"] if (c["label"], p[0], p[2]) in rate]


def per_year(rate_per_1000):
    return rate_per_1000 / 1000 * IPS * YEAR


def years_left(peak, rate_per_1000, i):
    """years until a process with this peak and rate runs out of slots with i slot bits (0 when its peak alone
    does not fit)"""
    return max(0.0, (2 ** i - peak) * 2.0 ** (ID_BITS - i)) / per_year(rate_per_1000)


def log_ticks(axis, lo, hi, base, step=None):
    """ticks at base^k, at multiples of the step (by default the smallest giving at most 9 ticks); returns the limits
    they span"""
    from math import ceil, floor, log
    k0, k1 = floor(log(lo, base) + 1e-9), ceil(log(hi, base) - 1e-9)
    steps = (1, 2, 4, 8, 16) if base == 2 else (1, 2, 3, 4, 5)   # base 2: 2^20 and 2^24 on ticks
    step = step or next(s for s in steps if (k1 - k0) / s <= 8)
    k0, k1 = k0 // step * step, -(-k1 // step) * step
    axis.set_ticks([float(base) ** k for k in range(k0, k1 + 1, step)],
                   [f"${base}^{{{k}}}$" for k in range(k0, k1 + 1, step)])
    return float(base) ** k0, float(base) ** k1


def scatter_figure(pairs, base, xlog=True, ylog=True):
    """log axes are base 2 from 2^0, linear ones from 0: the origin is always the first value"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    pts = [p[:4] + (per_year(p[4]),) for p in pairs if p[0] not in SCATTER_LEAVE_OUT]
    fig = plt.figure(figsize=(4.6, 3.0))
    ax = fig.add_axes((0.11, 0.14, 0.62, 0.8))
    xmax = max([max(p[3] for p in pts) * 1.5, 2.0 ** (SPLIT + 4) if xlog else 2.0 ** SPLIT * 1.15]
               + [2.0 ** i for i in OTHER_SPLITS])
    if xlog:
        ax.set_xscale("log", base=2)
        xlo, xhi = log_ticks(ax.xaxis, 1.0, X_TOP or xmax, 2, TICK_STEP)
        xhi = X_TOP or xhi
    else:
        xlo, xhi = 0.0, xmax
    if ylog:
        ax.set_yscale("log", base=2)
        ylo, _ = log_ticks(ax.yaxis, 1.0, 2.0 ** (ID_BITS + 3), 2, TICK_STEP)
        yhi = 2.0 ** (ID_BITS + 3)   # a little above the capacity, not a whole tick step
    else:
        ylo, yhi = 0.0, 2.0 ** ID_BITS * 1.1
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    from math import log2
    ah = AXES_WIDTH * log2(yhi / ylo) / log2(xhi / xlo) if xlog and ylog and EQUAL_SCALE else AXES_WIDTH
    fig.set_size_inches(AXES_WIDTH + 0.7, ah + 0.6)   # the legend sits inside the panel
    ax.set_position((0.55 / (AXES_WIDTH + 0.7), 0.45 / (ah + 0.6), AXES_WIDTH / (AXES_WIDTH + 0.7), ah / (ah + 0.6)))
    if ylog:   # the ceiling, named where it is (the nearest tick is another power of two)
        ax.annotate(f"$2^{{{ID_BITS}}}$", (xlo, 2.0 ** ID_BITS), xytext=(3, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=8)
    L = (np.logspace(0, SPLIT * np.log10(2), 600) if xlog else np.linspace(0, 2.0 ** SPLIT, 600))[:-1]
    cap = (2.0 ** SPLIT - L) * 2.0 ** (ID_BITS - SPLIT)
    if not BANDS:   # one line: the split's capacity for one year, and the gap to the run closest to it, in years
        ax.plot(np.append(L, 2.0 ** SPLIT), np.append(cap, ylo), color=INK, lw=1.3, zorder=1)
        fits = [p for p in pts if p[3] < 2 ** SPLIT]
        near = min(fits, key=lambda p: (2.0 ** SPLIT - p[3]) * 2.0 ** (ID_BITS - SPLIT) / p[4])
        line = (2.0 ** SPLIT - near[3]) * 2.0 ** (ID_BITS - SPLIT)
        ax.annotate("", (near[3], line), xytext=(near[3], near[4] * 1.4 if ylog else near[4]), zorder=2,
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
        mid = (line * near[4]) ** 0.5 if ylog else (line + near[4]) / 2
        gap = line / near[4]   # the margin in years, and in bits on the log2 axis
        ax.annotate(f"{gap:,.0f} years ({np.log2(gap):.1f} bits)", (near[3], mid), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8)
    else:           # the bands, painted from the top: the axes are the < 1 year gray (above it, right of the wall)
        ax.set_facecolor(BAND_GRAYS[0])
        for years, gray in zip(BANDS, BAND_GRAYS[1:] + ("white",)):
            ax.fill_between(L, ylo, cap / years, color=gray, lw=0, zorder=0)
        # each band named by its range, at the left where the bands are level (the middle of each band in log2)
        names = (["< 1 year"] + [f"{a:,}\u2013{b:,} years" for a, b in zip(BANDS, BANDS[1:])]
                 + [f"> {BANDS[-1]:,} years"])
        mids = ([2.0 ** ID_BITS * (yhi / 2.0 ** ID_BITS) ** 0.5]
                + [2.0 ** ID_BITS / (a * b) ** 0.5 for a, b in zip(BANDS, BANDS[1:])]
                + [2.0 ** ID_BITS / BANDS[-1] / 10 ** 0.5])
        for name, y in zip(names, mids):
            ax.annotate(name, (xlo, y), xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=7, color="#444444")
    for i in OTHER_SPLITS:   # other splits' capacity for one year, and the shaded split's split label
        Li = np.logspace(0, i * np.log10(2), 600)[:-1]
        ax.plot(Li, (2.0 ** i - Li) * 2.0 ** (ID_BITS - i), color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for i in (SPLIT,) + OTHER_SPLITS:
        ax.annotate(f"{i} slot + {ID_BITS - i} generation bits", (2.0 ** i, ylo), xytext=(-2, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7, rotation=90)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, length=3)
    ax.minorticks_off()
    for axis, log in ((ax.xaxis, xlog), (ax.yaxis, ylog)):
        if not log:   # linear: 0 at the origin, powers of ten as one offset label
            axis.get_major_formatter().set_powerlimits((0, 0))
    ax.set_xlabel("Peak live heap objects per process", fontsize=9)
    ax.set_ylabel("Modeled allocations per year", fontsize=9)
    for label, (color, marker) in SUITE_STYLE.items():
        sel = [p for p in pts if p[0] == label]
        if not sel:
            continue
        for program in {p[1] for p in sel}:   # a program's data set sizes, in size order
            run = sorted((p for p in sel if p[1] == program), key=lambda p: dataset_order(p[2]))
            if len(run) > 1:
                ax.plot([p[3] for p in run], [p[4] for p in run], color=color, lw=0.8, zorder=2)
        ax.scatter([p[3] for p in sel], [p[4] for p in sel], s=(6.5 if marker == "*" else 4.2) ** 2, marker=marker,
                   color=color, edgecolors="white", linewidths=0.4, zorder=3, label=label)
    # inside the empty lower right, left of the wall and its label
    wall = log2(2.0 ** SPLIT / xlo) / log2(xhi / xlo) if xlog else 2.0 ** SPLIT / xhi
    ax.legend(loc="lower right", bbox_to_anchor=(wall - 0.06, 0.02), frameon=False, fontsize=7,
              handletextpad=0.2, borderaxespad=0, labelspacing=0.3)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base + ".png", dpi=120, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_scatter_caption(pairs, path, peaks):
    """the caption draft, every number from the drawn data (reviewed 2026-10-26: the limit is a sufficient bound, the
    rate an assumption per process, the left-out suites named)"""
    shown = [p for p in pairs if p[0] not in SCATTER_LEAVE_OUT]
    left_out = sorted({p[0] for p in pairs if p[0] in SCATTER_LEAVE_OUT})
    g = ID_BITS - SPLIT
    fits = [p for p in shown if p[3] < 2 ** SPLIT]
    worst = min(fits, key=lambda p: years_left(p[3], p[4], SPLIT))
    past = sorted((p for p in shown if p[3] >= 2 ** SPLIT), key=lambda p: p[3])
    rated = {p[:3] for p in pairs}
    no_rate = [(s, n) for s in dict.fromkeys(q[0] for q in peaks)   # suites in figure order, runs without a rate
               if (n := sum(q[0] == s and q not in rated for q in peaks))]
    ips = f"{IPS:.0e}".replace("1e+10", "10^10")
    lines = [
        f"Peak live heap objects in one process against the heap allocations it would make in a year at {ips} "
        f"instructions per second (per process, summed over its threads, 365.25-day years). Allocations per "
        f"instruction are measured under DynamoRIO. A {ID_BITS}-bit lifetime id with {SPLIT} slot and {g} "
        f"generation bits supports at least (2^{SPLIT} - peak) x 2^{g} allocations before a process can run out of "
        f"slots. "
        + (f"The gray bands show how many years that lasts ({', '.join(f'{y:,}' for y in BANDS)}), darker is sooner."
           if BANDS else "The line is that bound for one year, the arrow the gap to the run closest to it.")
        + (f" The dashed curve is the one-year bound of the "
           f"{' and '.join(f'{i}/{ID_BITS - i}' for i in OTHER_SPLITS)} split." if OTHER_SPLITS else ""),
        f"The {len(shown)} runs shown are those of the {len(peaks)} in the census with a measured allocation rate "
        f"as well" + (f" ({', '.join(left_out)} left out)" if left_out else "") + ". "
        + (f"Without a rate: " + ", ".join(f"{s} ({n})" for s, n in no_rate) + ". " if no_rate else "")
        + f"With {SPLIT} slot and {g} generation bits the shortest bound among them is about "
        f"{years_left(worst[3], worst[4], SPLIT):,.0f} years, for {worst[1]} ({worst[0]}, {worst[2]}). "
        + (f"{len(past)} runs need more than 2^{SPLIT} slots: "
           + ", ".join(f"{p[1]} at {p[2]} ({p[3]:,} objects)" for p in past) + "." if past else ""),
    ]
    open(path, "w").write("\n\n".join(lines) + "\n")


def write_figures(rows, rate_rows, out):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("report: matplotlib not available, figures skipped")
        return
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    cols = figure_data(rows)
    write_caption(cols, os.path.join(out, "figures", "slots-caption.txt"))
    strip_figure(cols, "Peak live heap objects per process", (20, 24), os.path.join(out, "figures", "slots"))
    if rate_rows:
        rcols = [c for c in figure_data(rate_rows) if c["points"]]
        write_rate_caption(rcols, os.path.join(out, "figures", "rates-caption.txt"))
        strip_figure(rcols, "Allocations per 1,000 instructions", None, os.path.join(out, "figures", "rates"))
        pairs = scatter_pairs(cols, rcols)
        if pairs:
            scatter_figure(pairs, os.path.join(out, "figures", "peak-rate"))
            write_scatter_caption(pairs, os.path.join(out, "figures", "peak-rate-caption.txt"),
                                  [(c["label"], p[0], p[2]) for c in cols for p in c["points"]])
            print(f"report: peak-rate figure, {len(pairs)} cells with both a peak and a rate")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")
    out = sys.argv[2] if len(sys.argv) > 2 else here
    rows, excluded = collect(results)
    if not rows:
        print(f"report: no valid cells under {results}", file=sys.stderr)
        return 1
    rate_rows = collect_rates(results)
    write_report(rows, excluded, out,
                 scatter_pairs(figure_data(rows), [c for c in figure_data(rate_rows) if c["points"]]))
    write_figures(rows, rate_rows, out)
    print(f"report: {len(rows)} cells, {len(excluded)} excluded -> {os.path.join(out, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
