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


def write_report(rows, excluded, out):
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


def write_figures(rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    except ImportError:
        print("report: matplotlib not available, figures skipped")
        return
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    cols = figure_data(rows)
    n = len(cols)
    W, H, AX_W, AX_H = 7.4, 3.1, 6.3, 2.6   # double-column width: seven bold suite titles side by side
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.55 / W, 0.3 / H, AX_W / W, AX_H / H))
    from math import ceil, log10
    top = max((p[1] for c in cols for p in c["points"]), default=1)
    ymin, ymax = 1, 10.0 ** max(8, ceil(log10(top * 1.5)))   # every mark inside, Redis's largest data set included
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.5, n - 0.5)
    ax.axhspan(2 ** 20, 2 ** 24, color=LIGHT, lw=0, zorder=0)
    ax.axhline(2 ** 20, color=INK, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.axhline(2 ** 24, color=INK, lw=1.3, zorder=1)
    ax.grid(axis="y", which="major", color="#dddddd", lw=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, length=3)
    decades = round(log10(ymax))
    step = 3 if decades > 8 else 2
    ks = range(0, decades + 1, step)
    ax.set_yticks([10 ** k for k in ks], [f"$10^{{{k}}}$" for k in ks])
    ax.minorticks_off()
    ax.set_ylabel("Peak live heap objects per process", fontsize=9)
    # the thresholds are named just outside the right edge, level with their lines
    for i in (20, 24):
        ax.annotate(f"$2^{{{i}}}$ slots", (1, 2 ** i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9)

    col_in = AX_W / n
    for x, c in enumerate(cols):
        vals = [p[1] for p in c["points"]]
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

    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    write_caption(cols, os.path.join(out, "figures", "slots-caption.txt"))
    fig.savefig(os.path.join(out, "figures", "slots.pdf"), bbox_inches="tight", pad_inches=0.03)
    # the PDF is the figure (vector), the PNG a screen preview small enough for a terminal image viewer
    fig.savefig(os.path.join(out, "figures", "slots.png"), dpi=120, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")
    out = sys.argv[2] if len(sys.argv) > 2 else here
    rows, excluded = collect(results)
    if not rows:
        print(f"report: no valid cells under {results}", file=sys.stderr)
        return 1
    write_report(rows, excluded, out)
    write_figures(rows, out)
    print(f"report: {len(rows)} cells, {len(excluded)} excluded -> {os.path.join(out, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
