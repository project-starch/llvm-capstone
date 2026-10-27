"""The three figures and their caption drafts, drawn from report.py's rows (the only module that uses matplotlib).

  slots      peak live heap objects per run, suites as columns, with the 2^20 and 2^24 slot lines
  rates      heap allocations per 1,000 instructions per run, suites as columns
  peak-rate  each run with both: instructions per allocation (x) against its peak (y)
Every number in a caption is taken from the data the figure draws. Captions have no semicolons (the paper's style).
PDFs carry no creation date, so a figure changes only when what it draws changes."""
import importlib.util
import os
import re
from math import ceil, floor, log2, log10

from suites import dataset_order

INK, GRAY, LIGHT = "#000000", "#7a7a7a", "#efefef"
# Times-like text and mathematics (STIX ships with matplotlib), to sit beside a Times body (USENIX, IEEE)
FONT = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix"}


def pyplot():
    """matplotlib's pyplot, headless and in the paper's style"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({**FONT, "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    return plt


def save(plt, fig, base, png_dpi):
    """the PDF is the figure (vector), the PNG a screen preview"""
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03, metadata={"CreationDate": None})
    fig.savefig(base + ".png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def column_summary(cols):
    """one sentence per column: its input and how many marks it has"""
    return " ".join(f"{c['label']}: {c['shown']}, {len(c['points'])} {c['unit']}"
                    f"{'' if c['unit'] == 'programs' else ' of one program'}." for c in cols if c["points"])


# --- the strip figures (slots, rates) --------------------------------------------------------------------------------
# One message, one panel. Suites are columns: programs are dots, scripts on one interpreter or data set sizes of one
# program are ticks with a diamond at the maximum. No value labels (the caption carries the numbers), black and grays
# only, so it prints and reads without colour.

def beeswarm(values, x0, ax_in, xspan_in, d_in, ymin, ymax, height_in):
    """deterministic offsets: each point takes the free x nearest the column centre, in inches on the page"""
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


def strip_figure(cols, ylabel, thresholds, base):
    """one panel, suites as columns: programs as dots, scripts, sizes or runs of one program as ticks with a diamond
    at the maximum; thresholds are 2^i lines (the band between them shaded), drawn when given"""
    plt = pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    n = len(cols)
    W, H, AX_W, AX_H = 7.4, 3.1, 6.3, 2.6   # double-column width: up to seven bold suite titles side by side
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.55 / W, 0.3 / H, AX_W / W, AX_H / H))
    vals_all = [p.value for c in cols for p in c["points"] if p.value > 0]
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
        vals = [max(p.value, ymin) for p in c["points"]]
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
    save(plt, fig, base, png_dpi=120)   # small enough for a terminal image viewer


def write_slots_caption(cols, path):
    programs = [c for c in cols if c["points"] and c["label"] != "Redis"]
    top = max((p for c in programs for p in c["points"]), key=lambda p: p.value)
    over20 = sum(p.value > 2 ** 20 for c in programs for p in c["points"])
    marks = sum(len(c["points"]) for c in programs)
    lines = [
        "Peak number of simultaneously live heap objects in one process over a whole run, measured natively on x86 "
        "with an exact allocation tracer. Dots are benchmark programs. Ticks are scripts on one interpreter, data "
        "set sizes or benchmark runs of one program, with a diamond at the maximum. Marks at the same value overlap.",
        column_summary(cols),
        f"Without Redis, {over20} of {marks} marks exceed 2^20 slots and none reaches 2^24. The largest is "
        f"{top.program} ({top.value:,} objects, {100 * top.value / 2 ** 24:.0f}% of 2^24).",
    ]
    redis = next((c for c in cols if c["label"] == "Redis" and c["points"]), None)
    if redis:   # a least-squares line through live objects against records
        ks = [int(re.search(r"([\d,]+) records", p.dataset)[1].replace(",", "")) for p in redis["points"]]
        vs = [p.value for p in redis["points"]]
        k_mean, v_mean = sum(ks) / len(ks), sum(vs) / len(vs)
        a = sum((k - k_mean) * (v - v_mean) for k, v in zip(ks, vs)) / sum((k - k_mean) ** 2 for k in ks)
        b = v_mean - a * k_mean
        lines.append(f"Redis holds {a:.2f} live objects per record plus {b:,.0f} (a fit to the {len(ks)} sizes), so "
                     f"it passes 2^24 at about {(2 ** 24 - b) / a / 1e6:.2f} M records. At {ks[-1]:,} records it "
                     f"holds {vs[-1]:,} objects.")
    open(path, "w").write("\n\n".join(lines) + "\n")


def write_rates_caption(cols, path):
    label, top = max(((c["label"], p) for c in cols for p in c["points"]), key=lambda m: m[1].value)
    per_generation = 2 ** SPLIT / (top.value / 1000)
    lines = [
        "Heap allocations per 1,000 instructions in one process over a whole benchmark run, natively on x86. "
        "Allocations and instructions are counted in the same run under DynamoRIO 11.3 (slotins.c: every basic "
        "block's instructions, every successful malloc-family allocation and realloc move), so benchmarks that run "
        "for a fixed time are paired with their own work. Marks as in the peak figure.",
        column_summary(cols),
        f"The highest is {top.program} ({label}) at {top.value:.2f} allocations per 1,000 instructions. At that rate "
        f"one generation of 2^{SPLIT} slots lasts {per_generation / 1e9:.1f} billion instructions, and each "
        f"generation bit doubles that.",
    ]
    open(path, "w").write("\n\n".join(lines) + "\n")


# --- the peak-rate figure -----------------------------------------------------------------------------------------
# Each run with both measurements is one point: its instructions per heap allocation (x, the mean allocation
# interarrival time Zorn and Grunwald measure in instructions, CU-CS-604-92, 1992) against its census peak (y), both
# measured, both log2 from 2^0 with one power of two as long on each. With i slot bits and g generation bits a slot
# retires after 2^g uses, so a process with peak L can run out of slots only after (2^i - L) * 2^g allocations. The
# caption turns the closest run's margin into years at three instruction rates (review 2026-10-27: Astra, Gemini and
# Fable put the measurement on the axes and the model elsewhere; lines for the lifetimes and the slot ceiling were
# tried and dropped, user).
# Suites are colour and shape: five hues without blue or yellow, each at least 3:1 on white, the set with the widest
# all-pairs separation validate_palette.js found among the Okabe-Ito and reference hues (normal vision 16.4,
# colour-blind 7.6, so every suite also has its own marker). A program's data set sizes are joined by a line.
SUITE_STYLE = {   # column label: colour, marker
    "SPEC CPU2017": ("#d55e00", "o"), "PARSEC": ("#009e73", "s"), "pyperformance": ("#7b2d8e", "^"),
    "Octane": ("#cc79a7", "X"), "speedtest1": ("#8c2d04", "D"), "Redis": (INK, "*"),
}
DENSE = ("pyperformance", "speedtest1")   # many runs at one spot: drawn smaller, translucent and underneath
CROWDED = "pyperformance"                 # its scripts share the interpreter's peak, which the caption says
LEGEND_ORDER = ("SPEC CPU2017", "PARSEC", "speedtest1", "pyperformance", "Octane")   # compiled, then interpreted
ID_BITS = 55                   # slot + generation bits of the lifetime id
SPLIT = 25                     # slot bits (generation bits = ID_BITS - SPLIT)
LEAVE_OUT = ("Redis",)         # not in this figure or its caption (user, 2026-10-27)
YEAR = 365.25 * 86400
TICK_STEP = 5                  # both log2 axes tick at every TICK_STEP-th power of two (2^25 is a tick)
AXES_WIDTH = 2.55              # inches: with labels the figure is one column wide, at most 3.3 in


def instructions_left(peak, rate):
    """log2 of the instructions after which a process with this peak and rate (per 1,000 instructions) can run out
    of slots"""
    return log2((2.0 ** SPLIT - peak) * 2.0 ** (ID_BITS - SPLIT) / (rate / 1000))


def log2_ticks(axis, lo, hi):
    """ticks at every TICK_STEP-th power of two from lo to hi"""
    k0, k1 = floor(log2(lo) + 1e-9) // TICK_STEP * TICK_STEP, -(-ceil(log2(hi) - 1e-9) // TICK_STEP) * TICK_STEP
    ks = range(k0, k1 + 1, TICK_STEP)
    axis.set_ticks([2.0 ** k for k in ks], [f"$2^{{{k}}}$" for k in ks])


def peak_rate_figure(pairs, base):
    plt = pyplot()
    runs = [p for p in pairs if p.suite not in LEAVE_OUT]
    x = lambda p: 1000 / p.rate   # instructions per allocation
    xlo, xhi = 1.0, 2.0 ** (-(-log2(max(x(p) for p in runs) * 1.05) // TICK_STEP) * TICK_STEP)
    ylo, yhi = 1.0, 2.0 ** SPLIT   # the y axis ends at the slots, the limit every peak must stay under
    w, h = AXES_WIDTH, AXES_WIDTH * log2(yhi / ylo) / log2(xhi / xlo)
    fig = plt.figure(figsize=(w + 0.7, h + 0.55))
    ax = fig.add_axes((0.55 / (w + 0.7), 0.45 / (h + 0.55), w / (w + 0.7), h / (h + 0.55)))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    log2_ticks(ax.xaxis, xlo, xhi)
    log2_ticks(ax.yaxis, ylo, yhi)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.5, length=3)   # mathtext sets exponents at 0.7: 6.65 pt
    ax.minorticks_off()
    ax.set_xlabel("Instructions per heap allocation", fontsize=10)
    ax.set_ylabel("Peak live heap objects", fontsize=10)
    for label, (color, marker) in SUITE_STYLE.items():
        sel = [p for p in runs if p.suite == label]
        if not sel:
            continue
        for program in {p.program for p in sel}:   # a program's data set sizes, in size order
            sizes = sorted((p for p in sel if p.program == program), key=lambda p: dataset_order(p.dataset))
            if len(sizes) > 1:
                ax.plot([x(p) for p in sizes], [p.peak for p in sizes], color=color, lw=0.8, zorder=2)
        dense = label in DENSE
        ax.scatter([x(p) for p in sel], [p.peak for p in sel], s=(5.5 if marker == "*" else 2.6 if dense else 3.6) ** 2,
                   marker=marker, color=color, alpha=0.5 if dense else 1, linewidths=0, zorder=2.5 if dense else 3,
                   label=label)
    # above the panel, row by row: no corner inside is free of runs
    handles, names = ax.get_legend_handles_labels()
    order = sorted(range(len(names)), key=lambda k: LEGEND_ORDER.index(names[k]))
    rows = [order[k:k + 3] for k in range(0, len(order), 3)]
    order = [r[c] for c in range(3) for r in rows if c < len(r)]   # matplotlib fills a legend column by column
    legend = ax.legend([handles[k] for k in order], [names[k] for k in order], loc="lower center",
                       bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False, fontsize=8.5, handletextpad=0.2,
                       columnspacing=1.0, borderaxespad=0, labelspacing=0.3)
    for handle in legend.legend_handles:   # the key at full strength, though dense suites are translucent
        handle.set_alpha(1)
    save(plt, fig, base, png_dpi=200)


def write_peak_rate_caption(pairs, path):
    runs = [p for p in pairs if p.suite not in LEAVE_OUT]
    g = ID_BITS - SPLIT
    fit = [p for p in runs if p.peak < 2 ** SPLIT]
    closest = min(fit, key=lambda p: instructions_left(p.peak, p.rate))
    n = instructions_left(closest.peak, closest.rate)
    widest = max(fit, key=lambda p: p.peak)
    crowd = sorted(p.peak for p in runs if p.suite == CROWDED)
    crowd = [v for v in crowd if v <= 1.25 * crowd[0]] if crowd else []
    years = lambda ips: 2.0 ** n / ips / YEAR
    caption = (
        f"Instructions per heap allocation of one process (x, the mean allocation interarrival time in instructions "
        f"as Zorn and Grunwald measure it, all its threads, under DynamoRIO in one run) against its peak live heap "
        f"objects (y, measured natively with an exact allocation tracer). A {ID_BITS}-bit lifetime id with "
        f"{SPLIT} slot and {g} generation bits supports at least (2^{SPLIT} - y)·2^{g} allocations before a process "
        f"can run out of slots, and the y axis ends at the 2^{SPLIT} slots. "
        f"The closest run to that limit, {closest.program} ({closest.suite}, one allocation every "
        f"{1000 / closest.rate:,.0f} instructions), can run out only after 2^{n:.2f} instructions: about "
        f"{years(1e9):,.0f}, {years(1e10):,.0f} or {years(1e11):,.1f} years at 10^9, 10^10 or 10^11 instructions "
        f"per second. The largest peak, {widest.program} ({widest.peak:,} objects), is "
        f"{SPLIT - log2(widest.peak):.1f} bits short of 2^{SPLIT} slots. {len(runs)} runs"
        + (f", {len(crowd)} of them {CROWDED} scripts that overlap near {crowd[0]:,} objects" if len(crowd) > 1 else "")
        + ".")
    open(path, "w").write(caption + "\n")


def write_figures(peak_cols, rate_cols, pairs, out):
    """every figure with its caption into OUT/figures; the rates and peak-rate figures only when there are rates"""
    if importlib.util.find_spec("matplotlib") is None:
        print("report: matplotlib not available, figures skipped")
        return
    fig = lambda name: os.path.join(out, "figures", name)
    os.makedirs(fig(""), exist_ok=True)
    write_slots_caption(peak_cols, fig("slots-caption.txt"))
    strip_figure(peak_cols, "Peak live heap objects per process", (20, 24), fig("slots"))
    if rate_cols:
        write_rates_caption(rate_cols, fig("rates-caption.txt"))
        strip_figure(rate_cols, "Allocations per 1,000 instructions", None, fig("rates"))
    if pairs:
        peak_rate_figure(pairs, fig("peak-rate"))
        write_peak_rate_caption(pairs, fig("peak-rate-caption.txt"))
        print(f"report: peak-rate figure, {len(pairs)} cells with both a peak and a rate")
