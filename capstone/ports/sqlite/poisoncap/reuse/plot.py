#!/usr/bin/env python3
"""Plot the reuse distance at memsys5 for one system: its control (protection off) against the
protected configuration, from reuse-wrap.c's reports.

    plot.py OUT_STEM --control LABEL=SOURCE --protected LABEL=SOURCE [--note TEXT]

No title: the axes and the legend say what is shown; which system and run belongs in the caption.

SOURCE is a speedtest1/run.py result.json (PoisonCap side), or RESULTS_JSONL#ROW_NAME
(run-sublet.py rows, Sublet side). The REUSE-GAP line at exit gives issues and the 32 bins
(bin b: distances of bit length b+1, i.e. [2^b, 2^(b+1)-1]).

One panel: cumulative same-start reuses within a release-to-reissue gap of at most the bucket's upper
edge, as a share of ALL issues (first issues and reuses not seen within the run are in the denominator,
so the curve ends at reuses/issues), control and protected. Writes OUT_STEM.png and OUT_STEM.pdf, and
OUT_STEM.csv with the plotted values and their difference (control minus protected, percentage points).
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reuse_report  # noqa: E402

CONTROL, PROTECTED = '#2a78d6', '#eb6834'      # categorical slots 1 and 2 (validated pair)
INK, MUTED, GRID = '#1f1f1e', '#5c5b55', '#e4e3df'


def source_stderr(source):
    """The stderr a SOURCE recorded: a result.json's, or RESULTS_JSONL#ROW's."""
    if '#' not in source:
        return json.loads(Path(source).read_text()).get('stderr') or ''
    path, row = source.split('#', 1)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r['name'] == row:
            return r.get('stderr') or ''
    raise SystemExit(f'{row} not in {path}')


def report(source):
    """(issues, reuses, bins, failed) of SOURCE's exit report; exits unless the report is admissible."""
    d = reuse_report.snapshot(source_stderr(source))
    if not d:
        raise SystemExit(f'{source}: no REUSE-GAP report at exit')
    if d['installed'] != '1' or d['error'] != '0':
        raise SystemExit(f'{source}: not admissible (installed={d["installed"]} error={d["error"]})')
    if reuse_report.failed_identities(d):
        raise SystemExit(f'{source}: identities do not hold')
    return int(d['issues']), int(d['reuses']), d['bins'], int(d['failed'])


def cumulative(issues, bins, upto):
    out, acc = [], 0
    for b in range(upto):
        acc += bins[b]
        out.append(100.0 * acc / issues)
    return out


def draw(out, edges, yc, yp, cl, pl, xmax, identical, note):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 9,
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.4))
    ax.set_xscale('log', base=2)
    ticks = [t for t in (1, 8, 64, 512, 4096, 32768, 262144) if t <= xmax]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_xlim(0.8, xmax * 1.25)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    # Cumulative curves carry their last value to the shared right edge.
    xs = edges + ([xmax * 1.25] if xmax * 1.25 > edges[-1] else [])
    ext = lambda ys: ys + ys[-1:] * (len(xs) - len(ys))
    ax.step(xs, ext(yc), where='post', color=CONTROL, linewidth=1.8)
    ax.step(xs, ext(yp), where='post', color=PROTECTED, linewidth=1.8, linestyle=(0, (4, 3)))
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Address reuse (%)')
    ax.set_xlabel('Reuse distance')
    right = xmax * 1.2
    if identical:
        # Below the curve on the right: empty, because cumulative curves rise early.
        ax.text(right, yc[-1] / 2, f'{cl} (solid),\n{pl} (dashed):\nidentical curves', fontsize=8,
                color=INK, ha='right', va='center')
    else:
        # Above its curve's end, or below it when the curve ends near the top of the axis.
        for y, label, color in ((yc[-1], cl, CONTROL), (yp[-1], pl, PROTECTED)):
            above = y + 4 <= 92
            ax.text(right, y + 4 if above else y - 4, label, fontsize=8, color=color, ha='right',
                    va='bottom' if above else 'top')
    if note:
        fig.text(0.01, 0.005, note, fontsize=6.5, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    # No creation date: an unchanged plot stays byte-identical when it is redrawn.
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', metadata={'CreationDate': None})


def write_values(out, edges, yc, yp, cl, pl):
    with out.with_suffix('.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['gap_upper_edge', f'{cl} cum %', f'{pl} cum %', 'control - protected pp'])
        for e, c, q in zip(edges, yc, yp):
            w.writerow([e, f'{c:.4f}', f'{q:.4f}', f'{c - q:.4f}'])


def write_caption(out, control, protected):
    (cl, ci, cr, cf), (pl, pi, pr, pf) = control, protected
    out.with_suffix('.caption.txt').write_text(
        f'Address reuse: cumulative percentage of successful memsys5 allocations that return a previously freed '
        f'start address within the indicated reuse distance. Reuse distance: allocations since the preceding free of '
        f'that address, the reuse included (immediate reuse: 1); power-of-two bins, plotted at their inclusive '
        f'upper edges. First-time '
        f'allocations stay in the denominator. Spatial protection is on in both configurations. '
        f'{ci:,} allocations ({cl.lower()}), {pi:,} ({pl.lower()}); reuses {cr:,} and {pr:,}; failed allocations '
        f'(counted in the gaps): {cf:,} and {pf:,}.\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out')
    p.add_argument('--control', required=True)
    p.add_argument('--protected', required=True)
    p.add_argument('--note', default='')
    p.add_argument('--xmax', type=int, default=0,
                   help="shared right edge of the x axis (give both systems' plots the same); default: the data")
    a = p.parse_args()
    (cl, cs), (pl, ps) = a.control.split('=', 1), a.protected.split('=', 1)
    ci, cr, cb, cf = report(cs)
    pi, pr, pb, pf = report(ps)
    last = max([b for b in range(32) if cb[b] or pb[b]] + [0]) + 2   # one empty bucket past the last
    edges = [2 ** (b + 1) - 1 for b in range(last)]                  # upper edge of each bucket
    yc, yp = cumulative(ci, cb, last), cumulative(pi, pb, last)
    out = Path(a.out)
    draw(out, edges, yc, yp, cl, pl, a.xmax or edges[-1], cb == pb and ci == pi, a.note)
    write_values(out, edges, yc, yp, cl, pl)
    write_caption(out, (cl, ci, cr, cf), (pl, pi, pr, pf))
    diff = [c - q for c, q in zip(yc, yp)]
    print(f'{out}: control {cr}/{ci} reuses, protected {pr}/{pi}; max diff {max(diff + [0]):.2f} pp')


if __name__ == '__main__':
    main()
