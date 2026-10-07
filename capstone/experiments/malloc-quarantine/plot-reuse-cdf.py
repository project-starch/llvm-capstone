#!/usr/bin/env python3
"""How long a freed address stays unused: one CDF per program, one panel.

    plot-reuse-cdf.py RESULT_DIR OUT_PNG

Input: MQ_TRACK=1 runs under ./traced (MQ-HIST, log2 buckets of the number
of allocations between the free of an address and its next allocation).
Only 'on' runs are drawn as curves.  An 'off' run (the tracer's instrument
check, no quarantine) contributes one vertical marker at its median, as the
reference for what reuse looks like without a quarantine.  Also prints the share of allocations at an address
never handed out before.
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

COLORS = ['#c0392b', '#2471a3', '#d68910', '#148f77', '#7d3c98']


def hist(path):
    m = re.search(r'^MQ-HIST(.*)$', path.read_text(errors='replace'), re.M)
    return {int(b): int(c) for b, c in re.findall(r'(\d+):(\d+)', m.group(1))} if m else None


def cdf(h):
    total, acc, xs, ys = sum(h.values()), 0, [], []
    for b in range(64):
        acc += h.get(b, 0)
        xs.append(2 ** b); ys.append(acc / total)
    return xs, ys


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    i = 0
    for p in sorted(src.glob('*.txt')):
        r = parse(p)
        h = hist(p)
        if not (r['ok'] and r['rows'] and h):
            print(f'FAILED or no histogram: {p.name}', file=sys.stderr)
            continue
        last = r['rows'][-1]
        fresh = last['fresh_addr'] / max(last['fresh_addr'] + last['reused_addr'], 1)
        xs, ys = cdf(h)
        med = next(x for x, y in zip(xs, ys) if y >= 0.5)
        below = sum(c for b, c in h.items() if b <= 12) / sum(h.values())
        name = setting(r['argv'])
        print(f"{name:32} {r['arm']:4} reuses={sum(h.values())} median<={med} "
              f"share<=2^12={below:.4f} fresh_share={fresh:.4f}")
        if r['arm'] == 'off':
            ax.axvline(med, color='#555555', lw=1, ls=':')
            ax.text(med * 1.15, 0.5, f'without quarantine:\nmedian ≤ {med}\n({name.split()[0].lstrip("./")})',
                    fontsize=7, color='#555555', va='center')
            continue
        ax.step(xs, ys, where='post', color=COLORS[i % len(COLORS)], lw=1.6,
                label=f"{name.split()[0].lstrip('./')} (median ≤ $2^{{{med.bit_length() - 1}}}$)")
        i += 1
    ax.set_xscale('log', base=2)
    ax.set_xlim(1, 2 ** 25)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('allocations until a freed address is handed out again')
    ax.set_ylabel('share of reuses')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
