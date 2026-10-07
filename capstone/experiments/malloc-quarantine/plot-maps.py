#!/usr/bin/env python3
"""Who owns the resident pages of a process over time: jemalloc, MRS, other.

    plot-maps.py RESULT_DIR OUT_PREFIX [NAME]

NAME (a substring of the file name) draws one program only.

Input: files written by run-maps.sh (MAPS lines sampled every 20 s with
procstat -v, RES in 4 KiB pages), one file per program.  One panel per
program: stacked areas of resident MiB by mapping owner against time since
the first sample.  'mrs' are the mappings MRS names itself
(mrs:alloc_descriptor_slab, one 16-byte capability per quarantined object);
'jemalloc' are jemalloc's named extents; 'file' are the binary and libraries;
'anon' is unnamed anonymous memory (stack, the shadow bitmap, …).  Prints the
peak of each owner and the peak share of 'mrs'.  A file without MAPS-DONE is
reported and left out.
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OWNERS = [('jemalloc', 'jemalloc heap', '#9aa5ae'), ('mrs', 'MRS quarantine bookkeeping', '#c0392b'),
          ('anon', 'other anonymous', '#5d8aa8'), ('file', 'binary + libraries', '#d5b26b'),
          ('other', 'other', '#7d6b91')]
PAGE = 4096 / 2**20


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else None
    runs = []
    for p in sorted(src.glob('*.txt')):
        if want and want not in p.name:
            continue
        t = p.read_text(errors='replace')
        if 'MAPS-DONE' not in t:
            print(f'skipped (no MAPS-DONE): {p.name}', file=sys.stderr)
            continue
        rows = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', l))
                for l in t.splitlines() if l.startswith('MAPS t=')]
        rss = re.search(r'^\s*(\d+)\s+maximum resident set size', t, re.M)
        if not rows:
            print(f'skipped (no samples): {p.name}', file=sys.stderr)
            continue
        runs.append((p.stem, rows, int(rss.group(1)) / 1024 if rss else None))
    if not runs:
        sys.exit(f'no usable files in {src}')
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.4), squeeze=False)
    for ax, (name, rows, maxrss) in zip(axes[0], runs):
        t0 = rows[0]['t']
        ts = [(r['t'] - t0) / 60 for r in rows]
        bottom = [0.0] * len(rows)
        peaks = {}
        for key, label, color in OWNERS:
            ys = [r.get(key, 0) * PAGE for r in rows]
            if max(ys) < 0.5:
                continue
            top = [b + y for b, y in zip(bottom, ys)]
            ax.fill_between(ts, bottom, top, color=color, lw=0, label=label)
            bottom = top
            peaks[key] = max(ys)
        tot = max(r['total'] * PAGE for r in rows)
        mrs_share = max((r.get('mrs', 0) / r['total']) for r in rows if r['total'])
        print(f"{name:20} samples={len(rows)} peak total={tot:7.1f} MiB  " +
              ' '.join(f'{k}={v:.1f}' for k, v in peaks.items()) +
              f"  mrs peak share={mrs_share:.1%}  time -l maxrss={maxrss if maxrss is None else round(maxrss, 1)}")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('minutes of QEMU time (orientation only, not a performance axis)')
        ax.set_xlim(0, ts[-1])
        ax.set_ylim(0, None)
    axes[0][0].set_ylabel('resident MiB by mapping owner')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='upper center', ncol=len(l))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out + '-maps.png', dpi=150)


if __name__ == '__main__':
    main()
