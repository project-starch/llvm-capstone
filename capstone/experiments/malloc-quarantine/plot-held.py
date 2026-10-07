#!/usr/bin/env python3
"""What the program holds against what the allocator holds, over a few cycles.

    plot-held.py RUN_FILE OUT_PNG [FIRST_ALLOC LAST_ALLOC] [--model R]

One traced run (mqtrace.so sample lines).  Grey area: live bytes the program
holds (live_req).  Red area on top: jemalloc 'allocated' minus live, i.e. the
bytes the allocator still keeps for freed objects (MRS quarantine plus the
arena being revoked).  The window defaults to the whole run.  --model R draws
the asynchronous trigger model's bounds for ratio R around the mean live bytes
of the window: L(1-R)/(1-2R) and L/(1-2R) (only meaningful for a steady live
set).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

MIB = 2**20


def main():
    args = sys.argv[1:]
    model = None
    if '--model' in args:
        i = args.index('--model'); model = float(args[i + 1]); del args[i:i + 2]
    sys.argv[1:] = args
    run = parse(Path(sys.argv[1]))
    if not (run['ok'] and run['rows']):
        sys.exit(f'incomplete run: {sys.argv[1]}')
    lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, run['rows'][-1]['op'])
    rows = [x for x in run['rows'] if lo <= x['op'] <= hi]
    ops = [x['op'] for x in rows]
    live = [x['live_req'] / MIB for x in rows]
    alloc = [x['allocated'] / MIB for x in rows]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.fill_between(ops, 0, live, color='#9aa5ae', label='held by the program (live)', lw=0)
    ax.fill_between(ops, live, alloc, color='#c0392b', alpha=.75,
                    label='freed, still held by the allocator (quarantine)', lw=0)
    if model:
        L = sum(live) / len(live)
        for y, txt in ((L * (1 - model) / (1 - 2 * model), 'model min'), (L / (1 - 2 * model), 'model max')):
            ax.axhline(y, color='black', ls='--', lw=.8)
            ax.text(ops[-1], y, f' {txt} ({y / L:.2f}× live)', va='center', fontsize=7)
    ax.set_xlim(ops[0], ops[-1])
    ax.set_ylim(0, max(alloc) * 1.08)
    ax.set_xlabel('allocations')
    ax.set_ylabel('MiB')
    ax.set_title(setting(run['argv']), fontsize=9)
    ax.legend(fontsize=8, loc='lower left', framealpha=.9)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150)
    q = [a - l for a, l in zip(alloc, live)]
    print(f"window {ops[0]}..{ops[-1]}: live {min(live):.2f}-{max(live):.2f} MiB, "
          f"allocated {min(alloc):.2f}-{max(alloc):.2f} MiB, held for freed objects {min(q):.2f}-{max(q):.2f} MiB")


if __name__ == '__main__':
    main()
