#!/usr/bin/env python3
"""Where the bytes go: what jemalloc's footprint consists of, per program.

    plot-frag.py RESULT_DIR OUT_PREFIX

Input: runs under ./traced2 with MQ_TRACK=1 (mqtrace.so v2 prints
live_asked).  Layers are averaged over every sample at which the program
holds at least half of its peak live bytes (so neither the start-up ramp nor
a teardown, where everything sits in quarantine, is the picture; and the
quarantine and the holes, which trade against each other over a revocation
cycle, are both represented).  jemalloc's resident bytes split into
  asked       bytes the program asked for (sum of malloc n over live objects)
  rounding    usable - asked: jemalloc's size classes (internal fragmentation)
  quarantine  allocated - usable live: freed, still held (MRS quarantine and
              the arena being revoked); it also absorbs any allocation libc
              makes without going through its PLT, which the tracer cannot see
  holes       active - allocated: unused space in partly used slabs/extents
              (external fragmentation)
  dirty       resident - active: pages jemalloc has not returned to the OS
Bars are shares of mean resident (100% each); the MiB are printed under the
bars, with the number of samples averaged.
A run without live_asked, MQ-DONE or with a negative layer is reported on
stderr and left out, never drawn.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

LAYERS = [('asked', 'asked by the program (live)', '#9aa5ae'),
          ('rounding', 'size-class rounding', '#d5b26b'),
          ('quarantine', 'quarantine: freed, still held', '#c0392b'),
          ('holes', 'holes in slabs / extents', '#7d6b91'),
          ('dirty', 'dirty pages not yet returned', '#5d8aa8')]
MIB = 2**20


def layers(run):
    rows = [x for x in run['rows'] if 'live_asked' in x]
    if not rows:
        return None, 'no live_asked samples (old tracer or MQ_TRACK=0)'
    peak = max(x['live_asked'] for x in rows)
    cand = [x for x in rows if x['live_asked'] >= peak / 2] or rows
    n = len(cand)
    mean = lambda k: sum(x[k] for x in cand) / n
    x = {k: mean(k) for k in ('live_asked', 'live_req', 'allocated', 'active', 'resident')}
    x['n'] = n
    v = {'asked': x['live_asked'], 'rounding': x['live_req'] - x['live_asked'],
         'quarantine': x['allocated'] - x['live_req'], 'holes': x['active'] - x['allocated'],
         'dirty': x['resident'] - x['active']}
    bad = [k for k, b in v.items() if b < -0.005 * x['resident']]
    if bad:
        return None, f'negative layer(s) {bad} over {n} samples'
    for k in v:
        v[k] = max(v[k], 0.0)
    return v, x


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    data = []
    for p in sorted(src.glob('*.txt')):
        r = parse(p)
        if r['arm'] == 'off':
            continue
        if not (r['ok'] and r['rows']):
            print(f'FAILED {p.name}', file=sys.stderr)
            continue
        v, x = layers(r)
        if v is None:
            print(f'skipped {p.name}: {x}', file=sys.stderr)
            continue
        data.append((setting(r['argv']), v, x))
    if not data:
        sys.exit(f'no usable runs in {src}')
    data.sort(key=lambda d: d[1]['asked'])
    print(f"{'program':28} {'resident MiB':>12} {'asked MiB':>9} " + ' '.join(f'{k:>10}' for k, _, _ in LAYERS)
          + '   (share of mean resident)')
    for prog, v, x in data:
        tot = x['resident']
        print(f"{prog[:28]:28} {tot / MIB:12.2f} {v['asked'] / MIB:9.2f} " +
              ' '.join(f"{v[k] / tot:10.3f}" for k, _, _ in LAYERS) + f"   samples={x['n']}")
    fig, ax = plt.subplots(figsize=(1.1 * len(data) + 3.2, 4))
    bottom = [0.0] * len(data)
    for k, name, color in LAYERS:
        h = [v[k] / x['resident'] for _, v, x in data]
        ax.bar(range(len(data)), h, 0.62, bottom=bottom, color=color, label=name)
        for i, (b, hh) in enumerate(zip(bottom, h)):
            if hh >= 0.07:
                ax.text(i, b + hh / 2, f'{hh:.0%}', ha='center', va='center', fontsize=7,
                        color='white' if k in ('quarantine', 'holes', 'dirty') else '#1d2328')
        bottom = [b + hh for b, hh in zip(bottom, h)]
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{p.split()[0].lstrip('./')}\n{x['resident'] / MIB:.1f} MiB resident\n"
                        f"{v['asked'] / MIB:.2f} MiB asked\n({x['n']} samples)" for p, v, x in data], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1], ['0', '25%', '50%', '75%', '100%'])
    ax.set_ylabel("share of jemalloc's resident bytes")
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out + '-frag.png', dpi=150)


if __name__ == '__main__':
    main()
