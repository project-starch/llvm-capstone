#!/usr/bin/env python3
"""CheriBSD default (jemalloc + MRS quarantine) against virtual Capstone (musl mallocng,
revocation at free), the same mimalloc-bench programs, two figures:

    compare.py CHERI_RESULTS CAPSTONE_RUNS OUT_PREFIX [CHERI_OFF_DIR] [NATIVE_DIR]

  OUT-reuse.png     per program, the CDF of the reuse distance of an address: allocations
                    between the free of an address and its next allocation (log2 buckets,
                    from each tracer's MQ-HIST). With CHERI_OFF_DIR (run.sh output of the
                    `off` arm: MRS revocation disabled, jemalloc without quarantine) and
                    NATIVE_DIR (<prog>.txt of the same programs on native musl 1.2.5
                    mallocng, no revocation) the unprotected allocators are drawn dashed.
  OUT-footprint.png per program, bytes held per byte the program asked for, stacked:
      CheriBSD   asked | size-class rounding | quarantine | jemalloc holes + dirty pages |
                 MRS bookkeeping (procstat, mrs: mappings) | rest of max RSS
      Capstone   asked | pinned pages beyond asked (mallocng slack, image, stack) |
                 node table | rest of max RSS
    The jemalloc layers are means over the samples at which the program holds at least
    half of its peak asked bytes (as plot-frag.py); asked is that same mean. RSS is the
    kernel's maximum over the untraced runs (mean of the repetitions). A Capstone run that
    ended at its time limit has no CAPSTONE_VM_STATS line: its bar shows asked and the rest
    of RSS only, and is marked (limit).

CHERI_RESULTS is the experiments/malloc-quarantine results root (e3-mb, e9-rep, e17-rep,
e16-fit, e17-fit, e18-live, e19-reuse, e14-maps, e19-maps). CAPSTONE_RUNS is a run directory
of remote-probe.sh (work-<name>/serial.log). Every number drawn is printed; a missing input
is reported, never drawn as zero.
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROGS = ['glibc-simple', 'espresso', 'cfrac', 'mstress', 'barnes', 'sh6bench']
CHERI_ARGV = {'glibc-simple': 'glibc-simple', 'espresso': 'espresso_largest.espresso',
              'cfrac': 'cfrac_17545186520507317056371138836327483792789528',
              'mstress': 'mstress_1_50_25', 'barnes': 'barnes___input', 'sh6bench': 'sh6bench_1'}
CHERI_TRACED = {'glibc-simple': 'e16-fit', 'espresso': 'e16-fit', 'cfrac': 'e16-fit',
                'mstress': 'e16-fit', 'barnes': 'e17-fit', 'sh6bench': 'e18-live'}
CHERI_MAPS = {'glibc-simple': 'e14-maps/glibc-simple.txt', 'sh6bench': 'e14-maps/sh6bench.txt',
              'cfrac': 'e19-maps/cfrac.txt', 'espresso': 'e19-maps/espresso.txt',
              'barnes': 'e19-maps/barnes.txt', 'mstress': 'e19-maps/mstress_1_50_25.txt'}
COL = {'cheri': '#c0392b', 'capstone': '#2471a3'}
LIGHT = {'cheri': '#f3b7b0', 'capstone': '#a9cbe3'}
MIB = 2**20


def rows_of(text, prefix='MQ op='):
    return [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
            for line in text.splitlines() if line.startswith(prefix)]


def hist_of(text):
    m = None
    for m in re.finditer(r'^MQ-HIST(.*)$', text, re.M):
        pass
    if not m:
        return None
    h = {int(b): int(c) for b, c in re.findall(r'(\d+):(\d+)', m.group(1))}
    return h if h else None


def steady(rows, key):
    peak = max(r[key] for r in rows)
    return [r for r in rows if r[key] >= peak / 2] or rows


def mean(xs):
    return sum(xs) / len(xs)


def cheri(root, prog):
    """The CheriBSD inputs of one program: traced rows/hist, RSS reps, MRS pages."""
    out = {'notes': []}
    files = list((root / CHERI_TRACED[prog]).glob(f'*{CHERI_ARGV[prog]}.txt'))
    if not files:
        out['notes'].append(f'no traced run under {CHERI_TRACED[prog]}')
        return out
    text = files[0].read_text(errors='replace')
    rows = [r for r in rows_of(text) if r.get('live_asked')]
    out['rows'] = rows
    out['hist'] = hist_of(text)
    out['complete'] = 'MQ-EXIT rc=0' in text
    if prog == 'sh6bench':
        # E18's live table filled at op 176M of 193M, before the program's peak; the
        # table-free E19 run (1/16 address sample) saw the whole run. It reports usable
        # bytes only, so the asked layer is usable bytes and the rounding layer is 0.
        reuse = list((root / 'e19-reuse').glob(f'*{CHERI_ARGV[prog]}.txt'))
        if reuse:
            text = reuse[0].read_text(errors='replace')
            out['hist'] = hist_of(text)
            out['complete'] = 'MQ-EXIT rc=0' in text
            rows = [r for r in rows_of(text) if r.get('live_req')]
            for r in rows:
                r['live_asked'] = r['live_req']
            out['rows'] = rows
            out['notes'].append('reuse histogram and live bytes from the 1/16 address sample '
                                '(e19-reuse): asked = usable bytes, rounding not separable')
        else:
            out['hist'] = None
    rss = []
    for d in ('e3-mb', 'e9-rep/r1', 'e9-rep/r2', 'e9-rep/r3', 'e17-rep/r2', 'e17-rep/r3'):
        for f in (root / d).glob(f'on-._{CHERI_ARGV[prog]}.txt'):
            m = re.search(r'^\s*(\d+)\s+maximum resident set size', f.read_text(errors='replace'), re.M)
            if m:
                rss.append(int(m.group(1)) * 1024)
    out['rss'] = rss
    maps = root / CHERI_MAPS[prog]
    if maps.exists():
        samples = rows_of(maps.read_text(errors='replace'), 'MAPS t=')
        samples = [s for s in samples if s.get('jemalloc')]
        if samples:
            st = steady(samples, 'jemalloc')
            out['mrs'] = mean([s['mrs'] for s in st]) * 4096
            out['mrs_n'] = len(st)
    else:
        out['notes'].append(f'no mappings sample {CHERI_MAPS[prog]}')
    return out


def capstone(runs, prog):
    out = {'notes': []}
    traced = runs / f'work-{prog}-traced' / 'serial.log'
    plain = runs / f'work-{prog}' / 'serial.log'
    if traced.exists():
        text = traced.read_text(errors='replace')
        out['rows'] = [r for r in rows_of(text) if r.get('live_asked')]
        out['hist'] = hist_of(text)
        out['complete'] = 'MQ-DONE' in text
        m = re.search(r'^MQ-ADDR addr_sample=(\d+)', text, re.M)
        if m and int(m.group(1)) > 1:
            out['notes'].append(f'reuse histogram from a 1/{m.group(1)} address sample')
    else:
        out['notes'].append('no traced run')
    if plain.exists():
        text = plain.read_text(errors='replace')
        m = re.search(r'^MB_RUSAGE .*maxrss_kib=(\d+)', text, re.M)
        out['rss'] = [int(m.group(1)) * 1024] if m else []
        s = re.search(r'^CAPSTONE_VM_STATS .*peak=(\d+).*node_bytes=(\d+)', text, re.M)
        if s:
            out['peak_pages'] = int(s.group(1)) * 4096
            out['node_bytes'] = int(s.group(2))
        out['plain_complete'] = bool(re.search(r'^MB_END:\S+:0', text, re.M))
    else:
        out['notes'].append('no untraced run')
    return out


def cdf(h):
    total = sum(h.values())
    xs, ys, acc = [], [], 0
    for b in range(64):
        acc += h.get(b, 0)
        xs.append(2 ** b)
        ys.append(acc / total)
    return xs, ys, total


def unprotected(root, off_dir, native_dir, prog):
    """The same program on each allocator without its protection: CheriBSD `off` arm
    (run.sh file off-*<argv>.txt; glibc-simple's older run under e11a-reuse) and native
    musl mallocng (<prog>.txt). Each: hist, complete, or None."""
    out = {}
    files = []
    if off_dir:
        files = sorted(Path(off_dir).glob(f'off-*{CHERI_ARGV[prog]}.txt'))
    if not files:
        files = sorted((root / 'e11a-reuse').glob(f'off-*{CHERI_ARGV[prog]}.txt'))
    if files:
        text = files[0].read_text(errors='replace')
        out['cheri_off'] = {'hist': hist_of(text), 'complete': 'MQ-EXIT rc=0' in text,
                            'file': str(files[0])}
    if native_dir and (Path(native_dir) / f'{prog}.txt').exists():
        text = (Path(native_dir) / f'{prog}.txt').read_text(errors='replace')
        out['native'] = {'hist': hist_of(text), 'complete': 'MQ-DONE' in text,
                         'file': str(Path(native_dir) / f'{prog}.txt')}
    return out


def main():
    root, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    off_dir = sys.argv[4] if len(sys.argv) > 4 else None
    native_dir = sys.argv[5] if len(sys.argv) > 5 else None
    data = {p: {'cheri': cheri(root, p), 'capstone': capstone(runs, p)} for p in PROGS}
    for p in PROGS:
        data[p].update(unprotected(root, off_dir, native_dir, p))

    # --- reuse distance -------------------------------------------------------------
    # The unprotected allocator is a wide light band drawn first; the protected one a thin
    # dark line on top. Coincidence shows as a dark line centred in its band. Programs
    # without a histogram on either side (barnes: 20 objects, no reuse) are left out; a
    # run that ended at a limit contributes the sample up to there, said in the caption.
    from matplotlib.lines import Line2D
    STYLE = {'cheri_off': (LIGHT['cheri'], 4.0, 1), 'native': (LIGHT['capstone'], 4.0, 1),
             'cheri': (COL['cheri'], 1.4, 2), 'capstone': (COL['capstone'], 1.4, 2)}
    NAMES = {'cheri': 'CheriBSD, MRS quarantine', 'cheri_off': 'CheriBSD, no quarantine',
             'capstone': 'Capstone, revoke at free', 'native': 'mallocng native, no revocation'}
    plotted = [p for p in PROGS if any((data[p].get(s) or {}).get('hist')
                                       for s in ('cheri', 'capstone'))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0))
    for ax, prog in zip(axes.flat, plotted):
        for sysname in ('cheri_off', 'native', 'cheri', 'capstone'):
            d = data[prog].get(sysname)
            h = d.get('hist') if d else None
            if not h:
                note = '; '.join(d['notes']) if d and 'notes' in d else 'no file'
                print(f'{prog:13} {sysname:9} reuse: no histogram ({note or "none"})')
                continue
            xs, ys, total = cdf(h)
            med = next(x for x, y in zip(xs, ys) if y >= 0.5)
            col, lw, z = STYLE[sysname]
            ax.step(xs, ys, where='post', color=col, lw=lw, zorder=z, solid_capstyle='butt')
            print(f'{prog:13} {sysname:9} reuse: reused={total} median_distance<={med} complete={d.get("complete")}')
        ax.set_xscale('log', base=2)
        ax.set_xlim(1, 2 ** 26)
        ax.set_ylim(0, 1)
        ax.set_title(prog, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(plotted):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('Fraction of reuses')
    for ax in axes[1, :len(plotted) - 3]:
        ax.set_xlabel('Reuse distance (allocations, log2 bins)', fontsize=9)
    for ax in axes[0, len(plotted) - 3:]:
        ax.set_xlabel('Reuse distance (allocations, log2 bins)', fontsize=9)
    handles = [Line2D([0], [0], color=STYLE[s][0], lw=STYLE[s][1]) for s in NAMES]
    axes.flat[-1].legend(handles, [NAMES[s] for s in NAMES], fontsize=9, frameon=False,
                         loc='center')
    fig.tight_layout()
    fig.savefig(out + '-reuse.png', dpi=150)

    # --- footprint: what the max RSS consists of, per program --------------------------
    LAY = {'cheri': [('asked', 'asked (live)', '#9aa5ae'),
                     ('rounding', 'size-class rounding', '#d5b26b'),
                     ('quarantine', 'quarantine (freed, held)', '#c0392b'),
                     ('jemalloc_other', 'jemalloc holes, dirty pages, metadata', '#7d6b91'),
                     ('mrs', 'MRS bookkeeping (16 B/object)', '#e67e22'),
                     ('rest', 'rest of max RSS', '#c8ced4')],
           'capstone': [('asked', 'asked (live)', '#9aa5ae'),
                        ('slack', 'pinned pages beyond asked (mallocng slack, image, stack)', '#5d8aa8'),
                        ('nodes', 'node table', '#2471a3'),
                        ('rest', 'rest of max RSS (launcher)', '#e1e7ec')]}
    bars = {}
    for prog in PROGS:
        for sysname in ('cheri', 'capstone'):
            d = data[prog][sysname]
            rows = d.get('rows')
            if not d.get('rss') or not rows:
                # barnes on Capstone ends at the mapping limit before its first sample:
                # max RSS known, live bytes not, so no bar
                print(f'{prog:13} {sysname:9} footprint: missing ({"; ".join(d["notes"]) or "rows/rss"})')
                continue
            st = steady(rows, 'live_asked')
            asked = mean([r['live_asked'] for r in st])
            rss = mean(d['rss'])
            # layers in order; the stack is clipped at the max RSS, because jemalloc's
            # resident counter also counts pages the program never touched (barnes)
            if sysname == 'cheri':
                usable = mean([r['live_req'] for r in st])
                allocated = mean([r['allocated'] for r in st])
                resident = mean([r['resident'] for r in st])
                seq = [('asked', asked), ('rounding', usable - asked), ('quarantine', allocated - usable),
                       ('jemalloc_other', resident - allocated), ('mrs', d.get('mrs', 0.0))]
                if 'mrs' not in d:
                    d['notes'].append('MRS bookkeeping unknown: drawn as 0')
            elif 'peak_pages' in d:
                seq = [('asked', asked), ('slack', d['peak_pages'] - asked), ('nodes', d['node_bytes'])]
            else:
                seq = [('asked', asked), ('slack', 0.0), ('nodes', 0.0)]
                d['notes'].append('no CAPSTONE_VM_STATS (time limit): slack and node table inside rest')
            v, cum, clipped = {}, 0.0, False
            for k, x in seq:
                x = max(x, 0.0)
                if cum + x > rss:
                    x = max(rss - cum, 0.0)
                    clipped = True
                v[k] = x
                cum += x
            v['rest'] = rss - cum
            if clipped:
                d['notes'].append('layers clipped at max RSS (allocator counts pages never touched)')
            done = d.get('plain_complete', d.get('complete', True)) and d.get('complete', True)
            lab = '' if done else 'abgebrochen, '
            if clipped:
                lab += 'gekappt, '
            # the ratio uses what the program asked for, even where the drawn layer is clipped
            bars[(prog, sysname)] = (v, rss, len(st), lab, rss / asked if asked else None)
            print(f'{prog:13} {sysname:9} footprint: asked={asked / MIB:.2f}MiB rss={rss / MIB:.2f}MiB '
                  f'rss/asked={rss / asked if asked else float("nan"):.2f} '
                  + ' '.join(f'{k}={x / MIB:.2f}MiB' for k, x in v.items())
                  + f' samples={len(st)} notes={d["notes"]}')
    # Only programs with both bars are drawn (barnes has no Capstone bar: it stops at the
    # mapping limit before its first sample). Limits and clipping are said in the caption,
    # and printed above, not written into the figure.
    plotted = [p for p in PROGS if (p, 'cheri') in bars and (p, 'capstone') in bars]

    # --- share: the same three layers on both systems ---------------------------------
    # asked (live bytes the program requested) | protection (what the temporal-safety
    # mechanism itself holds: CheriBSD quarantine + MRS bookkeeping, Capstone node table) |
    # everything else (allocator slack, image, libc, stack, launcher). Each bar is 100 % of
    # the max RSS; the protection layer carries its bytes per live byte.
    if plotted:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.0))
        SH = [('asked', 'asked (live)', '#9aa5ae'), ('protection', 'protection: quarantine + MRS / node table', '#c0392b'),
              ('other', 'everything else: allocator slack, image, libc, stack, launcher', '#dfe4e8')]
        handles = {}
        for ax, prog in zip(axes.flat, plotted):
            names = []
            for i, sysname in enumerate(('cheri', 'capstone')):
                v, rss, n, lim, ratio = bars[(prog, sysname)]
                prot = v['quarantine'] + v['mrs'] if sysname == 'cheri' else v['nodes']
                asked = v['asked']
                known = sysname == 'cheri' or 'peak_pages' in data[prog][sysname]
                s = {'asked': asked / rss, 'protection': prot / rss, 'other': 1 - (asked + prot) / rss}
                bottom = 0.0
                for key, name, color in SH:
                    if s[key] <= 0:
                        continue
                    b = ax.bar(i, s[key] * 100, 0.7, bottom=bottom, color=color)
                    handles.setdefault(name, b)
                    bottom += s[key] * 100
                if known and asked > 0:
                    r = prot / asked
                    ax.text(i, 101, (f'{r:.0f}' if r >= 10 else f'{r:.2g}') + '× live', ha='center',
                            va='bottom', fontsize=8)
                names.append(f'{"CheriBSD" if sysname == "cheri" else "Capstone"}\n{rss / MIB:.1f} MiB')
                print(f'{prog:13} {sysname:9} share: asked={s["asked"] * 100:.1f}% protection={s["protection"] * 100:.1f}% '
                      f'other={s["other"] * 100:.1f}% protection/live={prot / asked if asked else float("nan"):.2f} known={known}')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(names, fontsize=9)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0, 112)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_title(prog, fontsize=10)
            ax.set_ylabel('share of max RSS (%)')
        for ax in axes.flat[len(plotted):]:
            ax.axis('off')
        axes.flat[-1].legend(handles.values(), handles.keys(), fontsize=8, frameon=False, loc='center')
        fig.tight_layout()
        fig.savefig(out + '-share.png', dpi=150)
    if plotted:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
        handles = {}
        for ax, prog in zip(axes.flat, plotted):
            names, tops = [], []
            for i, sysname in enumerate(('cheri', 'capstone')):
                v, rss, n, lim, ratio = bars[(prog, sysname)]
                bottom = 0.0
                for key, name, color in LAY[sysname]:
                    if v[key] <= 0:
                        continue
                    b = ax.bar(i, v[key] / MIB, 0.7, bottom=bottom, color=color)
                    handles.setdefault(name, b)
                    bottom += v[key] / MIB
                if ratio is not None:
                    ax.text(i, bottom, f'{ratio:.1f}× asked', ha='center', va='bottom', fontsize=8)
                names.append(f'{"CheriBSD" if sysname == "cheri" else "Capstone"}\n{rss / MIB:.1f} MiB')
                tops.append(bottom)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(names, fontsize=9)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0, max(tops) * 1.18)
            ax.set_title(prog, fontsize=10)
            ax.set_ylabel('max RSS (MiB)')
        for ax in axes.flat[len(plotted):]:
            ax.axis('off')
        axes.flat[-1].legend(handles.values(), handles.keys(), fontsize=8, frameon=False, loc='center')
        fig.tight_layout()
        fig.savefig(out + '-footprint.png', dpi=150)
    else:
        print('no program with both footprint bars')


if __name__ == '__main__':
    main()
