#!/usr/bin/env python3
"""Validate event ledgers/oracles and plot paired SQLite memory observations."""
import argparse
import csv
import gzip
import json
import re
from pathlib import Path

ARMS = ['capstone', 'capstone-sublet', 'poisoncap-spatial', 'poisoncap-temporal']
LABELS = dict(zip(ARMS, ['Capstone original', 'Sublet', 'CheriBSD original', 'PoisonCap corrected']))
COLORS = dict(zip(ARMS, ['#777777', '#00796b', '#6688aa', '#c75135']))
PAIRS = [('capstone', 'capstone-sublet'), ('poisoncap-spatial', 'poisoncap-temporal')]


def paper_layout(root, byarm, ratios):
    """Reformat the complete churn campaign; no failed-burst comparison.

    These are layout previews of a selected allocator ledger, not new
    measurements or a complete platform-memory result.
    """
    import hashlib
    import statistics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    for arm in ARMS:
        rs = byarm[arm]
        if {r['rep'] for r in rs} != {1, 2, 3}:
            raise ValueError(f'paper preview needs all three repetitions: {arm}')
        for rep in (1, 2, 3):
            subset = [r for r in rs if r['rep'] == rep]
            if len(subset) != 17 or {r['unit'] for r in subset} != set(range(17)):
                raise ValueError(f'paper preview needs the complete churn profile: {arm}')
    if len(ratios) != 6:
        raise ValueError('paper preview needs six paired ratio records')

    out = root/'paper-layout'
    out.mkdir(exist_ok=True)
    palette = {'capstone-sublet': '#0072B2', 'poisoncap-temporal': '#D55E00'}
    names = {'capstone-sublet': 'Sublet', 'poisoncap-temporal': 'PoisonCap (corrected)'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .6,
                         'lines.linewidth': 1.2, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'}), PdfPages(out/'sqlite-layout-preview.pdf') as book:
        def save(fig, name):
            for ext in ('pdf', 'svg', 'png'):
                fig.savefig(out/(name+'.'+ext), dpi=220)
            book.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(7.05, 1.9))
        fig.subplots_adjust(left=.12, right=.97, bottom=.28, top=.71, wspace=.42)
        for ax, metric, title in zip(axes, ['address_ratio', 'peak_H_ratio'],
                                     ['(a) Allocated address coverage', '(b) Peak selected allocator bytes']):
            for i, (_, protected) in enumerate(PAIRS):
                vals = [r[metric] for r in ratios if r['arm'] == protected]
                center = statistics.median(vals)
                y = .14 if i == 0 else -.14
                ax.errorbar(center, y, xerr=[[center-min(vals)], [max(vals)-center]],
                            fmt='o' if i == 0 else 's', color=palette[protected],
                            markersize=4.5, capsize=2)
                ax.annotate(f'{center:.2f}×', (center, y), xytext=(6, 0),
                            textcoords='offset points', va='center', fontsize=8)
            ax.axvline(1, color='.4', lw=.8, ls=':', zorder=0)
            ax.set(xlim=(.65, 5.6), ylim=(-.38, .38), yticks=[0],
                   yticklabels=['SQLite\nmain'], xticks=[1, 2, 3, 4, 5],
                   xlabel='Protected / own original (×)', title=title)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', color='.92', linewidth=.5)
        handles = [Line2D([], [], color=palette[a], marker=m, ls='', label=names[a])
                   for a, m in zip(palette, ['o', 's'])]
        fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
                   bbox_to_anchor=(.53, 1.01))
        save(fig, '01-paired-memory-cost')

        def curves(ax, field, arms):
            for arm in arms:
                base, prot = next(pair for pair in PAIRS if arm in pair)
                data = [[next(r for r in byarm[arm] if r['rep'] == rep and r['unit'] == u)
                         for u in range(17)] for rep in (1, 2, 3)]
                vals = [[field(r)/2**20 for r in run] for run in data]
                median = [statistics.median(v) for v in zip(*vals)]
                low = [min(v) for v in zip(*vals)]
                high = [max(v) for v in zip(*vals)]
                original = arm == base
                ax.fill_between(range(1, 18), low, high, color=palette[prot], alpha=.12)
                ax.plot(range(1, 18), median, color=palette[prot],
                        ls='--' if original else '-', marker='o' if original else 's',
                        mfc='white' if original else palette[prot], markersize=3,
                        markevery=((0 if original else 2) + (prot == 'poisoncap-temporal'), 4),
                        zorder=3 if original else 2)

        def work_axes(ax, upper):
            ax.axvspan(.5, 1.5, color='.94', zorder=0)
            ax.set(xlim=(.5, 17.5), ylim=(0, upper), xticks=[1, 5, 9, 13, 17],
                   xlabel='Completed workload units')
            ax.grid(axis='y', color='.90', linewidth=.5)

        def common_legend(fig):
            hs = []
            for base, prot in PAIRS:
                hs.extend([
                    Line2D([], [], color=palette[prot], ls='--', marker='o',
                           mfc='white', markersize=3,
                           label='Capstone original' if base == 'capstone' else 'CheriBSD original'),
                    Line2D([], [], color=palette[prot], ls='-', marker='s',
                           markersize=3, label=names[prot])])
            fig.legend(handles=hs, loc='upper center', ncol=2, frameon=False,
                       bbox_to_anchor=(.53, 1.01), columnspacing=2.5)

        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.6))
        fig.subplots_adjust(left=.09, right=.98, bottom=.20, top=.70, wspace=.24)
        for ax, (base, prot), title in zip(axes, PAIRS, ['(a) Capstone', '(b) CheriBSD']):
            curves(ax, lambda r: r['ever'], [base, prot])
            work_axes(ax, 5.5)
            ax.set_title(title)
        axes[0].set_ylabel('Address coverage (MiB)')
        common_legend(fig)
        save(fig, '02-address-coverage')

        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.6))
        fig.subplots_adjust(left=.09, right=.98, bottom=.20, top=.70, wspace=.24)
        for ax, field, title in zip(axes, ['peak_held', 'held'],
                                    ['(a) Peak within each workload unit', '(b) After database close']):
            curves(ax, lambda r: r[field]+r['metadata'], ARMS)
            work_axes(ax, 7.2)
            ax.set_title(title)
        axes[0].set_ylabel('Selected allocator bytes (MiB)')
        common_legend(fig)
        save(fig, '03-repeated-work')

    inputs = ['runs.json', 'oracles.json', 'unit-ledger.csv', 'validation.json', 'metrics.json']
    record = {'scope': 'Layout preview; complete SQLite churn runs only; selected allocator accounting',
              'new_measurements': False, 'width_inches': 7.05,
              'summary': 'median and full range of three paired repetitions',
              'input_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs},
              'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')


def readtext(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return f.read()
    return path.read_text(errors='replace')


def parse(raw):
    ledgers, oracles, begins, ends = [], {}, {}, {}
    unit = None
    for line in raw.splitlines():
        begin = re.search(r'STUDY-BEGIN unit=(\d+) size=(\d+)', line)
        if begin:
            unit, size = map(int, begin.groups())
            if unit in begins:
                raise ValueError(f'duplicate unit {unit}')
            begins[unit] = size
            oracles[unit] = []
        oracle = re.search(r'STUDY-ORACLE phase=(\d+) rows=(\d+) hash=([0-9a-f]+)', line)
        if oracle:
            if unit is None:
                raise ValueError('oracle without unit')
            oracles[unit].append(list(oracle.groups()))
        end = re.search(r'STUDY-END unit=(\d+) rc=(\d+)', line)
        if end:
            k, rc = map(int, end.groups())
            if k in ends:
                raise ValueError(f'duplicate end {k}')
            ends[k] = rc
        if 'STUDY-LEDGER ' in line:
            row = {k: int(v) for k,v in re.findall(r'(\w+)=(-?\d+)', line.split('STUDY-LEDGER ',1)[1])}
            required = {'unit','phase','live','held','peak_held','quarantine','metadata','free',
                        'largest_free','pool','ever','window','allocs','reused_starts','oom','observer'}
            if set(row) != required:
                raise ValueError('incomplete ledger fields')
            if (any(v<0 for k,v in row.items() if k!='phase') or
                row['pool']<=0 or row['peak_held']>row['pool'] or
                row['live']+row['quarantine'] != row['held'] or
                row['held']+row['free'] != row['pool'] or
                not 0 <= row['largest_free'] <= row['free'] or
                not 0 <= row['window'] <= row['ever'] <= row['pool'] or
                not 0 <= row['reused_starts'] <= row['allocs'] or
                row['peak_held'] < row['held']):
                raise ValueError(f'inconsistent ledger: {row}')
            if ledgers:
                prior=ledgers[-1]
                if row['ever']<prior['ever'] or row['pool']!=prior['pool'] or row['metadata']!=prior['metadata']:
                    raise ValueError('allocator configuration or cumulative footprint changed')
                if row['unit']==prior['unit'] and any(row[k]<prior[k] for k in ['peak_held','window','allocs','reused_starts']):
                    raise ValueError('nonmonotonic within-unit event counters')
            ledgers.append(row)
    return ledgers, begins, ends, oracles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--paper-layout', action='store_true',
                        help='write compact paper previews of complete churn data')
    args = parser.parse_args()
    root = args.campaign
    config = json.loads((root/'runs.json').read_text())
    reference = json.loads((root/'oracles.json').read_text())
    rows, validation = [], []
    for run in config['runs']:
        raw = readtext(root/run['stdout'])
        try:
            ledgers, begins, ends, oracles = parse(raw)
            mismatch = [u for u in ends if oracles[u] != reference[str(begins[u])]]
            if mismatch:
                raise ValueError(f'oracle mismatch in completed units: {mismatch}')
            if any(int(v) for v in re.findall(r'STUDY-REVOKE[^\n]* errors=(\d+)',raw)):
                raise ValueError('explicit revocation returned an error')
            complete = run['status'] == 'completed'
            if complete:
                count = 1 if run['profile']=='qualification' else 17 if run['profile']=='churn' else 13
                if set(ends) != set(range(count)) or any(ends.values()):
                    raise ValueError('missing or failed units')
                if f'STUDY-COMPLETE units={count}' not in raw:
                    raise ValueError('missing completion marker')
                releases = [r for r in ledgers if r['phase']==-1]
                if len(releases) != count or any(r['live'] or r['oom'] for r in releases):
                    raise ValueError('live allocations or OOM after close')
                if run['arm'].startswith('capstone') and 'DROPPED 0 RC 0' not in raw:
                    raise ValueError('domain output gate failed')
            for row in ledgers:
                rows.append(dict(arm=run['arm'], profile=run['profile'], rep=run['rep'],
                                 complete=complete, size=begins[row['unit']], **row))
            validation.append(dict(arm=run['arm'],profile=run['profile'],rep=run['rep'],
                                   status=run['status'],completed_units=len(ends),valid=True))
        except (KeyError,ValueError) as exc:
            validation.append(dict(arm=run['arm'],profile=run['profile'],rep=run['rep'],
                                   status=run['status'],valid=False,error=str(exc)))
    (root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    bad = [r for r in validation if not r['valid']]
    if bad:
        raise SystemExit(json.dumps(bad,indent=2))
    if not rows:
        raise SystemExit('no observed ledgers')
    with (root/'phase-ledger.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ends=[r for r in rows if r['phase']==-1]
    with (root/'unit-ledger.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(ends)
    churn=[r for r in ends if r['profile']=='churn' and r['complete']]
    byarm={arm: [r for r in churn if r['arm']==arm] for arm in ARMS}
    summary={arm: {'completed_churn_repetitions':sorted({r['rep'] for r in rs}),
                   'peak_H_bytes':max((r['peak_held']+r['metadata'] for r in rs if r['unit']>0),default=None),
                   'max_end_H_bytes':max((r['held']+r['metadata'] for r in rs if r['unit']>0),default=None),
                   'max_address_footprint_bytes':max((r['ever'] for r in rs),default=None),
                   'metadata_bytes':sorted({r['metadata'] for r in rs})}
             for arm,rs in byarm.items()}
    ratios=[]
    for base,prot in PAIRS:
        for rep in range(1,4):
            a=[r for r in byarm[base] if r['rep']==rep and r['unit']>0]
            b=[r for r in byarm[prot] if r['rep']==rep and r['unit']>0]
            if len(a)==len(b)==16:
                ratios.append(dict(arm=prot,rep=rep,peak_H_ratio=max(r['peak_held']+r['metadata'] for r in b)/max(r['peak_held']+r['metadata'] for r in a),
                                   address_ratio=max(r['ever'] for r in b)/max(r['ever'] for r in a)))
    (root/'metrics.json').write_text(json.dumps(dict(arms=summary,ratios=ratios),indent=2)+'\n')

    if args.paper_layout:
        paper_layout(root, byarm, ratios)
        print(root/'paper-layout'/'sqlite-layout-preview.pdf')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout(rect=(0,.04,1,1))
        fig.text(.01,.008,'SQLite 3.22.0 speedtest1 main · 129,055 × 64-byte atoms · selected allocator accounting; not RSS',fontsize=8)
        for ext in ('pdf','svg','png'):fig.savefig(figdir/(name+'.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,(base,prot) in enumerate(PAIRS):
        vals=[r['peak_H_ratio'] for r in ratios if r['arm']==prot]
        if vals:
            axes[0].bar(i,sum(vals)/len(vals),color=COLORS[prot],alpha=.65)
            axes[0].scatter([i]*len(vals),vals,color='black',s=18,zorder=3)
            axes[0].text(i,max(vals)+.08,f'{sum(vals)/len(vals):.2f}× (n={len(vals)})',ha='center')
        else:
            axes[0].text(i,1.1,'incomplete',ha='center',rotation=90)
    axes[0].set_ylim(0,max([r['peak_H_ratio'] for r in ratios]+[1])*1.15)
    axes[0].axhline(1,color='black',ls=':',lw=1)
    axes[0].set(xticks=[0,1],xticklabels=['Sublet / original','PoisonCap / original'],ylabel='Peak H: protected / original',title='Within-platform protection cost',xlim=(-.6,1.6))
    for i,arm in enumerate(ARMS):
        rs=byarm[arm]
        if rs:
            values=[max(r['peak_held']+r['metadata'] for r in rs if r['rep']==rep and r['unit']>0)/2**20 for rep in sorted({r['rep'] for r in rs})]
            axes[1].bar(i,sum(values)/len(values),color=COLORS[arm],alpha=.7)
            axes[1].scatter([i]*len(values),values,s=18,color='black')
    axes[1].set(xticks=range(4),xticklabels=['Cap. orig.','Sublet','CHERI orig.','PoisonCap'],ylabel='Peak H (MiB)',title='Absolute selected footprint')
    save(fig,'01-protection-cost')

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for arm in ARMS:
        for rep in sorted({r['rep'] for r in byarm[arm]}):
            rs=sorted([r for r in byarm[arm] if r['rep']==rep],key=lambda r:r['unit'])
            for ax,key in zip(axes,['peak_held','held']):
                ax.plot([r['unit'] for r in rs],[(r[key]+r['metadata'])/2**20 for r in rs],
                        color=COLORS[arm],label=LABELS[arm] if rep==1 else None,
                        ls='--' if 'original' in LABELS[arm] else '-',alpha=.8)
    for ax in axes:
        ax.axvspan(-.1,.5,color='#eeeeee',zorder=-1);ax.set(xlabel='Complete workload unit (0 = warmup)',ylabel='H (MiB)',xticks=[0,4,8,12,16]);ax.grid(alpha=.2)
    axes[0].set_title('Peak within each unit');axes[1].set_title('After database close')
    axes[1].legend(fontsize=8)
    save(fig,'02-retained-memory')

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for arm in ARMS:
        for rep in sorted({r['rep'] for r in byarm[arm]}):
            rs=sorted([r for r in byarm[arm] if r['rep']==rep],key=lambda r:r['unit'])
            for ax,key in zip(axes,['ever','window']):
                ax.plot([r['unit'] for r in rs],[100*r[key]/r['pool'] for r in rs],color=COLORS[arm],
                        label=LABELS[arm] if rep==1 else None,ls='--' if 'original' in LABELS[arm] else '-',alpha=.8)
    for ax in axes[:2]:
        ax.set(xlabel='Complete workload unit (0 = warmup)',ylabel='Distinct allocated arena bytes (% of pool)',ylim=(0,105),xticks=[0,4,8,12,16]);ax.grid(alpha=.2)
    for base,prot in PAIRS:
        for rep in range(1,4):
            denom={r['unit']:r['ever'] for r in byarm[base] if r['rep']==rep}
            rr=sorted([r for r in byarm[prot] if r['rep']==rep and r['unit'] in denom],key=lambda r:r['unit'])
            axes[2].plot([r['unit'] for r in rr],[r['ever']/denom[r['unit']] for r in rr],color=COLORS[prot],label=LABELS[prot] if rep==1 else None)
    axes[2].axhline(1,color='black',ls=':',lw=1)
    axes[2].set(xlabel='Complete workload unit (0 = warmup)',ylabel='Cumulative footprint: protected / original',title='Within-platform address expansion',ylim=(.8,4.3),xticks=[0,4,8,12,16]);axes[2].grid(alpha=.2);axes[2].legend(fontsize=8)
    axes[0].set_title('Cumulative address footprint');axes[1].set_title('Address footprint in each unit');axes[1].legend(fontsize=8)
    save(fig,'03-address-footprint')

    burst=[r for r in ends if r['profile']=='burst']
    if burst:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for arm in ARMS:
            rs=[r for r in burst if r['arm']==arm]
            for rep in sorted({r['rep'] for r in rs}):
                rr=sorted([r for r in rs if r['rep']==rep],key=lambda r:r['unit'])
                for ax,key in zip(axes,['peak_held','held']):
                    ax.plot([r['unit'] for r in rr],[(r[key]+r['metadata'])/2**20 for r in rr],color=COLORS[arm],label=LABELS[arm] if rep==1 else None,ls='--' if 'original' in LABELS[arm] else '-')
            failures=[r for r in validation if r['arm']==arm and r['profile']=='burst' and r['status']!='completed']
            for r in failures:
                for ax in axes:
                    ax.axvline(r['completed_units'],color=COLORS[arm],ls=':')
                    ax.text(r['completed_units']+.15,.5,LABELS[arm]+' failed',color=COLORS[arm],fontsize=8)
        for ax in axes:
            ax.axvspan(3.5,4.5,color='#eeeeee',label='Size-4 burst')
            ax.set(xlabel='Complete workload unit',ylabel='H (MiB)',xlim=(-.2,12.3),ylim=(0,None));ax.grid(alpha=.2)
        axes[0].set_title('Peak within each unit; failed burst not imputed')
        axes[1].set_title('After database close');axes[1].legend(fontsize=8)
        save(fig,'04-burst-diagnostic')
    print(json.dumps(dict(arms=summary,ratios=ratios),indent=2))


if __name__=='__main__':main()
