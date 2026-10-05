#!/usr/bin/env python3
"""Validate the complete matrix and export counters and reproducible figures."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent

def ratio(n,d):
    return n/d if d else None

def collect(roots):
    records=[]
    for root in roots:
        for path in sorted(root.glob('*/baseline-*/result.json')) + sorted(root.glob('*/observed-*/result.json')):
            value=json.loads(path.read_text())
            if value['status']!='pass': raise ValueError(f'incomplete run: {path}')
            value['raw_record_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(value)
    return validate(records)

def validate(records):
    plan=json.loads((HERE/'workloads.json').read_text())
    expected={}
    for spec in plan['workloads']:
        if spec['application']=='wireshark':
            for capture in spec['captures']: expected['tshark-'+Path(capture).stem]=spec['application']
        else: expected[spec['id']]=spec['application']
    keys=[(r['workload'],r['variant'],r['repetition']) for r in records]
    wanted={(w,v,i) for w in expected for v in ['baseline','observed'] for i in range(1,plan['repetitions']+1)}
    if len(keys)!=len(set(keys)) or set(keys)!=wanted: raise ValueError('missing or duplicate matrix cells')
    reference={}
    for row in records:
        if row['application']!=expected[row['workload']]: raise ValueError('application mapping mismatch')
        if row['oracle']!=reference.setdefault(row['workload'],row['oracle']): raise ValueError('oracle mismatch')
        if row['variant']=='observed':
            for c in row['measurement']['allocators']:
                if c['unknown_alloc'] or c['unknown_free'] or c['unknown_reuse']: raise ValueError('incomplete observation')
                if c['alloc']!=c['free']+c['live']: raise ValueError('lifetime partition failed')
                if c['reuse']!=c['inside']+c['outside']+c['unknown_reuse']: raise ValueError('reuse partition failed')
                if sum(c['gap_inside'])>c['inside']: raise ValueError('gap count overflow')
    return sorted(records,key=lambda r:(r['application'],r['workload'],r['repetition'],r['variant']))

def read_export(path):
    records=json.loads((path/'runs.json').read_text())
    for record in records:
        record['status']='pass'
        if record['variant']=='observed':
            record['measurement']={'allocators':record['allocators']}
    return validate(records)

def export(records,dest):
    dest.mkdir(parents=True,exist_ok=True)
    compact=[]
    groups=defaultdict(list)
    for run in records:
        row={key:run[key] for key in ['application','workload','variant','repetition','binary_sha256','oracle','raw_record_sha256']}
        row['command']=run['command']
        if run['variant']=='observed':
            row['allocators']=[]
            for measured in run['measurement']['allocators']:
                c={k:v for k,v in measured.items() if k not in ['live_bytes','peak_bytes']}
                row['allocators'].append(c)
                groups[(run['application'],run['workload'],c['family'])].append(c)
        compact.append(row)
    (dest/'runs.json').write_text(json.dumps(compact,indent=2)+'\n')
    aggregate=[]
    for (app,workload,family),runs in sorted(groups.items()):
        if len(runs)!=3: raise ValueError('allocator row missing from a repetition')
        row=dict(application=app,workload=workload,allocator=family,repetitions=3,
                 exercised=any(r['alloc'] for r in runs))
        for key in ['alloc','free','reuse','inside','outside','bulk_free','live','cross_instance']:
            row[key+'_min']=min(r[key] for r in runs)
            row[key+'_max']=max(r[key] for r in runs)
        for name,denominator in [('I_over_A','alloc'),('I_over_R','reuse')]:
            values=[ratio(r['inside'],r[denominator]) for r in runs]
            known=[v for v in values if v is not None]
            row[name+'_mean']=statistics.mean(known) if known else None
            row[name+'_min']=min(known) if known else None
            row[name+'_max']=max(known) if known else None
        row['gap_inside_count']=sum(sum(r['gap_inside']) for r in runs)
        row['inside_count']=sum(r['inside'] for r in runs)
        row['gap_coverage']=ratio(row['gap_inside_count'],row['inside_count'])
        aggregate.append(row)
    with (dest/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(aggregate)
    (dest/'summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    return aggregate,groups

def figures(rows,groups,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
                         'svg.hashsalt':'native-lifetime-survey'})
    def save(fig,stem):
        for suffix in ['pdf','svg','png']:
            metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else ({'Date':None} if suffix=='svg' else None)
            path=dest/(stem+'.'+suffix)
            fig.savefig(path,dpi=180,metadata=metadata)
            if suffix=='svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    apps=['sqlite','postgresql','cpython','mruby','perl','ffmpeg','wireshark','memcached']
    active=sorted([r for r in rows if r['exercised']],key=lambda r:(apps.index(r['application']),r['workload'],r['allocator']))
    labels=[r['workload'].replace('python-json-','Python JSON ').replace('postgres-pgbench','PostgreSQL pgbench')+
            ' / '+r['allocator'].replace('sqlite-','').replace('cpython-','').replace('postgres-','').replace('ffmpeg-','') for r in active]
    y=np.arange(len(active))
    height=max(6,.28*len(active)+1.8)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,height),sharey=True,gridspec_kw={'width_ratios':[1,1]})
    for axis,key,color,title in [(ax,'I_over_A','#167D8D','Within-backing reuse / all object issues'),
                                  (bx,'I_over_R','#355C9A','Within-backing reuse / same-start reissues')]:
        vals=np.array([r[key+'_mean'] if r[key+'_mean'] is not None else 0 for r in active])
        lo=np.array([r[key+'_min'] if r[key+'_min'] is not None else 0 for r in active])
        hi=np.array([r[key+'_max'] if r[key+'_max'] is not None else 0 for r in active])
        axis.barh(y,vals,color=color,height=.7,zorder=3)
        axis.errorbar(vals,y,xerr=np.vstack([np.maximum(0,vals-lo),np.maximum(0,hi-vals)]),
                      fmt='none',ecolor='#17252B',capsize=2,lw=.8,zorder=4)
        axis.set_xlim(0,1.04); axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_title(title,loc='left',fontsize=10,pad=12)
        axis.grid(axis='x',color='#E1E6EA',zorder=0)
        for index,r in enumerate(active):
            if r[key+'_mean'] is None: axis.text(.03,index,'undefined: no reuse',va='center',fontsize=8)
    ax.set_yticks(y,labels); ax.invert_yaxis()
    fig.suptitle('Native object reuse inside live system allocations',x=.025,ha='left',fontsize=15,fontweight='bold')
    fig.text(.025,.02,'Three fresh observed processes per profile. Bars show means, whiskers show the observed range.\n'
             'Whole-process windows. Profiles and allocator levels stay separate. Exact-start reuse only.',fontsize=9,color='#46555C')
    fig.tight_layout(rect=[0,.065,1,.96])
    save(fig,'reuse-fractions')
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6.2),sharex=True,sharey=True)
    palette=['#167D8D','#355C9A','#B7792F','#835A97','#5B8457','#B34F54','#63747E']
    max_bin=max((b for runs in groups.values() for r in runs for b,n in enumerate(r['gap_inside']) if n),default=0)
    maximum=2**max(20,max_bin+1)
    for axis,app in zip(axes.flat,apps):
        index=0
        shown=set()
        for (a,workload,family),runs in sorted(groups.items()):
            if a!=app: continue
            hist=np.sum(np.array([r['gap_inside'] for r in runs],dtype=np.int64),axis=0)
            if not hist.sum(): continue
            last=max(np.flatnonzero(hist))
            upper=np.array([2**(b+1)-1 for b in range(last+1)],dtype=float)
            cdf=np.cumsum(hist[:last+1])/hist.sum()
            # Extend a CDF after its final bin. A distribution entirely at gap
            # one otherwise produces a single invisible point with step().
            upper=np.append(upper,maximum)
            cdf=np.append(cdf,1.)
            label=workload.replace(app+'-','').replace('python-','')+' / '+family.split('-')[-1]
            color=palette[index%len(palette)]
            style='-'
            if app=='ffmpeg':
                label=workload.replace('ffmpeg-','')+' / '+family.replace('ffmpeg-','').replace('-pool','')
            if app=='wireshark':
                label=family.replace('wmem-','') if family not in shown else None
                shown.add(family)
                color=palette[0 if family=='wmem-block' else 1]
                style='-' if family=='wmem-block' else '--'
            axis.step(upper,cdf,where='post',label=label,color=color,linestyle=style,lw=1.4)
            index+=1
        axis.set_title(app,loc='left',fontweight='bold')
        axis.set_xscale('log',base=2); axis.set_xlim(1,maximum); axis.set_ylim(0,1.02)
        exponents=list(range(0,max(20,max_bin+1)+1,6))
        ticks=[2**e for e in exponents]
        labels=[str(n) if n<1024 else (str(n//1024)+'K' if n<1048576 else str(n//1048576)+'M') for n in ticks]
        axis.set_xticks(ticks,labels)
        axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(color='#E1E6EA',lw=.6)
        if index: axis.legend(fontsize=5.8,loc='lower right',frameon=False)
    fig.suptitle('How soon retained storage is issued again',x=.025,ha='left',fontsize=15,fontweight='bold')
    fig.supxlabel('Successful object issues in the same allocator instance after retirement',y=.055)
    fig.supylabel('Cumulative share of within-backing reissues',x=.01)
    fig.text(.025,.013,'CDF at logarithmic bin upper bounds, three runs per curve. Cross-instance reuse is excluded. A gap of 1 means the next issue.\n'
             'Wireshark: one curve per input, color and line style denote allocator. Coverage and raw denominators are in summary.csv.',fontsize=8,color='#46555C')
    fig.tight_layout(rect=[.025,.10,1,.94])
    save(fig,'reuse-gaps')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',nargs='*',type=Path)
    parser.add_argument('--from-export',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args()
    if bool(args.roots)==bool(args.from_export):
        parser.error('provide raw campaign roots or --from-export')
    data=read_export(args.from_export) if args.from_export else collect(args.roots)
    rows,groups=export(data,args.output)
    if args.plots: figures(rows,groups,args.output)
    print(f'PASS {len(data)} processes, {len({r["workload"] for r in data})} profiles, {len({r["application"] for r in data})} applications')
