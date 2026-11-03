import re,glob,os,json,sys
os.chdir(sys.argv[1] if len(sys.argv) > 1 else '.')
ARM={'capstone-spatial':'MapLife','capstone-lifetimes':'Temporal'}
rows=[]
for d in sorted(glob.glob('outlc/*/')):
    f=d+'guest/serial.log'
    if not os.path.exists(f) or 'died' in d: continue
    wins={}
    for l in open(f,errors='replace'):
        m=re.match(r'capstone: lc-model window=(\d+) (.*)',l.strip())
        if m and int(m.group(1)) not in wins:
            wins[int(m.group(1))]={k:list(map(int,v.split(','))) for k,v in re.findall(r'(\w\.\w+)=([\d,]+)',m.group(2))}
            wins[int(m.group(1))]['disagree']=[int(re.search(r'\bdisagree=(\d+)',m.group(2)).group(1))]+[0]*6
    r=json.load(open(d+'result.json'))
    arms=list(r['arms'].items()); w=0; name=os.path.basename(d[:-1]).split('-test')[0]
    for lab,a in arms:
        n=len(a['commands']); ws=list(range(w,w+n)); w+=n
        if any(i not in wins for i in ws): rows.append((name,ARM.get(lab,lab),'MISSING windows')); continue
        tot={}
        for i in ws:
            for k,v in wins[i].items(): tot[k]=[x+y for x,y in zip(tot.get(k,[0]*7),v)]
        def rate(inst):
            n=sum(sum(tot[f'{inst}.{c}'][:6]) for c in ('ldc','stc'))
            miss=sum(sum(tot[f'{inst}.{c}'][2:6]) for c in ('ldc','stc'))
            conf=sum(tot[f'{inst}.{c}'][3] for c in ('ldc','stc'))
            dead=sum(tot[f'{inst}.{c}'][1] for c in ('ldc','stc'))
            return n,miss,conf,dead
        n,miss,conf,dead=rate('R'); nE,missE,_,_=rate('E')
        rows.append((name,ARM.get(lab,lab),f"LDC+STC {n/1e6:9.1f}M  miss R {miss/n*100:6.3f}% (conflict {conf/n*100:.3f})  E {missE/nE*100:6.3f}%  misses {miss/1e6:.2f}M  hit_dead {dead}  disagree {tot['disagree'][0]}"))
    if w!=len(wins): rows.append((name,'?',f'window count {len(wins)} != commands {w}'))
if not rows: sys.exit('NO DATA: no outlc/*/guest/serial.log under '+os.getcwd())
for r in rows: print(f"{r[0]:16s} {r[1]:9s} {r[2]}")
