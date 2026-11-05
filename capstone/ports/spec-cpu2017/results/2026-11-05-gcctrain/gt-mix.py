import sys,json
import os
P=os.path.join(os.environ['CAPSTONE_WORK'],'spec','prof')
exec(open(os.path.join(P,'capmix-summary.py')).read().split("BENCH={")[0])
out={}
for cfg,name,arm in (('Temporal','502.gcc_r-train01-capstone-lifetimes-devmk-m','capstone-lifetimes'),('Plain','502.gcc_r-train01-plain-rv64-m','plain-rv64')):
    W=windows(name); C=cmds(name); acc=dict(prog=[0]*len(CL),ker=0,fw=0,oth=0,ref=0)
    for w,(a,c) in zip(W,C):
        if a!=arm or c['tag']!='run': continue
        prog,ker,fw,oth=split(w, cfg=='Plain')
        if cfg!='Plain': acc['ref']+=c['instret']
        acc['prog']=[x+y for x,y in zip(acc['prog'],prog)]; acc['ker']+=ker; acc['fw']+=fw; acc['oth']+=oth
    out[cfg]=acc
T,Pl=out['Temporal'],out['Plain']; pt,pp=sum(T['prog']),sum(Pl['prog'])
print(f"gcc train01: program instr Temporal {pt/1e9:.3f}e9 (instret {T['ref']/1e9:.3f}e9)  Plain {pp/1e9:.3f}e9  ratio {pt/pp:.3f}")
print(f"   kernel Temporal {T['ker']/1e9:.3f}e9 Plain {Pl['ker']/1e9:.3f}e9 | firmware {T['fw']/1e6:.1f}M/{Pl['fw']/1e6:.1f}M | other {T['oth']/1e6:.1f}M/{Pl['oth']/1e6:.1f}M")
print("   class      Plain/1k  Temporal/1k(of Plain)")
for i,n in enumerate(CL):
    if Pl['prog'][i] or T['prog'][i]: print(f"   {n:10s} {Pl['prog'][i]/pp*1e3:8.1f} {T['prog'][i]/pp*1e3:10.1f}")
json.dump(out,open(f'{P}/gt-mix.json','w'))
