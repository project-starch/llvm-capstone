#!/usr/bin/env python3
"""Pair A check for one bench dir name: G (out-nat-g-...) vs P1 (out-nat-p1-...). exit 0 PASS, 1 FAIL, 4 NO DATA."""
import sys, re, os, filecmp, hashlib
B=os.environ.get('SPEC_BOARD_DIR')   # the runtime lane's spec/board directory (<mnt-hdd>/capstone/spec/board)
if not B or not os.path.isdir(B) or len(sys.argv) != 2: print('NO DATA: set SPEC_BOARD_DIR; usage: pa-check.py <bench>'); sys.exit(4)
bench=sys.argv[1]
D={a: f'{B}/out-nat-{a}-hpmnest-frcdkms-b1/{bench}' for a in ('g','p1')}
D={a: os.environ.get('PA_'+a.upper(), D[a]) for a in D}   # override for negative controls
BIT={'g':'tagoverlap6c_9fd9d2530.bit','p1':'pagedtbl_d83d3a633.bit'}
fails=[]; info={}
for a,d in D.items():
    if not os.path.isdir(d+'/read/files'): print('NO DATA:', d); sys.exit(4)
    rl=open(d+'/read.log').read().splitlines()[0]; sm=open(d+'/smoke-verdict.txt').read()
    t=open(d+'/board.txt',errors='replace').read(); dl=open(d+'/driver.log',errors='replace').read()
    bit=re.findall(r"'nv_bitstream_name': '([^']+)'",dl)
    traps=re.findall(r'^.*TRAP LOG.*$',t,re.M)
    L=re.findall(r'^@@NATIVE-CNT 0 (begin|end) (.*)$',t,re.M)
    ok_spec=bool(re.search(r'rc!=0: 0,.*specdiff (\d+)/\1 PASS',rl))
    ok_sm='PASS: all 9 steps' in sm
    ok_bit=bool(bit) and all(x.endswith(BIT[a]) for x in bit)
    if [x[0] for x in L]!=['begin','end']: print('NO DATA: NATIVE-CNT lines', a, [x[0] for x in L]); sys.exit(4)
    kv=[dict(s.split('=',1) for s in x[1].split()) for x in L]
    info[a]=dict(traps=traps, cnt={k:int(kv[1][k])-int(kv[0][k]) for k in kv[0] if kv[0][k].isdigit()},
                 kern=re.findall(r'^@@NS-KERNEL.*$',t,re.M))
    print(f'{a:3} specdiff {"PASS" if ok_spec else "FAIL"} | smoke {"9/9" if ok_sm else "FAIL"} | bitstream {"ok" if ok_bit else "WRONG "+str(bit)} | traps {len(traps)} | {info[a]["kern"]}')
    print('    ', rl)
    if not (ok_spec and ok_sm and ok_bit): fails.append(a+' functional')
dc=filecmp.dircmp(D['g']+'/read/files', D['p1']+'/read/files')
def cmp_tree(x,y):
    bad=[]
    for r,_,fs in os.walk(x):
        for f in fs:
            p=os.path.join(r,f); q=os.path.join(y,os.path.relpath(p,x))
            if not os.path.exists(q) or open(p,'rb').read()!=open(q,'rb').read(): bad.append(os.path.relpath(p,x))
    for r,_,fs in os.walk(y):
        for f in fs:
            if not os.path.exists(os.path.join(x,os.path.relpath(os.path.join(r,f),y))): bad.append('only P1: '+f)
    return bad
bad=cmp_tree(D['g']+'/read/files', D['p1']+'/read/files')
nf=sum(len(fs) for _,_,fs in os.walk(D['g']+'/read/files'))
print(f'outputs: {nf} files, {"identical" if not bad else "DIFFER "+str(bad)}')
if bad or nf==0: fails.append('outputs')
if [re.sub(r'^\S*\s*','',x) for x in info['g']['traps']]!=[re.sub(r'^\S*\s*','',x) for x in info['p1']['traps']]: fails.append('traps differ'); print('traps', info)
else: print('traps identical:', info['p1']['traps'])
for f in ('manifest.txt','native-kernel.config','native-tree.check'):
    same=open(D['g']+'/'+('image/' if f=='manifest.txt' else '')+f,'rb').read()==open(D['p1']+'/'+('image/' if f=='manifest.txt' else '')+f,'rb').read()
    print(f'kit {f}: {"same" if same else "DIFFER"}')
g,p=info['g']['cnt'],info['p1']['cnt']
for k,n in [('run_cycles','run_cycles'),('hpm8','ev23'),('hpm4','L1D'),('hpm3','L1I'),('hpm5','DTLB'),('hpm6','mispredict'),('hpm7','issue stall'),('between_cycles','between'),('runs','runs')]:
    print(f'  {n:12} G {g[k]:>16,} P1 {p[k]:>16,} d {p[k]-g[k]:>+14,} ({100*(p[k]-g[k])/g[k]:+.3f} %)')
dr=100*(p['run_cycles']-g['run_cycles'])/g['run_cycles']
print(f'run_cycles {dr:+.3f} % vs band 0.2 %:', 'PASS' if abs(dr)<=0.2 else 'FAIL')
if abs(dr)>0.2: fails.append('run_cycles')
print('[key 5 NOT MET] PAIR A', bench, 'PASS' if not fails else 'FAIL '+str(fails)); sys.exit(1 if fails else 0)
