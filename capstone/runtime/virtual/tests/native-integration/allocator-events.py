#!/usr/bin/env python3
"""Generate replay events from validated GCC allocator lifecycle traces.
The Capstone relocation is runtime image base minus ELF text origin.
"""
import argparse,pathlib,json,subprocess,bisect,collections
p=argparse.ArgumentParser(description=__doc__)
for name in ['plain-trace','capstone-trace','plain-exe','capstone-exe','out']:
 p.add_argument('--'+name,type=pathlib.Path,required=True)
p.add_argument('--capstone-relocation',type=lambda s:int(s,0),required=True)
opts=p.parse_args()
opts.out.mkdir(parents=True,exist_ok=False)
w=opts.out;lines=[];maxids=0;stats={}
for arm,exe,base in [('plain',opts.plain_exe,0),('native',opts.capstone_exe,opts.capstone_relocation)]:
 syms=[]
 for line in subprocess.check_output(['nm','-S',exe],text=True).splitlines():
  a=line.split()
  if len(a)==4 and a[2].lower() in ['t','w']:syms.append((int(a[0],16),int(a[1],16),a[3]))
 syms.sort();starts=[a for a,b,n in syms]
 def caller(r):
  pc=r['ra']-base;i=bisect.bisect_right(starts,pc)-1
  assert i>=0 and pc<syms[i][0]+syms[i][1]
  return syms[i][2]
 d=json.loads((opts.plain_trace if arm=='plain' else opts.capstone_trace).read_text());events=d['events'];label='cap' if arm=='native' else 'plain';maxids=max(maxids,d['ids']);hist=collections.Counter()
 lines += [f'#define {label.upper()}_COUNT {len(events)}',f'#define {label.upper()}_LIVE {d["live_at_end"]}',f'static const struct event {label}_events[]={{']
 for r in events:
  n=r['size'];cohort=0
  if r['phase']==1:
   if r['op']=='m' and n==(6400 if arm=='native' else 3200) and caller(r)=='xrealloc':cohort=1
   if r['op']=='c' and n==(4096 if arm=='native' else 2048) and caller(r)=='xcalloc':cohort=2
  hist[cohort]+=1
  lines.append("{'%s',%d,%d,%d,%d,%d,%d},"%(r['op'],r['phase'],r['id'],r.get('old_id',0),cohort,n,r.get('alignment',0)))
 lines+=['};'];stats[arm]=dict(events=len(events),ids=d['ids'],live=d['live_at_end'],cohort_events=dict(hist))
 assert hist[1]==195 and hist[2]==65,stats
lines.append(f'#define MAX_IDS {maxids}')
(w/'events.h').write_text('\n'.join(lines)+'\n');(w/'generation.json').write_text(json.dumps(stats,indent=2)+'\n');print(stats)
