import json, subprocess, itertools, sys
sys.path.insert(0,'/tmp/capstone')
from extract_layout import layout
S="/tmp/claude-1005/-home-alexey-dev-llvm-capstone/fea3eec8-e31e-459d-82aa-366f35932b14/scratchpad/wit"
import os
# build -> (victim_slot or None, delta, victim_known)
OUT = {
 "c0":(None,0,True), "c4":(0x1c,+333,True), "c8":(0x1c,-9,True),
 "rs0":(None,0,True), "rs4":(0x1c,-72,True), "rs8":(0x1c,+9,True),
 "t12":(0x1c,+330,True), "t16":(None,0,True), "t0b":(0x1c,+333,True),
 "bs16":(None,0,True), "nr16":(None,0,True), "dp0":(0x1c,-9,True),
 "ka0":(0x18,-558,True), "kb12":(0x28,-9,True),
 "gp0":(0x1c,-9,False), "gp16":(None,0,False), "gp32":(None,0,False),
 "sep12":(0x1c,-9,False), "sep20":(0x1c,+330,False),
}
rows=[]
for n,(vic,delta,known) in OUT.items():
    p=f"{S}/{n}.dom"
    if not os.path.exists(p): p=f"{S}/overlay-backup/{n}.dom"
    if not os.path.exists(p): continue
    L=layout(p)
    if not L: continue
    rows.append(dict(name=n, frame=L["frame"], store=L["store"], rmw=L["rmw"],
                     victim=vic, delta=delta, known=known, damaged=vic is not None))
print(f"dataset: {len(rows)} builds ({sum(r['damaged'] for r in rows)} damaged)\n")

def rowof(a): return a & ~0xF
# candidate ANCHOR rules: victim is the rmw slot at a given offset expression
anchors = {
 "store+0x1c": lambda r: (r["store"] or 0)+0x1c,
 "store+0x18": lambda r: (r["store"] or 0)+0x18,
 "sp+0x1c":    lambda r: 0x1c,
 "s0-0x34":    lambda r: r["frame"]-0x34,
 "s0-0x38":    lambda r: r["frame"]-0x38,
 "lowest_rmw": lambda r: r["rmw"][0] if r["rmw"] else None,
 "highest_rmw":lambda r: r["rmw"][-1] if r["rmw"] else None,
 "2nd_rmw":    lambda r: r["rmw"][1] if len(r["rmw"])>1 else None,
}
print("ANCHOR RULES  (predict victim == anchor; correct iff anchor unoccupied)")
for nm,fn in anchors.items():
    ok=[]; bad=[]
    for r in rows:
        a=fn(r)
        pred_dam = a in r["rmw"]
        if r["damaged"]!=pred_dam: bad.append(r["name"]+":dam?"); continue
        if r["damaged"] and r["victim"]!=a: bad.append(r["name"]+":slot"); continue
        ok.append(r["name"])
    print(f"  {nm:12} {len(ok):>2}/{len(rows)}   fails: {' '.join(bad[:7])}")

print("\nSTRUCTURAL PREDICATES on the victim slot (damaged builds with KNOWN victim only)")
kn=[r for r in rows if r["known"]]
for r in kn:
    if not r["damaged"]: continue
    v=r["victim"]; rw=r["rmw"]
    inrow=[x for x in rw if rowof(x)==rowof(v)]
    idx=rw.index(v)
    print(f"  {r['name']:5} frame=0x{r['frame']:02x} store=0x{r['store']:02x} rmw={[hex(x) for x in rw]}"
          f" victim=0x{v:02x} idx={idx}/{len(rw)-1} inrow={len(inrow)} v-store=0x{v-(r['store'] or 0):02x}"
          f" dword_off={(v&7)} row_off={(v&0xF)} delta={r['delta']:+d}")
print("\n  undamaged builds:")
for r in kn:
    if r["damaged"]: continue
    print(f"  {r['name']:5} frame=0x{r['frame']:02x} rmw={[hex(x) for x in r['rmw']]}"
          f" rows={sorted(set(hex(rowof(x)) for x in r['rmw']))}")
