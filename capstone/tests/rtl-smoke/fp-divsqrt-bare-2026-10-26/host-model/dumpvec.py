import re
def parse(log, w, ldf, divop, sqop):
    L=open(log,errors="replace").read().splitlines(); it=[]; cur={}
    for i,l in enumerate(L):
        if not l.startswith("3 "): continue
        m=re.match(r"3 0x[0-9a-f]+ \(0x[0-9a-f]+\) (?:f|x) ?\d+ 0x([0-9a-f]+)",l)
        if not m: continue
        ins=L[i-1]; v=int(m.group(1),16)&((1<<w)-1)
        if f"{ldf}     fa0" in ins: cur={"a":v}
        elif f"{ldf}     fa1" in ins and "a" in cur: cur["b"]=v
        elif divop in ins and "a" in cur: cur["q"]=v
        elif sqop in ins and "q" in cur: cur["s"]=v; it.append(cur); cur={}
    return it
# usage: dumpvec.py <fp-ieee-divsqrt retire trace .log> <fp-ieee-divsqrt-s retire trace .log>  (writes vec.h)
import sys
d=parse(sys.argv[1],64,"fld","fdiv.d","fsqrt.d")
s=parse(sys.argv[2],32,"flw","fdiv.s","fsqrt.s")
assert len(d)==512 and len(s)==512, (len(d),len(s))
with open("vec.h","w") as f:
    f.write("static const unsigned long long VD[512][4]={\n"+",\n".join(f"{{{c['a']:#x}ULL,{c['b']:#x}ULL,{c['q']:#x}ULL,{c['s']:#x}ULL}}" for c in d)+"};\n")
    f.write("static const unsigned VS[512][4]={\n"+",\n".join(f"{{{c['a']:#x}u,{c['b']:#x}u,{c['q']:#x}u,{c['s']:#x}u}}" for c in s)+"};\n")
print("ok",len(d),len(s))
