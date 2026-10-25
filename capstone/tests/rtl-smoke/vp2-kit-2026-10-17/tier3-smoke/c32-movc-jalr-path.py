#!/usr/bin/env python3
# c32-movc-jalr-path.py <llvm-objdump -d --no-show-raw-insn listing of ONE function>: the C-32 shape (silicon MOVC nulls an
# integer source) as `movc rd, rs` followed by `jalr rs` with rs unwritten in between.
import re,sys
# For each `movc rd, rs` with rd != rs: follow straight-line code AND unconditional `j` within the function, stopping at a
# write to rs; report a `jalr rs` (or any read of rs as a jump target). Path-limited, so a hit is a candidate, 0 is not absence.
L=[l for l in open(sys.argv[1]) if re.match(r'\s+[0-9a-f]+:\s',l)]
ins=[];idx={}
for l in L:
    m=re.match(r'\s+([0-9a-f]+):\s+(\S+)\s*(.*)',l); a=int(m.group(1),16); ins.append((a,m.group(2),[x.strip() for x in m.group(3).split(',')])); idx[a]=len(ins)-1
hits=0;movcs=0
for i,(a,op,args) in enumerate(ins):
    if op!='movc' or len(args)<2 or args[0]==args[1]: continue
    movcs+=1; rs=args[1]; j=i+1; steps=0
    while j<len(ins) and steps<200:
        b,op2,a2=ins[j]; steps+=1
        if op2=='jalr' and a2 and a2[0]==rs: print(f"HIT movc {args[0]},{rs} @{a:x} -> jalr {rs} @{b:x}"); hits+=1; break
        if a2 and a2[0]==rs and op2 not in ('stc','sd','sw','sh','sb','beq','bne','blt','bge','bltu','bgeu','beqz','bnez','jalr','j'): break
        if op2=='j':
            t=int(a2[0].split()[0],16)
            if t in idx: j=idx[t]; continue
            break
        if op2 in ('ret',) or (op2=='jalr' and a2 and a2[0] in ('ra','zero')): break
        j+=1
print(f"movc rd!=rs: {movcs}, movc-then-jalr-of-source: {hits}")
