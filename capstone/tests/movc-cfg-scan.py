#!/usr/bin/env python3
"""C-32 shape over a real control-flow graph. For each `movc rd, rs` (rd != rs) in an image:
  source: the definitions of rs REACHING the movc (backward search over predecessors within the function;
          function entry = unknown, a call's return = unknown for caller-saved regs) -- classified
          'int' if every reaching definition is an integer-producing instruction other than `li rs, 0`,
          'mixed' if some are, 'cap/unknown' otherwise;
  sink:   whether some path FORWARD from the movc reaches a READ of rs before any redefinition of rs.
A hit needs both. Necessary, not sufficient: tagged-ness is dynamic. Prints per-image counts and the 'int' hits."""
import re,subprocess,sys,collections
OD='llvm/cmake-build-debug/bin/llvm-objdump'
INT_OPS={'addi','add','addw','addiw','sub','subw','li','lui','auipc','lw','ld','lbu','lb','lh','lhu','lwu','sext.w','slli','srli','srai','sll','srl','sra','slliw','srliw','sraiw','sllw','srlw','sraw','and','andi','or','ori','xor','xori','mul','mulw','mulh','mulhu','div','divu','divw','divuw','rem','remu','remw','remuw','sltu','slt','slti','sltiu','seqz','snez','sgtz','sltz','neg','negw','not','mv','lcc','zext.b','sext.b','sext.h','zext.h','csrr','csrrs','csrrw','csrrsi','csrrci','min','max','minu','maxu','sh1add','sh2add','sh3add','andn','orn','xnor','rev8','clz','ctz','cpop','bexti','bset','bclr','binv','bext','sh1add.uw','sh2add.uw','sh3add.uw','add.uw','slli.uw','fmv.x.d','fmv.x.w','fcvt.w.d','fcvt.l.d','fcvt.wu.d','fcvt.lu.d','feq.d','flt.d','fle.d','fclass.d'}
STORES={'sd','sw','sh','sb','stc','fsd','fsw'}
BR=re.compile(r'^(beq|bne|blt|bge|bltu|bgeu|beqz|bnez|blez|bgez|bltz|bgtz|bgt|ble|bgtu|bleu)$')
END={'ret','jr','tail','ecall','ebreak','mret','sret','wfi','unimp'}
CALLER_SAVED=set(['ra','t0','t1','t2','t3','t4','t5','t6']+['a%d'%i for i in range(8)])
def regs_in(ops): return re.findall(r'\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b',ops)
def parse(path):
    out=subprocess.run([OD,'-d','--triple=capstone64-unknown-elf',path],capture_output=True,text=True).stdout
    func=None; funcs=collections.OrderedDict()
    for ln in out.splitlines():
        m=re.match(r'^([0-9a-f]+) <([^>]+)>:',ln)
        if m:
            if not m.group(2).startswith('.L'): func=m.group(2); funcs.setdefault(func,[])
            continue
        m=re.match(r'^\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} ){2,4}\s*(\S+)\s*(.*)$',ln)
        if m and func is not None: funcs[func].append((int(m.group(1),16),m.group(2),m.group(3).strip()))
    return funcs
def analyse(insns):
    idx={a:i for i,(a,_,_) in enumerate(insns)}
    def target(ops):
        m=re.search(r'\b0x([0-9a-f]+)\b',ops); return idx.get(int(m.group(1),16)) if m else None
    succ=[[] for _ in insns]
    for i,(a,op,ops) in enumerate(insns):
        if op in END: continue
        if op=='j':
            t=target(ops); succ[i]=[t] if t is not None else []
        elif BR.match(op):
            t=target(ops); succ[i]=[i+1]+([t] if t is not None else [])
        else: succ[i]=[i+1]
        succ[i]=[s for s in succ[i] if s is not None and s<len(insns)]
    pred=[[] for _ in insns]
    for i,ss in enumerate(succ):
        for s in ss: pred[s].append(i)
    return succ,pred
JALR={'jalr','cjalr'}
def defs_reads(op,ops):
    r=regs_in(ops)
    if not r: return None,[]
    if op in STORES or BR.match(op): return None,r
    if op=='<unknown>': return None,r
    # `jalr rs` is the pseudo for `jalr ra, rs, 0`: it DEFINES ra and READS rs. Returning r[0] as the
    # def made an indirect call look like a definition of its own TARGET register, which terminated the
    # backward walk there and injected a spurious 'cap' reaching definition -- so the reaching-def union
    # was a lower bound and a site could read MIXED on the strength of a call it merely passed through.
    # It bit only when the movc's source is CALLEE-saved, because the caller-saved case is already
    # short-circuited at the 'call' branch in scan(). Found by the compiler lane, 2026-09-17.
    # Only the one-operand form is wrong; `jalr rd, rs, imm` already has r[0] == rd.
    if op in JALR and len(r)==1: return 'ra',r
    return r[0],r[1:]
def scan(path,label):
    funcs=parse(path); n_movc=0; strong=[]; mixed=0; sinks_total=0; opaque=0
    for f,insns in funcs.items():
        if not insns: continue
        succ,pred=analyse(insns)
        for i,(a,op,ops) in enumerate(insns):
            if op!='movc': continue
            r=regs_in(ops)
            if len(r)<2 or r[0]==r[1]: continue
            n_movc+=1; rd,rs=r[0],r[1]
            # backward: reaching definitions of rs
            kinds=set(); seen=set(); stack=list(pred[i])
            while stack:
                j=stack.pop()
                if j in seen: continue
                seen.add(j); a2,op2,ops2=insns[j]; d,_=defs_reads(op2,ops2)
                if op2 in ('jal','jalr','call','cjalr','cjal') and rs in CALLER_SAVED: kinds.add('call'); continue
                if d==rs:
                    if op2 in INT_OPS and not (op2=='li' and re.search(r',\s*(0x0|0)$',ops2)): kinds.add('int')
                    elif op2=='li': kinds.add('zero')
                    else: kinds.add('cap')
                    continue
                if not pred[j]: kinds.add('entry')
                stack.extend(pred[j])
            if not kinds: kinds.add('entry')
            src='int' if kinds<= {'int'} else ('mixed' if 'int' in kinds else 'other')
            # forward: a read of rs before a redefinition
            hit=None; seen=set(); stack=list(succ[i])
            while stack and hit is None:
                j=stack.pop()
                if j in seen: continue
                seen.add(j); a2,op2,ops2=insns[j]; d,reads=defs_reads(op2,ops2)
                if rs in reads: hit=(a2,op2,ops2); break
                if d==rs: continue
                if op2 in ('jal','jalr','call','cjalr','cjal') and rs in CALLER_SAVED: continue
                stack.extend(succ[j])
            if hit: sinks_total+=1
            if hit and src=='int': strong.append((f,a,rd,rs,hit,'INT-ONLY'))
            elif hit and src=='mixed': mixed+=1; strong.append((f,a,rd,rs,hit,'MIXED '+','.join(sorted(kinds))))
            elif hit and (kinds & {'call','entry'}) and 'cap' not in kinds: opaque+=1   # a call return or a function argument: taggedness not static
    print(f"== {label}: {n_movc} movc(rd!=rs); {sinks_total} with the source read again on some path; of those {len(strong)-mixed} have ONLY integer reaching definitions, {mixed} mixed (integer on some path), {opaque} whose source is only a call return or a function argument (taggedness not static; not classified)")
    for f,a,rd,rs,h,k in strong: print(f"   [{k}] {f}+{a:#x}: movc {rd}, {rs}  -> read @{h[0]:#x} {h[1]} {h[2]}")
args=sys.argv[1:]
if not args or len(args)%2:
    sys.exit("usage: movc-cfg-scan.py <image> <label> [<image> <label> ...]  -- pairs; a lone path would otherwise print nothing and exit 0, which reads like a clean image")
for p,l in zip(args[0::2],args[1::2]): scan(p,l)
