#!/usr/bin/env python3
"""c32-cfg-scan.py DISASM... [--all]: the C-32 census with control flow, for `objdump -d` output with raw bytes.
It reuses c32-movc-scan.py's parse/defreg/usereg and INT_DEF/CAP_DEF sets (the sibling file). For every `movc rd, rs` (rd != rs) it collects
the REACHING definitions of rs over the function's CFG (branches and jumps followed; a def through another movc is
followed to that movc's own source, depth <= 4), and checks whether rs is READ on some path after the movc before
being redefined. A candidate has an integer-producing reaching def other than `li rs, 0`, and a live read.
The linear scanner misses a loop-carried integer: cpython 20bcfdd1's demonstrated site 0x331c0c, `movc s2, a3; jalr a3`
with a3 = `mv a3, a1` on the loop's entry path, is its must-fire. Reads that happen only through a call's argument
registers are counted apart (call-arg), not as candidates. --all prints those too. Exit 2 if any file could not be
parsed (no data is not zero).

Controls (2026-10-20, the corpus SDK's frozen compiler):
  * cpython 20bcfdd1 (built with +movc-keeps-integer-source): 12 candidates, 0x331c0c among them;
  * llvm/test/CodeGen/Capstone/c32-movc-untagged-live.ll at -O2: 2 with +movc-keeps-integer-source
    (bridged_phi_residue, bridged_callarg_plus_phi), 0 without; real_cap_copy never.
LIMITS: path-insensitive. In a large function whose callee-saved register takes many definitions, an integer def can
"reach" along paths that never run together, so a count is an upper bound, not a defect list (postgres 64010720: 285;
its rebuild without the feature: 163, dominated by ATRewriteCatalogs). The executed gate is the window counter's
movc-null sites under CAPSTONE_MOVC_NULL_SCALAR=1 (capstone-qemu cap-window-counter)."""
import importlib.util, pathlib, re, sys, collections
spec = importlib.util.spec_from_file_location('c32', pathlib.Path(__file__).with_name('c32-movc-scan.py'))
c32 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c32)
COND = {'beq','bne','blt','bge','bltu','bgeu','beqz','bnez','blez','bgez','bltz','bgtz','ble','bgt','bleu','bgtu'}
CALLER = {'ra','t0','t1','t2','t3','t4','t5','t6','a0','a1','a2','a3','a4','a5','a6','a7'}
ARGS = {'a0','a1','a2','a3','a4','a5','a6','a7'}
TGT = re.compile(r'0x([0-9a-f]+)\s*<')   # the branch/jump target, never the <sym+0x..> offset

def is_call(m, o):
    if m == 'jal': return True
    if m == 'jalr':
        r = c32.regs(o); return not r or r[0] in ('ra',) or len(r) == 1
    return False

def analyse(insns):
    by_func = collections.defaultdict(list)
    for t in insns: by_func[t[3]].append(t)
    out = []; stats = collections.Counter()
    for func, F in by_func.items():
        idx = {a: i for i, (a, m, o, f) in enumerate(F)}
        succ = [[] for _ in F]
        for i, (a, m, o, f) in enumerate(F):
            t = TGT.findall(o or ''); tgt = idx.get(int(t[-1], 16)) if t else None
            if m in ('j',): succ[i] = [tgt] if tgt is not None else []
            elif m in ('ret', 'jr', 'cjr', 'mret', 'sret', 'unimp', 'ebreak') or (m == 'jalr' and c32.regs(o)[:1] == ['zero']): succ[i] = []
            elif m in COND: succ[i] = [x for x in (i + 1 if i + 1 < len(F) else None, tgt) if x is not None]
            else: succ[i] = [i + 1] if i + 1 < len(F) else []
        pred = [[] for _ in F]
        for i, s in enumerate(succ):
            for j in s: pred[j].append(i)
        def reaching(i, r, depth):
            """classes of the defs of r reaching instruction i"""
            seen, todo, cls = set(), list(pred[i]), set()
            if not pred[i]: cls.add('entry')
            while todo and len(seen) < 4000:
                j = todo.pop()
                if j in seen: continue
                seen.add(j); a, m, o, f = F[j]
                if c32.defreg(m, o) == r and not (m in ('jalr', 'jr', 'cjr') and len(c32.regs(o)) == 1):
                    if m == 'movc':
                        src = c32.regs(o)[1:2]
                        cls |= reaching(j, src[0], depth - 1) if src and depth > 0 else {'cap'}
                    elif m in c32.INT_DEF:
                        cls.add('zero' if m == 'li' and re.search(r',\s*0x0$|,\s*0$', o) else 'int')
                    elif m in c32.CAP_DEF: cls.add('cap')
                    else: cls.add('other:' + m)
                    continue
                if is_call(m, o) and r in CALLER: cls.add('call'); continue
                if not pred[j]: cls.add('entry')
                todo.extend(pred[j])
            return cls
        def live_after(i, r):
            seen, todo = set(), list(succ[i])
            while todo and len(seen) < 4000:
                j = todo.pop()
                if j in seen: continue
                seen.add(j); a, m, o, f = F[j]
                reads = c32.regs(o) if m in ('jalr', 'jr', 'cjr') and len(c32.regs(o)) == 1 else c32.usereg(m, o)
                if r in reads: return ('read', a, m, o)   # a single-operand jalr/jr READS its register
                if is_call(m, o) and r in ARGS: return ('call-arg', a, m, o)
                if (c32.defreg(m, o) == r and not (m in ('jalr', 'jr', 'cjr') and len(c32.regs(o)) == 1)) or (is_call(m, o) and r in CALLER): continue
                todo.extend(succ[j])
            return None
        for i, (a, m, o, f) in enumerate(F):
            if m != 'movc': continue
            r = c32.regs(o)
            if len(r) < 2 or r[0] == r[1]: continue
            stats['movc'] += 1; rs = r[1]
            cls = reaching(i, rs, 4)
            if 'int' not in cls: continue
            lv = live_after(i, rs)
            if not lv: stats['int-def dead after'] += 1; continue
            stats['call-arg' if lv[0] == 'call-arg' else 'CANDIDATE'] += 1
            out.append(('call-arg' if lv[0] == 'call-arg' else 'CANDIDATE', a, func, o, sorted(cls), lv[1:]))
    return out, stats

unread = []
for path in [a for a in sys.argv[1:] if not a.startswith('--')]:
    insns = c32.parse(path)
    if not insns: print(f'{path}: COULD NOT PARSE (needs objdump -d with raw bytes)'); unread.append(path); continue
    out, stats = analyse(insns)
    print(f'{path}: {len(insns)} insns, ' + ', '.join(f'{k} {v}' for k, v in sorted(stats.items())))
    for kind, a, func, o, cls, lv in out:
        if '--all' in sys.argv or kind == 'CANDIDATE':
            print(f'  {kind} {a:#x} {func}: movc {o} defs={cls} read {lv[0]:#x} {lv[1]} {lv[2]}')
sys.exit(2 if unread else 0)
