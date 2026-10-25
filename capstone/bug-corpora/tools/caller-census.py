#!/usr/bin/env python3
"""Static caller census: every call to every copy of a function, by RAW word decode, enclosed in its FUNC symbol.
Usage: caller-census.py <llvm-bin> <image> <function-name>. Decodes jal (direct target) and jalr whose register was
built by auipc (+ addi) in the same function, tracked per register and reset at each FUNC start and on any other write
we cannot model. Prints the parse count, every call to the target's copies with its enclosing function, every jalr whose
target it could not resolve, and every materialisation of a target address that does not feed a call (address taken).
Also scans every loaded data word for a target address. Exits 2 on no parse; 1 if the target's address is taken in
code or held in data (the census then cannot be exhaustive); 0 otherwise, unresolved jalr being function pointers that
cannot hold an address never taken."""
import re, struct, subprocess, sys, collections
LB, IMG, NAME = sys.argv[1:4]
syms = []
for l in subprocess.run([f'{LB}/llvm-readelf', '-sW', IMG], capture_output=True, text=True).stdout.splitlines():
    p = l.split()
    if len(p) >= 8 and p[3] == 'FUNC' and p[1] != '0':
        syms.append((int(p[1], 16), int(p[2]), p[7]))
syms.sort()
targets = {a for a, s, n in syms if n == NAME}
def enclosing(pc):
    best = None
    for a, s, n in syms:
        if a <= pc < a + max(s, 1): best = (n, a)
    return best
def sx(v, b): return v - (1 << b) if v & (1 << (b - 1)) else v
starts = {a for a, s, n in syms}
out = subprocess.run([f'{LB}/llvm-objdump', '-d', '--no-print-imm-hex', IMG], capture_output=True, text=True).stdout
n = n32 = 0; reg = {}; calls = []; unresolved = []; taken = []
pending = {}   # reg -> (value, pc) built but not yet consumed by a jalr
for line in out.splitlines():
    m = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){2,4})', line)
    if not m: continue
    pc = int(m.group(1), 16); b = bytes.fromhex(m.group(2).replace(' ', '')); n += 1
    if pc in starts:
        for r, (v, at) in pending.items():
            if v in targets: taken.append((at, enclosing(at)))
        reg.clear(); pending.clear()
    if len(b) != 4:
        continue
    n32 += 1
    w = int.from_bytes(b, 'little'); op = w & 0x7f; rd = (w >> 7) & 31; f3 = (w >> 12) & 7; rs1 = (w >> 15) & 31
    if op == 0x17:                                   # auipc
        reg[rd] = pc + sx(w & 0xfffff000, 32); pending[rd] = (reg[rd], pc); continue
    if op == 0x13 and f3 == 0 and rs1 in reg:        # addi rd, rs1, imm
        reg[rd] = reg[rs1] + sx(w >> 20, 12); pending.pop(rs1, None) if rs1 != rd else None; pending[rd] = (reg[rd], pc); continue
    if op == 0x6f:                                   # jal
        t = pc + sx(((w >> 31) << 20) | (((w >> 12) & 0xff) << 12) | (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3ff) << 1), 21)
        if t in targets: calls.append((pc, 'jal', t, enclosing(pc)))
        reg.pop(rd, None); pending.pop(rd, None); continue
    if op == 0x67:                                   # jalr
        if rs1 in reg:
            t = reg[rs1] + sx(w >> 20, 12)
            if t in targets: calls.append((pc, 'jalr', t, enclosing(pc)))
            pending.pop(rs1, None)
        elif not (rd == 0 and rs1 == 1):             # plain `ret` is not a call
            unresolved.append((pc, enclosing(pc)))
        reg.pop(rd, None); continue
    if rd in reg and op in (0x03, 0x13, 0x33, 0x37, 0x1b, 0x3b, 0x5b):  # any other write to a tracked register
        if rd in pending and pending[rd][0] in targets: taken.append((pending[rd][1], enclosing(pending[rd][1])))
        reg.pop(rd, None); pending.pop(rd, None)
for r, (v, at) in pending.items():
    if v in targets: taken.append((at, enclosing(at)))
print(f"== {IMG.split('/')[-1]}: {n} instructions parsed ({n32} 32-bit); {NAME} copies at {[hex(t) for t in sorted(targets)]}")
by = collections.Counter((e[0] if e else '?', hex(t)) for pc, k, t, e in calls)
for (fn, t), c in sorted(by.items()): print(f"   caller {fn:40} -> copy {t}  x{c}")
# A function pointer in data could reach the target through an unresolved jalr: scan every loaded, non-executable,
# non-NOBITS section for an 8-byte word equal to a target copy's address.
d = open(IMG, 'rb').read()
shoff = struct.unpack_from('<Q', d, 0x28)[0]; shentsize, shnum, shstrndx = struct.unpack_from('<HHH', d, 0x3a)
words = in_data = 0
for i in range(shnum):
    nm, ty, fl, ad, off, sz = struct.unpack_from('<IIQQQQ', d, shoff + i * shentsize)
    if fl & 2 and not fl & 4 and ty != 8:
        for o in range(0, sz - 7, 8):
            words += 1; in_data += struct.unpack_from('<Q', d, off + o)[0] in targets
print(f"   calls {len(calls)}; unresolved jalr {len(unresolved)}; address-taken {len(taken)}; "
      f"data words holding a target address {in_data} of {words}")
for pc, e in unresolved[:8]: print(f"     unresolved jalr at {pc:#x} in {e[0] if e else '?'}")
for pc, e in taken[:8]: print(f"     address taken at {pc:#x} in {e[0] if e else '?'}")
if n == 0: sys.exit(2)
# exhaustive when no address is taken in code or data: an unresolved jalr then cannot reach the target
sys.exit(1 if taken or in_data else 0)
