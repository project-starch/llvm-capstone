#!/usr/bin/env python3
"""Data slots that hold an address and that no capability initializer writes (C-75).

    capinit-unwritten-slots.py FILE...      objects or archives (llvm-ar from $CAPSTONE_LLVM_BIN)

A pointer in initialized data is, in a capstone64 object, a 64-bit address relocation
(R_Capstone_64); the tag comes from the object's __capstone_cap_init, which takes a capability to
each slot it writes (an R_Capstone_PCREL_HI20 against the slot's object inside the initializer). A
slot inside an object the initializer never addresses keeps its link-time address, untagged, and
faults when it is loaded through. C-75 was one: a table of weak-alias addresses (musl's
atfork_locks), which the initializer skipped.

Prints one line per such slot: file, section, containing object ('?' when no sized symbol contains
it), offset in it, and the relocation's target. Left out: .gct (the loader fills it) and
.init_array/.fini_array (the runtime's constructor walk reads link addresses by design). A slot
that is an integer on purpose is reported too; the SDK has one, __capstone_init_fini_anchor_link.

Controls: a table of a weak alias's address beside a table of a variable's reports exactly the
alias slot on 7d01722aab88 and nothing with C-75's fix; musl's stdio and atexit objects and
CPython's _struct.o report nothing. Exits 2 when it read no object."""
import struct, subprocess, sys, tempfile, os
AR = os.path.join(os.environ.get('CAPSTONE_LLVM_BIN', ''), 'llvm-ar')
R_64, R_PCREL_HI20 = 2, 23
SHT_SYMTAB, SHT_RELA = 2, 4
SHF_ALLOC, SHF_EXEC = 2, 4

def parse(blob):
    shoff, = struct.unpack_from('<Q', blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', blob, 0x3a)
    secs = [struct.unpack_from('<IIQQQQIIQQ', blob, shoff + i * shentsize) for i in range(shnum)]
    def name(strtab, off):
        s = secs[strtab]; b = blob[s[4] + off:]; return b[:b.index(b'\0')].decode()
    names = [name(shstrndx, s[0]) for s in secs]
    symtab = next(i for i, s in enumerate(secs) if s[1] == SHT_SYMTAB)
    st = secs[symtab]
    syms = []
    for off in range(st[4], st[4] + st[5], 24):
        n, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', blob, off)
        syms.append((name(st[6], n), info & 0xf, shndx, value, size))
    relas = []
    for i, s in enumerate(secs):
        if s[1] != SHT_RELA: continue
        for off in range(s[4], s[4] + s[5], 24):
            r_off, info, addend = struct.unpack_from('<QQq', blob, off)
            relas.append((s[7], r_off, info & 0xffffffff, info >> 32, addend))  # applies-to section
    return secs, names, syms, relas

def census(label, blob):
    secs, names, syms, relas = parse(blob)
    inits = [(sh, v, v + sz) for n, t, sh, v, sz in syms if n == '__capstone_cap_init']
    targets = set()
    for sec, off, typ, sym, addend in relas:
        if typ == R_PCREL_HI20 and any(sec == sh and a <= off < b for sh, a, b in inits):
            n, t, sh, v, sz = syms[sym]
            targets.add((sh, v + addend))
    objs = {}
    for n, t, sh, v, sz in syms:
        if 0 < sh < len(secs) and sz > 0 and t in (1, 0):   # OBJECT or NOTYPE with a size
            objs.setdefault(sh, []).append((v, v + sz, n))
    out = []
    for sec, off, typ, sym, addend in relas:
        if typ != R_64: continue
        flags, nm = secs[sec][2], names[sec]
        if (not flags & SHF_ALLOC or flags & SHF_EXEC or
                nm.startswith(('.eh_frame', '.capstone_cap_init', '.gct', '.init_array', '.fini_array'))):
            continue
        box = next(((a, b, n) for a, b, n in objs.get(sec, []) if a <= off < b), None)
        tn = syms[sym][0] or names[syms[sym][2]] if syms[sym][2] < len(names) else syms[sym][0]
        if box is None:
            out.append((label, nm, '?', off, tn)); continue
        a, b, n = box
        if not any(sh == sec and a <= t < b for sh, t in targets):
            out.append((label, nm, n, off - a, tn))
    return out

seen, rows = 0, []
for path in sys.argv[1:]:
    if path.endswith('.a'):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([AR, 'x', os.path.abspath(path)], cwd=d, check=True)
            for m in sorted(os.listdir(d)):
                seen += 1; rows += census(f'{path}({m})', open(os.path.join(d, m), 'rb').read())
    else:
        seen += 1; rows += census(path, open(path, 'rb').read())
for r in rows:
    print('\t'.join(map(str, r)))
print(f'# objects read: {seen}, unwritten slots: {len(rows)}', file=sys.stderr)
sys.exit(2 if seen == 0 else 0)
