#!/usr/bin/env python3
"""capstone-size-assume-check.py <elf>...: check a statically linked Capstone image against the size assumptions in its
.capstone.size_assume section.

With -mllvm -capstone-drop-redundant-shrink-declarations the compiler drops the SHRINKTO / SHRINK narrowing of a
global that a translation unit only DECLARES, when every access through it lies within the declared size. The size is
that unit's belief. It records, per such global, the bytes from the symbol's start that the remaining accesses reach:
an 8-byte little-endian size, then the symbol's NUL-terminated name. This script fails an image in which a named
global is defined smaller than a record requires, missing, defined more than once, or of unknown size. That turns an
ODR size mismatch, which the dropped narrowing could have caught at the image's end, into a link-time error.

Run it on the linked image before stripping: it needs .symtab.
Exit 0: every record holds, or the image carries no records (it prints how many it checked).
Exit 1: a record fails.
Exit 2: an image could not be read, or it has records but no symbol table.
"""
import struct, sys

def sections(data):
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        raise ValueError('not a little-endian ELF64 file')
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3a)
    hdrs = []
    for i in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
            '<IIQQQQIIQQ', data, shoff + i * shentsize)
        hdrs.append(dict(name=name, type=typ, off=off, size=size, link=link, entsize=entsize))
    strtab = hdrs[shstrndx]
    def nm(h):
        s = data[strtab['off'] + h['name']:]
        return s[:s.index(b'\0')].decode()
    for h in hdrs:
        h['sname'] = nm(h)
    return hdrs

def check(path):
    data = open(path, 'rb').read()
    secs = sections(data)
    sa = [h for h in secs if h['sname'] == '.capstone.size_assume']
    if not sa:
        print(f'{path}: no .capstone.size_assume section, 0 assumptions')
        return 0
    recs = []
    for h in sa:
        b = data[h['off']:h['off'] + h['size']]; i = 0
        while i < len(b):
            need, = struct.unpack_from('<Q', b, i); j = b.index(b'\0', i + 8)
            recs.append((b[i + 8:j].decode(), need)); i = j + 1
    symtab = [h for h in secs if h['type'] == 2]   # SHT_SYMTAB
    if not symtab:
        print(f'{path}: {len(recs)} assumptions but no .symtab (stripped?): cannot check')
        return 2
    st = symtab[0]; strs = secs[st['link']]
    defs = {}
    for k in range(st['size'] // 24):
        name, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', data, st['off'] + 24 * k)
        bind, vis = info >> 4, other & 3
        # Skip undefined symbols and file-static locals. A LOCAL symbol with HIDDEN or INTERNAL visibility is a
        # global the static link demoted (musl's __libc, __malloc_context, ...): it is the definition.
        if shndx == 0 or (bind == 0 and vis not in (1, 2)):
            continue
        s = data[strs['off'] + name:]; s = s[:s.index(b'\0')].decode()
        defs.setdefault(s, []).append(size)
    need = {}
    for n, k in recs:
        need[n] = max(need.get(n, 0), k)
    bad = 0
    for n, k in sorted(need.items()):
        d = defs.get(n)
        if not d:
            print(f'{path}: FAIL {n}: assumed >= {k} B, no definition in the image'); bad += 1
        elif len(d) != 1:
            print(f'{path}: FAIL {n}: assumed >= {k} B, {len(d)} definitions'); bad += 1
        elif d[0] == 0 or d[0] < k:
            print(f'{path}: FAIL {n}: assumed >= {k} B, defined {d[0]} B'); bad += 1
    print(f'{path}: {len(need)} symbols / {len(recs)} records checked, {bad} failing')
    return 1 if bad else 0

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rc = 0
    for p in sys.argv[1:]:
        try:
            rc = max(rc, check(p))
        except Exception as e:
            print(f'{p}: cannot read ({e})'); rc = max(rc, 2)
    sys.exit(rc)

main()
