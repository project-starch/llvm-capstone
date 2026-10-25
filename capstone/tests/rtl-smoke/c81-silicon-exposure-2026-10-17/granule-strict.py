"""Strict object-copy check: a run is flagged when 16 contiguous destination bytes [d, d+16) from one base are each
stored from a register loaded in the same run from ONE source base at the SAME relative offset (src_off - dst_off
constant over the granule): what a copy of one 16-byte object through scalar registers looks like."""
import re
def flag(t):
    ins = [x.strip() for x in t.split(";") if x.strip()]
    src = {}; cov = {}   # reg -> (sbase, soff, width); dbase -> {dst byte: (sbase, delta)}
    for x in ins:
        m = re.match(r"(lb|lbu|lh|lhu|lw|lwu|ld) (\w+), (-?0x[0-9a-f]+|-?\d+)\((\w+)\)", x)
        if m:
            w = {"lb":1,"lbu":1,"lh":2,"lhu":2,"lw":4,"lwu":4,"ld":8}[m.group(1)]
            src[m.group(2)] = (m.group(4), int(m.group(3), 0), w); continue
        m = re.match(r"(sb|sh|sw|sd) (\w+), (-?0x[0-9a-f]+|-?\d+)\((\w+)\)", x)
        if m:
            w = {"sb":1,"sh":2,"sw":4,"sd":8}[m.group(1)]; r = m.group(2); off = int(m.group(3), 0); base = m.group(4)
            if r in src and src[r][2] >= w:
                sb, so, _ = src[r]
                for i in range(w): cov.setdefault(base, {})[off + i] = (sb, so - off)
            continue
        m = re.match(r"\w+ (\w+),", x)
        if m: src.pop(m.group(1), None)
    for base, d in cov.items():
        for start in d:
            g = [d.get(b) for b in range(start, start + 16)]
            if all(g) and len(set(g)) == 1: return f"16-byte object copy {g[0][0]}+{start+g[0][1]:#x} -> {base}+{start:#x}"
    return None
assert flag("ld a2, 0x0(a1); ld a3, 0x8(a1); sd a2, 0x0(a0); sd a3, 0x8(a0)")
assert not flag("ld a4, 0x0(a5); sd a4, 0xa8(a1); ld a4, 0x0(a6); sd a4, 0xb0(a1)")
assert not flag("sd zero, 0x0(a0); sd zero, 0x8(a0)")
