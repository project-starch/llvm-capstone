#!/usr/bin/env python3
"""Over rescan-nosplit.py's plain-only runs: flag any run in which COPY-stores (a store whose value register was
loaded earlier in the same run and not overwritten since) cover 16 contiguous bytes from one destination base. That is
the least a scalar copy needs to move a whole capability. Positive control: the compiler lane's repro.c
copy32_unknown shape (two ld then two sd) must be flagged; a zero fill must not."""
import re, sys
def flag(t):
    ins = [x.strip() for x in t.split(";") if x.strip()]
    loaded = {}; cov = {}
    for x in ins:
        m = re.match(r"(lb|lbu|lh|lhu|lw|lwu|ld) (\w+), (-?0x[0-9a-f]+|-?\d+)\((\w+)\)", x)
        if m: loaded[m.group(2)] = True; continue
        m = re.match(r"(sb|sh|sw|sd) (\w+), (-?0x[0-9a-f]+|-?\d+)\((\w+)\)", x)
        if m:
            w = {"sb":1,"sh":2,"sw":4,"sd":8}[m.group(1)]; r = m.group(2); off = int(m.group(3), 0); base = m.group(4)
            if loaded.get(r): cov.setdefault(base, set()).update(range(off, off + w))
            continue
        m = re.match(r"\w+ (\w+),", x)
        if m and m.group(1) in loaded: loaded[m.group(1)] = False   # register overwritten by something else
    for base, s in cov.items():
        for start in s:
            if all(b in s for b in range(start, start + 16)): return f"16 contiguous copied bytes at {base}+{start:#x}"
    return None
if __name__ == "__main__":
    ctl = "ld a2, 0x0(a1); ld a3, 0x8(a1); sd a2, 0x0(a0); sd a3, 0x8(a0)"
    neg = "sd zero, 0x0(a0); sd zero, 0x8(a0); ld a1, 0x0(a2)"
    assert flag(ctl) and not flag(neg), "control failed"
    print("control: copy32 shape flagged, zero fill clean")
    L = open(sys.argv[1]).read().splitlines(); n = hits = 0
    for i, l in enumerate(L):
        if re.match(r"^  [A-Za-z_]", l):
            n += 1; f = flag(L[i + 1])
            if f: hits += 1; print("FLAG", l.strip(), "--", f)
    print(f"{sys.argv[1]}: {n} plain-only runs, {hits} flagged")
    if n == 0: sys.exit("NO DATA")
