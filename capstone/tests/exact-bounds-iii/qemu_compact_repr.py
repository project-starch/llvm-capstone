#!/usr/bin/env python3
"""Is a CDERIVE child representable? -- the predicate helper_csderive applies.

Faithful port of capstone-qemu cap_compress/cap_uncompress (target/riscv/cap_compress.c at
b290f5656600, the kit's QEMU). All arithmetic masked to 64 bits: unbounded Python hid a 64-bit
overflow once before (R-54), and the decoder's own comment records an int-shift overflow at E=18.

helper_csderive (op_helper.c:1232-1247) sets child.base = child.cursor, child.end = base + size,
then compresses, uncompresses, and refuses with 29 if base or end did not survive. Cursor == base
for the child, which matters because the decoder reconstructs the window from the cursor.
"""
M64 = (1 << 64) - 1

def _clz64(x):
    assert x > 0
    return 64 - x.bit_length()

def compress(base, end, cursor):
    """Returns (lo, hi_fields) as (cursor, (B, T, iE)) -- only the bounds fields matter here."""
    base &= M64; end &= M64; cursor &= M64
    length = (end - base) & M64
    leading = (63 - _clz64(length)) if (length >> 13) else 12
    E = leading - 12
    if E == 0 and ((length >> 12) & 1) == 0:
        iE = 0
        B = base & ((1 << 14) - 1)
        T = end & ((1 << 12) - 1)
    else:
        iE = 1
        B = (base >> E) & ~7 & M64
        T = (end >> E) & ~7 & M64
        if end & (((1 << (E + 3)) - 1)):
            T += 8
        B |= E & 7
        T |= (E >> 3) & 7
    # the field widths the struct actually stores: bE 3, b 11 -> B is 14 bits; tE 3, t 9 -> T is 12
    B &= (1 << 14) - 1
    T &= (1 << 12) - 1
    return cursor, B, T, iE

def uncompress(cursor, B_field, T_field, iE):
    """Mirrors cap_uncompress: B = b<<3 | bE, i.e. the stored 14/12 bits."""
    cursor &= M64
    B = ((B_field >> 3) & ((1 << 11) - 1)) << 3
    T = ((T_field >> 3) & ((1 << 9) - 1)) << 3
    if iE:
        E = (((T_field) & 7) << 3) | (B_field & 7)
        msb = 1
    else:
        E = 0
        T |= T_field & 7
        B |= B_field & 7
        msb = 0
    carry_out = 1 if (T & 4095) < (B & 4095) else 0
    T |= (((B >> 12) + carry_out + msb) & 3) << 12
    span = (1 << (E + 14)) & M64 if E + 14 < 64 else 0
    window = (~(span - 1)) & M64 if span else 0
    b = (cursor & window) | ((B << E) & M64)
    t = (cursor & window) | ((T << E) & M64)
    A3 = (cursor >> (E + 11)) & 7 if E + 11 < 64 else 0
    B3 = (B >> 11) & 7
    T3 = (T >> 11) & 7
    R = (B3 - 1) & 7
    if A3 >= R and T3 < R:   t = (t + span) & M64
    elif A3 < R and T3 >= R: t = (t - span) & M64
    if A3 >= R and B3 < R:   b = (b + span) & M64
    elif A3 < R and B3 >= R: b = (b - span) & M64
    return b & M64, t & M64

def representable(base, size):
    """True iff a CDERIVE child at [base, base+size) survives the round trip."""
    end = (base + size) & M64
    cur, B, T, iE = compress(base, end, base)   # child's cursor IS its base
    b, t = uncompress(cur, B, T, iE)
    return b == base and t == end, (b, t, iE)

if __name__ == "__main__":
    print("=== POSITIVE CONTROLS: a predicate that cannot reproduce a known refusal is worthless\n")
    # tests/sublet-lifetimes case 27 `compact_bounds_refusal`: INC(a4, s2, 1) then size 4096,
    # documented to refuse with 29. Base is offset by 1 from an aligned region start.
    for label, base, size, want in [
        ("case 27 shape: base+1, size 4096 (documented REFUSAL)", 0x80001001, 4096, False),
        ("same size, aligned base 0x80001000           (expect OK)", 0x80001000, 4096, True),
        ("small object, odd base, size 64              (expect OK)", 0x80001001,   64, True),
        ("small object, odd base, size 4095            (expect OK)", 0x80001001, 4095, True),
    ]:
        ok, (b, t, iE) = representable(base, size)
        flag = "PASS" if ok == want else "*** PREDICATE WRONG ***"
        print(f"  {flag:6} {label}")
        print(f"         base={base:#x} size={size} -> representable={ok} (iE={iE}, decoded base={b:#x} end={t:#x})")
    print("\n=== where the boundary actually is: smallest size needing alignment, per power of two\n")
    print("  size        iE  base alignment required for exactness")
    for shift in range(5, 25):
        size = 1 << shift
        need = None
        for a in range(0, 25):
            align = 1 << a
            if all(representable(0x80000000 + align * k, size)[0] for k in (1, 3, 5)):
                need = align; break
        ok_any, (_, _, iE) = representable(0x80000000 + 1, size)
        print(f"  {size:>10}  {iE}   {'any (exact for unaligned base)' if ok_any else f'{need} bytes'}")
