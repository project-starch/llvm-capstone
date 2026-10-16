#!/usr/bin/env python3
"""The slot's last IB bytes are the NEXT slot's in-band header, so "inside the slot" is the wrong
containment test. enframe (meta.h:196-201) puts slot idx at `storage + stride*idx` and ends its
usable region at `p + stride - IB`; those final 4 bytes are `p'[-4..-1]` for `p' = storage +
stride*(idx+1)` -- the check byte, index and offset bits enframe itself reads. A rounded end that
reaches them gives an object authority over its neighbour's allocator header, which exact bounds
never allowed.

Counted here over the REPAIRED geometry, at the lowest AND the highest surviving offset (the
peer lane ran offset 0 only), and then re-counted with the fix.
"""
import load_model, sweep as S
bm = load_model.load()
UNIT, IB, PAGE, G0 = S.UNIT, S.IB, 4096, 0x40000000
SC = list(S.SIZE_CLASSES)

def gran(base, n):
    iE, B, T, E = bm.compress(base, base + n)
    return 1 if iE == 0 else (1 << (E + 3))

def dec(base, n):
    iE, B, T, E = bm.compress(base, base + n)
    return bm.decompress(iE, B, T, base)

# the repaired table, recomputed here so this script stands alone
NEW = list(SC)
for sc, c in enumerate(SC):
    stride = UNIT * c
    if stride < PAGE:
        continue
    g = gran(G0 + UNIT, stride - IB)
    s2 = ((stride + g - 1) // g) * g
    g2 = gran(G0 + UNIT, s2 - IB)
    if g2 != g:
        s2 = ((stride + g2 - 1) // g2) * g2
        g = g2
    NEW[sc] = s2 // UNIT

def sweep(table, hdr_granule, fix):
    """-> (escapes, next_header_hits, promoted, checked). fix: classify on repr_round(n)
    against stride-IB, promoting to the next class that can hold it."""
    esc = hit = promoted = checked = 0
    per_class = {}
    for sc in range(28, len(table)):
        stride = UNIT * table[sc]
        cap = stride - IB
        g = gran(G0 + UNIT, cap)
        hdr = (((UNIT + g - 1) // g) * g) if hdr_granule else UNIT
        storage = G0 + hdr
        lo = UNIT * table[sc - 1] - IB + 1
        c_hit = 0
        for n in range(max(1, lo), cap + 1):
            use_sc, use_stride, use_n = sc, stride, n
            if fix:
                r = n
                g0 = gran(storage, n)
                r = ((n + g0 - 1) // g0) * g0
                j = sc
                while j < len(table) and r > UNIT * table[j] - IB:
                    j += 1
                if j != sc:
                    promoted += 1
                    if j >= len(table):
                        continue          # pushed onto the mmap path
                    use_sc, use_stride = j, UNIT * table[j]
                use_n = r
            ustride = use_stride
            ucap = ustride - IB
            ug = gran(G0 + UNIT, ucap)
            uhdr = (((UNIT + ug - 1) // ug) * ug) if hdr_granule else UNIT
            ustorage = G0 + uhdr
            slack = (ustride - IB - use_n) // UNIT
            offs = [o for o in range(0, max(slack, 0) + 1) if (UNIT * o) % ug == 0] or [0]
            for off in {offs[0], offs[-1]}:
                slot = ustorage + ustride * 1          # slot 1: it has a neighbour on both sides
                p = slot + UNIT * off
                b, t = dec(p, use_n)
                checked += 1
                if b < slot or t > slot + ustride:
                    esc += 1
                if t > slot + ustride - IB:
                    hit += 1; c_hit += 1
        per_class[sc] = (g, c_hit)
    return esc, hit, promoted, checked, per_class

print("=" * 96)
print("POSITIVE CONTROL: the ORIGINAL table with a 16-byte header must show escapes")
print("=" * 96)
e, h, p_, c, _ = sweep(SC, hdr_granule=False, fix=False)
print("  escapes %d, next-header hits %d, of %d checks" % (e, h, c))
assert e > 0, "the escape check cannot fire -- instrument suspect"

print()
print("=" * 96)
print("REPAIRED geometry, NO length fix -- the peer lane's finding")
print("=" * 96)
e, h, p_, c, per = sweep(NEW, hdr_granule=True, fix=False)
print("  escapes from the SLOT:            %d of %d" % (e, c))
print("  reaching the NEXT SLOT's HEADER:  %d" % h)
print()
print("  %-4s %8s %8s %10s" % ("sc", "stride", "granule", "hits"))
tot = 0
for sc in sorted(per):
    g, n = per[sc]
    tot += n
    if n: print("  %-4d %8d %8d %10d" % (sc, UNIT*NEW[sc], g, n))
print("  per-class total: %d  (hits are the sizes with n > stride - granule)" % tot)

print()
print("=" * 96)
print("REPAIRED geometry WITH the length fix (classify repr_round(n) against stride - IB)")
print("=" * 96)
e, h, p_, c, _ = sweep(NEW, hdr_granule=True, fix=True)
print("  escapes from the SLOT:            %d of %d" % (e, c))
print("  reaching the NEXT SLOT's HEADER:  %d" % h)
print("  sizes promoted one class:         %d" % p_)
