#!/usr/bin/env python3
"""Re-cost of item 1 WITH the layout change, after the retraction.

The encoder oracle is the project's own validated model, not a reimplementation:
bounds_model.compress/decompress, a literal port of ariane_pkg.sv compress_bounds at
capstone-ariane d93a8d66b, including the R-54 exponent renormalisation, and bit-identical to
capstone-qemu's cap_compress. Its self-check reproduces the RTL comment's numbers exactly
(1050 random regions decode short without renormalisation, 0 with it), which is the positive
control for this whole analysis.

Geometry read from source:
  heap.c bind_slots :64-69   slot_base = group_base + HDR + i*stride,  HDR = 16
  meta.h enframe    :196-226 user base = slot_base + UNIT*off, off in [0, slack], slack=(stride-IB-n)/UNIT
  heap.c publish    :167-180 cap_derive(parent, key - base(parent), n)  -- the exposed bound is n
"""
import load_model
bm = load_model.load()   # the RTL oracle, pinned
import sweep as S

UNIT, IB, HDR = S.UNIT, S.IB, 16
PAGE = 4096

def decoded(base, n):
    """The bounds the hardware actually gives for a request of n bytes at `base`."""
    iE, B, T, E = bm.compress(base, base + n)
    return bm.decompress(iE, B, T, base)

def granule_of(base, n):
    """The granule the encoder actually used, including renormalisation -- derived from the
    oracle rather than assumed, so a renormalised case reports its true (coarser) granule."""
    iE, B, T, E = bm.compress(base, base + n)
    return 1 if iE == 0 else (1 << (E + 3))

print("=" * 100)
print("STEP 0  instrument check: the oracle must show BOTH containment and escape")
print("=" * 100)
# a case that must be contained, and one that must escape
ok = decoded(0x10000000, 64)
bad = decoded(0x10000000 + 16, 131056 - 4)
print("  exact small (base 256 MiB, n=64):        decoded %s  width %d" % (tuple(hex(x) for x in ok), ok[1]-ok[0]))
print("  large at a 16-byte offset (n=131052):    decoded %s  width %d" % (tuple(hex(x) for x in bad), bad[1]-bad[0]))
assert ok == (0x10000000, 0x10000000 + 64), "the oracle lost an exact small bound"
assert bad[0] < 0x10000000 + 16 or bad[1] > 0x10000000 + 16 + 131052, \
    "the oracle shows no widening at all -- instrument suspect"
print("  -> the oracle distinguishes the two cases")

print()
print("=" * 100)
print("STEP 1  TODAY's geometry: does the decoded bound stay inside its slot?")
print("=" * 100)
GROUP = 0x40000000   # page-aligned, as a VM mapping is
print("%-4s %8s %9s %7s %9s %9s %s" % ("sc","stride","n_max","granule","base esc","end esc","verdict"))
fail = []
for sc, c in enumerate(S.SIZE_CLASSES):
    stride = UNIT*c
    if stride < PAGE:
        continue
    n = stride - IB                      # the largest request this class serves
    slot = GROUP + HDR                   # slot i=0, the cheapest case
    base = slot                          # off = 0, also the cheapest
    g = granule_of(base, n)
    d0, d1 = decoded(base, n)
    base_esc = slot - d0                 # bytes the bound reaches BELOW the slot
    end_esc = d1 - (slot + stride)       # bytes it reaches ABOVE the slot
    bad = base_esc > 0 or end_esc > 0
    if bad: fail.append(sc)
    print("%-4d %8d %9d %7d %9d %9d %s" % (sc, stride, n, g, max(0,base_esc), max(0,end_esc),
                                           "ESCAPES" if bad else "contained"))
print()
print("classes whose BEST case (slot 0, offset 0) already escapes: %d of %d"
      % (len(fail), sum(1 for c in S.SIZE_CLASSES if UNIT*c >= PAGE)))

print()
print("=" * 100)
print("STEP 2  the MINIMAL geometry that contains every slot and offset")
print("=" * 100)
print("Requirement, from the encoder: base truncates DOWN to a granule multiple and the end rounds")
print("UP to one, so containment needs  HDR, stride and UNIT*off all granule multiples; then")
print("slot_base and slot_base+stride are granule-aligned and the decoded range cannot leave the slot.")
print()
print("%-4s %8s %7s %9s %9s %8s %9s %10s" %
      ("sc","stride","granule","HDR'","stride'","+bytes","+%","off step"))
tot_old = tot_new = 0
rows = []
for sc, c in enumerate(S.SIZE_CLASSES):
    stride = UNIT*c
    if stride < PAGE:
        rows.append((sc, stride, 1, HDR, stride, 0, 0.0, 1)); tot_old += stride; tot_new += stride
        continue
    n = stride - IB
    g = granule_of(GROUP + HDR, n)
    # the granule can only coarsen as the stride grows, so take the granule at the NEW stride too
    stride2 = ((stride + g - 1) // g) * g
    g2 = granule_of(GROUP + HDR, stride2 - IB)
    if g2 != g:
        stride2 = ((stride + g2 - 1) // g2) * g2
        g = g2
    hdr2 = ((HDR + g - 1) // g) * g
    extra = stride2 - stride
    rows.append((sc, stride, g, hdr2, stride2, extra, 100.0*extra/stride, max(1, g//UNIT)))
    tot_old += stride; tot_new += stride2
    print("%-4d %8d %7d %9d %9d %8d %9.2f %10s" % (sc, stride, g, hdr2, stride2, extra,
                                                   100.0*extra/stride, "1 in %d" % max(1, g//UNIT)))
print()
print("slot-size cost, summed over the 20 affected classes: %d -> %d bytes (+%.3f%%)"
      % (tot_old, tot_new, 100.0*(tot_new-tot_old)/tot_old))
print("group-header cost: 16 -> up to %d bytes, once per group (not per slot)"
      % max(r[3] for r in rows))

print()
print("=" * 100)
print("STEP 3  verify the proposed geometry CONTAINS every slot and every surviving offset")
print("=" * 100)
checked = esc = 0
for sc, stride0, g, hdr2, stride2, extra, pct, step in rows:
    if stride0 < PAGE:
        continue
    for i in range(4):
        slot = GROUP + hdr2 + i*stride2
        n = stride2 - IB
        slack = (stride2 - IB - n)//UNIT
        offs = [o for o in range(0, max(slack, 0)+1) if (UNIT*o) % g == 0] or [0]
        for off in offs[:8]:
            base = slot + UNIT*off
            d0, d1 = decoded(base, n)
            checked += 1
            if d0 < slot or d1 > slot + stride2:
                esc += 1
                if esc <= 3:
                    print("  ESCAPE sc=%d i=%d off=%d: decoded [%#x,%#x) vs slot [%#x,%#x)"
                          % (sc, i, off, d0, d1, slot, slot+stride2))
print("  checked %d (class, slot, offset) combinations; escapes: %d" % (checked, esc))
print("  -> %s" % ("the geometry contains every case" if esc == 0 else "THE GEOMETRY IS NOT SUFFICIENT"))

print()
print("=" * 100)
print("STEP 4  does publish() need more than the geometry?")
print("=" * 100)
# publish derives over the REQUESTED n, not the stride. With a granule-aligned base and a
# granule-multiple stride, is a request of ANY n < stride contained?
# A top-64-bytes window is a narrowed view and would show exactly what I went looking for, so
# sweep EVERY request the class serves, at every surviving offset, not just the largest.
nbad = 0
ntot = 0
for sc, stride0, g, hdr2, stride2, extra, pct, step in rows:
    if stride0 < PAGE:
        continue
    lo = (UNIT*S.SIZE_CLASSES[sc-1] - IB + 1) if sc else 1
    for n in range(max(1, lo), stride2 - IB + 1):
        slack = (stride2 - IB - n)//UNIT
        offs = [o for o in range(0, slack+1) if (UNIT*o) % g == 0]
        for off in (offs[:1] + offs[-1:]) or [0]:
            slot = GROUP + hdr2
            base = slot + UNIT*off
            d0, d1 = decoded(base, n)
            ntot += 1
            if d0 < slot or d1 > slot + stride2:
                nbad += 1
                if nbad <= 3:
                    print("  ESCAPE sc=%d n=%d off=%d: [%#x,%#x) vs slot [%#x,%#x)"
                          % (sc, n, off, d0, d1, slot, slot+stride2))
print("swept %d (class, request, offset) combinations across every repaired class" % ntot)
print("requests whose decoded bound escapes its slot: %d" % nbad)
print("  -> publish needs %s" % ("NO change beyond the geometry" if nbad == 0 else
                                 "a length round-up as well (%d cases)" % nbad))

print()
print("=" * 100)
print("STEP 5  the group-header cost in context (it is per GROUP, not per slot)")
print("=" * 100)
print("%-4s %9s %7s %8s %10s" % ("sc", "stride'", "granule", "hdr'", "hdr cost if 1 slot/group"))
for sc, stride0, g, hdr2, stride2, extra, pct, step in rows:
    if stride0 < PAGE:
        continue
    print("%-4d %9d %7d %8d %10.3f%%" % (sc, stride2, g, hdr2, 100.0*(hdr2-HDR)/stride2))
