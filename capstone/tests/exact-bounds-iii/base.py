#!/usr/bin/env python3
"""RETRACTION CHECK. The cost document said option (iii) needs "no alignment change". The kernel's
own predicate (modcapstone capstone.c:338) says: "A region is representable only if BOTH its length
and its base are granule multiples", enforced at :344 and process.c:124.

So the child capability `publish` derives must have a granule-ALIGNED BASE as well as a
granule-multiple length. Where does that base come from?

  heap.c bind_slots (:64-69):  slot_base = group_base + 16 + i*stride
  enframe (meta.h:196-226):    user base = slot_base + UNIT*off,  off in [0, slack]
  heap.c publish (:167-180):   cap_derive(parent, key - base(parent), n)

group_base comes from a VM mapping, so it is at least 4096-aligned. The `+ 16` group header is
the problem as soon as the granule exceeds 16.
"""
import sweep as S

UNIT, IB = S.UNIT, S.IB

def g_of(L):
    return S.repr_granule(L)

print("=" * 92)
print("Is the length alone enough? -- what the cost doc assumed")
print("=" * 92)
bad_stride = []
for sc, c in enumerate(S.SIZE_CLASSES):
    stride = UNIT*c
    if stride < 4096:
        continue
    g = g_of(stride)
    if stride % g:
        bad_stride.append((sc, stride, g, stride % g))
print("size-class STRIDES that are not a granule multiple of their own length: %d" % len(bad_stride))
for sc, stride, g, rem in bad_stride:
    print("   sc=%-3d stride=%-7d granule=%-5d remainder=%d" % (sc, stride, g, rem))

print()
print("=" * 92)
print("The BASE requirement -- what the cost doc missed")
print("=" * 92)
# For a request n: length L = repr_round(n), granule g = g_of(L).
# user base = group_base + 16 + i*stride + UNIT*off, with group_base a multiple of 4096.
# Base is representable iff (16 + i*stride + UNIT*off) % g == 0  (for g <= 4096).
print("group_base is page-aligned, so the base is representable iff")
print("    (16 + i*stride + UNIT*off) mod granule == 0")
print()
print("%-5s %8s %9s %7s %-28s" % ("sc", "stride", "L=rnd(n)", "granule", "16 + i*stride mod g, i=0..3"))
affected = []
for sc, c in enumerate(S.SIZE_CLASSES):
    stride = UNIT*c
    if stride < 4096:
        continue
    n_max = stride - IB
    L = S.repr_round(n_max)
    g = g_of(L)
    mods = [(16 + i*stride) % g for i in range(4)]
    ok_i0 = mods[0] == 0
    if not ok_i0:
        affected.append((sc, stride, g))
    print("%-5d %8d %9d %7d %-28s %s" % (sc, stride, L, g, mods,
                                         "" if ok_i0 else "<- slot 0 base NOT aligned"))

print()
print("classes whose FIRST slot already has an unrepresentable base: %d of %d classes with stride >= 4096"
      % (len(affected), sum(1 for c in S.SIZE_CLASSES if UNIT*c >= 4096)))

print()
print("=" * 92)
print("How much of the offset cycling survives?")
print("=" * 92)
# enframe may place the user pointer at any off in [0, slack]; only granule-aligned ones work.
print("%-5s %8s %7s %9s %9s" % ("sc", "stride", "granule", "off mult", "usable offs"))
for sc, c in enumerate(S.SIZE_CLASSES):
    stride = UNIT*c
    if stride < 4096:
        continue
    L = S.repr_round(stride - IB)
    g = g_of(L)
    # off must satisfy (16 + i*stride + 16*off) % g == 0; with g >= 32 and 16*off stepping by 16,
    # only every (g/16)th off can work, and only if (16 + i*stride) mod 16 == 0 (it is).
    step = max(1, g // UNIT)
    print("%-5d %8d %7d %9s %9s" % (sc, stride, g, step,
                                    "1 in %d" % step if step > 1 else "all"))

print()
print("=" * 92)
print("Does the cheap fix still work? Round the length UP and the base DOWN to the granule.")
print("=" * 92)
# publish can derive over [base_down, base_down + L'] where base_down = base & ~(g-1) and
# L' = repr_round(n + (base - base_down)). That stays inside the slot iff
#   (base - base_down) + L' <= stride - (base - slot_base) ... compute the worst case per class.
worst = []
for sc, c in enumerate(S.SIZE_CLASSES):
    stride = UNIT*c
    if stride < 4096:
        continue
    n_max = stride - IB
    L0 = S.repr_round(n_max)
    g = g_of(L0)
    # worst case: base sits g-16 above a granule boundary
    back = g - UNIT if g > UNIT else 0
    L = S.repr_round(n_max + back)
    worst.append((sc, stride, g, back, L, L <= stride))
    print("sc=%-3d stride=%-7d granule=%-5d base rounded back by up to %-6d -> length %-8d %s"
          % (sc, stride, g, back, L, "fits the slot" if L <= stride else "DOES NOT FIT"))
print()
print("classes where rounding the base down still fits the slot: %d of %d"
      % (sum(1 for w in worst if w[5]), len(worst)))
