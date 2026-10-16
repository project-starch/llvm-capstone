#!/usr/bin/env python3
"""Option (iii): compressed 27-bit bounds (exact < 4 KiB, granule 2^(ilog2(L)-9) above).
What must change in musl 1.2.5 mallocng so that every exposed object bound, after outward
rounding, stays inside its own slot?

Constants and arithmetic are READ FROM SOURCE, printed below so the inputs can be checked:
  musl-1.2.5 src/malloc/mallocng/{malloc.c,meta.h}
  granule rule: modcapstone capstone.c capstone_repr_granule/_round
"""
import sys

# ---- inputs, printed so a reader can check them -------------------------------------------
UNIT = 16            # meta.h:14
IB   = 4             # meta.h:15
MMAP_THRESHOLD = 131052   # meta.h:12
SIZE_CLASSES = [     # malloc.c:12-24, units of UNIT bytes
    1,2,3,4,5,6,7,8, 9,10,12,15, 18,20,25,31, 36,42,50,63,
    72,84,102,127, 146,170,204,255, 292,340,409,511,
    584,682,818,1023, 1169,1364,1637,2047, 2340,2730,3276,4095,
    4680,5460,6552,8191,
]

def ilog2(x):
    assert x > 0
    return x.bit_length() - 1

def repr_granule(L):
    """capstone.c:274 -- below 4096 the encoder takes its exact branch, granule 1."""
    if L < 4096:
        return 1
    return 1 << (ilog2(L) - 9)

def repr_round(L):
    """capstone.c:282 -- round up to the granule, then re-check (the kernel does two passes)."""
    g = repr_granule(L)
    r = ((L + g - 1) // g) * g
    if r % repr_granule(r):
        g2 = repr_granule(r)
        r = ((r + g2 - 1) // g2) * g2
    return r

def clz32(x):
    return 32 - x.bit_length()

def size_to_class(n):
    """meta.h:229-238, verbatim. Returns None when no class can hold n -- mallocng only calls
    this for n < MMAP_THRESHOLD, so an out-of-range index means the length has been pushed onto
    the individually-mmapped big path. That is a RESULT of rounding, not an error."""
    u = (n + IB - 1) >> 4
    if u < 10:
        return u
    u += 1
    i = (28 - clz32(u)) * 4 + 8
    if i + 1 >= len(SIZE_CLASSES):
        return None
    if u > SIZE_CLASSES[i+1]:
        i += 2
    if i >= len(SIZE_CLASSES) or u > SIZE_CLASSES[i]:
        i += 1
    if i >= len(SIZE_CLASSES):
        return None
    return i

# ---- self-checks on the rounding rule ------------------------------------------------------
def selfcheck():
    # exact below 4 KiB
    for L in (1, 17, 4095):
        assert repr_round(L) == L, L
    # granule at the first few decades
    # 131052 < 2^17, so ilog2 = 16 and the granule is 2^7 = 128 (my first guess of 256 was
    # wrong and this assertion caught it, which is the point of having it).
    cases = {4096: 8, 8191: 8, 8192: 16, 131052: 128, 4198400: 8192}
    for L, g in cases.items():
        assert repr_granule(L) == g, (L, repr_granule(L), g)
    # the second pass must be exercised: find an L where one pass is not enough
    second = [L for L in range(4096, 1 << 22)
              if (lambda g: ((L + g - 1)//g*g) % repr_granule((L + g - 1)//g*g))(repr_granule(L))]
    # rounded result is always a multiple of ITS OWN granule after the function returns
    for L in list(range(1, 300000)) + [1 << 22, (1 << 22) + 1, 4198400]:
        r = repr_round(L)
        assert r >= L and r % repr_granule(r) == 0, (L, r)
    return len(second)

n_second = selfcheck()

print("=" * 96)
print("INPUTS   UNIT=%d  IB=%d  MMAP_THRESHOLD=%d  %d size classes, max %d units = %d bytes"
      % (UNIT, IB, MMAP_THRESHOLD, len(SIZE_CLASSES), SIZE_CLASSES[-1], UNIT*SIZE_CLASSES[-1]))
print("         granule: 1 below 4096, else 2^(ilog2(L)-9);  lengths needing a 2nd rounding pass: %d"
      % n_second)
print("=" * 96)

# ---- the small path: per size class ---------------------------------------------------------
# enframe (meta.h:196-226):  stride = UNIT*size_classes[sc];  slack = (stride-IB-n)//UNIT
#   user pointer p = slotstart + UNIT*off, off in [0, slack];  end = slotstart + stride - IB
# room from p to end  =  stride - IB - UNIT*off
#   off=slack  (worst) -> n + ((stride-IB-n) % UNIT)      <= n+15
#   off=0      (best)  -> stride - IB
print("\n### SMALL PATH (n < MMAP_THRESHOLD): does repr_round(n) stay inside the slot?\n")
print("%-4s %7s %8s %9s %9s %9s %9s" %
      ("sc", "units", "stride", "n_max", "gran@nmax", "fits off=0", "fits worst"))
rows = []
for sc, c in enumerate(SIZE_CLASSES):
    stride = UNIT * c
    # largest n that this class serves: size_to_class(n) == sc
    n_max = max((n for n in range(1, min(stride + 64, MMAP_THRESHOLD))
                 if size_to_class(n) == sc), default=None)
    if n_max is None:
        continue
    g = repr_granule(n_max)
    room_best = stride - IB
    room_worst = n_max + ((stride - IB - n_max) % UNIT)
    fb = repr_round(n_max) <= room_best
    fw = repr_round(n_max) <= room_worst
    rows.append((sc, c, stride, n_max, g, fb, fw))
    if stride >= 2048 or not fb or not fw:
        print("%-4d %7d %8d %9d %9d %9s %9s" % (sc, c, stride, n_max, g,
                                                "yes" if fb else "NO", "yes" if fw else "NO"))

bad_best = [r for r in rows if not r[5]]
bad_worst = [r for r in rows if not r[6]]
print("\nclasses where repr_round(n_max) leaves the slot:  off=0 -> %d of %d,  worst off -> %d of %d"
      % (len(bad_best), len(rows), len(bad_worst), len(rows)))
print("first affected class (worst off): sc=%s" % (bad_worst[0][0] if bad_worst else None))

# how many individual n values are affected, and the max overshoot
aff_worst = aff_best = 0
max_over_worst = max_over_best = 0
for n in range(1, MMAP_THRESHOLD):
    sc = size_to_class(n)
    stride = UNIT * SIZE_CLASSES[sc]
    r = repr_round(n)
    rb = stride - IB
    rw = n + ((stride - IB - n) % UNIT)
    if r > rb:
        aff_best += 1
        max_over_best = max(max_over_best, r - rb)
    if r > rw:
        aff_worst += 1
        max_over_worst = max(max_over_worst, r - rw)
print("individual n in 1..%d affected:  off=0 -> %d (max overshoot %d B),  worst off -> %d (max %d B)"
      % (MMAP_THRESHOLD - 1, aff_best, max_over_best, aff_worst, max_over_worst))

# ---- fix (a): classify on repr_round(n) -----------------------------------------------------
print("\n### FIX (a): classify on repr_round(n) instead of n\n")
promoted = 0
to_big = 0
tot_extra = 0
max_extra = 0
still_bad = 0
worst_ratio = (0.0, None)
for n in range(1, MMAP_THRESHOLD):
    r = repr_round(n)
    sc0, sc1 = size_to_class(n), size_to_class(r)
    if sc1 is None:
        to_big += 1
        continue
    s0, s1 = UNIT*SIZE_CLASSES[sc0], UNIT*SIZE_CLASSES[sc1]
    if sc1 != sc0:
        promoted += 1
        tot_extra += s1 - s0
        max_extra = max(max_extra, s1 - s0)
        ratio = (s1 - s0) / n
        if ratio > worst_ratio[0]:
            worst_ratio = (ratio, n)
    # The check "r <= r + ((s1-IB-r) % UNIT)" would be VACUOUS -- it can never fire, because
    # once enframe is handed r the slack is computed FROM r and the offset cannot push the
    # bound out. The non-vacuous condition is whether the chosen class can hold r at all:
    # the slot must have room for r plus the in-band header.
    if r > s1 - IB:
        still_bad += 1
print("n values promoted to a larger class: %d of %d (%.2f%%)"
      % (promoted, MMAP_THRESHOLD - 1, 100.0*promoted/(MMAP_THRESHOLD - 1)))
print("extra bytes: total %d over the swept range, max %d for one allocation" % (tot_extra, max_extra))
print("worst single-allocation overhead: %.1f%% at n=%d" % (100.0*worst_ratio[0], worst_ratio[1]))
print("n values still not fitting after (a), at the worst slot offset: %d" % still_bad)
print("n values pushed off the size-class table onto the mmap path by rounding: %d" % to_big)

# ---- fix (b): pin off=0 for large classes ---------------------------------------------------
print("\n### FIX (b): keep classification, pin off=0 for classes with stride >= 4096\n")
b_bad = 0
for n in range(1, MMAP_THRESHOLD):
    sc = size_to_class(n)
    stride = UNIT*SIZE_CLASSES[sc]
    room = (stride - IB) if stride >= 4096 else (n + ((stride - IB - n) % UNIT))
    if repr_round(n) > room:
        b_bad += 1
print("n values still not fitting under (b): %d" % b_bad)
print("classes that would lose offset cycling (the double-free trap): %d of %d"
      % (sum(1 for c in SIZE_CLASSES if UNIT*c >= 4096), len(SIZE_CLASSES)))

# ---- the big path ---------------------------------------------------------------------------
# malloc.c:308-327   needed = n + IB + UNIT ; maplen = ceil(needed/4096)
# get_stride (meta.h:180)  stride = maplen*4096 - UNIT
print("\n### BIG PATH (n >= MMAP_THRESHOLD): individually mmapped\n")
def big(n):
    needed = n + IB + UNIT
    maplen = (needed + 4095) // 4096
    stride = maplen*4096 - UNIT
    return maplen, stride

samples = [MMAP_THRESHOLD, MMAP_THRESHOLD + 1, 1 << 18, (1 << 20) - 1, 1 << 20,
           1 << 22, (1 << 22) + 4096, 4198400, 1 << 24]
print("%12s %7s %10s %10s %8s %10s %10s %s" %
      ("n", "maplen", "stride", "repr_rnd", "gran", "room_best", "room_wst", "fits best/worst"))
for n in samples:
    maplen, stride = big(n)
    r = repr_round(n)
    rb = stride - IB
    rw = n + ((stride - IB - n) % UNIT)
    print("%12d %7d %10d %10d %8d %10d %10d  %s/%s"
          % (n, maplen, stride, r, repr_granule(n), rb, rw,
             "yes" if r <= rb else "NO", "yes" if r <= rw else "NO"))

# how often does the rounded bound exceed the whole MAPPING (unmapped VA, not a neighbour)?
step = 4096
over_map = 0
over_map_max = 0
checked = 0
n = MMAP_THRESHOLD
while n < (1 << 26):
    maplen, stride = big(n)
    r = repr_round(n)
    checked += 1
    if r > maplen*4096:
        over_map += 1
        over_map_max = max(over_map_max, r - maplen*4096)
    n += step
print("\nsampled every %d B from %d to 64 MiB (%d points):" % (step, MMAP_THRESHOLD, checked))
print("  rounded bound exceeds the MAPPING: %d points, max overshoot %d B" % (over_map, over_map_max))
# and the fix: size the mapping from repr_round(n) instead of n
extra_pages = {}
n = MMAP_THRESHOLD
while n < (1 << 26):
    m0 = big(n)[0]
    m1 = ((repr_round(n) + IB + UNIT) + 4095) // 4096
    extra_pages[m1 - m0] = extra_pages.get(m1 - m0, 0) + 1
    n += step
print("  fix: size the mapping from repr_round(n). extra pages needed, by count: %s"
      % dict(sorted(extra_pages.items())))

# the page-rounding claim in mallocng.md:96, checked directly
first_bad_pages = None
for k in range(1, 1 << 16):
    L = k*4096
    if L % repr_granule(L):
        first_bad_pages = (k, L)
        break
print("\nmallocng.md:96 says mappings are a 'page-rounded requested length'.")
print("  smallest page-rounded length that is NOT representable: %s pages = %s bytes"
      % (first_bad_pages[0], first_bad_pages[1]))
bad = sum(1 for k in range(1, 1 << 14) if (k*4096) % repr_granule(k*4096))
print("  of the first %d page counts, %d are not representable (%.1f%%)"
      % ((1 << 14) - 1, bad, 100.0*bad/((1 << 14) - 1)))
