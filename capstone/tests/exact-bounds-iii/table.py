#!/usr/bin/env python3
"""Does changing size_classes[] break mallocng's class lookup?

size_to_class (meta.h:229-238) finds the octave with a bit trick,
  i = (28 - clz32(u))*4 + 8
then refines with two comparisons against the table. The refinement is what makes the lookup
correct, so the question is whether the EDITED table still satisfies the invariant the allocator
relies on: every request maps to a class whose slot can hold it, and the mapping is monotone.
"""
import sweep as S
import recost  # reuse its computed strides

UNIT, IB = S.UNIT, S.IB
OLD = list(S.SIZE_CLASSES)
NEW = list(OLD)
for sc, stride0, g, hdr2, stride2, extra, pct, step in recost.rows:
    NEW[sc] = stride2 // UNIT

changed = [(i, OLD[i], NEW[i]) for i in range(len(OLD)) if OLD[i] != NEW[i]]
print("table entries changed: %d of %d" % (len(changed), len(OLD)))
for i, a, b in changed:
    print("   size_classes[%d]: %d -> %d units  (%d -> %d bytes)" % (i, a, b, UNIT*a, UNIT*b))

print("\nmonotone strictly increasing? old=%s new=%s"
      % (all(OLD[i] < OLD[i+1] for i in range(len(OLD)-1)),
         all(NEW[i] < NEW[i+1] for i in range(len(NEW)-1))))

def s2c(n, tbl):
    u = (n + IB - 1) >> 4
    if u < 10: return u
    u += 1
    i = (28 - (32 - u.bit_length())) * 4 + 8
    if i + 1 >= len(tbl): return None
    if u > tbl[i+1]: i += 2
    if i >= len(tbl) or u > tbl[i]: i += 1
    return i if i < len(tbl) else None

print("\ninvariant: the chosen class must be able to hold the request (n <= stride - IB)")
bad_old = bad_new = 0
moved = 0
for n in range(1, S.MMAP_THRESHOLD):
    a, b = s2c(n, OLD), s2c(n, NEW)
    if a is not None and n > UNIT*OLD[a] - IB: bad_old += 1
    if b is not None and n > UNIT*NEW[b] - IB: bad_new += 1
    if a != b: moved += 1
print("   requests the OLD table mis-sizes: %d" % bad_old)
print("   requests the NEW table mis-sizes: %d" % bad_new)
print("   requests that change class:       %d of %d" % (moved, S.MMAP_THRESHOLD-1))

print("\nthe index bit-trick must still land within one step of the right octave:")
off_by = {}
for n in range(1, S.MMAP_THRESHOLD, 7):
    b = s2c(n, NEW)
    if b is None: continue
    # the smallest class that fits
    want = next((i for i, c in enumerate(NEW) if UNIT*c - IB >= n), None)
    if want is not None:
        off_by[b - want] = off_by.get(b - want, 0) + 1
print("   chosen class minus smallest-fitting class: %s" % dict(sorted(off_by.items())))
print("   (0 = tight, 1 = one class of slack; a NEGATIVE value would be a bug)")
assert bad_new == 0, "the edited table mis-sizes %d requests" % bad_new
assert min(off_by) >= 0, "the edited table can choose a class that is too small"
print("\nOK: the edited table preserves the invariant and never under-sizes.")

# POSITIVE CONTROL: the mis-sizing check must be able to fire. Shrink one entry below what the
# octave arithmetic will send to it, and the invariant must break.
BROKEN = list(NEW); BROKEN[43] = NEW[43] - 8      # 65536 -> 65408 bytes
bad = sum(1 for n in range(1, S.MMAP_THRESHOLD)
          if (lambda b: b is not None and n > UNIT*BROKEN[b] - IB)(s2c(n, BROKEN)))
print("\nPOSITIVE CONTROL: with size_classes[43] shrunk by 8 units, mis-sized requests = %d" % bad)
assert bad > 0, "the mis-sizing check cannot fire -- it proves nothing about the edited table"
print("  -> the check fires, so the 0 above is a result and not a dead check")
