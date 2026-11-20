#!/usr/bin/env python3
"""large-port.py DIR: GC objects of 4 KiB or more as lifetimes too (nested lane, 2026-11-20). The collaborator's port
derives a GC object only below 4 KiB, where every bound is exact. sublet_exact (added to sublet-port.h, following QEMU
9e69171f1c's cap_compress: a region of 4 KiB or more is exact when its base and end are multiples of its granule,
2^(msb(len) - 9) for len of 8 KiB and more, 8 bytes below) decides instead, at allocation, at release and where a
free list renews a GC object, so a GC object has a lifetime of its own exactly when it is exactly representable.
ggc-page.c's large objects are powers of two on page boundaries, so all of them are; an obstack object of 4 KiB or
more is derived where its bounds happen to be exact (none on train01). Every edit is asserted."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:60], s.count(old))
    p.write_text(s.replace(old, new))
edit('sublet-port.h', """#define SUBLET_EXACT_LIMIT 4096
""", """#define SUBLET_EXACT_LIMIT 4096

/* Whether CDERIVE holds [BASE, BASE + LEN) exactly (QEMU 9e69171f1c cap_compress): every LEN below 4 KiB; above, BASE
   and BASE + LEN must be multiples of the granule 2^(E + 3), E = msb(LEN) - 12 (0 below 8 KiB), one exponent more
   where the rounded span would reach 1,024 granules.  */
static inline int
sublet_exact (unsigned long base, unsigned long len)
{
  unsigned long e, g, q, sr, lr;
  if (len < SUBLET_EXACT_LIMIT)
    return 1;
  e = (len >> 13) ? (unsigned long) (63 - __builtin_clzl (len)) - 12 : 0;
  g = 1ul << (e + 3);
  q = len >> (e + 3); sr = base & (g - 1); lr = len & (g - 1);
  if ((q == 1023 && (sr | lr) != 0) || (q == 1022 && sr + lr > g))
    g <<= 1;
  return (base & (g - 1)) == 0 && ((base + len) & (g - 1)) == 0;
}
""")
edit('ggc-page.c', """  if (object_size < SUBLET_EXACT_LIMIT)
    result = sublet_derive (result, object_size);""", """  if (sublet_exact ((unsigned long) result, object_size))
    result = sublet_derive (result, object_size);""")
edit('ggc-page.c', """  if (size < SUBLET_EXACT_LIMIT)
    sublet_release (p);""", """  if (sublet_exact ((unsigned long) p, size))
    sublet_release (p);""")
edit('tree-phinodes.c', """  if (ggc_get_size (phi) < SUBLET_EXACT_LIMIT)
    phi = (gimple) sublet_renew (phi);""", """  if (sublet_exact ((unsigned long) phi, ggc_get_size (phi)))
    phi = (gimple) sublet_renew (phi);""")
edit('obstack.c', """  if (size >= SUBLET_EXACT_LIMIT)
    return object;""", """  if (!sublet_exact ((unsigned long) object, size))
    return object;	/* an obstack object of 4 KiB or more keeps its chunk's lifetime unless exact */""")
print('large GC and obstack objects ported')
