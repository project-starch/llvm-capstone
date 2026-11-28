#!/usr/bin/env python3
"""large-count.py DIR: COUNTING BUILD (nested lane, 2026-11-20). Tallies, at exit, the GC and obstack objects of 4 KiB
or more that the port leaves with their page's or chunk's lifetime, and how many of them the reference format could
bound exactly where they are (sublet_exact below follows QEMU 9e69171f1c's cap_compress). Prints one SUBLET_TALLY line.
Every edit is asserted."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:60], s.count(old))
    p.write_text(s.replace(old, new))
EXACT = r"""
/* COUNTING BUILD: whether [base, base + len) is a bound the reference format holds exactly (cap_compress). */
static int
sublet_exact (unsigned long base, unsigned long len)
{
  unsigned long e, g, q, sr, lr;
  if (len < 4096)
    return 1;
  e = (len >> 13) ? (unsigned long) (63 - __builtin_clzl (len)) - 12 : 0;
  g = 1ul << (e + 3);
  q = len >> (e + 3); sr = base & (g - 1); lr = len & (g - 1);
  if ((q == 1023 && (sr | lr) != 0) || (q == 1022 && sr + lr > g))
    g <<= 1;
  return (base & (g - 1)) == 0 && ((base + len) & (g - 1)) == 0;
}
"""
edit('ggc-page.c', '#include "sublet-port.h"\n', '#include "sublet-port.h"\n' + EXACT + r"""
unsigned long sublet_large_tally[8];	/* [0] ggc count [1] ggc bytes [2] ggc exact [3] obstack count [4] bytes [5] exact */
static void
sublet_large_print (void)
{
  fprintf (stderr, "SUBLET_TALLY ggc_large=%lu ggc_large_bytes=%lu ggc_large_exact=%lu obstack_large=%lu obstack_large_bytes=%lu obstack_large_exact=%lu\n",
	   sublet_large_tally[0], sublet_large_tally[1], sublet_large_tally[2],
	   sublet_large_tally[3], sublet_large_tally[4], sublet_large_tally[5]);
}
""")
edit('ggc-page.c', """  if (object_size < SUBLET_EXACT_LIMIT)
    result = sublet_derive (result, object_size);""", """  if (object_size < SUBLET_EXACT_LIMIT)
    result = sublet_derive (result, object_size);
  else
    {
      static int registered;
      if (!registered)
	{
	  registered = 1;
	  atexit (sublet_large_print);
	}
      sublet_large_tally[0]++;
      sublet_large_tally[1] += object_size;
      sublet_large_tally[2] += sublet_exact ((unsigned long) result, object_size);
    }""")
edit('obstack.c', """  if (size >= SUBLET_EXACT_LIMIT)
    return object;""", """  if (size >= SUBLET_EXACT_LIMIT)
    {
      extern unsigned long sublet_large_tally[8];
      unsigned long base = (unsigned long) object, len = size;
      unsigned long e = (len >> 13) ? (unsigned long) (63 - __builtin_clzl (len)) - 12 : 0;
      unsigned long g = 1ul << (e + 3), q = len >> (e + 3), sr = base & (g - 1), lr = len & (g - 1);
      if ((q == 1023 && (sr | lr) != 0) || (q == 1022 && sr + lr > g))
	g <<= 1;
      sublet_large_tally[3]++;
      sublet_large_tally[4] += size;
      sublet_large_tally[5] += (base & (g - 1)) == 0 && ((base + len) & (g - 1)) == 0;
      return object;
    }""")
print('large-object counting build')
