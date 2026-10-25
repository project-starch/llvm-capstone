/* Used by run.sh; see there. */
/* For representable bounds [b,e) and a source cursor inside them, is every cursor in [b-2048, e+2048] representable
 * under QEMU's R-60 predicate (capstone_cursor_representable)? Reports the worst slack below and above. */
#include <stdio.h>
#include <stdlib.h>
#include "cap_compress.h"
static uint64_t rng = 88172645463325252ull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
int main(int argc, char **argv) {
  long n = argc > 1 ? atol(argv[1]) : 200000, regions = 0, skipped = 0, bad = 0;
  uint64_t worst_lo = ~0ull, worst_hi = ~0ull;
  for (long i = 0; i < n; i++) {
    unsigned lg = 4 + xr() % 30;                       /* lengths 16 B .. 2^33 */
    uint64_t len = (1ull << lg) + (xr() % (1ull << lg));
    len &= ~15ull; if (!len) len = 16;
    uint64_t base = (0x80000000ull + (xr() % (1ull << 36))) & ~15ull;
    capfat_t c = {0}; c.type = CAP_TYPE_NONLIN; c.perms = CAP_PERMS_RW;
    c.bounds.base = base; c.bounds.end = base + len;
    int cur = xr() % 3;
    c.bounds.cursor = cur == 0 ? base : cur == 1 ? base + (xr() % len) : base + len - 8;
    uint64_t lo, hi; capfat_t d;
    cap_compress(&c, &lo, &hi); cap_uncompress(lo, hi, &d);
    if (d.bounds.base != c.bounds.base || d.bounds.end != c.bounds.end) { skipped++; continue; } /* not exact: skip */
    regions++;
    uint64_t b = d.bounds.base, e = d.bounds.end, s;
    for (s = 1; s <= 2048; s++) if (!capstone_cursor_representable(&c, b - s)) break;
    if (s - 1 < worst_lo) worst_lo = s - 1;
    for (s = 0; s <= 2048; s++) if (!capstone_cursor_representable(&c, e + s)) break;
    if (s < worst_hi) worst_hi = s;   /* cursors e .. e+s-1 representable */
    if (worst_lo < 2048 || worst_hi <= 2048) {
      if (bad++ < 5) printf("SHORT base=%#lx len=%#lx cursor=%#lx below=%lu above=%lu\n", (unsigned long)b,
                            (unsigned long)(e - b), (unsigned long)c.bounds.cursor, (unsigned long)worst_lo, (unsigned long)worst_hi);
      worst_lo = worst_hi = ~0ull;
    }
  }
  printf("regions %ld (inexact requests skipped %ld) short %ld; worst slack below %lu above %lu (cap 2048/2049)\n",
         regions, skipped, bad, (unsigned long)worst_lo, (unsigned long)worst_hi);
  return bad != 0;
}
