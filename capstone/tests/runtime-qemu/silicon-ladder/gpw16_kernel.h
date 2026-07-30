#ifndef GPW16_KERNEL_H
#define GPW16_KERNEL_H
/* SIZE SWEEP across the 16-byte point, kernel shape held fixed.
   Every silicon rung that fails has globals of EXACTLY 16 bytes and nothing else;
   every rung that passes has none (passing sizes seen: 4, 13, 256, 512, 2400).
   16 bytes is one capability and, on this RTL, exactly one dcache line
   (DcacheLineWidth = 128 bits, one cap_tag_q bit per line). This rung is 64 bytes.
   Prediction on the board: gpw4 (16 B) FAILS, gpw2/gpw8/gpw16 PASS. */
static unsigned g[16];
static unsigned gpw16_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<16;i++) g[i] = (unsigned)(i + 1);
  for (int i=0;i<16;i++) { h ^= g[i]; h *= 16777619u; }
  return h;
}
#endif
