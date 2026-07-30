#ifndef GPW16B_KERNEL_H
#define GPW16B_KERNEL_H
/* Same SIZE as gpw16 (16 words, 64 B) and the same element count, but the stored value
   is decoupled from the index so the compiler does not reuse the address register as
   the loop counter.
     gpw16  stores i+1  -> `cincoffset a4, a3, a4` makes a4 a CAPABILITY, then
                           `movc a4, a6` puts a scalar back into that same register and
                           the next iteration uses it as an integer index. FAILS.
     gpsz   stores i*3+7 -> index stays in its own integer register, never aliased with
                           a capability. PASSES at 64 elements.
   This rung is gpsz's shape at gpw16's size, isolating the register-reuse pattern from
   both size and element count. Prediction: PASSES where gpw16 fails. */
static unsigned g[16];
static unsigned gpw16b_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<16;i++) g[i] = (unsigned)i*3u+7u;
  for (int i=0;i<16;i++) { h ^= g[i]; h *= 16777619u; }
  return h;
}
#endif
