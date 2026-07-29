#ifndef GPN4_KERNEL_H
#define GPN4_KERNEL_H
/* COUNT-ONLY bisection for the multi-global defect. Every global is the same shape
   (4 words, zero-init) so the sole variable is HOW MANY. Single-global rungs all pass on
   silicon and gpstress (6 mixed globals) returns wrong data, so the fault is in
   per-global bookkeeping -- the record walk, the storage carve, or the cap-table index --
   not in any one initializer path. This says at what count it starts. */
static unsigned g0[4];
static unsigned g1[4];
static unsigned g2[4];
static unsigned g3[4];
static unsigned gpn4_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<4;i++) g0[i] = (unsigned)(i + 0*7 + 1);
  for (int i=0;i<4;i++) g1[i] = (unsigned)(i + 1*7 + 1);
  for (int i=0;i<4;i++) g2[i] = (unsigned)(i + 2*7 + 1);
  for (int i=0;i<4;i++) g3[i] = (unsigned)(i + 3*7 + 1);
  for (int i=0;i<4;i++) { h ^= g0[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g1[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g2[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g3[i]; h *= 16777619u; }
  return h;
}
#endif
