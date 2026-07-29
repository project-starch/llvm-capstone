#ifndef GPN16_KERNEL_H
#define GPN16_KERNEL_H
/* COUNT-ONLY bisection, see gpn2/gpn8/gpn64. */
static unsigned g0[4];
static unsigned g1[4];
static unsigned g2[4];
static unsigned g3[4];
static unsigned g4[4];
static unsigned g5[4];
static unsigned g6[4];
static unsigned g7[4];
static unsigned g8[4];
static unsigned g9[4];
static unsigned g10[4];
static unsigned g11[4];
static unsigned g12[4];
static unsigned g13[4];
static unsigned g14[4];
static unsigned g15[4];
static unsigned gpn16_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<4;i++) g0[i] = (unsigned)(i + 0*7 + 1);
  for (int i=0;i<4;i++) g1[i] = (unsigned)(i + 1*7 + 1);
  for (int i=0;i<4;i++) g2[i] = (unsigned)(i + 2*7 + 1);
  for (int i=0;i<4;i++) g3[i] = (unsigned)(i + 3*7 + 1);
  for (int i=0;i<4;i++) g4[i] = (unsigned)(i + 4*7 + 1);
  for (int i=0;i<4;i++) g5[i] = (unsigned)(i + 5*7 + 1);
  for (int i=0;i<4;i++) g6[i] = (unsigned)(i + 6*7 + 1);
  for (int i=0;i<4;i++) g7[i] = (unsigned)(i + 7*7 + 1);
  for (int i=0;i<4;i++) g8[i] = (unsigned)(i + 8*7 + 1);
  for (int i=0;i<4;i++) g9[i] = (unsigned)(i + 9*7 + 1);
  for (int i=0;i<4;i++) g10[i] = (unsigned)(i + 10*7 + 1);
  for (int i=0;i<4;i++) g11[i] = (unsigned)(i + 11*7 + 1);
  for (int i=0;i<4;i++) g12[i] = (unsigned)(i + 12*7 + 1);
  for (int i=0;i<4;i++) g13[i] = (unsigned)(i + 13*7 + 1);
  for (int i=0;i<4;i++) g14[i] = (unsigned)(i + 14*7 + 1);
  for (int i=0;i<4;i++) g15[i] = (unsigned)(i + 15*7 + 1);
  for (int i=0;i<4;i++) { h ^= g0[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g1[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g2[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g3[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g4[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g5[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g6[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g7[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g8[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g9[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g10[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g11[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g12[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g13[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g14[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g15[i]; h *= 16777619u; }
  return h;
}
#endif
