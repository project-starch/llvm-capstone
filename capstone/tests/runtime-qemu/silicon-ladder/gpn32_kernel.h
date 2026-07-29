#ifndef GPN32_KERNEL_H
#define GPN32_KERNEL_H
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
static unsigned g16[4];
static unsigned g17[4];
static unsigned g18[4];
static unsigned g19[4];
static unsigned g20[4];
static unsigned g21[4];
static unsigned g22[4];
static unsigned g23[4];
static unsigned g24[4];
static unsigned g25[4];
static unsigned g26[4];
static unsigned g27[4];
static unsigned g28[4];
static unsigned g29[4];
static unsigned g30[4];
static unsigned g31[4];
static unsigned gpn32_compute(void) {
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
  for (int i=0;i<4;i++) g16[i] = (unsigned)(i + 16*7 + 1);
  for (int i=0;i<4;i++) g17[i] = (unsigned)(i + 17*7 + 1);
  for (int i=0;i<4;i++) g18[i] = (unsigned)(i + 18*7 + 1);
  for (int i=0;i<4;i++) g19[i] = (unsigned)(i + 19*7 + 1);
  for (int i=0;i<4;i++) g20[i] = (unsigned)(i + 20*7 + 1);
  for (int i=0;i<4;i++) g21[i] = (unsigned)(i + 21*7 + 1);
  for (int i=0;i<4;i++) g22[i] = (unsigned)(i + 22*7 + 1);
  for (int i=0;i<4;i++) g23[i] = (unsigned)(i + 23*7 + 1);
  for (int i=0;i<4;i++) g24[i] = (unsigned)(i + 24*7 + 1);
  for (int i=0;i<4;i++) g25[i] = (unsigned)(i + 25*7 + 1);
  for (int i=0;i<4;i++) g26[i] = (unsigned)(i + 26*7 + 1);
  for (int i=0;i<4;i++) g27[i] = (unsigned)(i + 27*7 + 1);
  for (int i=0;i<4;i++) g28[i] = (unsigned)(i + 28*7 + 1);
  for (int i=0;i<4;i++) g29[i] = (unsigned)(i + 29*7 + 1);
  for (int i=0;i<4;i++) g30[i] = (unsigned)(i + 30*7 + 1);
  for (int i=0;i<4;i++) g31[i] = (unsigned)(i + 31*7 + 1);
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
  for (int i=0;i<4;i++) { h ^= g16[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g17[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g18[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g19[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g20[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g21[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g22[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g23[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g24[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g25[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g26[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g27[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g28[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g29[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g30[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g31[i]; h *= 16777619u; }
  return h;
}
#endif
