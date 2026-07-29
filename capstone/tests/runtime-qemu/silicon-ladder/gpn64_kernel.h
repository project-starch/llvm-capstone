#ifndef GPN64_KERNEL_H
#define GPN64_KERNEL_H
/* COUNT-ONLY bisection for the multi-global defect. Every global is the same shape
   (4 words, zero-init) so the sole variable is HOW MANY. Single-global rungs all pass on
   silicon and gpstress (6 mixed globals) returns wrong data, so the fault is in
   per-global bookkeeping -- the record walk, the storage carve, or the cap-table index --
   not in any one initializer path. This says at what count it starts. */
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
static unsigned g32[4];
static unsigned g33[4];
static unsigned g34[4];
static unsigned g35[4];
static unsigned g36[4];
static unsigned g37[4];
static unsigned g38[4];
static unsigned g39[4];
static unsigned g40[4];
static unsigned g41[4];
static unsigned g42[4];
static unsigned g43[4];
static unsigned g44[4];
static unsigned g45[4];
static unsigned g46[4];
static unsigned g47[4];
static unsigned g48[4];
static unsigned g49[4];
static unsigned g50[4];
static unsigned g51[4];
static unsigned g52[4];
static unsigned g53[4];
static unsigned g54[4];
static unsigned g55[4];
static unsigned g56[4];
static unsigned g57[4];
static unsigned g58[4];
static unsigned g59[4];
static unsigned g60[4];
static unsigned g61[4];
static unsigned g62[4];
static unsigned g63[4];
static unsigned gpn64_compute(void) {
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
  for (int i=0;i<4;i++) g32[i] = (unsigned)(i + 32*7 + 1);
  for (int i=0;i<4;i++) g33[i] = (unsigned)(i + 33*7 + 1);
  for (int i=0;i<4;i++) g34[i] = (unsigned)(i + 34*7 + 1);
  for (int i=0;i<4;i++) g35[i] = (unsigned)(i + 35*7 + 1);
  for (int i=0;i<4;i++) g36[i] = (unsigned)(i + 36*7 + 1);
  for (int i=0;i<4;i++) g37[i] = (unsigned)(i + 37*7 + 1);
  for (int i=0;i<4;i++) g38[i] = (unsigned)(i + 38*7 + 1);
  for (int i=0;i<4;i++) g39[i] = (unsigned)(i + 39*7 + 1);
  for (int i=0;i<4;i++) g40[i] = (unsigned)(i + 40*7 + 1);
  for (int i=0;i<4;i++) g41[i] = (unsigned)(i + 41*7 + 1);
  for (int i=0;i<4;i++) g42[i] = (unsigned)(i + 42*7 + 1);
  for (int i=0;i<4;i++) g43[i] = (unsigned)(i + 43*7 + 1);
  for (int i=0;i<4;i++) g44[i] = (unsigned)(i + 44*7 + 1);
  for (int i=0;i<4;i++) g45[i] = (unsigned)(i + 45*7 + 1);
  for (int i=0;i<4;i++) g46[i] = (unsigned)(i + 46*7 + 1);
  for (int i=0;i<4;i++) g47[i] = (unsigned)(i + 47*7 + 1);
  for (int i=0;i<4;i++) g48[i] = (unsigned)(i + 48*7 + 1);
  for (int i=0;i<4;i++) g49[i] = (unsigned)(i + 49*7 + 1);
  for (int i=0;i<4;i++) g50[i] = (unsigned)(i + 50*7 + 1);
  for (int i=0;i<4;i++) g51[i] = (unsigned)(i + 51*7 + 1);
  for (int i=0;i<4;i++) g52[i] = (unsigned)(i + 52*7 + 1);
  for (int i=0;i<4;i++) g53[i] = (unsigned)(i + 53*7 + 1);
  for (int i=0;i<4;i++) g54[i] = (unsigned)(i + 54*7 + 1);
  for (int i=0;i<4;i++) g55[i] = (unsigned)(i + 55*7 + 1);
  for (int i=0;i<4;i++) g56[i] = (unsigned)(i + 56*7 + 1);
  for (int i=0;i<4;i++) g57[i] = (unsigned)(i + 57*7 + 1);
  for (int i=0;i<4;i++) g58[i] = (unsigned)(i + 58*7 + 1);
  for (int i=0;i<4;i++) g59[i] = (unsigned)(i + 59*7 + 1);
  for (int i=0;i<4;i++) g60[i] = (unsigned)(i + 60*7 + 1);
  for (int i=0;i<4;i++) g61[i] = (unsigned)(i + 61*7 + 1);
  for (int i=0;i<4;i++) g62[i] = (unsigned)(i + 62*7 + 1);
  for (int i=0;i<4;i++) g63[i] = (unsigned)(i + 63*7 + 1);
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
  for (int i=0;i<4;i++) { h ^= g32[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g33[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g34[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g35[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g36[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g37[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g38[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g39[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g40[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g41[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g42[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g43[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g44[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g45[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g46[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g47[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g48[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g49[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g50[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g51[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g52[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g53[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g54[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g55[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g56[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g57[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g58[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g59[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g60[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g61[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g62[i]; h *= 16777619u; }
  for (int i=0;i<4;i++) { h ^= g63[i]; h *= 16777619u; }
  return h;
}
#endif
