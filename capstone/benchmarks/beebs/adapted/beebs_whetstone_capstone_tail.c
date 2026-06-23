/*
 * Capstone adapted tail for BEEBS `whetstone`.
 *
 * Upstream `verify_benchmark` returns -1 and there is no built-in check; the
 * per-module results flow only through POUT (printf, gated on PRINTOUT).  We
 * build the domain with -DPRINTOUT so every `IF(JJ==II)POUT(...)` fires, strip
 * the upstream printf POUT definition, and provide this capturing POUT that
 * folds each module's four double outputs into a running FNV-1a checksum.
 * `verify_benchmark` compares that checksum against a host reference computed
 * from the *same* source + the *same* soft-float libm (gcc -O0
 * -ffp-contract=off): all accumulators are double, and IEEE-754 double ops are
 * bit-identical between host hardware float and target compiler-rt soft-float,
 * so the comparison is exact (the libm's absolute accuracy is irrelevant — both
 * sides use the same `sin/cos/atan/log/exp/sqrt`).
 *
 * The build macro-renames the upstream verify stub; we re-take the name here.
 */
#undef verify_benchmark

static unsigned long whet_fnv = 1469598103934665603UL;

static void whet_fold(double v) {
  union {
    double d;
    unsigned long u;
  } b;
  b.d = v;
  for (int i = 0; i < 8; i++) {
    whet_fnv ^= (unsigned char)(b.u >> (i * 8));
    whet_fnv *= 1099511628211UL;
  }
}

void POUT(long N, long J, long K, double X1, double X2, double X3, double X4) {
  (void)N;
  (void)J;
  (void)K;
  whet_fold(X1);
  whet_fold(X2);
  whet_fold(X3);
  whet_fold(X4);
}

int verify_benchmark(int res) {
  (void)res;
  return whet_fnv == 0x2f975c4609a1bfbbUL;
}
