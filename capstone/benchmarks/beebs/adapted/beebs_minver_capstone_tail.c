/*
 * Capstone adapted tail for BEEBS `minver`.
 *
 * Upstream `verify_benchmark` returns -1 ("no verification").  minver inverts
 * the global 3x3 matrix `a` into `a_i` (with determinant `det`).  Single-
 * precision +,-,*,/ are correctly rounded, so the soft-float result is
 * bit-identical to a native float reference; we FNV-1a-checksum `a_i` + `det`
 * against that reference. The build script macro-renames the upstream stub.
 */
#undef verify_benchmark

extern float a_i[3][3];
extern float det;

/* Native reference (cc -O0 -fno-builtin): det=-16.6666718 */
#define MINVER_EXPECTED_FNV 0x628628bb22f4d413UL

int verify_benchmark(int res) {
  (void)res;
  unsigned long h = 1469598103934665603UL;
  const unsigned char *p = (const unsigned char *)&a_i[0][0];
  for (unsigned i = 0; i < sizeof(float) * 9; i++) {
    h ^= p[i];
    h *= 1099511628211UL;
  }
  const unsigned char *q = (const unsigned char *)&det;
  for (unsigned i = 0; i < sizeof(float); i++) {
    h ^= q[i];
    h *= 1099511628211UL;
  }
  return (h == MINVER_EXPECTED_FNV) ? 1 : 0;
}
