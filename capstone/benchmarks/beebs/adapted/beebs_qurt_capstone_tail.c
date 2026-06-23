/*
 * Capstone adapted tail for BEEBS `qurt`.
 *
 * Upstream `verify_benchmark` returns -1.  `benchmark()` solves three quadratics;
 * the last (in3 = {1,-4,8}, discriminant < 0) leaves complex-conjugate roots in
 * the globals: flag=-1, x1 = 2 + 2i, x2 = 2 - 2i.  qurt uses its own approximate
 * `qurt_sqrt`, so we check within a tolerance.  The build script macro-renames
 * the upstream stub.
 */
#undef verify_benchmark

extern float x1[2], x2[2];
extern int flag;

static int qurt_approx(float a, float b) {
  float d = a - b;
  if (d < 0)
    d = -d;
  return d < 1e-3f;
}

int verify_benchmark(int res) {
  (void)res;
  if (flag != -1)
    return 0;
  if (!qurt_approx(x1[0], 2.0f) || !qurt_approx(x1[1], 2.0f))
    return 0;
  if (!qurt_approx(x2[0], 2.0f) || !qurt_approx(x2[1], -2.0f))
    return 0;
  return 1;
}
