/*
 * Capstone adapted harness for BEEBS `cubic`.
 *
 * Upstream `basicmath_small.c` provides initialise/benchmark/verify, but its
 * `verify_benchmark` returns -1 ("no verification").  We do not compile it;
 * instead this tail drives `SolveCubic` (from the double-pivoted libcubic.c)
 * and verifies against the *exact mathematical roots* of the documented
 * polynomials — a strong correctness oracle for the whole soft-float + libm
 * stack:
 *   x^3 - 10.5x^2 + 32x - 30 = (x-2)(x-2.5)(x-6)  -> 3 real roots {2, 2.5, 6}
 *   x^3 -  4.5x^2 + 17x - 30                       -> 1 real root  {2.5}
 */

extern void SolveCubic(double a, double b, double c, double d, int *solutions,
                       double *x);

static int cub_sol1, cub_sol2;
static double cub_x1[3], cub_x2[3];

void initialise_benchmark(void) {}

int benchmark(void) {
  double x[48];
  int sol;

  /* Documented calls with known roots (checked in verify_benchmark). */
  SolveCubic(1.0, -10.5, 32.0, -30.0, &cub_sol1, cub_x1);
  SolveCubic(1.0, -4.5, 17.0, -30.0, &cub_sol2, cub_x2);

  /* Remaining upstream calls for execution coverage (results unchecked). */
  SolveCubic(1.0, -3.5, 22.0, -31.0, &sol, x);
  SolveCubic(1.0, -13.7, 1.0, -35.0, &sol, x);
  for (double a = 1; a < 3; a++)
    for (double b = 10; b > 8; b--)
      for (double c = 5; c < 6; c += 0.5)
        for (double d = -1; d > -3; d--)
          SolveCubic(a, b, c, d, &sol, x);

  return 0;
}

static int approx(double a, double b) {
  double diff = a - b;
  if (diff < 0)
    diff = -diff;
  return diff < 1e-4;
}

static int has_root(const double *xs, double r) {
  for (int i = 0; i < 3; i++)
    if (approx(xs[i], r))
      return 1;
  return 0;
}

int verify_benchmark(int res) {
  (void)res;
  if (cub_sol1 != 3)
    return 0;
  if (!has_root(cub_x1, 2.0) || !has_root(cub_x1, 2.5) || !has_root(cub_x1, 6.0))
    return 0;
  if (cub_sol2 != 1)
    return 0;
  if (!approx(cub_x2[0], 2.5))
    return 0;
  return 1;
}
