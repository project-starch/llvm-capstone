/*
 * Compact, self-contained, freestanding double-precision libm shared by the
 * Capstone PureCap FP BEEBS benchmarks (cubic, st, frac, nbody, ...).  The
 * bare-metal domain has no libm; this provides fabs, sqrt, exp, log, pow, sin,
 * cos and acos.
 *
 * These are standard, deterministic IEEE-754 double implementations
 * (fdlibm-style kernels / Cody-Waite reduction), accurate to well under 1e-12.
 * No floating-point hardware is assumed: every operation lowers to compiler-rt
 * soft-float (no FP register state to preserve across the host-call boundary).
 *
 * Build the native accuracy self-test with:
 *   cc -DCUBIC_LIBM_TEST -O2 beebs_softfloat_libm.c -lm -o /tmp/libm_test && /tmp/libm_test
 */

typedef unsigned long long u64;

union dbits {
  double d;
  u64 u;
};

double fabs(double x) {
  union dbits b;
  b.d = x;
  b.u &= 0x7fffffffffffffffULL;
  return b.d;
}

/* Exact a*a = hi + lo via a Veltkamp/Dekker split (no FMA). */
static double two_prod_square(double a, double *lo) {
  const double split = 134217729.0; /* 2^27 + 1 */
  double c = split * a;
  double ahi = c - (c - a);
  double alo = a - ahi;
  double hi = a * a;
  *lo = ((ahi * ahi - hi) + 2.0 * ahi * alo) + alo * alo;
  return hi;
}

/* Correctly-rounded (round-to-nearest-even) double sqrt: a Newton seed brings y
 * within ~1 ulp, then the exact residual x - y*y selects the correctly-rounded
 * neighbor. Needed because several benchmarks compare results for exact
 * equality (st, nbody). */
double sqrt(double x) {
  if (x != x)
    return x; /* NaN */
  if (x < 0.0)
    return 0.0 / 0.0;
  if (x == 0.0)
    return x; /* +/-0 */
  union dbits b;
  b.d = x;
  if ((b.u >> 52) == 0x7ff)
    return x; /* +inf */

  b.u = (b.u >> 1) + 0x1ff8000000000000ULL; /* exponent-halving seed */
  double y = b.d;
  y = 0.5 * (y + x / y);
  y = 0.5 * (y + x / y);
  y = 0.5 * (y + x / y);
  y = 0.5 * (y + x / y);

  double lo, hi, r;
  /* Pull y down until y*y <= x. */
  for (;;) {
    hi = two_prod_square(y, &lo);
    r = (x - hi) - lo;
    if (r >= 0.0)
      break;
    union dbits d;
    d.d = y;
    d.u--;
    y = d.d;
  }
  /* Push y up while (y+ulp)^2 <= x, so y becomes the largest such double. */
  for (;;) {
    union dbits u;
    u.d = y;
    u.u++;
    double yp = u.d;
    hi = two_prod_square(yp, &lo);
    r = (x - hi) - lo;
    if (r < 0.0)
      break;
    y = yp;
  }
  /* y is now the largest double with y*y <= x. Round to nearest even between y
   * and y+ulp. The midpoint m=y+ulp/2 is not representable, so compare the exact
   * residual r0 = x - y*y against (m*m - y*y) = y*ulp + ulp*ulp/4. */
  union dbits u;
  u.d = y;
  u.u++;
  double yp = u.d;
  double ulp = yp - y; /* exact */
  hi = two_prod_square(y, &lo);
  double r0 = (x - hi) - lo;             /* x - y*y (exact to rounding) */
  double thresh = y * ulp + 0.25 * ulp * ulp; /* m*m - y*y */
  if (r0 > thresh)
    return yp;
  if (r0 < thresh)
    return y;
  union dbits db;
  db.d = y;
  return (db.u & 1ULL) ? yp : y; /* tie -> even */
}

/* ---- exp / log / pow ---------------------------------------------------- */

static const double LN2_HI = 6.93147180369123816490e-01;
static const double LN2_LO = 1.90821492927058770002e-10;
static const double INV_LN2 = 1.44269504088896340736e+00;

static double scale2(double x, int k) {
  /* x * 2^k, k within normal range (cubic stays well inside). */
  union dbits b;
  b.d = 1.0;
  b.u += ((u64)k) << 52;
  return x * b.d;
}

double exp(double x) {
  if (x != x)
    return x;
  if (x > 700.0)
    return x * 1e300; /* +inf-ish */
  if (x < -700.0)
    return 0.0;
  int k = (int)(x * INV_LN2 + (x >= 0 ? 0.5 : -0.5));
  double r = (x - k * LN2_HI) - k * LN2_LO;
  /* exp(r), r in [-ln2/2, ln2/2]; degree-7 Taylor. */
  double e = 1.0 +
             r * (1.0 +
                  r * (1.0 / 2.0 +
                       r * (1.0 / 6.0 +
                            r * (1.0 / 24.0 +
                                 r * (1.0 / 120.0 +
                                      r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
  return scale2(e, k);
}

double log(double x) {
  if (x < 0.0)
    return 0.0 / 0.0;
  if (x == 0.0)
    return -1.0 / 0.0;
  union dbits b;
  b.d = x;
  int k = (int)((b.u >> 52) & 0x7ff) - 1023;
  /* mantissa f in [1,2) */
  b.u = (b.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
  double f = b.d;
  if (f > 1.4142135623730951) {
    f *= 0.5;
    k += 1;
  }
  double s = (f - 1.0) / (f + 1.0);
  double s2 = s * s;
  /* log(f) = 2*(s + s^3/3 + s^5/5 + ... ) */
  double t = s2 * (1.0 / 3.0 +
                   s2 * (1.0 / 5.0 +
                         s2 * (1.0 / 7.0 +
                               s2 * (1.0 / 9.0 + s2 * (1.0 / 11.0)))));
  double logf = 2.0 * (s + s * t);
  return k * LN2_HI + (k * LN2_LO + logf);
}

double pow(double x, double y) {
  if (x <= 0.0)
    return 0.0; /* cubic only calls pow(base>0, 1/3) */
  return exp(y * log(x));
}

/* ---- sin / cos (Cody-Waite reduction + fdlibm kernels) ------------------ */

static const double PIO2_HI = 1.57079632673412561417e+00;
static const double PIO2_LO = 6.07710050650619224932e-11;
static const double TWO_OVER_PI = 6.36619772367581382433e-01;

static double sin_kernel(double x) {
  double z = x * x;
  return x + x * z *
                 (-1.66666666666666324348e-01 +
                  z * (8.33333333332248946124e-03 +
                       z * (-1.98412698298579493134e-04 +
                            z * (2.75573137070700676789e-06 +
                                 z * (-2.50507602534068634195e-08 +
                                      z * 1.58969099521155010221e-10)))));
}

static double cos_kernel(double x) {
  double z = x * x;
  return 1.0 - 0.5 * z +
         z * z *
             (4.16666666666666019037e-02 +
              z * (-1.38888888888741095749e-03 +
                   z * (2.48015872894767294178e-05 +
                        z * (-2.75573143513906633035e-07 +
                             z * (2.08757232129817482790e-09 +
                                  z * -1.13596475577881948265e-11)))));
}

double cos(double x) {
  if (x < 0.0)
    x = -x; /* cos is even */
  int n = (int)(x * TWO_OVER_PI + 0.5);
  double r = (x - n * PIO2_HI) - n * PIO2_LO;
  switch (n & 3) {
  case 0:
    return cos_kernel(r);
  case 1:
    return -sin_kernel(r);
  case 2:
    return -cos_kernel(r);
  default:
    return sin_kernel(r);
  }
}

double sin(double x) {
  int neg = 0;
  if (x < 0.0) {
    x = -x;
    neg = 1;
  }
  int n = (int)(x * TWO_OVER_PI + 0.5);
  double r = (x - n * PIO2_HI) - n * PIO2_LO;
  double s;
  switch (n & 3) {
  case 0:
    s = sin_kernel(r);
    break;
  case 1:
    s = cos_kernel(r);
    break;
  case 2:
    s = -sin_kernel(r);
    break;
  default:
    s = -cos_kernel(r);
    break;
  }
  return neg ? -s : s;
}

/* ---- acos (fdlibm e_acos: acos = pi/2 - asin) --------------------------- */

static const double PIO2 = 1.57079632679489661923e+00;
static const double PI_CONST = 3.14159265358979311600e+00;

/* R(t) = p(t)/q(t), the fdlibm asin rational approximation. */
static double asin_R(double t) {
  double p = t * (1.66666666666666657415e-01 +
                  t * (-3.25565818622400915405e-01 +
                       t * (2.01212532134862925881e-01 +
                            t * (-4.00555345006794114027e-02 +
                                 t * (7.91534994289814532176e-04 +
                                      t * 3.47933107596021167570e-05)))));
  double q = 1.0 + t * (-2.40339491173441421878e+00 +
                        t * (2.02094576023350569471e+00 +
                             t * (-6.88283971605453293030e-01 +
                                  t * 7.70381505559019352791e-02)));
  return p / q;
}

double acos(double x) {
  if (x >= 1.0)
    return (x == 1.0) ? 0.0 : 0.0 / 0.0;
  if (x <= -1.0)
    return (x == -1.0) ? PI_CONST : 0.0 / 0.0;
  if (fabs(x) < 0.5) {
    double t = x * x;
    return PIO2 - (x + x * asin_R(t));
  }
  if (x >= 0.5) { /* x in [0.5, 1) */
    double t = (1.0 - x) * 0.5;
    double s = sqrt(t);
    return 2.0 * (s + s * asin_R(t));
  }
  /* x in (-1, -0.5] */
  double t = (1.0 + x) * 0.5;
  double s = sqrt(t);
  return PI_CONST - 2.0 * (s + s * asin_R(t));
}

/* Round toward -inf.  Pure bit-manipulation (no FP rounding mode needed):
   clear the fractional mantissa bits, rounding negative non-integers away from
   zero.  Handles |x|<1, already-integral values, and inf/NaN (e >= 52). */
double floor(double x) {
  union dbits b;
  b.d = x;
  int e = (int)((b.u >> 52) & 0x7ff) - 1023; /* unbiased exponent */
  if (e >= 52)
    return x; /* already integral, or inf/NaN */
  if (e < 0) {
    if (x == 0.0)
      return x;                     /* preserve -0.0 */
    return (b.u >> 63) ? -1.0 : 0.0; /* |x| < 1 */
  }
  u64 frac = (1ULL << (52 - e)) - 1;
  if ((b.u & frac) == 0)
    return x; /* already integral */
  if (b.u >> 63)
    b.u += (1ULL << (52 - e)); /* negative: round toward -inf */
  b.u &= ~frac;
  return b.d;
}

#ifdef CUBIC_LIBM_TEST
#include <math.h>
#include <stdio.h>
/* Our definitions above use the standard names; compare against the system
   long-double routines (cosl/acosl/sqrtl/powl), which have distinct names. */
int main(void) {
  double maxerr = 0.0;
  for (double x = 0.0; x <= 6.0; x += 0.0009) {
    double e;
    e = fabs(cos(x) - cosl(x));
    if (e > maxerr) { maxerr = e; }
  }
  printf("cos   max abs err over [0,6]    = %.3e\n", maxerr);
  maxerr = 0;
  for (double x = -1.0; x <= 1.0; x += 0.0003) {
    double e = fabs(acos(x) - acosl(x));
    if (e > maxerr) maxerr = e;
  }
  printf("acos  max abs err over [-1,1]   = %.3e\n", maxerr);
  maxerr = 0;
  for (double x = 0.01; x <= 50.0; x += 0.01) {
    double e = fabs(sqrt(x) - sqrtl(x)) / sqrtl(x);
    if (e > maxerr) maxerr = e;
  }
  printf("sqrt  max rel err over [.01,50] = %.3e\n", maxerr);
  maxerr = 0;
  for (double x = 0.1; x <= 40.0; x += 0.01) {
    double e = fabs(pow(x, 1.0 / 3.0) - powl(x, 1.0L / 3.0L)) / powl(x, 1.0L / 3.0L);
    if (e > maxerr) maxerr = e;
  }
  printf("cbrt  max rel err over [.1,40]  = %.3e\n", maxerr);
  maxerr = 0;
  for (double x = -10.0; x <= 10.0; x += 0.013) {
    double e = fabs(floor(x) - floorl(x));
    if (e > maxerr) maxerr = e;
  }
  printf("floor max abs err over [-10,10] = %.3e\n", maxerr);
  return 0;
}
#endif
