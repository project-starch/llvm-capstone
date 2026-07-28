#ifndef BEEBS_AHA_MONT64_KERNEL_H
#define BEEBS_AHA_MONT64_KERNEL_H
/* Silicon-ladder rung: BEEBS `aha-mont64` -- Montgomery modular multiplication.
 *
 * Source: BEEBS `src/aha-mont64/mont64.c` (Hacker's Delight, public domain; BEEBS
 * harness GPLv3). mulul64 / modul64 / montmul / xbinGCD are VERBATIM, including the
 * branch-free conditional subtract and the `volatile` result temporaries. Only the
 * driver differs.
 *
 * WHY THIS ONE. It is the cleanest possible test of the mechanism the ladder is
 * trying to establish: **overhead is a property of DATA ACCESS, not execution.**
 * mont64 has NO arrays and NO tables -- 24 B of scalar globals (upstream's three
 * inputs) and otherwise a dozen 64-bit locals in registers. `beebs_cover` already probes
 * the no-data end of that axis, but cover is control-flow (180 switch dispatches);
 * this is straight-line 64-bit arithmetic. If the mechanism claim holds, both land
 * near 1.00x for completely different execution profiles, which is a much stronger
 * statement than one rung making it alone.
 *
 * SHAPE SCREEN (this is a shape argument, not a prediction -- shape screening was
 * measured NON-predictive on 2026-07-28, passes and fails land in the same bucket):
 *   - C-4/C-5 (large read-only data delivery, 4 KiB code window): 24 B of
 *     initialized data (vs a 256 B copy-path threshold) and ~1 KB of code at -O1.
 *     Should not come near either.
 *   - R-1 (register-indexed load with an intervening store to the same object):
 *     there is no array to store into. The only memory traffic is the stack.
 *
 * WHAT IT ADDS to the mix. Every rung so far is integer arithmetic over arrays,
 * calls, or branches. This is 64-bit multiply/shift/carry-chain dominated: modul64
 * is a 64-iteration shift-and-subtract, xbinGCD a 64-iteration binary GCD, and
 * montmul three 64x64->128 multiplies. Long dependency chains, no memory.
 *
 * ADAPTATION, and it is the only one: upstream `benchmark()` returns an ERROR COUNT
 * (0 on success), which is useless as a checksum -- it cannot distinguish a correct
 * run from a run that silently computed nothing. This folds the actual computed
 * quantities (p, p1, rinv, mprime) into an FNV hash over MONT_REPS distinct inputs,
 * so a miscompute changes the returned value. The oracle is the same header compiled
 * natively, exactly like every other rung. Inputs are re-derived per rep with the
 * kernel's OWN modul64, so `a`,`b` < `m` stays true by construction and no libcall
 * (__umoddi3) is needed. `m` is fixed because it must be odd. */

typedef unsigned long long mont_u64;
typedef long long          mont_i64;

#define MONT_REPS 32

/* ---------------------------- mulul64 ----------------------------- */
/* 64 * 64 ==> 128, Knuth Algorithm M. VERBATIM. */
static void mont_mulul64(mont_u64 u, mont_u64 v, mont_u64 *whi, mont_u64 *wlo) {
  mont_u64 u0, u1, v0, v1, k, t;
  mont_u64 w0, w1, w2;

  u1 = u >> 32; u0 = u & 0xFFFFFFFF;
  v1 = v >> 32; v0 = v & 0xFFFFFFFF;

  t = u0*v0;
  w0 = t & 0xFFFFFFFF;
  k = t >> 32;

  t = u1*v0 + k;
  w1 = t & 0xFFFFFFFF;
  w2 = t >> 32;

  t = u0*v1 + w1;
  k = t >> 32;

  *wlo = (t << 32) + w0;
  *whi = u1*v1 + w2 + k;
}

/* ---------------------------- modul64 ----------------------------- */
/* Divides (x || y) by z giving the remainder. Must have x < z. VERBATIM. */
static mont_u64 mont_modul64(mont_u64 x, mont_u64 y, mont_u64 z) {
  mont_i64 i, t;

  for (i = 1; i <= 64; i++) {           /* Do 64 times. */
    t = (mont_i64)x >> 63;              /* All 1's if x(63) = 1. */
    x = (x << 1) | (y >> 63);           /* Shift x || y left */
    y = y << 1;                         /* one bit. */
    if (((mont_u64)((mont_i64)x | t)) >= z) {
      x = x - z;
      y = y + 1;
    }
  }
  return x;                             /* Quotient is y. */
}

/* ---------------------------- montmul ----------------------------- */
static mont_u64 mont_montmul(mont_u64 abar, mont_u64 bbar, mont_u64 m,
                             mont_u64 mprime) {
  mont_u64 thi, tlo, tm, tmmhi, tmmlo, uhi, ulo, ov;

  mont_mulul64(abar, bbar, &thi, &tlo);   /* t = abar*bbar. */

  tm = tlo*mprime;

  mont_mulul64(tm, m, &tmmhi, &tmmlo);    /* tmm = tm*m. */

  ulo = tlo + tmmlo;                      /* Add t to tmm */
  uhi = thi + tmmhi;                      /* (128-bit add). */
  if (ulo < tlo) uhi = uhi + 1;           /* Allow for a carry. */

  ov = (uhi < thi) | ((uhi == thi) & (ulo < tlo));

  ulo = uhi;                              /* Shift u right */
  uhi = 0;                                /* 64 bit positions. */

  ulo = ulo - (m & -(ov | (ulo >= m)));   /* Branch-free `if u >= m, u -= m`. */
  (void)uhi;

  return ulo;
}

/* ---------------------------- xbinGCD ----------------------------- */
/* Extended binary GCD, simplified for a a power of 2 and b odd. VERBATIM. */
static void mont_xbinGCD(mont_u64 a, mont_u64 b, volatile mont_u64 *pu,
                         volatile mont_u64 *pv) {
  mont_u64 alpha, beta, u, v;

  u = 1; v = 0;
  alpha = a; beta = b;          /* alpha is even and beta is odd. */

  /* Invariant from here on: a = u*2*alpha - v*beta. */
  while (a > 0) {
    a = a >> 1;
    if ((u & 1) == 0) {                   /* Delete a common */
      u = u >> 1; v = v >> 1;             /* factor of 2 in u and v. */
    } else {
      /* We want u = (u + beta) >> 1, but that can overflow, so use
         Dietz's method. */
      u = ((u ^ beta) >> 1) + (u & beta);
      v = (v >> 1) + alpha;
    }
  }

  *pu = u;
  *pv = v;
}

/* ------------------------------ driver ---------------------------- */
/* Upstream's fixed inputs, kept as file-scope variables exactly as upstream holds
   them (`static uint64 in_a, in_b, in_m;` assigned by initialise_benchmark). m must
   be ODD; a and b must be < m.
   NOT `static`, and initialized rather than assigned: the gp cap-table glue lives in
   a separate translation unit and references each delivered global BY NAME, so a
   file-local symbol fails to link -- the same constraint documented on rv8_sha512's
   K table. 3 x 8 B = 24 B, far below the 256 B large-RO copy threshold, so this rung
   needs no opt-in knobs.
   NOT `const`, for the same reason upstream's are not: a `const` global with an
   initializer is constant-folded into immediates at -O1 (writing to it is UB, so
   clang may assume the value), the cap table is then never read, zero `ldc gp[i]`
   are emitted and the rung silently stops exercising the ABI. Measured, not guessed:
   the const version built clean and gated `ldc-gp=0`.
   Keeping them as globals rather than folding them into constants is deliberate: a
   domain with zero globals emits no `ldc gp[i]` at all, does not exercise the
   gp-captable ABI this ladder measures, and is rejected by build-ladder-domain.sh's
   static gate. */
mont_u64 mont_in_m = 0xfae849273928f89full;   /* must be odd */
mont_u64 mont_in_b = 0x14736defb9330573ull;   /* must be < m */
mont_u64 mont_in_a = 0x0549372187237fefull;   /* must be < m */

#define MONT_M mont_in_m
#define MONT_B mont_in_b
#define MONT_A mont_in_a

static unsigned mont_compute(void) {
  unsigned h = 2166136261u;
  mont_u64 m = MONT_M;

  for (int rep = 0; rep < MONT_REPS; rep++) {
    mont_u64 a, b, hr, p1hi, p1lo, p1, p, abar, bbar, phi, plo;
    volatile mont_u64 rinv, mprime;

    /* Distinct inputs per rep, reduced mod m by the kernel's own modul64 so the
       a,b < m precondition holds by construction and no 64-bit division libcall
       is needed (the freestanding domain cannot link __umoddi3). */
    a = mont_modul64(0, MONT_A + (mont_u64)rep * 0x9E3779B97F4A7C15ull, m);
    b = mont_modul64(0, MONT_B + (mont_u64)rep * 0xC2B2AE3D27D4EB4Full, m);

    /* The simple calculation: (a*b)**4 (mod m), correct for all a,b,m < 2**64. */
    mont_mulul64(a, b, &p1hi, &p1lo);       /* a*b (mod m). */
    p1 = mont_modul64(p1hi, p1lo, m);
    mont_mulul64(p1, p1, &p1hi, &p1lo);     /* (a*b)**2 (mod m). */
    p1 = mont_modul64(p1hi, p1lo, m);
    mont_mulul64(p1, p1, &p1hi, &p1lo);     /* (a*b)**4 (mod m). */
    p1 = mont_modul64(p1hi, p1lo, m);

    /* r is the smallest power of 2 larger than m; hr is half of r, because r can
       be 2**64 which does not fit in one word. */
    hr = 0x8000000000000000ull;

    /* r*rinv - m*mprime = 1. */
    mont_xbinGCD(hr, m, &rinv, &mprime);    /* in effect doubles hr. */

    /* Now the Montgomery route to the same answer. */
    abar = mont_modul64(a, 0, m);           /* a*r (mod m) */
    bbar = mont_modul64(b, 0, m);           /* b*r (mod m) */

    p = mont_montmul(abar, bbar, m, mprime);   /* a*b (mod m) */
    p = mont_montmul(p, p, m, mprime);         /* (a*b)**2 (mod m) */
    p = mont_montmul(p, p, m, mprime);         /* (a*b)**4 (mod m) */

    /* Convert p back to a normal number: p = (p*rinv) % m. */
    mont_mulul64(p, rinv, &phi, &plo);
    p = mont_modul64(phi, plo, m);

    /* Fold everything that was actually computed. p must equal p1 -- the two
       independent routes to (a*b)**4 mod m -- so a miscompute in EITHER shows up. */
    h ^= (unsigned)(p   & 0xffffffffu); h *= 16777619u;
    h ^= (unsigned)(p   >> 32);         h *= 16777619u;
    h ^= (unsigned)(p1  & 0xffffffffu); h *= 16777619u;
    h ^= (unsigned)(p1  >> 32);         h *= 16777619u;
    h ^= (unsigned)(rinv   & 0xffffffffu); h *= 16777619u;
    h ^= (unsigned)(mprime & 0xffffffffu); h *= 16777619u;
    h ^= (unsigned)(p == p1);           h *= 16777619u;
  }
  return h;
}
#endif
