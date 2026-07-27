#ifndef CTRSANITY_KERNEL_H
#define CTRSANITY_KERNEL_H
/* Counter-sanity probe: is a cycle measured in a DOMAIN comparable to a cycle
 * measured in the Linux BASELINE at all?
 *
 * WHY THIS EXISTS (2026-07-27). `beebs_cnt` is silicon-correct and retires
 * 1.138x the baseline's instructions, yet takes 0.684x the CYCLES -- capability
 * CPI 1.68 against baseline CPI 2.79. Read literally that says pervasive
 * capability safety makes code 32% FASTER, which is not a result, it is an
 * uncontrolled confound (issue I-2). Every overhead ratio we publish assumes the
 * two counters mean the same thing. Nothing has ever tested that assumption, and
 * if it is false then EVERY row is wrong -- including beebs_prime's 1.032%,
 * which a confound in this direction would make too flattering to us.
 *
 * THE TEST. Run work that compiles to the SAME RISC-V instructions on both
 * targets, and compare.
 *
 *   - The measured loop is pure REGISTER arithmetic: no loads, no stores, no
 *     array, no call. A capability target and a plain riscv64 target have
 *     nothing to differ about here -- there is no pointer to make 128 bits wide
 *     and no global to route through a cap table.
 *   - The one global store (`cs_sink`) sits OUTSIDE the loop. It exists only
 *     because the ladder build gate rejects a domain with no `ldc gp[i]` (a rung
 *     that never touches the cap table would not be exercising the ABI). At a
 *     few instructions against ~1M, it cannot move the ratio.
 *   - The loop body carries NO inline asm. A first version pinned `acc` with an
 *     empty `asm volatile("" : "+r"(acc))` each iteration, and that DEFEATED the
 *     probe: the Capstone backend emitted two redundant `mv a4, a4` around the
 *     constraint, making the loop 7 instructions against the baseline's 5 --
 *     a 1.4x instruction difference manufactured by the probe itself. The opaque
 *     trip count already prevents constant folding and `acc` feeds the returned
 *     hash, so neither the loop nor the value can be optimized away without it.
 *     (Those no-op moves are a real if minor codegen wart, logged as C-9.)
 *     **Identical instret between the two builds is the check that the probe is
 *     valid at all -- verify it before reading the cycle numbers.**
 *
 * HOW TO READ THE RESULT
 *
 *   instret ratio ~= 1.00 and cycles ratio ~= 1.00
 *       -> the counters agree. I-2 is specific to beebs_cnt and the existing
 *          rows stand.
 *   instret ratio ~= 1.00 but cycles ratio << 1.00
 *       -> a domain cycle is CHEAPER than a baseline cycle for identical work.
 *          Every published ratio understates capability overhead by that factor
 *          and must be corrected or withdrawn. This is the outcome that would
 *          hurt, which is exactly why it is worth one boot.
 *   instret ratio != 1.00
 *       -> the two builds are NOT doing the same work; this probe is void and
 *          the codegen must be compared before anything is concluded.
 *
 * CTRSANITY_N is overridden to 4x by ctrsanity4 to separate a PROPORTIONAL
 * effect (a counter that scales -- ratio stays put as the work grows) from a
 * FIXED one (one-off interference or entry cost -- ratio moves toward 1.0 as the
 * work grows). One length cannot tell those apart. */

#ifndef CTRSANITY_N
#define CTRSANITY_N 100000L
#endif

static volatile int cs_sink;    /* satisfies the ldc gp[i] build gate, outside the loop */

static unsigned cs_compute(void) {
  long n = CTRSANITY_N;
  __asm__ volatile("" : "+r"(n));   /* opaque bound: no constant folding */

  long acc = 0;
  for (long i = 0; i < n; i++)
    acc += i ^ (acc >> 3);

  cs_sink = (int)acc;               /* the only memory op, and it is outside */

  unsigned h = 2166136261u;
  h ^= (unsigned)acc;        h *= 16777619u;
  h ^= (unsigned)(acc >> 32); h *= 16777619u;
  return h;
}
#endif
