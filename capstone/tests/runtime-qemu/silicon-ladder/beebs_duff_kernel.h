#ifndef BEEBS_DUFF_KERNEL_H
#define BEEBS_DUFF_KERNEL_H
/* Silicon-ladder rung: BEEBS duff -- Duff's device byte copy.
 *
 * Source: Bristol/Embecosm BEEBS `duff`. Verbatim compute (the switch-into-loop
 * is the point of the benchmark and is kept exactly).
 *
 * SHAPE PREDICTION under issue R-1 (ref/ISSUES.md): PASS, for the same
 * cross-object reason as `beebs_cnt`, but through a different mechanism and so
 * worth its own boot. duffcopy walks TWO pointers -- `*to++ = *from++` -- which
 * is precisely the "pointer walk" form that FAILED as a mitigation in the R-1
 * sweep (rawhazard7 P2, both accesses through pointers, returned stale). The
 * difference is that in rawhazard7 both pointers were derived from ONE object,
 * and here they address two distinct arrays.
 *
 * So this rung separates two readings of the failed mitigation:
 *   - "a pointer walk is unsafe"                          -> duff FAILS
 *   - "a pointer walk into the SAME object is unsafe"     -> duff PASSES
 * The second is what the registry currently claims. Nothing else in the corpus
 * distinguishes them.
 *
 * Note it is also the only rung exercising byte-granular (`lb`/`sb`) access
 * through the cap table. Narrow accesses are the surviving static candidate for
 * coremark_matrix's second fault, which no probe has isolated yet -- if duff
 * fails while cnt passes, narrow access is implicated and that is a new lead.
 * -fno-jump-tables is already unconditional in the ladder build, so the switch
 * lowers to a compare chain rather than a jump table. */

#define DUFF_ARRAYSIZE       100
#define DUFF_INVOCATION_CNT   43

static char duff_source[DUFF_ARRAYSIZE];
static char duff_target[DUFF_ARRAYSIZE];

static void duff_duffcopy(char *to, char *from, int count) {
  int n = (count + 7) / 8;
  switch (count % 8) {
  case 0: do {    *to++ = *from++;
  case 7:         *to++ = *from++;
  case 6:         *to++ = *from++;
  case 5:         *to++ = *from++;
  case 4:         *to++ = *from++;
  case 3:         *to++ = *from++;
  case 2:         *to++ = *from++;
  case 1:         *to++ = *from++;
          } while (--n > 0);
  }
}

static unsigned duff_compute(void) {
  unsigned h = 2166136261u;
  for (int rep = 0; rep < 64; rep++) {
    int i;
    for (i = 0; i < DUFF_ARRAYSIZE; i++) {
      duff_source[i] = (char)(DUFF_ARRAYSIZE - i + rep);
      duff_target[i] = 0;
    }
    int cnt = DUFF_INVOCATION_CNT;
    __asm__ volatile("" : "+r"(cnt));   /* keep the count opaque */
    duff_duffcopy(duff_target, duff_source, cnt);
    /* Fold the WHOLE target, not just the copied prefix: the bytes past the
       count must still be zero, so a runaway copy is caught rather than
       averaged away. */
    for (i = 0; i < DUFF_ARRAYSIZE; i++) {
      h ^= (unsigned char)duff_target[i];
      h *= 16777619u;
    }
  }
  return h;
}
#endif
