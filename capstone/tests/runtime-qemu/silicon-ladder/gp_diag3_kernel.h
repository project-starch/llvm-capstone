#ifndef GP_DIAG3_KERNEL_H
#define GP_DIAG3_KERNEL_H
/* Silicon-ladder DIAGNOSTIC rung v3: isolate ITERATED SHARED-REGION ACCESS.
 *
 * WHERE THIS COMES FROM. gp_diag v2 ran 11 probes on the board and ALL of them
 * were correct -- scalar globals via the gp cap-table, global arrays, globals
 * read from a noinline callee, deep and mutual recursion, array-store-with-live-
 * accumulator, initialized globals, function-local statics, a counted loop with
 * no call, and a counted loop WITH a call. Every mechanism we had suspected is
 * fine in isolation. The ONLY thing still wrong was res[0], produced by the one
 * remaining computation: a loop that reads back through the SHARED-REGION
 * capability `res`. See
 *   history/25-07-2026_18-44-49_gp-diag-v2-all-probes-correct-fault-is-shared-region-loop.md
 *
 * So v3 tests exactly that, and nothing else. Every probe below is a sum over 8
 * words; the only differences between them are (a) which memory the words live
 * in, (b) whether the access is in a loop, and (c) how the address is formed.
 *
 * WHY THE MAGIC NUMBERS. The seeds are DISTINCT POWERS OF TWO (256<<i), so every
 * expected value is a subset sum with a unique decomposition. A wrong result is
 * therefore self-describing: read it in binary and you know exactly which
 * iterations ran and which element each one read. That is the whole point --
 * v1/v2 returned FNV checksums, which are not injective, so a wrong value told
 * us nothing (see the checksum-inversion dead end in the v1 note).
 *
 * Slot map (controller prints res[3..] as dbg0..):
 *   dbg0 A  loop over a GLOBAL array (gp cap-table)         expect 65280
 *   dbg1 B  loop over res[]          (SHARED REGION)        expect 65280  <-- suspect
 *   dbg2 C  STRAIGHT-LINE read of res[] (no loop)           expect 65280
 *   dbg3 D  loop over res[] at a CONSTANT index             expect 2048  (8*256)
 *   dbg4 E  loop over a LOCAL STACK array                   expect 65280
 *   dbg5 F  loop STORING into res[], straight-line readback  expect 65280
 *   dbg6 G  loop over res[] via a WALKING POINTER (p++)     expect 65280
 *   dbg7 H  NESTED loop over res[] (byte extract, v2 shape) expect 255
 *   dbg8 I  canary constant                                 expect 0xC0FFEE
 *
 * HOW TO READ THE OUTCOME:
 *   A,E correct + B wrong  -> the fault needs the SHARED-REGION cap, not loops
 *                             and not the gp cap-table. That is the reproducer.
 *   C wrong                -> not about loops at all: a straight-line read-back
 *                             of the shared region is already stale/wrong.
 *   D wrong but C correct  -> the per-iteration CAPABILITY RELOAD is the fault
 *                             (D varies nothing but the loop counter).
 *   B wrong but D correct  -> the varying index (cincoffset) is the fault.
 *   F wrong but B correct  -> the STORE side, not the load side.
 *   G vs B                 -> walking pointer vs indexed addressing.
 *   H is the v2 fold's exact shape, kept as a POSITIVE CONTROL: if nothing else
 *   fails, H (and res[0], same shape) must still fail, or v3 did not reproduce
 *   the bug and the result is inconclusive rather than exculpatory.
 *
 * Discipline carried over from v2: everything that must be trustworthy is
 * STRAIGHT-LINE, so a probe's value is never confounded by the very construct
 * under test. Keep this kernel small -- all domain code must fit the monitor's
 * 4 KiB PCC window (link-gpfree.ld hard-fails past it). */

#define GPD3_CANARY   0xC0FFEEUL
#define GPD3_N        8           /* words per window */
#define GPD3_W1       32          /* res[32..40): read window  */
#define GPD3_W2       40          /* res[40..48): write window */
#define GPD3_NPROBE   9           /* raw slots res[3 .. 3+9) */

#define GPD3_SEED(i)  (256UL << (i))      /* 256,512,...,32768 -- all distinct 2^k */
#define GPD3_SUM      65280UL             /* 256+512+...+32768 */
#define GPD3_CONSTSUM (GPD3_N * 256UL)    /* 2048: 8 reads of seed 0 */
#define GPD3_BYTESUM  255UL               /* 1+2+4+...+128: one set byte per seed */

static unsigned long gpd3_garr[GPD3_N];   /* global array -> gp cap-table */

/* Seed one window of res[] with STRAIGHT-LINE stores. v2 proved straight-line
   stores through the shared-region cap land correctly (the controller read all
   11 of them back), so this is a trustworthy starting state for the probes. */
#define GPD3_SEED_RES(base)                                                    \
  do {                                                                         \
    res[(base) + 0] = GPD3_SEED(0);  res[(base) + 1] = GPD3_SEED(1);           \
    res[(base) + 2] = GPD3_SEED(2);  res[(base) + 3] = GPD3_SEED(3);           \
    res[(base) + 4] = GPD3_SEED(4);  res[(base) + 5] = GPD3_SEED(5);           \
    res[(base) + 6] = GPD3_SEED(6);  res[(base) + 7] = GPD3_SEED(7);           \
  } while (0)

#define GPD3_SUM_RES_STRAIGHT(base)                                            \
  (res[(base) + 0] + res[(base) + 1] + res[(base) + 2] + res[(base) + 3] +     \
   res[(base) + 4] + res[(base) + 5] + res[(base) + 6] + res[(base) + 7])

/* Runs every probe and returns the v2-shaped FNV fold over the raw slots.
   `res` must have at least GPD3_W2 + GPD3_N words (the 4 KiB region has 512). */
static unsigned gpd3_run(unsigned long *res) {
  unsigned long larr[GPD3_N];
  unsigned long s;
  int i;

  /* ---- seeding, all straight-line ------------------------------------- */
  GPD3_SEED_RES(GPD3_W1);
  gpd3_garr[0] = GPD3_SEED(0);  gpd3_garr[1] = GPD3_SEED(1);
  gpd3_garr[2] = GPD3_SEED(2);  gpd3_garr[3] = GPD3_SEED(3);
  gpd3_garr[4] = GPD3_SEED(4);  gpd3_garr[5] = GPD3_SEED(5);
  gpd3_garr[6] = GPD3_SEED(6);  gpd3_garr[7] = GPD3_SEED(7);
  larr[0] = GPD3_SEED(0);  larr[1] = GPD3_SEED(1);
  larr[2] = GPD3_SEED(2);  larr[3] = GPD3_SEED(3);
  larr[4] = GPD3_SEED(4);  larr[5] = GPD3_SEED(5);
  larr[6] = GPD3_SEED(6);  larr[7] = GPD3_SEED(7);

  /* A: loop over a GLOBAL array -- control for "loop over the gp cap-table". */
  s = 0;
  for (i = 0; i < GPD3_N; i++) s += gpd3_garr[i];
  res[3 + 0] = s;

  /* B: the SUSPECT -- same loop, but the words live in the shared region. */
  s = 0;
  for (i = 0; i < GPD3_N; i++) s += res[GPD3_W1 + i];
  res[3 + 1] = s;

  /* C: same words, same cap, NO loop. Separates "shared region" from "loop",
     and doubles as a read-after-write visibility check on the region. */
  res[3 + 2] = GPD3_SUM_RES_STRAIGHT(GPD3_W1);

  /* D: loop over the shared region with a CONSTANT index. Reloads the region
     capability every iteration exactly like B, but never offsets it. */
  s = 0;
  for (i = 0; i < GPD3_N; i++) s += res[GPD3_W1];
  res[3 + 3] = s;

  /* E: identical loop over a LOCAL STACK array -- control for "loop over any
     memory reached through a capability that is not the shared region". */
  s = 0;
  for (i = 0; i < GPD3_N; i++) s += larr[i];
  res[3 + 4] = s;

  /* F: the STORE side -- write the window from inside a loop, then read it back
     straight-line (C already showed whether straight-line reads are sound). */
  for (i = 0; i < GPD3_N; i++) res[GPD3_W2 + i] = GPD3_SEED(i);
  res[3 + 5] = GPD3_SUM_RES_STRAIGHT(GPD3_W2);

  /* G: walking pointer instead of an indexed access. */
  {
    unsigned long *p = res + GPD3_W1;
    s = 0;
    for (i = 0; i < GPD3_N; i++) { s += *p; p++; }
    res[3 + 6] = s;
  }

  /* H: the exact shape of the v2 checksum fold (outer loop over shared-region
     words, inner loop over their bytes) -- the positive control. */
  s = 0;
  for (i = 0; i < GPD3_N; i++) {
    unsigned long v = res[GPD3_W1 + i];
    int b;
    for (b = 0; b < 8; b++) s += (v >> (8 * b)) & 0xffUL;
  }
  res[3 + 7] = s;

  /* I: canary -- a wrong value here means the slot plumbing, not the compiler. */
  res[3 + 8] = GPD3_CANARY;

  /* Fold, in the v2 shape (a loop reading back through the shared-region cap),
     so res[0] is itself a second instance of the positive control. */
  {
    unsigned h = 2166136261u;
    for (i = 0; i < GPD3_NPROBE; i++) {
      unsigned v = (unsigned)res[3 + i];
      int b;
      for (b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
    }
    return h;
  }
}
#endif
