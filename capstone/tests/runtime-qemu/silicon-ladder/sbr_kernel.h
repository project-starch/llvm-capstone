#ifndef SBR_H
#define SBR_H
/* sbr = Stale Branch operand Repro.
 *
 * Minimal standalone repro for the S-03 root cause: a conditional branch whose source
 * register is written by the IMMEDIATELY PRECEDING instruction appears to resolve on the
 * STALE (pre-write) operand value on the FPGA. Found in sqlite3InsertBuiltinFuncs, where
 *
 *      stc  a1, 0x0(a0)      ; spill pOther (NULL) to a stack slot
 *      ld   a0, 0x0(a0)      ; read it back as an integer -> 0
 *      beqz a0, <else>       ; NOT taken on silicon, so `if (pOther)` runs with pOther NULL
 *
 * sent a NULL pointer down the non-NULL path and wedged on the deref.
 *
 * WHY THIS SHAPE. In SQLite the failure is a WEDGE: one bit per session, and it takes the
 * core with it. Here every arm produces a WRONG NUMBER instead -- each sets its own bit in a
 * returned bitmask -- so a single run reports all six arms and nothing can hang.
 *
 * Every arm seeds its destination register NON-ZERO, then makes it architecturally ZERO,
 * then branches on zero. Correct hardware takes the branch and contributes 0. Hardware that
 * reads the pre-write operand sees the seed, does not take the branch, and sets the bit.
 *
 * The arms are three matched PAIRS. Within each pair the only difference is one inserted
 * `nop`, so the difference between the pair IS the variable:
 *
 *   bit 0 / 1   store to the slot, load it back, branch      <- the exact SQLite shape
 *   bit 2 / 3   ALU producer, no memory at all               <- is a load needed?
 *   bit 4 / 5   load with NO preceding store to that address <- is the store needed?
 *
 * Reading the result:
 *   any ODD bit (1,3,5) SET  -> the INSTRUMENT is broken, not the silicon. With a nop
 *                               between producer and branch there is no hazard to observe,
 *                               so a control arm must read the true value. Believe nothing
 *                               else in the run.
 *   0 set, 4 clear           -> the preceding store is REQUIRED: store-to-load forwarding,
 *                               not a branch-operand hazard.
 *   0 and 4 both set         -> the store is irrelevant; the branch operand is the defect.
 *   2 set                    -> not even a load is needed; any producer triggers it.
 *   nothing set              -> hazard NOT reproduced in isolation; it needs something the
 *                               SQLite context supplies that this rung does not.
 *
 * The BASE constant matters: a bare 0 is indistinguishable from "the rung was not compiled
 * in" and from "the harness reported nothing", both of which have produced false CLEAN
 * verdicts on this project. A nonzero base proves the code ran.
 */

#define SBR_BASE 0xB0000u

/* R-16 REDRAW KNOB. The SHA5 entry stall is per-image and deterministic, so retrying a
 * stalling binary is futile -- a fresh DRAW is needed instead. SBR_PAD emits that many bytes
 * of never-executed nops ahead of the code, which shifts the image layout while leaving
 * sbr_compute byte-identical. Verify that identity before trusting a draw: compare the
 * function's bytes across draws, and abort if any two whole images hash the same. */
#ifndef SBR_PAD
#define SBR_PAD 0
#endif
#define SBR_STR2(x) #x
#define SBR_STR(x) SBR_STR2(x)
#if SBR_PAD > 0
__asm__(".pushsection .text\n\t.rept " SBR_STR(SBR_PAD) "\n\tnop\n\t.endr\n\t.popsection");
#endif

static volatile unsigned long sbr_slot = 1; /* seeded NON-zero: arm 0's store is what makes
                                             * it zero, so a skipped store cannot masquerade
                                             * as a passing arm. */

static unsigned sbr_compute(void)
{
  unsigned long p = (unsigned long)&sbr_slot;
  unsigned long a0, a1, a2, a3, a4, a5;

  /* PAIR A -- the exact SQLite shape: store, load it back, branch immediately. */
  __asm__ volatile(
      "li   %0, -1\n\t"          /* seed the branch register non-zero */
      "sd   zero, 0(%1)\n\t"     /* slot := 0 */
      "ld   %0, 0(%1)\n\t"       /* architecturally 0 */
      "beqz %0, 1f\n\t"          /* correct: taken */
      "li   %0, 1\n\t"           /* not taken => the branch saw the stale seed */
      "j    2f\n"
      "1:\tli %0, 0\n"
      "2:"
      : "=&r"(a0) : "r"(p) : "memory");

  __asm__ volatile(              /* CONTROL for A: one nop, nothing else changed */
      "li   %0, -1\n\t"
      "sd   zero, 0(%1)\n\t"
      "ld   %0, 0(%1)\n\t"
      "nop\n\t"
      "beqz %0, 1f\n\t"
      "li   %0, 1\n\t"
      "j    2f\n"
      "1:\tli %0, 0\n"
      "2:"
      : "=&r"(a1) : "r"(p) : "memory");

  /* PAIR B -- ALU producer, no memory operand anywhere. */
  __asm__ volatile(
      "li   %0, -1\n\t"
      "addi %0, zero, 0\n\t"
      "beqz %0, 1f\n\t"
      "li   %0, 1\n\t"
      "j    2f\n"
      "1:\tli %0, 0\n"
      "2:"
      : "=&r"(a2) : : );

  __asm__ volatile(              /* CONTROL for B */
      "li   %0, -1\n\t"
      "addi %0, zero, 0\n\t"
      "nop\n\t"
      "beqz %0, 1f\n\t"
      "li   %0, 1\n\t"
      "j    2f\n"
      "1:\tli %0, 0\n"
      "2:"
      : "=&r"(a3) : : );

  /* PAIR C -- load and branch, but NO store to that address in this arm. The slot is
   * already 0 here because pair A stored 0 to it, so the load still reads 0; what differs
   * from pair A is only that no store immediately precedes the load. */
  __asm__ volatile(
      "li   %0, -1\n\t"
      "ld   %0, 0(%1)\n\t"
      "beqz %0, 1f\n\t"
      "li   %0, 1\n\t"
      "j    2f\n"
      "1:\tli %0, 0\n"
      "2:"
      : "=&r"(a4) : "r"(p) : "memory");

  __asm__ volatile(              /* CONTROL for C */
      "li   %0, -1\n\t"
      "ld   %0, 0(%1)\n\t"
      "nop\n\t"
      "beqz %0, 1f\n\t"
      "li   %0, 1\n\t"
      "j    2f\n"
      "1:\tli %0, 0\n"
      "2:"
      : "=&r"(a5) : "r"(p) : "memory");

  return SBR_BASE
       | ((unsigned)a0 << 0) | ((unsigned)a1 << 1)
       | ((unsigned)a2 << 2) | ((unsigned)a3 << 3)
       | ((unsigned)a4 << 4) | ((unsigned)a5 << 5);
}
#endif
