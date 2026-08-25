#ifndef S12_KERNEL_H
#define S12_KERNEL_H
/* S-12 MINIMAL REPRO -- the WhereCode NOT_CAP window, inside a real capability domain.
 *
 * WHY THIS EXISTS. The only reproducer for S-12 is all of SQLite, which is not a handover.
 * This is the same instruction window in ~40 lines.
 *
 * WHY IT IS A DOMAIN KERNEL AND NOT A BARE-METAL TEST. Five directed bare-metal simulations
 * fail to reproduce this -- the bare four-instruction shape, an intervening-store sweep across
 * the write-buffer depth, adjacent-granule scalar stores, a full offset-for-offset replay of
 * the window, and a scalar WAW/occupancy test. The last three carry proven-firing positive
 * controls, so their clean results are real. What they all lack is DOMAIN CONTEXT: M-mode
 * instead of running after capenter, register-resident capabilities instead of a cap table, a
 * .data buffer instead of a monitor-carved stack. This closes that gap, which is the largest
 * one left, while staying small enough to hand over.
 *
 * WHAT IS ESTABLISHED, so nobody re-derives it. At the wedge the memory is INTACT AND TAGGED
 * -- the granule holds the stored cursor and its shadow tag byte reads 1 -- while tval says the
 * consumer received cursor 0. The value was never lost; it was never delivered. That excludes a
 * software NULL and excludes a memory-path loss. The mechanism is open.
 *
 * WHY IT CANNOT WEDGE. The reloaded value is inspected with `lcc` selector 1, the TOTAL type
 * query, which answers 7 for NOT_CAP WITHOUT raising (capstone_dyn_unit.anvil:195). So a bad
 * reload is COUNTED rather than fatal and every arm returns a number, which is bisectable where
 * a hang is not. The production shape uses `cincoffsetimm`, which RAISES -- that difference is
 * deliberate and is stated in the risk note at the bottom.
 *
 * THE ARMS. Arm 2 MUST fail; a batch of negatives from an instrument that has never produced a
 * positive is worth nothing, and this project has published exactly that mistake.
 *
 *   S12_ARM 0  CONTROL          spill; reload; type-check.            EXPECT bad == 0
 *   S12_ARM 1  THE SHAPE        spill; the 9 real intervening stores; movc dest,zero;
 *                               reload; type-check.                   ANY bad == the defect
 *   S12_ARM 2  POSITIVE CONTROL spill; scribble the slot scalar-wise; reload; type-check.
 *                               EXPECT bad == REPS. Architecturally CORRECT -- a plain store
 *                               legitimately clears the granule tag -- and it exists only to
 *                               prove the detector can report a bad reload at all.
 *   S12_ARM 3  SEPARATION       arm 1 plus one nop before the reload's consumer. R-20 was cured
 *                               by exactly one nop, so arm 1 failing while arm 3 does not is
 *                               the scheduling discriminator.
 *
 * RETURN VALUE: 0xC12A0000 | (arm << 12) | (bad & 0xFFF), bad saturating at 0xFFF.
 */

#ifndef S12_ARM
#define S12_ARM 1
#endif
#ifndef S12_REPS
#define S12_REPS 512
#endif

/* The frame. Sized and aligned so the slot at -0x70 and every intervening-store target down to
 * -0x5e0 stay inside it, exactly as they do relative to s0 on the board. */
static unsigned char volatile s12_frame[0x800] __attribute__((aligned(16)));
static void *volatile s12_subject;
static unsigned long volatile s12_sink[8];

/* lcc selector 1: the TOTAL type query. Returns 7 for NOT_CAP and does NOT raise.
 *
 * always_inline is LOAD-BEARING, not tidiness. Left as a plain static function the compiler
 * emitted a real CALL, which puts an entire function call -- prologue, jump, return -- between
 * the reload and its consumer. Adding instructions there is the single perturbation known to
 * make this fault disappear, so the repro would have been silently useless while looking
 * correct. Verified in the artifact: the lcc must appear INSIDE s12_compute, not in a separate
 * s12_type symbol. */
__attribute__((always_inline))
static inline unsigned long s12_type(const void *p) {
  unsigned long v = 0;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(v) : "r"(p));
  return v;
}
#define S12_NOT_CAP 7u

static unsigned s12_compute(void)
{
  unsigned long bad = 0;
  /* fp plays s0. The slot is fp-0x70 and is 16-byte aligned by construction, which the
   * hardware requires of a capability store and which the board's s0-0x70 also satisfies. */
  unsigned char volatile *fp = s12_frame + 0x700;
  void *volatile *slot = (void *volatile *)(fp - 0x70);

  for (unsigned r = 0; r < S12_REPS; r++) {
    void *v = (void *)&s12_subject;      /* any tagged capability; its identity is irrelevant */

#if S12_ARM == 0
    *slot = v;
#elif S12_ARM == 1 || S12_ARM == 3
    *slot = v;                            /* +0x40  THE SUBJECT STORE */
    /* The nine intervening stores, at the board's own offsets: five capability-sized and four
     * scalar, in the same order. Kept as C stores through volatile so the compiler may not
     * sink or reorder them across the reload. */
    *(void *volatile *)(fp - 0x5d0) = v;             /* +0x48 */
    *(unsigned volatile *)(fp - 0x74) = 0x5A5A5A5Au; /* +0x4c  scalar, granule BELOW the slot */
    *(void *volatile *)(fp - 0x5b0) = v;             /* +0x54 */
    *(void *volatile *)(fp - 0x90)  = v;             /* +0x58 */
    *(unsigned long volatile *)(fp - 0x98) = 0x3C3Cu;/* +0x60  scalar */
    *(void *volatile *)(fp - 0x5a0) = (void *)0;     /* +0x6c  a ZEROED value, as movc a4,zero */
    *(unsigned volatile *)(fp - 0x10c) = 0u;         /* +0x70  scalar, of that zeroed value */
    *(unsigned volatile *)(fp - 0x110) = 0u;         /* +0x78  scalar, again */
    *(void *volatile *)(fp - 0x120) = (void *)0;     /* +0x84 */
#elif S12_ARM == 2
    *slot = v;
    /* POSITIVE CONTROL: destroy the slot with scalar stores. This is correct architecture --
     * a plain store clears the granule's tag -- so it must report bad, and if it does not,
     * the detector is blind and no other arm carries a verdict. */
    ((unsigned long volatile *)slot)[0] = 0;
    ((unsigned long volatile *)slot)[1] = 0;
#endif

#if S12_ARM == 3
    __asm__ volatile("nop");              /* the separation control */
#endif

    /* THE RELOAD and its consumer. Reading through the volatile slot is the ldc; the type
     * query is the consumer, standing in for cincoffsetimm without the raise. */
    void *back = *slot;
    if (s12_type(back) == S12_NOT_CAP)
      bad++;

    s12_sink[r & 7] = (unsigned long)back;
  }

  if (bad > 0xFFF) bad = 0xFFF;
  return (unsigned)(0xC12A0000u | ((unsigned)S12_ARM << 12) | (unsigned)bad);
}

/* RISK, stated rather than discovered later. Two things here differ from the production shape
 * and either could hide the defect:
 *   1. the consumer is `lcc` (non-raising) instead of `cincoffsetimm` (raising), so the exact
 *      consuming instruction differs;
 *   2. this is a loop, whereas the board's window executes twice inside a much longer call
 *      chain at a deterministic depth.
 * Both were accepted so that every arm RETURNS -- a wedge yields one bit and this yields a
 * rate. If arm 1 comes back clean while the SQLite repro still wedges, that is evidence about
 * THIS test and not about the silicon, and the next step is to move the consumer back to
 * cincoffsetimm and accept wedging arms. Do not read a clean arm 1 as an exoneration. */

#endif /* S12_KERNEL_H */
