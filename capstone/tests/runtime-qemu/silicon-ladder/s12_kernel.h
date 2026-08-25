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
/* REDRAW SEED. R-16 (entry stall) is PER-IMAGE: the domain never enters, so retrying the same
 * image is futile and the skill says to REDRAW instead -- rebuild with a harmless constant
 * varied so the code under test stays byte-identical across draws.
 *
 * This seeds an unused volatile global, so it changes .data and therefore the image hash while
 * not altering a single instruction. That is stronger than padding with nops, which would shift
 * alignment inside the window under test and make the draws differ in something that matters. */
#ifndef S12_DRAW
#define S12_DRAW 0
#endif
static unsigned long volatile s12_draw_seed = 0xD2A0000UL + S12_DRAW;

/* PAGE-ALIGNED so every arm places the slot at the SAME address.
 *
 * The matched-pair experiment arms ONE watchpoint address and runs both the raising and the
 * non-raising arm against it -- which only works if both arms put the slot in the same place.
 * At 16-byte alignment they did not: the lcc arm landed the slot at VA 0x11750 and the
 * cincoffsetimm arm at 0x11790, 64 bytes apart, because the two bodies differ in size and shift
 * everything after them. One armed address could not have served both, and arming per-arm would
 * reintroduce exactly the wrong-allocation failure class this static array exists to remove.
 *
 * 4096 swamps the inter-arm drift, so the frame -- and therefore the slot -- lands identically.
 * VERIFY IT rather than assume it: the addresses are compared across arms before any run. */
static unsigned char volatile s12_frame[0x800] __attribute__((aligned(4096)));
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
  (void)s12_draw_seed;
#ifdef S12_SENTINEL
  /* ENTRY-STALL DISCRIMINATOR. Returns before touching the frame, the slot or the consumer.
   *
   * This exists because the lpc controller emits NO post-entry marker, so `SHA5` last on this
   * path CANNOT tell "the body wedged" from "the domain never ran" -- locagg_kernel.h:34-36
   * says so in as many words, and the tree records every lpc-hosted domain dying in share #1
   * on 2026-08-06 regardless of content. Arm 4 fell silent at SHA5 on two successive draws and
   * I read that as an entry stall; that reading is unsupported without this arm.
   *
   *   returns 0xC12A4E17 -> glue, cap-init, entry and return all work, so arm 4's silence is
   *                         attributable to its BODY -- i.e. the window faulted
   *   wedges             -> the rung is an entry stall and says NOTHING about the window
   *
   * The sentinel value cannot collide with a real result: a real arm-4 return is
   * 0xC12A4000 | bad, and arm 4 never increments bad, so it returns exactly 0xC12A4000. */
  return 0xC12A0000u | (4u << 12) | 0xE17u;
#endif
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
#if S12_ARM != 4
    if (s12_type(back) == S12_NOT_CAP)
      bad++;
#endif

#if S12_ARM == 4
    /* THE RAISING CONSUMER -- the production shape, and the reason this arm exists.
     *
     * Arms 0/1/3 inspect the reload with `lcc` selector 1, which is TOTAL: it answers 7 for
     * NOT_CAP without raising, so a bad reload is counted and the arm still returns a rate.
     * That is what makes them bisectable, and it is also a real deviation -- production uses
     * `cincoffsetimm`, which RAISES. A clean arm 1 therefore cannot rule out a fault that only
     * the raising consumer exposes, and the kernel said so before arm 1 was ever run.
     *
     * So this arm consumes the reload EXACTLY as sqlite3WhereCodeOneLoopStart+0x8c does:
     * cincoffsetimm on the loaded value, same 0xb0 displacement. If the operand arrives
     * NOT_CAP the domain takes mcause 25 and WEDGES -- one bit per boot instead of a rate,
     * which is the price of matching production.
     *
     * The counter is written to the sink BEFORE the consumer runs, so a wedge still leaves
     * evidence of how many iterations completed rather than dying silently. */
    s12_sink[0] = r;
    {
      void *volatile _c = back;
      __asm__ volatile(".insn i 0x5b, 0x2, %0, %1, 0xb0" : "=r"(_c) : "r"(_c));
      /* cursor half only; a capability does not fit a long, and this is a liveness
         marker rather than a value under test */
      s12_sink[1] = (unsigned long)(__UINTPTR_TYPE__)_c;
    }
#endif

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
