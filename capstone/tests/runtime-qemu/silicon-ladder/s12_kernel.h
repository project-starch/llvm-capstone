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
 * WHAT IS ESTABLISHED -- REWRITTEN 2026-08-25, because the previous version was not.
 *
 * It used to read: "at the wedge the memory is INTACT AND TAGGED ... the value was never lost, it
 * was never delivered, which excludes a software NULL and excludes a memory-path loss." Every
 * clause of that is withdrawn. It rested on two things that did not survive: the shadow-tag read,
 * which is DRAM and not the L1 tag the load actually consumed, and arm 4, which never wrote the
 * slot -- so the granule was untouched because nothing had touched it, not because the value
 * survived. "Delivered but not lost" was an inference from a measurement of uninitialised memory.
 *
 * WHAT IS ACTUALLY ESTABLISHED, and it is less:
 *   - This repro does NOT reproduce S-12. Arm 2's positive control fires (bad == REPS), so the
 *     detector works, and arm 1 returns bad == 0 across three builds. Arm 4 -- the same shape
 *     with the production RAISING consumer -- runs 512 iterations clean once the subject store
 *     is actually emitted.
 *   - The repro's `v` is NONLIN (arm 5, rebuilt with the store; QEMU 0xC12A5100). NONLIN is not
 *     in the LDC clear set (load_unit.sv:225-226), so the move-clear does not fire HERE.
 *   - What cap type SQLite's value carries at the fault site is UNMEASURED, and it is the
 *     discriminating unknown: if it is in that clear set, this repro never exercised the
 *     mechanism it was built to test. The line below calling `v`'s identity "irrelevant" is the
 *     weakest assumption in this file for exactly that reason.
 *
 * The mechanism is open, and so is whether this kernel is even the right instrument for it.
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

#ifdef S12_SELF_ARM_WP
  /* ARM THE WATCHPOINT FROM INSIDE THE DOMAIN, using its own capability.
   *
   * The watchpoint compares a PHYSICAL address, and page-aligning the frame only made the
   * VIRTUAL address identical across arms -- DBAS is per-ARM, not per-boot (measured: arms in
   * one boot took 0x82400000 and 0x82800000). So one csrw before the boot cannot serve both
   * arms of the matched pair, and whichever arm it was not computed for yields an empty record,
   * which the decision table reads as "the load was fine".
   *
   * The host cannot fix this: create_dom returns only a domain id, and DBAS is printed by the
   * monitor, so userspace never sees it. But the DOMAIN does not need it. Capabilities here
   * carry physical addresses -- the trap mepc is physical, and the monitor's BASE: trace is
   * physical -- so `lcc` selector 2 on a capability pointing at the slot returns the slot's
   * PHYSICAL address directly. No DBAS, no host, no race with the driver.
   *
   * Placed BEFORE the loop, outside the measured window. Layout changes ahead of the window are
   * known survivable: page-aligning the frame moved everything after it and the fault still
   * fired. That is evidence, not an assumption -- but it is evidence about a DIFFERENT change,
   * so this arm's ability to still fault is verified on the board before it is relied on. */
  {
    unsigned long _pa = 0;
#ifdef S12_WP_FRAME_OFF
    /* COMPUTE THE SUBJECT SLOT AT RUNTIME FROM s0. This is the only form that survives a rebuild.
     *
     * The subject is a COMPILER spill at `s0 - OFF`, and OFF is chosen by the compiler: it was
     * 0xc0 in one build and 0xa0 in the next, because adding the arming code moved the frame.
     * So a hardcoded physical address is stale the moment anything is rebuilt -- which is
     * circular, since arming it IS a rebuild.
     *
     * s0 is the frame pointer and is a CAPABILITY register, so the domain can read it, offset it
     * and query its cursor for the PHYSICAL address -- no DBAS, no host, no measured constant.
     * Only OFF stays a build constant, and OFF is an immediate: changing it does not change code
     * size, so the layout is stable across the derive-then-rebuild step and the offset can be
     * confirmed unchanged afterwards.
     *
     *   movc tmp, s0        ; tmp = the frame pointer capability
     *   cincoffsetimm       ; tmp = s0 - OFF, the subject slot
     *   lcc  sel 2          ; cursor = its PHYSICAL address
     */
    { void *volatile _fp;
      __asm__ volatile(".insn r 0x5b, 0x1, 0xa, %0, x8, x0" : "=r"(_fp));
      __asm__ volatile(".insn i 0x5b, 0x2, %0, %1, %2"
                       : "=r"(_fp) : "r"(_fp), "i"(-(S12_WP_FRAME_OFF)));
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(_pa) : "r"(_fp)); }
#elif defined(S12_WP_ADDR)
    /* SUPERSEDED -- READ THIS BEFORE ARMING ANYTHING.
     *
     * The paragraph below concluded "the subject slot is NOT this kernel's array, it is a
     * compiler spill at s0-0xc0". That conclusion came from arm 4, which never wrote the array,
     * so of course the array's granule looked untouched. In every arm that DOES write --
     * which, since the hoist above, is every arm -- the kernel's array store IS the subject.
     * Arming a compiler spill instead arms a granule the subject never touches, which is the
     * same class of error the retraction is about. Kept for the derivation technique only.
     *
     * ORIGINAL TEXT FOLLOWS, and its premise is false:
     * EXPLICIT PHYSICAL ADDRESS, because the subject slot is NOT this kernel's array.
     *
     * The fault is on a COMPILER-GENERATED spill at s0-0xc0, not on `s12_frame`. The kernel's
     * own store is a different, earlier one, and computing the array's address -- however
     * carefully -- arms a granule the fault never touches. That yields an EMPTY record, which
     * the decision table reads as "the load was fine": a confident wrong answer, not a null one.
     *
     * The compiler picks the frame slot and does not consult us, so the address has to come from
     * a measured wedge (s0 at the halt, minus the build's frame offset) rather than from source.
     * It is therefore a hardcoded PHYSICAL address and only valid for the DBAS it was measured
     * at -- the driver's DBAS guard and the s07 STC-recorder cross-check are what make it safe.
     */
    _pa = (unsigned long)(S12_WP_ADDR);
#else
    void *volatile _p = (void *)(s12_frame + 0x700 - 0x70);
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(_pa) : "r"(_p));
#endif
    __asm__ volatile("csrw 0x811, %0" :: "r"(_pa));
    { unsigned long _bk = 0;
      __asm__ volatile("csrr %0, 0x811" : "=r"(_bk));
      s12_sink[2] = _bk;          /* readback, so a failed arm is visible rather than assumed */
      s12_sink[3] = _pa; }
    /* group 9 (store watchpoint) so the arming is PROVEN on this boot: an empty LDC record only
       means "the load was fine" if group 9 fired at the subject store. */
    { unsigned long _m = 0x200UL;
      __asm__ volatile("csrw 0x810, %0" :: "r"(_m)); }
  }
#endif
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
#if S12_ARM == 7
    void *volatile _c7;
#endif

    /* THE SUBJECT STORE, HOISTED ABOVE THE ARM CHAIN ON PURPOSE.
     *
     * It used to sit inside the #if/#elif chain below, once per branch. Arm 4 was added and
     * matched no branch, so it read a slot it never wrote -- zero-initialised memory, a NOT_CAP
     * reload, and a raising consumer taking mcause 25 for a trivially correct reason. Arms 5 and
     * 6 had the identical defect. Three boots were spent measuring it.
     *
     * Every arm that is not the entry-stall discriminator needs this store, so making it
     * conditional was never buying anything -- it only created a way to forget it. Hoisted, an
     * arm CANNOT miss it: the failure mode is impossible rather than merely detected. The
     * #error at the reload site is kept as a tripwire in case someone re-conditionalises this. */
    *slot = v;                            /* +0x40  THE SUBJECT STORE */
#define S12_SLOT_WRITTEN 1

#if S12_ARM == 0
    /* nothing further: store, reload, type-check */
#elif S12_ARM == 1 || S12_ARM == 3 || S12_ARM == 4
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
    /* POSITIVE CONTROL: destroy the slot with scalar stores. This is correct architecture --
     * a plain store clears the granule's tag -- so it must report bad, and if it does not,
     * the detector is blind and no other arm carries a verdict. */
    ((unsigned long volatile *)slot)[0] = 0;
    ((unsigned long volatile *)slot)[1] = 0;
#endif

#if S12_ARM == 3
    __asm__ volatile("nop");              /* the separation control */
#endif



#if S12_ARM == 5
    /* WHICH CAP TYPE IS THE STORED VALUE? This decides between the two live hypotheses.
     *
     * An LDC of a LINEAR-class capability performs a MOVE: load_unit.sv:225-226 fires a clear
     * for {LINEAR, REVOKE, UNINIT, SEALED, SEALEDRET} with write permission, and that clear
     * writes cursor 0, metadata 0, tag 0 (store_unit.sv:462-469) -- bit-for-bit create_cnull,
     * exactly the operand observed at the fault. If the clear fires every iteration, its write
     * competes with the next iteration's stc for the same granule in an 8-entry write buffer
     * that this kernel structurally overflows every iteration, and a merge landing out of order
     * reads back as a null.
     *
     * If the type is NONLIN the clear NEVER fires and that whole mechanism is dead, leaving the
     * stale-FLU-operand account standing alone.
     *
     * Post-shift encoding: LINEAR 0, NONLIN 1, REVOKE 2, UNINIT 3, SEALED 4, SEALEDRET 5,
     * EXIT 6, NOT_CAP 7. Reported in bits 8-11 of the return value, which are otherwise unused
     * (bad occupies 0-11 but is 0 on this arm, and the arm number sits at 12-15).
     *
     * WARNING -- THIS PACKING IS WHY THE FIRST READING OF THIS ARM WAS WRONG. The type shares one
     * word with the NOT_CAP counter, and arm 5 also runs the counting consumer below. So the
     * observed 0x200 had TWO readings -- type 2 (REVOKE) with a clean reload, or type 1 (NONLIN)
     * with the reload NOT_CAP on all 512 iterations -- and the folder recorded REVOKE for days.
     * It is NONLIN. With the subject store now hoisted above the arm chain the reload is clean,
     * so the low byte is 0 and the encoding is unambiguous again; if you ever reintroduce a
     * failing reload on this arm, the ambiguity comes back. Report the type in its own field. */
    bad = (bad & ~0xF00UL) | ((s12_type(v) & 0xF) << 8);
#endif

#if S12_ARM == 6
    /* DOES THE LDC MOVE-CLEAR ACTUALLY FIRE ON SILICON? Measure its CONSEQUENCE, not its type.
     *
     * The stored value is REVOKE-typed (measured: arm 5 returned type 2 post-shift = raw 3),
     * which is in the clear set at load_unit.sv:225-226. If the clear fires, an LDC of that slot
     * is a MOVE: it returns the value AND writes the granule to all-zero
     * (store_unit.sv:462-469 -- cursor 0, metadata 0, tag 0 = create_cnull).
     *
     * So reload the SAME slot twice. If the clear fires, the second reload MUST come back
     * NOT_CAP, every time. If it does not fire, the second reload is a valid capability.
     *
     * Retyping was the obvious alternative and is not available here: CAPTYPE lives on opcode
     * 0x7b, which no domain kernel uses and which is unverified inside a domain, and DELIN
     * raises UNEXPECTED_CAP_TYPE on anything that is not LINEAR (capstone_dyn_unit.anvil:476-477)
     * -- ours is REVOKE. This probe needs neither: it uses only lcc selector 1, which is TOTAL
     * and cannot raise, so the arm returns a RATE rather than wedging.
     *
     *   count == REPS -> the clear FIRES every iteration; the granule is zeroed by the reload
     *   count == 0    -> the clear never fires, and the whole clear-ordering candidate is DEAD
     *   0 < count     -> it fires SOMETIMES, which would itself be the defect
     */
    { void *back2 = *slot;
      if (s12_type(back2) == S12_NOT_CAP) bad++;
      s12_sink[4] = (unsigned long)(__UINTPTR_TYPE__)back2; }
#endif

#if S12_ARM == 7
    /* THE STALE-OPERAND SHAPE, AT PRODUCTION SPACING. This is the first arm built from a
     * MECHANISM rather than from the instruction list.
     *
     * WHAT THE BOARD ACTUALLY SHOWS, six wedges for six:
     *   - the granule holds a LIVE capability (cursor 0x827e4cd0, non-zero metadata)
     *   - the LATCHED tval is 0 and cap_type is NOT_CAP, i.e. bit-for-bit create_cnull
     *   - a4 at the halt holds the slot's cursor exactly, so THE LOAD DID WRITE IT
     * The driver's own pre-registered discriminator calls that "STALE-OPERAND CONFIRMED": the
     * load landed, and the consumer read something else.
     *
     * WHAT THAT SOMETHING ELSE IS. Production runs
     *     movc a4, zero      <- a4 := create_cnull, ALL ZERO
     *     stc  a4, 0(a5)
     *     ldc  a4, 0(a0)     <- same register, two instructions later
     *     cincoffsetimm a4, a4, 0xb0
     * so a4's PRE-LOAD value is exactly the operand observed at the fault. A consumer reading
     * the stale destination register instead of the load's result reproduces every measured
     * fact, including the bit pattern, with memory untouched.
     *
     * WHY ARMS 0-4 CANNOT REPRODUCE IT. They contain the same register reuse but the compiler
     * places `movc` NINE instructions before the reload; production places it TWO. The window
     * is the variable, and no C-level arm controls instruction spacing. Hence raw asm: one
     * block, four instructions, nothing schedulable between them.
     *
     * The consumer RAISES, as production does, so this arm reports by wedging or returning --
     * it cannot count. Returns the sentinel if it survives. */
    {
      void *volatile *_sp = slot;
      unsigned char volatile *_scr = fp - 0x120;
      __asm__ volatile(
          ".insn r 0x5b, 0x1, 0xa, %[t], x0, x0\n"          /* movc          t, zero      */
          ".insn s 0x5b, 0x4, %[t], 0(%[scr])\n"            /* stc           t, 0(scr)    */
          ".insn i 0x5b, 0x3, %[t], 0(%[sp])\n"             /* ldc           t, 0(sp)     */
          ".insn i 0x5b, 0x2, %[t], 0xb0(%[t])\n"           /* cincoffsetimm t, t, 0xb0   */
          : [t] "=&r"(_c7)
          : [scr] "r"(_scr), [sp] "r"(_sp)
          : "memory");
      s12_sink[5] = (unsigned long)(__UINTPTR_TYPE__)_c7;
    }
#endif

    /* ACCEPTANCE CRITERION, and it FAILS THE BUILD rather than the run.
     *
     * Arm 4 was added as "arm 1's shape with the production raising consumer", but the
     * `*slot = v` store lives inside the #if/#elif chain above and arm 4 matched NO branch
     * of it. So arm 4 read a slot it had never written. `s12_frame` is zero-initialised BSS,
     * which makes `back` a genuine NOT_CAP and the `cincoffsetimm` below raise mcause 25 --
     * EXACTLY the production symptom, produced by correct hardware doing what the ISA says.
     * Three boots of group-9 watchpoint measurements were spent walking that null backwards
     * up the chain; every one of them was measuring uninitialised memory.
     *
     * The failure is invisible to every check we had: the artifact is correct, the fault is
     * the right mcause at the right PC, the ladder is monotone, and the repro is beautifully
     * deterministic. Only the ABSENCE of a store distinguishes it, and nothing was looking
     * for an absence. Hence a compile-time assertion: an arm that reads the slot must have
     * declared that it wrote it. */
#if !defined(S12_SLOT_WRITTEN)
#  error "This arm reads the subject slot but no branch above writes it. The reload would return zero-initialised BSS and the consumer would raise mcause 25 for a trivially correct reason, which is indistinguishable from the defect under test. Add this arm to a branch that does `*slot = v` (and defines S12_SLOT_WRITTEN), or give it its own."
#endif

    /* THE RELOAD and its consumer. Reading through the volatile slot is the ldc; the type
     * query is the consumer, standing in for cincoffsetimm without the raise. */
    void *back = *slot;
#if S12_ARM != 4 && S12_ARM != 6
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
