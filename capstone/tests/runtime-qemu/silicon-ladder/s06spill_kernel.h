#ifndef S06SPILL_KERNEL_H
#define S06SPILL_KERNEL_H
/* Does a CAPABILITY SPILLED TO THE STACK come back from `ldc` still tagged?
 *
 * WHY. With the S-06 codegen guard enabled, SQLite stops faulting at
 * vdbeMemClearExternAndSetNull and instead faults in three different places with the SAME shape,
 * all mcause 25 (UNEXPECTED_OPERAND):
 *
 *   sqlite3_strnicmp+0x134   ldc from s0-0x70 -> cincoffsetimm a0, a0, 1
 *   memcpy+0x298             ldc from s0-0x70 -> cincoffset a0, a0, a1
 *
 * The strnicmp one is decisive about WHICH operand is wrong: `cincoffsetimm` is the IMMEDIATE
 * form, so rs2 is not involved and the stale-tag-in-rs2 reading cannot apply -- the reloaded
 * capability itself is NOT_CAP. (CINCOFFSET's register form raises on EITHER rs1 being NOT_CAP
 * or rs2 being tagged, which is why the memcpy site alone would have been ambiguous.)
 *
 * The guard did not generate that code: sqlite3_strnicmp is BYTE-IDENTICAL guarded and
 * unguarded, and the guarded build runs end-to-end under QEMU, which clears a granule's tag on a
 * plain store and would therefore catch a pass that untagged capabilities. What the guard does is
 * add a branch per granule, multiplying basic blocks and hence capability SPILLS -- so the
 * hypothesis is that it EXPOSES a spill/reload defect rather than causing one.
 *
 * TWO READINGS, AND THIS RUNG SEPARATES THEM:
 *   A  spill/reload of a capability loses the tag, at some or all stack offsets
 *   B  nothing is wrong with spilling; the fault moves because the IMAGE moved (S-01
 *      image-perturbation sensitivity), and s0-0x70 twice is a coincidence of frame layout
 *
 * Under A this rung fails and names the offsets. Under B it returns a clean 16 and the SQLite
 * fault must be attributed to perturbation instead -- which is a very different next step, so
 * conflating them would send the whole investigation the wrong way.
 *
 * METHOD. Sixteen separate pointer LOCALS, each holding a genuine capability, each therefore
 * living in its own stack slot at -O0. They are written, then a barrier keeps the compiler from
 * folding the reload into the store, then each is read back and its type queried. The slots span
 * several hundred bytes of frame, so an offset-dependent failure shows up as a PATTERN of bits
 * rather than a single yes/no.
 *
 * retval = a 16-bit mask, bit i set if local i survived the round trip as a capability.
 *   0xFFFF (65535)  every spill survived -> reading B, spilling is sound
 *   anything else   reading A, and the zero bits name which slots lost the tag
 *
 * The type query is TOTAL, so a lost tag is REPORTED rather than faulting -- this rung returns a
 * number in both cases instead of wedging, which is the whole point of using LCC to ask.
 *
 * Deliberately NOT dereferencing the reloaded pointers: an untagged one would fault and the run
 * would tell us nothing about the remaining slots. The question here is how many survive, and a
 * wedge answers it for at most one.
 */

/* REDRAW PADDING. R-16 (entry stall) is per-image, not per-binary: retrying the same bytes is
   futile, and the documented remedy is to rebuild with a harmless constant varied so the code
   under test stays byte-identical while the image moves. S06SPILL_REDRAW=<n> emits n unused
   words; it is referenced by a volatile read that is never reached, so the tested loop is
   untouched. sha256sum every draw and abort if two match. */
#ifdef S06SPILL_REDRAW
__attribute__((used, aligned(16))) static volatile unsigned long s06spill_pad[S06SPILL_REDRAW];
#endif

__attribute__((aligned(16))) static unsigned long s06spill_target[2];

#define S06SPILL_LCC_TYPE(out, cap) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out) : "r"(cap))

/* A compiler barrier, not a hardware one: it stops the reload being folded away, which would
   make the rung pass without ever spilling anything -- a gate that cannot fire. */
#define S06SPILL_BARRIER() __asm__ volatile("" ::: "memory")

#define S06SPILL_DECL(i) void *p##i = (void *)s06spill_target;
#define S06SPILL_CHECK(i)                          \
  do {                                             \
    unsigned long ty_;                             \
    S06SPILL_LCC_TYPE(ty_, p##i);                  \
    if (ty_ != 7UL) r |= (1u << (i));              \
  } while (0)

static unsigned s06spill_compute(void)
{
  unsigned r = 0;

  s06spill_target[0] = 0x5A5A5A5A5A5A5A5AUL;

  S06SPILL_DECL(0)  S06SPILL_DECL(1)  S06SPILL_DECL(2)  S06SPILL_DECL(3)
  S06SPILL_DECL(4)  S06SPILL_DECL(5)  S06SPILL_DECL(6)  S06SPILL_DECL(7)
  S06SPILL_DECL(8)  S06SPILL_DECL(9)  S06SPILL_DECL(10) S06SPILL_DECL(11)
  S06SPILL_DECL(12) S06SPILL_DECL(13) S06SPILL_DECL(14) S06SPILL_DECL(15)

  S06SPILL_BARRIER();

  S06SPILL_CHECK(0);  S06SPILL_CHECK(1);  S06SPILL_CHECK(2);  S06SPILL_CHECK(3);
  S06SPILL_CHECK(4);  S06SPILL_CHECK(5);  S06SPILL_CHECK(6);  S06SPILL_CHECK(7);
  S06SPILL_CHECK(8);  S06SPILL_CHECK(9);  S06SPILL_CHECK(10); S06SPILL_CHECK(11);
  S06SPILL_CHECK(12); S06SPILL_CHECK(13); S06SPILL_CHECK(14); S06SPILL_CHECK(15);

#ifdef S06SPILL_SELFTEST
  /* POSITIVE CONTROL, added 2026-08-14. This rung shipped without one while the three rungs
     built after it (s06bnds, s06wr, s06pld) all carry the same six lines -- and 65535 from a
     query that cannot return anything else is not a measurement, it is an unproven instrument.
     Query a value that is NOT a capability and require the mask to collapse to 0.

     The control is behind an #ifdef, so the clean build is byte-identical to the one already
     run on silicon; adding this does not invalidate that 65535. */
  {
    unsigned long junk_ = 0x5A5Aul, tj_;
    S06SPILL_LCC_TYPE(tj_, junk_);
    if (tj_ == 7UL) r = 0u;
  }
#endif

  return r;
}
#endif /* S06SPILL_KERNEL_H */
