#ifndef S06PLD_KERNEL_H
#define S06PLD_KERNEL_H
/* Does a PLAIN 8-byte LOAD from a granule that holds a capability cost that capability its tag?
 *
 * WHERE THIS CAME FROM -- disassembly, not a guess. The board fault is mcause 25 at memcpy+0x2a8,
 * on a destination pointer reloaded from the stack slot at s0-0x60 and found NOT_CAP. Listing
 * every instruction in memcpy that touches that granule gives exactly five:
 *
 *     153af4  stc    <- the spill: the capability is written
 *     153b18  ld     <- a PLAIN 8-byte load of its low half
 *     153bdc  ldc
 *     153c28  ldc
 *     153d64  ldc    <- the faulting reload
 *
 * ZERO PLAIN STORES. Nothing overwrites the granule between the store and the failing reload, so
 * the tag is lost with nothing writing to it -- which rules out ordinary, correct tag-clearing on
 * a partial overwrite, and rules out the write-buffer `.user` clobber (wt_dcache_wbuffer.sv:602),
 * since that requires a coalescing plain STORE to the same word and there is none.
 *
 * What IS there is a plain `ld`. A load must never disturb a tag.
 *
 * AND IT IS THE ONE INGREDIENT THE PASSING RUNGS LACK. s06spill (tag survives), s06bnds (bounds
 * survive) and s06wr (survives byte stores written through the pointer) all round-trip the slot
 * with `stc`/`ldc` only; not one of them reads the granule with a scalar load in between. Each
 * returns 65535 on silicon with a firing control, so the shapes they DO cover are sound.
 *
 * It also matches two older, separately recorded observations of the same shape: the -O0 strlen
 * defect ("ldc, lbu, ldc again from the same slot") and S-04, both of which differ from their
 * working -O1 forms precisely by round-tripping a capability through a stack slot that plain
 * accesses also touch.
 *
 * METHOD. Sixteen pointer locals, each in its own stack slot at -O0. For each: read the slot's
 * low half with a PLAIN load by type-punning the pointer variable's own storage, then query the
 * pointer's type. A barrier on either side keeps the compiler from folding the round trip away,
 * which would make the rung pass without ever creating the condition.
 *
 * retval = a 16-bit mask, bit i SET if the pointer was still a capability after the plain load.
 *   0xFFFF (65535)  a plain load does not disturb the tag -- this reading is refuted too
 *   anything else   the zero bits name which slots lost it, and the rung IS the minimal
 *                   out-of-SQLite reproduction the investigation has been after
 *
 * RESULT ON SILICON: 65535. A plain load does not disturb the tag either. FOUR shapes are now
 * refuted -- tag, bounds, byte stores written through the pointer, and a scalar load of the
 * granule -- so every simple construct in memcpy's inner loop is sound on this hardware and the
 * trigger needs something the rungs still do not have. Scale is the obvious remaining candidate:
 * memcpy in SQLite runs over large buffers with cache pressure, where a spilled line can be
 * EVICTED and refilled between the store and the reload; sixteen adjacent slots touched in a
 * tight loop never leave the cache. See s06evict.
 *
 * The type query is TOTAL, so a lost tag is REPORTED rather than faulted on: the rung returns a
 * number either way and measures all sixteen slots instead of dying at the first bad one.
 */

__attribute__((aligned(16))) static unsigned long s06pld_target[2];

#define S06PLD_LCC_TYPE(out_, cap_) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out_) : "r"(cap_))

#define S06PLD_BARRIER() __asm__ volatile("" ::: "memory")

/* The plain load is of the POINTER VARIABLE'S OWN STORAGE -- its low 8 bytes read as a scalar,
   which is exactly what `ld` at 153b18 does to memcpy's spilled `d`. Reading some other object
   would not create the condition. */
#define S06PLD_DECL(i) void *p##i = (void *)s06pld_target;
#define S06PLD_CHECK(i)                                                       \
  do {                                                                        \
    unsigned long ty_;                                                        \
    S06PLD_BARRIER();                                                         \
    scratch += *(volatile unsigned long *)(void *)&p##i;   /* the PLAIN ld */ \
    S06PLD_BARRIER();                                                         \
    S06PLD_LCC_TYPE(ty_, p##i);                                               \
    if (ty_ != 7UL) r |= (1u << (i));                                         \
  } while (0)

static unsigned s06pld_compute(void)
{
  unsigned r = 0;
  unsigned long scratch = 0;

  s06pld_target[0] = 0x5A5A5A5A5A5A5A5AUL;

  S06PLD_DECL(0)  S06PLD_DECL(1)  S06PLD_DECL(2)  S06PLD_DECL(3)
  S06PLD_DECL(4)  S06PLD_DECL(5)  S06PLD_DECL(6)  S06PLD_DECL(7)
  S06PLD_DECL(8)  S06PLD_DECL(9)  S06PLD_DECL(10) S06PLD_DECL(11)
  S06PLD_DECL(12) S06PLD_DECL(13) S06PLD_DECL(14) S06PLD_DECL(15)

  S06PLD_CHECK(0);  S06PLD_CHECK(1);  S06PLD_CHECK(2);  S06PLD_CHECK(3);
  S06PLD_CHECK(4);  S06PLD_CHECK(5);  S06PLD_CHECK(6);  S06PLD_CHECK(7);
  S06PLD_CHECK(8);  S06PLD_CHECK(9);  S06PLD_CHECK(10); S06PLD_CHECK(11);
  S06PLD_CHECK(12); S06PLD_CHECK(13); S06PLD_CHECK(14); S06PLD_CHECK(15);

#ifdef S06PLD_SELFTEST
  /* POSITIVE CONTROL: query a value that is NOT a capability and require the mask to collapse.
     65535 is otherwise indistinguishable from a query that cannot report 7 -- the failure this
     project keeps hitting. */
  {
    unsigned long junk_ = 0x5A5Aul, tj_;
    S06PLD_LCC_TYPE(tj_, junk_);
    if (tj_ == 7UL) r = 0u;
  }
#endif

  /* Consume the scalar so the plain loads cannot be optimised away. */
  if (scratch == 0x1234UL)
    r ^= 0x8000u;
  return r;
}
#endif /* S06PLD_KERNEL_H */
