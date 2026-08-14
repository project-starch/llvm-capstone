#ifndef S06WR_KERNEL_H
#define S06WR_KERNEL_H
/* Does a spilled capability keep its tag while BYTE STORES ARE WRITTEN THROUGH IT?
 *
 * WHY, and why the two rungs before this one do not answer it. s06spill shows a spilled
 * capability comes back still TAGGED (0xFFFF on silicon); s06bnds shows its BOUNDS survive too
 * (65535 on silicon), both with positive controls that fire. So a bare spill/reload is sound on
 * this hardware and the obvious reading of the memcpy fault is already refuted.
 *
 * But neither rung writes anything between the spill and the reload. memcpy does, and that is the
 * one structural difference. The measured fault, extended phase 2->3 with the granule guard on,
 * mcause 25 UNEXPECTED_OPERAND at memcpy+0x2a8:
 *
 *     ld            a1, 0x0(a1)
 *     cincoffset    a0, a0, a1
 *     lbu           a0, 0x0(a0)        ; read a source byte
 *     cincoffsetimm a2, s0, -0x60
 *     ldc           a2, 0x0(a2)        ; reload the DEST pointer from its stack slot
 *     cincoffset    a1, a2, a1         <== raises: a2 is NOT_CAP
 *     sb            a0, 0x0(a1)        ; store the byte
 *
 * That is memcpy's byte tail loop: reload the destination pointer from a stack slot, offset it,
 * store one byte, repeat. At -O0 the pointer round-trips through its slot on every iteration, so
 * the loop is a spill/reload interleaved with plain stores through the very pointer being spilled.
 * This rung is that shape and nothing else.
 *
 * retval = a 16-bit mask, bit i SET if the pointer was still a capability on iteration i.
 *   0xFFFF (65535)  the tag survives byte stores through it -- this shape is exonerated too, and
 *                   whatever distinguishes memcpy is still elsewhere
 *   anything else   the zero bits name the iteration at which the tag died, and the rung IS the
 *                   minimal out-of-SQLite reproduction
 *
 * RESULT ON SILICON: 65535. Writing through a spilled pointer does not damage its tag either, so
 * this reading is refuted as well. Three shapes now pass -- tag, bounds, write-through -- and
 * memcpy still faults, which means the difference is in memcpy's own code rather than in any
 * simple property of a spill.
 *
 * DO NOT TRY TO TEST THE REAL memcpy AS A LADDER RUNG. build-ladder-domain.sh compiles exactly
 * ONE C file, so a rung does not link beebs_freestanding_string.c and `memcpy` there is whatever
 * the compiler emits -- a different primitive from the one SQLite calls. An attempt was made and
 * deleted. The vehicle for exercising the real memcpy is the SQLite domain, which links it.
 *
 * The type query is TOTAL, so a lost tag is REPORTED, not faulted on -- the rung returns a number
 * either way and measures all sixteen iterations instead of dying at the first bad one. It never
 * issues a bounds selector, which would raise on exactly the NOT_CAP input it exists to catch.
 */

__attribute__((aligned(16))) static unsigned char s06wr_dst[256];
__attribute__((aligned(16))) static unsigned char s06wr_src[256];

#define S06WR_LCC_TYPE(out_, cap_) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out_) : "r"(cap_))

/* Compiler barrier: forces the pointer back through its stack slot each iteration, which is the
   whole construct under test. Without it the compiler keeps it in a register and the rung passes
   without ever spilling -- a gate that cannot fire. */
#define S06WR_BARRIER() __asm__ volatile("" ::: "memory")

static unsigned s06wr_compute(void)
{
  unsigned r = 0;
  unsigned i;
  unsigned char *d = s06wr_dst;
  const unsigned char *s = s06wr_src;

  for (i = 0; i < 16u; i++)
    s06wr_src[i] = (unsigned char)(0x40u + i);

  for (i = 0; i < 16u; i++) {
    unsigned long ty_;

    /* Force the spill, then read the tag back through the reloaded value -- the same order memcpy
       has: the pointer is live in its slot across a plain store made through it. */
    S06WR_BARRIER();
    S06WR_LCC_TYPE(ty_, d);
    if (ty_ != 7UL)
      r |= (1u << i);

#ifdef S06WR_SELFTEST
    /* POSITIVE CONTROL: query a value that is NOT a capability, so the mask must come back 0. A
       rung returning 0xFFFF proves nothing until the query is shown able to report NOT_CAP, and
       "everything was tagged" is exactly what a query on the wrong operand would also print. */
    {
      unsigned long junk_ = 0x5A5A5A5AUL, tj_;
      S06WR_LCC_TYPE(tj_, junk_);
      if (tj_ != 7UL) r |= (1u << i); else r &= ~(1u << i);
    }
#endif

    /* The store through the spilled pointer -- the thing s06spill and s06bnds never do. */
    d[i] = s[i];
    S06WR_BARRIER();
  }

  /* Consume the destination so the stores cannot be optimised away. */
  if (s06wr_dst[15] == 0xFFu)
    r ^= 0x8000u;

  return r;
}
#endif /* S06WR_KERNEL_H */
