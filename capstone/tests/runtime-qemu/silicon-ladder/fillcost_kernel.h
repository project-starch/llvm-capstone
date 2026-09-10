#ifndef FILLCOST_KERNEL_H
#define FILLCOST_KERNEL_H
/* Silicon-ladder rung: what does the RECLAIM FILL actually cost?
 *
 * WHY THIS RUNG EXISTS. The monitor's reclaim (R-30/R-31 firmware half) overwrites a revoked
 * region before reusing it: one capability store per 16 bytes, 256 of them for the 4 KiB every
 * region in this tree is. Whether that is affordable was the open question in the reclaim
 * decision, and the honest answer was a bracket -- roughly 3 % to 25 % of the boundary path --
 * whose whole width came from ONE unmeasured quantity: what 4 KiB of capability stores costs on
 * this silicon. Everything else in that estimate is measured.
 *
 * WHY IT DOES NOT NEED THE FLASH, which is the point of measuring it this way. The reclaim PATH
 * cannot be exercised on the current bitstream: revoke there returns LINEAR for a writable region
 * (the R-31 defect), so the monitor's UNINIT guard never fires. But the fill's PHYSICAL WORK --
 * 256 capability stores walking a page -- is a pure memory-system quantity and does not depend on
 * the capability's type at all. So it is measurable today, on the flashed bitstream.
 *
 * WHY RAW ASM AND NOT A C LOOP. The first version of this rung was a C `for` loop over a
 * capability-typed double pointer. It emitted the right instruction, and it measured the wrong
 * thing: at -O0 the domain build spills everything, so the loop body was NINETEEN instructions
 * per store, of which one was the stc. A cycle count dominated 18:1 by loop overhead cannot
 * answer "what does the store cost", and backing the store out of it would have meant stacking
 * an assumed CPI for the overhead on top of a measured total. The monitor's own fill is a
 * four-instruction asm loop (`beq / stc / addi / j`), so this rung is a four-instruction asm loop
 * too, and measures the thing the decision turns on.
 *
 * THE MATCHED CONTROL IS THE MEASUREMENT. `fillnop` includes this same header with FILL_PAYLOAD
 * set to a nop, so the two rungs differ by EXACTLY ONE INSTRUCTION -- same iteration count, same
 * pointer walk, same branch, same instruction count -- and
 *
 *     (fillcost cycles - fillnop cycles) / 256
 *
 * is the marginal cost of one 16-byte capability store on this silicon, with the loop overhead
 * cancelled rather than assumed. Neither arm alone gives that number.
 *
 * PREDICTED READING, written before the run: 4 instructions x 256 = ~1024 in each arm's bracket,
 * plus a few for the prologue. Cycles: `fillnop` should sit near 1.1-1.5 cyc/instr (three trivial
 * ALU ops and a predictable backward branch); `fillcost` above it by whatever a capability store
 * costs. The measured 1024-byte copy rate of 3.52 cyc/byte extrapolates to a 14,400-cycle ceiling
 * for the fill, which OVERSTATES it because a copy loads and stores where a fill only stores.
 * A per-store delta near 3 cycles makes the reclaim a few percent of the boundary path; near 50
 * makes it a quarter of it.
 *
 * A READING NEAR ZERO IS NOT GOOD NEWS -- it means the loop was optimised away or never ran.
 * The RETURN VALUE is the positive control for exactly that, and it reports two separate facts:
 *
 *     +256   the loop executed 256 iterations (the asm counter is returned, not assumed)
 *     +512   slot 0, seeded NON-ZERO before the loop, read back as zero -- the store LANDED
 *
 * so fillcost must return 768 and fillnop 256. A deleted loop reads 0 in both. This matters
 * because the buffer is BSS: a version of this rung that only checked "does it read back zero"
 * would have passed with the loop removed entirely, since BSS is already zero. */

#ifndef FILL_PAYLOAD
/* stc x0, 0(p) -- store the null capability, exactly what the monitor's C_RECLAIM stores */
#define FILL_PAYLOAD ".insn s 0x5b, 0x4, x0, 0(%[p])\n"
#endif
#ifndef FILL_STORES
#define FILL_STORES 1
#endif
/* FILL_PASSES > 1 runs the loop again over the SAME buffer, so (that rung - fillcost) is the cost of
   a pass over a region the fill has just touched. That is the only way to settle whether "cold"
   drives the 23.6 cycles/store figure: this cache is write-through with NO write-allocate
   (wt_dcache_wbuffer.sv:43-44), so a warm region is NOT obviously cheaper and the answer cannot be
   assumed in either direction. */
#ifndef FILL_PASSES
#define FILL_PASSES 1
#endif
/* FILL_TAG keeps every rung's retval distinct, so "not compiled in" can never read as another
   arm's pass. */
#ifndef FILL_TAG
#define FILL_TAG 0
#endif

#define FILLCOST_BYTES 4096
#define FILLCOST_SLOTS (FILLCOST_BYTES / 16)

static void *fillcost_buf[FILLCOST_SLOTS];

static unsigned fillcost_compute(void)
{
  void **p = fillcost_buf;
  unsigned long n = FILLCOST_SLOTS;
  unsigned long k = 0;
  unsigned r = 0;
  unsigned pass = 0;

  /* Seed slot 0 AND slot 255 non-zero. Slot 0 alone was not enough: it is the FIRST slot the loop
     touches, so it reads back zero whether or not the pointer walk advanced -- the control could
     not tell "walked 4 KiB" from "stored 256 times to the same granule". Slot 255 is reached only
     if the walk works. (Audit, 2026-09-10.) */
  fillcost_buf[0] = (void *)fillcost_buf;
  fillcost_buf[FILLCOST_SLOTS - 1] = (void *)fillcost_buf;

/* THE GUARD MACRO IS `__CAPSTONE__`, NOT `__riscv`. The domain target is
   capstone64-unknown-elf and it does NOT predefine __riscv, so the first version of this rung
   guarded on __riscv, silently fell through to the portable C loop, and built the very
   19-instruction shape this rewrite exists to remove. Nothing caught it at build time and the
   oracle could not: the C loop stores too, so QEMU returned the same 768. Only the disassembly
   showed it. Hence the #error below -- a target that is neither the domain nor the declared
   native oracle now fails to compile instead of quietly measuring the wrong thing. */
  for (pass = 0; pass < FILL_PASSES; pass += 1) {
  p = fillcost_buf;
  k = 0;
#if defined(__CAPSTONE__)
  /* Four instructions, nothing schedulable between them, and the loop counter is an OUTPUT so
     the iteration count is measured rather than trusted. cincoffsetimm walks the pointer by 16
     because the fill has to cover the page; the monitor gets that walk free from the UNINIT
     cursor advance, which this bitstream does not provide for a LINEAR capability. */
  __asm__ volatile(
      "1: " FILL_PAYLOAD
      "   .insn i 0x5b, 0x2, %[p], 16(%[p])\n"   /* cincoffsetimm p, p, 16 */
      "   addi %[k], %[k], 1\n"
      "   bltu %[k], %[n], 1b\n"
      : [p] "+r"(p), [k] "+r"(k)
      : [n] "r"(n)
      : "memory");
#elif defined(FILLCOST_NATIVE_ORACLE)
  /* Native oracle: the same iteration count, so the expected value is computed and not typed in.
     x86 cannot execute stc, so the STORE half is modelled by FILL_STORES rather than emitted. */
  for (k = 0; k < n; k += 1) {
#if FILL_STORES
    p[k] = 0;
#endif
  }
#else
#error "fillcost: neither the Capstone domain target nor a declared native oracle build"
#endif
  r += (unsigned)k;                             /* +256 per pass : the loop ran */
  }

  if (fillcost_buf[0] == 0)
    r += 512;                                   /* +512 : the store landed at the FIRST slot */
  if (fillcost_buf[FILLCOST_SLOTS - 1] == 0)
    r += 1024;                                  /* +1024 : and at the LAST -- the walk covered 4 KiB */
  r += FILL_TAG;                                /* per-rung sentinel */
  return r;
}
#endif
