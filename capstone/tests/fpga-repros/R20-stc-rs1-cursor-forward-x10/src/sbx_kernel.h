#ifndef SBX_H
#define SBX_H
/* sbx = Store/load tied-register repro, x10 vs a control register.
 *
 * THE POINT. An earlier probe (`sbb`) built the S-03 instruction sequence in a 13 KB domain and
 * came back CLEAN, which was read for hours as "the defect is context-dependent, the tiny rung
 * lacks something SQLite has". That was wrong, and cost real board time. `sbb`'s tied-register
 * arm was written on **t1/x6**. The defect is specific to **x10/a0**:
 *
 *   issue_read_operands.sv:566  marks an in-flight capability op's rs1 as clobbered, so a reader
 *                               of that register STALLS;
 *   issue_read_operands.sv:568  then unconditionally OVERWRITES that entry for x10 alone,
 *                               keeping the claim only for CAPENTER;
 *   issue_read_operands.sv:674-677 + check_fwd_rs1 (ariane_pkg.sv:929-935, which includes STC)
 *                               serve a reader whose rs1 matches an in-flight STC's rs1 with
 *                               that STC's rs1_cursor -- i.e. the STORE'S BASE ADDRESS.
 *
 * So on x10 the reader does not stall, issues alongside the in-flight STC, and is handed the
 * store's base address instead of the loaded value. On any other register the stall stands.
 *
 * This rung runs the SAME sequence twice -- once tied on a0, once on t1 -- so a single run shows
 * the defect AND its register-specificity, and the t1 arm doubles as an internal control.
 *
 * Every arm returns a bit; nothing can hang. A bit is SET when the branch was NOT taken, i.e. it
 * saw a nonzero value -- which is WRONG, because the slot was just stored with a NULL capability.
 *
 *   bit 0  a0 tied, 0 nops before the branch
 *   bit 1  a0 tied, 1 nop
 *   bit 2  a0 tied, 2 nops
 *   bit 3  a0 tied, 4 nops       -- the separation sweep: where does the window close?
 *   bit 4  t1 tied, 0 nops       <- NEGATIVE CONTROL: same sequence, different register
 *   bit 5  a0, plain `sd` store instead of `stc`, 0 nops  <- is a CAPABILITY store required?
 *   bit 6  a0, no store at all, 0 nops                    <- is the store required at all?
 *
 * retval = 0xD0000000 | bits.
 * INSTRUMENT VALIDATION is bits 4 and 6, NOT the nop arms. If bit 4 (same sequence on t1) or
 * bit 6 (no store) is SET, the probe cannot tell a good case from a bad one and the whole run is
 * void. The nop arms are a MEASUREMENT, not a control: how much separation closes the window is
 * one of the things being measured, and a nonzero answer there is a result, not a fault.
 *
 *   bit 0 set, bit 4 clear   the defect, reproduced in 13 KB and confined to x10.
 *   bits 0-3 all set         one, two and four nops all fail to close the window here.
 *   bit 5 set                a plain scalar store is enough -- NOT capability-specific.
 *   bit 5 clear, bit 0 set   a CAPABILITY store is required.
 *   bit 4 or 6 set           INSTRUMENT VOID -- believe nothing in this run.
 *
 * The 0xD magic distinguishes a real 0 result from "rung not compiled in" / "harness reported
 * nothing", both of which have produced false CLEAN verdicts on this project.
 */

#define SBX_MAGIC 0xD0000000u

/* R-16 redraw knob: never-executed nops shift image layout while leaving sbx_compute
 * byte-identical. Verify that identity across draws before trusting any draw. */
#ifndef SBX_PAD
#define SBX_PAD 0
#endif
#define SBX_STR2(x) #x
#define SBX_STR(x) SBX_STR2(x)
#if SBX_PAD > 0
__asm__(".pushsection .text\n\t.rept " SBX_STR(SBX_PAD) "\n\tnop\n\t.endr\n\t.popsection");
#endif

/* The tied register is named explicitly and listed as a clobber, so the compiler cannot allocate
 * an operand to it. `sep` is the only difference between an arm and its control. */
#define SBX_ARM(r, slot, REG, sep)                        \
  __asm__ volatile("movc " REG ", %1\n\t"                 \
                   "movc t0, zero\n\t"                    \
                   "stc  t0, 0(" REG ")\n\t"              \
                   "ld   " REG ", 0(" REG ")\n\t"         \
                   sep                                    \
                   "beqz " REG ", 1f\n\t"                 \
                   "li   " REG ", 1\n\t"                  \
                   "j    2f\n"                            \
                   "1:\tli " REG ", 0\n"                  \
                   "2:\tmv %0, " REG                      \
                   : "=r"(r) : "r"(slot) : REG, "t0", "memory")

/* Same shape, but a PLAIN SCALAR store -- does the trigger need a capability store? */
#define SBX_SD(r, slot, REG, sep)                         \
  __asm__ volatile("movc " REG ", %1\n\t"                 \
                   "sd   zero, 0(" REG ")\n\t"            \
                   "ld   " REG ", 0(" REG ")\n\t"         \
                   sep                                    \
                   "beqz " REG ", 1f\n\t"                 \
                   "li   " REG ", 1\n\t"                  \
                   "j    2f\n"                             \
                   "1:\tli " REG ", 0\n"                   \
                   "2:\tmv %0, " REG                       \
                   : "=r"(r) : "r"(slot) : REG, "t0", "memory")

/* Same shape with NO adjacent store -- negative control; must read the true 0. */
#define SBX_NOST(r, slot, REG, sep)                       \
  __asm__ volatile("movc " REG ", %1\n\t"                 \
                   "ld   " REG ", 0(" REG ")\n\t"         \
                   sep                                    \
                   "beqz " REG ", 1f\n\t"                 \
                   "li   " REG ", 1\n\t"                  \
                   "j    2f\n"                             \
                   "1:\tli " REG ", 0\n"                   \
                   "2:\tmv %0, " REG                       \
                   : "=r"(r) : "r"(slot) : REG, "t0", "memory")

/* the rung builder requires at least one gp[i] global access */
static volatile unsigned sbx_tag = 0;

static unsigned sbx_compute(void)
{
  volatile unsigned long slot[8];
  unsigned long a, b, c, d, e, f, g;

  slot[0] = 1; slot[2] = 1; slot[4] = 1;  /* seeded NON-zero: a store that never happened
                                           * cannot masquerade as a clean read */

  SBX_ARM(a, &slot[0], "a0", "");
  SBX_ARM(b, &slot[0], "a0", "nop\n\t");
  SBX_ARM(c, &slot[0], "a0", "nop\n\tnop\n\t");
  SBX_ARM(d, &slot[0], "a0", "nop\n\tnop\n\tnop\n\tnop\n\t");
  SBX_ARM(e, &slot[2], "t1", "");                       /* negative control: other register */
  SBX_SD (f, &slot[4], "a0", "");                       /* scalar store instead of stc */
  /* zero slot[6] in a SEPARATE block so no store is adjacent inside the measured one */
  __asm__ volatile("sd zero, 0(%0)" :: "r"(&slot[6]) : "memory");
  SBX_NOST(g, &slot[6], "a0", "");                      /* negative control: no store at all */

  return SBX_MAGIC
       | (sbx_tag & 0xFF80u)
       | ((unsigned)a << 0) | ((unsigned)b << 1) | ((unsigned)c << 2)
       | ((unsigned)d << 3) | ((unsigned)e << 4) | ((unsigned)f << 5)
       | ((unsigned)g << 6);
}
#endif
