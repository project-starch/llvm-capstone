#ifndef SBB_H
#define SBB_H
/* sbb = Store, Branch, Bitmask. The minimal repro of the S-03 failure itself.
 *
 * WHAT IS ALREADY KNOWN, so this probe tests only what is left:
 *   - The architectural value is CORRECT. sbr (the value-returning probe) does
 *     `movc t0,zero; stc t0,0(p); ld rd,0(p)` and returns rd; silicon returned 0 on three
 *     independent draws. The load is fine.
 *   - Yet at the S-03 site the branch IMMEDIATELY after that load behaves as if the value
 *     were nonzero, and inserting one `nop` before the branch fixes it (board arms
 *     gap/gap2 RETURN, gapN/n0 WEDGE, base WEDGE).
 *   - The general form "any branch after its producer reads stale" is REFUTED: `strlen`
 *     (0x14bdb4) has `lbu a2,0(a2); bnez a2` and terminates normally.
 *
 * So the surviving hypothesis is CONDITIONAL, and this probe is built to falsify it:
 * the branch mis-resolves only when its operand comes from an `ld` that itself immediately
 * follows a store TO THE SAME ADDRESS. That condition holds at the S-03 site and at
 * sqlite3Strlen30 (0x16b18, where it cannot be observed because both readings give the same
 * branch direction), and does NOT hold at strlen or at sqlite3FunctionSearch (0x144b04,
 * whose store is to a different address two instructions earlier).
 *
 * The slot is a STACK local, matching the S-03 site (`cincoffsetimm a0, s0, -0x50`) rather
 * than sbr's global, because provenance of the address capability is one of the few
 * remaining differences between the two.
 *
 * EVERY ARM RETURNS A BIT; nothing branches to a hang. A bit is SET when the branch was NOT
 * taken, i.e. it saw the value as nonzero -- which is WRONG, because the value is 0.
 *
 *   bit 0  A  stc ; ld ; beqz              <- the exact S-03 shape
 *   bit 1  B  stc ; ld ; nop ; beqz        <- control for A
 *   bit 2  C  sd  ; ld ; beqz              <- is a CAPABILITY store required?
 *   bit 3  D  sd  ; ld ; nop ; beqz        <- control for C
 *   bit 4  E  ld ; beqz, no adjacent store <- is the store required at all?
 *   bit 5  F  ld ; nop ; beqz              <- control for E
 *   bit 6  G  stc ; ld rA,0(rA) ; beqz rA        <- EXACT S-03 shape: the load's DESTINATION
 *                                                  register is its own ADDRESS register
 *   bit 7  H  stc ; ld rA,0(rA) ; nop ; beqz rA  <- control for G
 *   bit 8  I  ld rA,0(rA) ; beqz rA              <- tied register, no adjacent store
 *   bit 9  J  ld rA,0(rA) ; nop ; beqz rA        <- control for I
 *   bit 10 K  STORE-BUFFER PRESSURE, then stc ; ld rA,0(rA) ; beqz rA
 *   bit 11 L  control for K
 *   bit 12 M  STORE-BUFFER PRESSURE, then sd ; ld ; beqz
 *   bit 13 N  control for M
 *
 * K-N exist because arms A-J are all CLEAN on silicon while the identical sequence in SQLite
 * wedges, so the defect is context-dependent. The RTL gives a candidate context: a load whose
 * page offset [11:3] matches ANY pending store stalls in WAIT_PAGE_OFFSET until that store
 * drains (load_unit.sv:354-389, store_buffer.sv:257-280), so the load's completion latency --
 * and therefore which forwarding tier a dependent instruction can observe -- depends on how
 * busy the store buffer is. A tiny cold rung drains instantly; SQLite does not. K-N put real
 * pressure in the buffer first.
 *
 * Arms A-F all read CLEAN on silicon (retval 0xC0000000, three draws), so the condition is
 * NOT "a branch after a load after a same-address store" on its own. G-J add the one
 * remaining structural difference between the repro and the real site: at 0x13cb68 the load
 * overwrites the very register supplying its address. `strlen` also does this (lbu a2,0(a2))
 * and works -- but it has no store to that address immediately before it, so G is the first
 * arm carrying BOTH properties at once.
 *
 * Reading retval = 0xC0000000 | bits:
 *   0xC0000000  nothing reproduced -- the condition needs something this rung lacks.
 *   any ODD bit (1,3,5) set  -> the INSTRUMENT is broken, not the silicon. With a nop there
 *                               is no hazard to observe, so a control MUST read the true 0.
 *                               Believe nothing else in the run.
 *   bit 0 set, bit 4 clear   -> the adjacent same-address store IS the condition.
 *   bit 0 and bit 2 set      -> a plain scalar store suffices; not capability-specific.
 *   bit 0 set, bit 2 clear   -> a CAPABILITY store is required: dual-bank / R-18-R-19 family.
 *   bit 4 set                -> no store needed; the store is a red herring.
 */

#define SBB_MAGIC 0xC0000000u

/* R-16 redraw knob: never-executed nops shift image layout while leaving sbb_compute
 * byte-identical. Verify that identity across draws before trusting any of them. */
#ifndef SBB_PAD
#define SBB_PAD 0
#endif
#define SBB_STR2(x) #x
#define SBB_STR(x) SBB_STR2(x)
#if SBB_PAD > 0
__asm__(".pushsection .text\n\t.rept " SBB_STR(SBB_PAD) "\n\tnop\n\t.endr\n\t.popsection");
#endif

/* `r` ends up 1 when the branch was NOT taken, i.e. the branch disagreed with the value. */
#define SBB_ARM(r, slot, store, sep)                     \
  __asm__ volatile(store "\n\t"                          \
                   "ld   %0, 0(%1)\n\t"                  \
                   sep                                   \
                   "beqz %0, 1f\n\t"                     \
                   "li   %0, 1\n\t"                      \
                   "j    2f\n"                           \
                   "1:\tli %0, 0\n"                      \
                   "2:"                                  \
                   : "=&r"(r) : "r"(slot) : "t0", "memory")

/* The rung builder requires at least one gp[i] global access; this also doubles as proof
 * the cap-table path ran, since a failed carve would not read back 0. */
static volatile unsigned sbb_tag = 0;

/* Tied-register arm: the load's destination IS its address register, as at the S-03 site.
 * t1 carries the address capability in, the loaded integer out, and the verdict out. */
#define SBB_TIED(r, slot, store, sep)                    \
  __asm__ volatile("movc t1, %1\n\t"                     \
                   store "\n\t"                          \
                   "ld   t1, 0(t1)\n\t"                  \
                   sep                                   \
                   "beqz t1, 1f\n\t"                     \
                   "li   t1, 1\n\t"                      \
                   "j    2f\n"                            \
                   "1:\tli t1, 0\n"                       \
                   "2:\tmv %0, t1"                        \
                   : "=r"(r) : "r"(slot) : "t0", "t1", "memory")

static unsigned sbb_compute(void)
{
  volatile unsigned long slot[8];
  unsigned long a, b, c, d, e, f, g, h, i, j, k, l, m, n;

  slot[0] = 1; slot[2] = 1; slot[4] = 1;   /* seeded NON-zero: a store that never happened
                                            * cannot masquerade as a clean read */

  SBB_ARM(a, &slot[0], "movc t0, zero\n\tstc t0, 0(%1)", "");
  SBB_ARM(b, &slot[0], "movc t0, zero\n\tstc t0, 0(%1)", "nop\n\t");
  SBB_ARM(c, &slot[2], "sd zero, 0(%1)",                 "");
  SBB_ARM(d, &slot[2], "sd zero, 0(%1)",                 "nop\n\t");
  /* E/F: slot[4] was zeroed by nothing yet -- zero it in a SEPARATE asm block so no store
   * sits adjacent to the load inside the measured block. */
  __asm__ volatile("sd zero, 0(%0)" :: "r"(&slot[4]) : "memory");
  SBB_ARM(e, &slot[4], "nop",                            "");
  SBB_ARM(f, &slot[4], "nop",                            "nop\n\t");

  slot[6] = 1;
  SBB_TIED(g, &slot[6], "movc t0, zero\n\tstc t0, 0(t1)", "");
  SBB_TIED(h, &slot[6], "movc t0, zero\n\tstc t0, 0(t1)", "nop\n\t");
  SBB_TIED(i, &slot[6], "nop",                            "");
  SBB_TIED(j, &slot[6], "nop",                            "nop\n\t");

  /* Fill the store buffer with traffic to OTHER addresses, then run the sequence. The stores
   * are to distinct 8-byte words so they occupy separate entries rather than coalescing. */
#define SBB_PRESSURE                                          \
  "sd zero, 0x40(%1)\n\tsd zero, 0x48(%1)\n\t"               \
  "sd zero, 0x50(%1)\n\tsd zero, 0x58(%1)\n\t"               \
  "sd zero, 0x60(%1)\n\tsd zero, 0x68(%1)\n\t"               \
  "sd zero, 0x70(%1)\n\tsd zero, 0x78(%1)\n\t"
  slot[6] = 1;
  SBB_TIED(k, &slot[6], SBB_PRESSURE "movc t0, zero\n\tstc t0, 0(t1)", "");
  SBB_TIED(l, &slot[6], SBB_PRESSURE "movc t0, zero\n\tstc t0, 0(t1)", "nop\n\t");
  slot[6] = 1;
  SBB_TIED(m, &slot[6], SBB_PRESSURE "sd zero, 0(t1)",                   "");
  SBB_TIED(n, &slot[6], SBB_PRESSURE "sd zero, 0(t1)",                   "nop\n\t");

  return SBB_MAGIC
       | (sbb_tag & 0xC000u)
       | ((unsigned)a << 0) | ((unsigned)b << 1)
       | ((unsigned)c << 2) | ((unsigned)d << 3)
       | ((unsigned)e << 4) | ((unsigned)f << 5)
       | ((unsigned)g << 6) | ((unsigned)h << 7)
       | ((unsigned)i << 8) | ((unsigned)j << 9)
       | ((unsigned)k <<10) | ((unsigned)l <<11)
       | ((unsigned)m <<12) | ((unsigned)n <<13);
}
#endif
