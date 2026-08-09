#ifndef SBR_H
#define SBR_H
/* sbr = Store-then-load Byte-Return probe.
 *
 * Minimal standalone probe for the S-03 site. In sqlite3InsertBuiltinFuncs the machine does
 *
 *      stc  a1, 0x0(a0)      ; capability store of pOther (NULL) to a stack slot
 *      ld   a0, 0x0(a0)      ; read the SAME address back as a plain 64-bit integer
 *      beqz a0, <else>       ; NOT taken on silicon, so `if (pOther)` runs with pOther NULL
 *
 * and the subsequent deref of the NULL pOther wedges. Board arms established that pOther IS
 * NULL and that the branch does not take; they did NOT establish why.
 *
 * STOP INFERRING, MEASURE. Every previous arm inferred the loaded value from whether a branch
 * was taken, which needs a model of the branch to interpret -- and the model that fit all
 * seventeen arms was refuted by `strlen`, which has the same branch-after-producer shape and
 * plainly works. So this probe contains NO BRANCH on the value under test: it performs the
 * store/load pair and RETURNS THE LOADED BITS. One number, no model required.
 *
 * Reading the result. retval = 0xA0000000 | sd_flag<<28 | (stc_loaded & 0x0FFFFFFF):
 *
 *   0xA0000000  both paths read back 0. The load is FINE; the defect is elsewhere and the
 *               branch becomes the suspect again.
 *   0xA8000000  the SCALAR store/load path is broken too (sd then ld did not read 0).
 *               Instrument or a far broader defect -- believe nothing else in the run.
 *   0xA8000000-ish with low bits set, e.g. 0xA0000000|0x08000000 -> 0xA8000000 collision:
 *               see below; bit 27 of the payload is what matters and is reported separately
 *               by the 0x0FFFFFFF mask, so a payload of 0x08000000 shows as 0xA8000000 ONLY
 *               if sd_flag is clear -- disambiguate with the sd_flag bit at 28.
 *   any nonzero payload  the `ld` after a same-address `stc` returns the WRONG BITS. A payload
 *               of 0x08000000 specifically is the compressed metadata word of a NULL
 *               capability (bit 27 = `cursorless`), i.e. the load returned the metadata half
 *               instead of the cursor half -- the dual-bank D-cache path, R-18/R-19 family.
 *
 * The 0xA magic in the top nibble is deliberate: a bare 0 is indistinguishable from "the rung
 * was not compiled in" and from "the harness reported nothing", both of which have produced
 * false CLEAN verdicts on this project.
 */

#define SBR_MAGIC 0xA0000000u

/* R-16 REDRAW KNOB. The SHA5 entry stall is per-image and deterministic, so retrying a
 * stalling binary is futile -- a fresh DRAW is needed. SBR_PAD emits that many never-executed
 * nops ahead of the code, shifting image layout while leaving sbr_compute byte-identical.
 * Verify that identity across draws before trusting any of them. */
#ifndef SBR_PAD
#define SBR_PAD 0
#endif
#define SBR_STR2(x) #x
#define SBR_STR(x) SBR_STR2(x)
#if SBR_PAD > 0
__asm__(".pushsection .text\n\t.rept " SBR_STR(SBR_PAD) "\n\tnop\n\t.endr\n\t.popsection");
#endif

static volatile unsigned long sbr_slot[4] = { 1, 1, 1, 1 };  /* seeded NON-zero, so a store
                                                              * that never happened cannot
                                                              * masquerade as a clean read */

static unsigned sbr_compute(void)
{
  unsigned long v_stc, v_sd;

  /* ARM 1 -- THE SQLITE SHAPE. Capability store of NULL, then a plain integer load of the
   * SAME address. No branch: the loaded bits are the result. */
  __asm__ volatile(
      "movc t0, zero\n\t"        /* t0 = NULL capability            */
      "stc  t0, 0(%1)\n\t"       /* capability store to the slot    */
      "ld   %0, 0(%1)\n\t"       /* scalar load of the same address */
      : "=r"(v_stc) : "r"(&sbr_slot[0]) : "t0", "memory");

  /* ARM 2 -- CONTROL. Identical except the store is a plain scalar `sd`. If this reads back
   * nonzero the instrument is broken, not the silicon. */
  __asm__ volatile(
      "sd   zero, 0(%1)\n\t"
      "ld   %0, 0(%1)\n\t"
      : "=r"(v_sd) : "r"(&sbr_slot[2]) : "memory");

  return SBR_MAGIC
       | ((v_sd != 0) ? 0x10000000u : 0u)
       | ((unsigned)v_stc & 0x0FFFFFFFu);
}
#endif
