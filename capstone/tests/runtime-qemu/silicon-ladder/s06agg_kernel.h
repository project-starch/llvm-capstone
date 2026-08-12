#ifndef S06AGG_KERNEL_H
#define S06AGG_KERNEL_H
/* MINIMAL REPRO of S-06 reaching a program through the COMPILER'S OWN struct copy.
 *
 * S-06 was fixed once, in the C memcpy (beebs_freestanding_string.c, BEEBS_CHUNK_COPY, which now
 * asks LCC whether a granule really holds a capability before issuing the `stc`). That covered
 * the library and nothing else. The compiler independently lowers 16-byte-aligned aggregate
 * copies to bare `ldc`/`stc` granule pairs -- 283 such stores across 41 copy runs in the SQLite
 * domain, including a 112-byte `Mem` copy inlined into sqlite3VdbeExec -- and those carry the
 * identical defect. Measured on the board 2026-08-12: repairing 217 of them made SQLite's wild
 * `Mem*` disappear entirely.
 *
 * This rung is that bug in ~20 lines instead of 1.4 MB of SQLite, so the fix can be gated on
 * something that runs in seconds and says exactly what broke.
 *
 * THE TRIGGER IS `low half % 0x4000 == 0`, NOT `low half == 0`. Measured in RTL simulation
 * (s06-lowhalf-zero.S / -swap.S, TESTNUM 5): `ldc` of an untagged granule loads metadata 0, the
 * pipeline re-encodes that zero against the CURSOR, and compress_bounds switches to its
 * cursorless scheme when bounds.start == cursor -- which decompress_bounds(0, cursor) makes true
 * exactly when the cursor's low 14 bits are zero. The manufactured nonzero metadata then asserts
 * st_wr_cap, both banks are written, and the high half is destroyed. So the probe granule below
 * uses 0x4000, a NONZERO value, deliberately: a repro keyed on zero would miss the general case
 * and would also be satisfied by a narrower fix that is still wrong.
 *
 * SELF-CHECKING, WITH THE CONTROL IN THE SAME COPY. Both granules go through ONE struct
 * assignment, so they share the copy, the alignment and the code path; the only difference is the
 * low half's value.
 *
 *   granule 0   lo = 0x4000      (0x4000-ALIGNED)      hi = sentinel A   <- the PROBE
 *   granule 1   lo = 0x12345678  (not aligned)         hi = sentinel B   <- the CONTROL
 *
 * retval is a BITMASK, one bit per 8-byte half, so a partial result names exactly which halves
 * survived. 15 is a correct copy.
 *
 *   bit 0  probe   LOW  half intact (0x4000)
 *   bit 1  probe   HIGH half intact (sentinel A)
 *   bit 2  control LOW  half intact (0x12345678)
 *   bit 3  control HIGH half intact (sentinel B)
 *
 *   15  correct copy -- what QEMU and the native oracle return, and what the codegen fix must
 *       make the board return
 *    5  MEASURED ON SILICON 2026-08-12 (control green): both LOW halves intact, BOTH HIGH halves
 *       lost. Not just the 0x4000-aligned one -- see below, this is why.
 *    0  nothing survived -- the copy did not happen at all; check the raw words before reading
 *       it as a result
 *
 * A BARE ldc/stc COPY LOSES EVERY PLAIN GRANULE'S HIGH HALF, by either of two mechanisms, and
 * the distinction matters because only one of them is the 0x4000 condition:
 *
 *   cursor NOT 0x4000-aligned -> the recompressed metadata is ZERO, so st_wr_cap is 0, bank 1 is
 *                               NOT WRITTEN AT ALL, and the destination keeps whatever it held
 *                               before -- here, the poison. Lost by omission.
 *   cursor IS  0x4000-aligned -> compress_bounds' cursorless branch manufactures a nonzero
 *                               metadata, st_wr_cap fires, and bank 1 is written with that
 *                               garbage. Lost by corruption.
 *
 * The 0x4000 condition therefore applies to the FIXUP sequence (ld,ld,sd,sd,ldc,stc), where the
 * plain stores have already laid both halves down correctly and only the trailing stc can undo
 * it -- which is what s06-lowhalf-zero.S measures, and why its CONTROL granule survives. It does
 * NOT apply to a bare copy, where there are no plain stores and the high half is never written
 * correctly in the first place. Conflating the two understates the compiler's exposure by
 * assuming only 0x4000-aligned granules are at risk; every plain granule is.
 *
 * AN EARLIER VERSION RETURNED 0 FOR TWO DIFFERENT THINGS -- "a low half is wrong" and "both high
 * halves lost" -- and the first board run returned exactly that 0, which said nothing. Conflating
 * outcomes in a verdict is the same mistake as a gate that cannot fire. Hence one bit per half,
 * and the RAW WORDS reported alongside (see s06agg_fpga_app.c), so a surprising verdict can be
 * read rather than guessed at.
 *
 * A CLEAN 15 IS ONLY EVIDENCE IF THE COPY REALLY USED ldc/stc. Disassemble before believing one:
 * if the compiler emitted plain 8-byte moves, or called memcpy (already fixed), this rung passes
 * without ever creating the condition it exists to detect. Check for an adjacent `ldc`/`stc` pair
 * inside s06agg_compute.
 */

/* Statics, not locals: a 16-byte-aligned LOCAL forces dynamic stack realignment, which this
 * backend cannot legalize (clang dies in LegalizeDAG). A static gets its alignment from the
 * linker for free. Same reasoning as s06lcc_kernel.h. */
typedef struct {
  unsigned long lo;    /* granule 0, low  */
  unsigned long hi;    /* granule 0, high */
  unsigned long lo2;   /* granule 1, low  */
  unsigned long hi2;   /* granule 1, high */
} s06agg_pair_t;

__attribute__((aligned(16))) static s06agg_pair_t s06agg_src;
__attribute__((aligned(16))) static s06agg_pair_t s06agg_dst;

#define S06AGG_HI_A 0xAAAA1111AAAA1111UL
#define S06AGG_HI_B 0xBBBB2222BBBB2222UL

static unsigned s06agg_compute(void)
{
  unsigned r = 0;

  s06agg_src.lo  = 0x4000UL;          /* NONZERO but 0x4000-aligned -- the trigger */
  s06agg_src.hi  = S06AGG_HI_A;
  s06agg_src.lo2 = 0x12345678UL;      /* not 0x4000-aligned -- the control */
  s06agg_src.hi2 = S06AGG_HI_B;

  /* Poison, so a copy that never happened cannot masquerade as a copy that succeeded. */
  s06agg_dst.lo  = 0xDEADDEADDEADDEADUL;
  s06agg_dst.hi  = 0xDEADDEADDEADDEADUL;
  s06agg_dst.lo2 = 0xDEADDEADDEADDEADUL;
  s06agg_dst.hi2 = 0xDEADDEADDEADDEADUL;

  /* THE SUBJECT: one struct assignment, which this backend lowers to capability-grained
     ldc/stc chunks for a 16-byte-aligned aggregate. Deliberately NOT a memcpy call -- the
     library memcpy is already fixed, and calling it would test the wrong thing. */
  s06agg_dst = s06agg_src;

  if (s06agg_dst.lo  == 0x4000UL)     r |= 1u;   /* probe   LOW  */
  if (s06agg_dst.hi  == S06AGG_HI_A)  r |= 2u;   /* probe   HIGH */
  if (s06agg_dst.lo2 == 0x12345678UL) r |= 4u;   /* control LOW  */
  if (s06agg_dst.hi2 == S06AGG_HI_B)  r |= 8u;   /* control HIGH */
  return r;
}
#endif /* S06AGG_KERNEL_H */
