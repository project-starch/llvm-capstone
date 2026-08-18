#ifndef WBUF_H
#define WBUF_H
/* WBUF -- does a PLAIN STORE to a granule's HIGH word destroy the tag of a capability stored to
 * the granule's LOW word AFTERWARDS?
 *
 * THE HYPOTHESIS UNDER TEST (RTL lane, 2026-08-18). The write buffer hits at 64-bit WORD
 * granularity (wt_dcache_wbuffer.sv:444 compares wtag == {address_tag, address_index[11:3]}),
 * so G+0 and G+8 occupy SEPARATE entries -- but every entry writes the whole 16-byte GRANULE's
 * single tag bit on drain (:410 wr_idx_o = wr_paddr[11:4], :416 wr_ctag_o, and
 * wt_dcache_mem.sv:459 cap_tag_q[wr_idx_i][j] <= wr_ctag_i). Drain is `rr_arb_tree` over
 * `dirty` -- ROTATION order, not program order. So an older plain store to G+8 can land AFTER a
 * younger stc to G, and the tag the stc set is overwritten with the plain store's zero.
 *
 * WHY THIS SHAPE AND NOT A LOOP. The two stores must be in the buffer SIMULTANEOUSLY, so they
 * are issued back to back per slot. tagsweep stores every slot and only then clobbers a few,
 * which is a different experiment: by the time the clobber issues, the stc entry is long gone.
 * That is why tagsweep's seeded arm shows loss and says nothing about ordering.
 *
 * WHY IT CANNOT WEDGE. The reload is checked with lcc field 1, the TOTAL type query, which
 * returns 7 for NOT_CAP WITHOUT raising (capstone_dyn_unit.anvil:195). A lost tag is COUNTED,
 * never fatal, so every arm returns a number. Same idiom as tagsweep_type; deliberately not
 * re-derived.
 *
 * THE ARMS, and the point is that ARM 2 MUST FAIL. A batch of negatives from an instrument
 * that has never produced a positive is worth nothing, and this project has published exactly
 * that mistake before.
 *
 *   WBUF_ARM 0  CONTROL, no plain store at all.  stc G; ldc G.        EXPECT loss == 0
 *   WBUF_ARM 1  TEST.  plain store G+8; stc G; ldc G.                 program order says the
 *                      tag survives, so ANY loss is a REORDER and confirms the mechanism
 *   WBUF_ARM 2  POSITIVE CONTROL. stc G; plain store G+8; ldc G.      EXPECT loss == N.
 *                      In program order the plain store legitimately clears the granule tag,
 *                      so this is CORRECT architecture, not a defect -- it exists solely to
 *                      prove the detector can report a loss at all.
 *   WBUF_ARM 3  SPACED. plain store G+8; ~64 unrelated stores; stc G. EXPECT loss == 0 if the
 *                      buffer has drained. Arm 1 losing while arm 3 does not IS the
 *                      buffer-residency discriminator, and is the strongest single result
 *                      this test can produce.
 *   WBUF_ARM 4  NEIGHBOUR GRANULE. plain store to G+16 (the NEXT granule); stc G; ldc G.
 *                      EXPECT loss == 0. Proves any effect is granule-scoped rather than
 *                      "a nearby store hurts".
 *
 * READING IT. Arms 0/3/4 non-zero, or arm 2 zero, means the instrument is wrong and NOTHING
 * else in the boot may be read. Only with arm 2 == N and arms 0/4 == 0 does arm 1 carry a
 * verdict.
 *
 * QEMU. CORRECTED: only ARM 2 aborts under emulation, not arms 1-4. op_helper.c:719 asserts
 * rs1_v->tag before any selector check, so a type query aborts only when a tag has ACTUALLY
 * been lost -- and QEMU's capability store is one atomic 16-byte-plus-tag operation with no
 * write buffer, no per-word entries and no drain arbiter, so the reordering under test cannot
 * occur there. Arms 0/1/3/4 therefore run clean under QEMU and return 0.
 *
 * That makes QEMU the NO-REORDER BASELINE rather than a weaker copy of the board: if arm 1
 * returns 0 under QEMU and non-zero on silicon, the difference IS the mechanism, measured
 * against an oracle that structurally cannot exhibit it. Arm 2 stays board-only, the same
 * divergence tagsweep records for its SEED arm.
 */

#ifndef WBUF_N
#define WBUF_N 256u             /* slots; x16 B = 4 KiB, comfortably inside one way */
#endif
#ifndef WBUF_REPS
#define WBUF_REPS 64u
#endif
#ifndef WBUF_ARM
#define WBUF_ARM 0
#endif
#ifndef WBUF_SCRUB
/* Distinctive and non-zero on purpose: a zero scrub cannot be distinguished from metadata that
   was already zero, and a zeroing memset is exactly the case of interest. */
#define WBUF_SCRUB 0xD15CA5DBAD5C2BA1ul
#endif
#ifndef WBUF_FIELDS
#define WBUF_FIELDS 0           /* 1 = also verify start/end/perm/cursor on surviving caps */
#endif

#define WBUF_OK    0xB0000000u
#define WBUF_FAULT 0xEE000000u

/* volatile per element: both stores and the reload must survive to the artifact. A
 * repeat-the-load-N-times ladder on this project was once CSE'd into ONE ldc regardless of N
 * and reported with total confidence having tested nothing. The disassembly is checked before
 * the boot as well -- belt and braces on purpose. */
static void *volatile wbuf_slots[WBUF_N + 2];
static unsigned wbuf_anchor;
static unsigned long volatile wbuf_sink[64];

static unsigned long wbuf_type(const void *p) {
  unsigned long v = 0;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(v) : "r"(p));
  return v;
}

/* FIELD QUERIES, FOR VALIDATING A FIX RATHER THAN DETECTING THE DEFECT.
 *
 * The plain store lands on the granule's HIGH word -- the METADATA half -- so start, end and
 * perm are AT RISK and the cursor (low word) is NOT. The cursor is queried anyway as a NEGATIVE
 * CONTROL: it should survive every arm, and if it ever does not, the corruption is not the
 * mechanism described above.
 *
 * WHY THIS EXISTS. The obvious fix -- propagate the youngest store's tag to a co-resident
 * same-granule entry so drain order stops mattering -- leaves the older plain entry writing its
 * stale scalar over the metadata half. With ctag=1 that yields a capability with a VALID TAG
 * over CORRUPTED METADATA: tag loss converted into tag forgery. A tag-only test reports that as
 * a total success. This turns two outcome buckets into three:
 *
 *     NOT_CAP                       -> LOST                  (the S-07 direction)
 *     capability, fields match      -> INTACT
 *     capability, any field wrong   -> CORRUPTED-BUT-TAGGED  (what a naive fix produces)
 *
 * ORDERING IS NOT OPTIONAL. Selector 1 is TOTAL and answers 7 for NOT_CAP WITHOUT raising
 * (capstone_dyn_unit.anvil:195 -- the NOT_CAP guard excludes zimm != 1). EVERY OTHER SELECTOR
 * RAISES on a NOT_CAP operand. So the type must be checked FIRST and the field queries reached
 * only when the value is still a capability; otherwise a tag-loss arm traps on the bounds query
 * and the trap masks the measurement.
 *
 * The reference values are CAPTURED FROM A LIVE CAPABILITY at startup rather than assumed, so
 * the comparison does not depend on knowing what the cap table hands out. They are non-zero for
 * a real global, which matters because the memset case makes ZERO the most likely corrupt
 * value -- a field whose correct value were zero could not detect a zero overwrite. */
#define WBUF_SEL(name, zimm)                                                    \
  static unsigned long name(const void *p) {                                    \
    unsigned long v = 0;                                                        \
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x" #zimm                  \
                     : "=r"(v) : "r"(p));                                       \
    return v;                                                                   \
  }
WBUF_SEL(wbuf_cursor, 2)   /* NOT at risk -- negative control */
WBUF_SEL(wbuf_start,  3)   /* at risk */
WBUF_SEL(wbuf_end,    4)   /* at risk */
WBUF_SEL(wbuf_perm,   5)   /* at risk */

static unsigned wbuf_compute(void)
{
  unsigned long lost = 0, corrupt = 0;
  unsigned i, r, k;
  void *base = (void *)&wbuf_anchor;
#if WBUF_FIELDS
  /* Captured from a live capability, not assumed. */
  const unsigned long ref_start  = wbuf_start(base);
  const unsigned long ref_end    = wbuf_end(base);
  const unsigned long ref_perm   = wbuf_perm(base);
  const unsigned long ref_cursor = wbuf_cursor(base);
#endif

  for (r = 0; r < WBUF_REPS; r++) {
    for (i = 0; i < WBUF_N; i++) {
      /* hi = the HIGH 8 bytes of slot i's own 16-byte granule */
      volatile unsigned long *hi =
          (volatile unsigned long *)((volatile char *)&wbuf_slots[i] + 8);
      /* nxt = the LOW 8 bytes of the NEXT granule, for the neighbour arm */
      volatile unsigned long *nxt =
          (volatile unsigned long *)((volatile char *)&wbuf_slots[i] + 16);

#if WBUF_ARM == 1
      *hi = 0xB8B8B8B8ul + i;          /* plain store, HIGH word, BEFORE the stc */
      wbuf_slots[i] = base;            /* stc, LOW word */
#elif WBUF_ARM == 2
      wbuf_slots[i] = base;            /* stc first ... */
      *hi = 0xB8B8B8B8ul + i;          /* ... then the plain store clears the tag: MUST show */
#elif WBUF_ARM == 3
      *hi = 0xB8B8B8B8ul + i;          /* plain store, HIGH word */
      for (k = 0; k < 64u; k++)        /* drain the buffer with unrelated traffic */
        wbuf_sink[k] = (unsigned long)k + r;
      wbuf_slots[i] = base;            /* stc, long after the plain store retired */
#elif WBUF_ARM == 4
      *nxt = 0xB8B8B8B8ul + i;         /* plain store to the NEXT granule */
      wbuf_slots[i] = base;            /* stc, LOW word of THIS granule */
#elif WBUF_ARM == 5
      /* THE SCRUB ARM. Same store order as arm 2 -- capability first, plain store second --
         but the plain store now carries a DISTINCTIVE NON-ZERO pattern and is READ BACK.
         Scrubbing with zeros would be ambiguous: a granule whose metadata is already zero
         cannot be told from one where the store landed. */
      wbuf_slots[i] = base;            /* stc: the capability the program then tries to destroy */
      *hi = WBUF_SCRUB ^ (unsigned long)i;   /* the scrub -- YOUNGER, so it MUST clobber */
#else
      wbuf_slots[i] = base;            /* arm 0: no plain store anywhere near */
#endif
    }

#if WBUF_ARM == 5
    /* Did the scrub actually land? Read the granule's high word back as a scalar and compare
       against the pattern that was written. A mismatch means the plain store was DROPPED and
       the capability survived the operation intended to destroy it.

       This is measured independently of the type query below, so the two counts are a
       cross-check on each other: every dropped scrub should correspond to a surviving
       capability, and the two numbers should agree. */
    for (i = 0; i < WBUF_N; i++) {
      volatile unsigned long *hi =
          (volatile unsigned long *)((volatile char *)&wbuf_slots[i] + 8);
      if (*hi != (WBUF_SCRUB ^ (unsigned long)i)) corrupt++;   /* scrub DROPPED */
    }
#endif

    for (i = 0; i < WBUF_N; i++) {
      void *p = wbuf_slots[i];         /* ldc */
      if (wbuf_type(p) == 7ul) {       /* 7 == NOT_CAP. MUST be tested before any other */
        lost++;                        /* selector, which would RAISE on this value.    */
        continue;
      }
#if WBUF_FIELDS && WBUF_ARM != 5
      if (wbuf_start(p)  != ref_start ||
          wbuf_end(p)    != ref_end   ||
          wbuf_perm(p)   != ref_perm  ||
          wbuf_cursor(p) != ref_cursor)
        corrupt++;                     /* tagged, but the metadata is not the original */
#endif
    }
  }

  /* Two 12-bit counters. WBUF_N * WBUF_REPS must stay <= 4095 per counter for this to be
     lossless; both saturate rather than wrap, so a saturated read is visibly at the limit
     instead of silently aliasing to a small number. */
  if (lost    > 0xFFFul) lost    = 0xFFFul;
  if (corrupt > 0xFFFul) corrupt = 0xFFFul;
  return WBUF_OK | (unsigned)(corrupt << 12) | (unsigned)lost;
}
#endif
