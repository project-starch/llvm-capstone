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
#ifndef WBUF_DRAIN
#define WBUF_DRAIN 300u         /* iterations between scrub and check, arm 7 only */
#endif
#ifndef WBUF_FIELDS
#define WBUF_FIELDS 0           /* 1 = also verify start/end/perm/cursor on surviving caps */
#endif
#ifndef WBUF_EVICT_WAYS
/* ARM 8's eviction walk. The D$ is 32 KiB / 8-way / 16 B lines (capstone_cv64a6_imafdc_sv39
 * _config_pkg.sv:48-49), so 2048 lines over 256 sets and the set index is paddr[11:4]. To evict
 * ONE target line you must touch more distinct lines in ITS set than there are ways: 8 + 1 = 9
 * is the floor, 12 is margin against the replacement policy not being true LRU.
 *
 * STRIDE 4096 IS THE POINT, and the reason it works survives translation: the cache is
 * PHYSICALLY indexed on paddr[11:4], and with 4 KiB pages paddr[11:0] == vaddr[11:0], so a
 * 4096-byte virtual stride preserves the set index whatever the mapping does. A linear 40 KiB
 * sweep would also evict, but touches ~2560 lines instead of 12 to achieve the same thing. */
#define WBUF_EVICT_WAYS 12u
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
/* 4096-ALIGNED so that element (k*4096 + setoff) has set index == setoff>>4 for every k, which
 * is what makes the walk hit ONE set instead of smearing across all of them. */
static unsigned char volatile wbuf_evict[WBUF_EVICT_WAYS * 4096u] __attribute__((aligned(4096)));

#if WBUF_ARM == 8
/* mcycle is readable from a domain -- gp_diag_fpga_app.c and regloop_diag_fpga_app.c both do
 * exactly this. Used ONLY by arm 8's eviction proof. */
static unsigned long wbuf_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}
#endif

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
  unsigned ctl_ok = 0u, subj_type = 0u;
  unsigned i, r, k;
#if WBUF_ARM == 8
  /* THE EVICTION PROOF. See the arm-8 block below: without it a green wr8 is produced by
     three different situations and cannot tell them apart. */
  unsigned long cyc_cold = 0, cyc_warm = 0;
#endif
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
#elif WBUF_ARM == 9
      /* THE WHERECODE SHAPE. Reproduces the SQLite wedge window with NO same-granule plain
         store at all -- which is exactly what makes it a different question from arms 1-4.
         The faulting window in sqlite3WhereCodeOneLoopStart is:
             stc <cap> -> subject granule
             9 stores to OTHER granules, of which TWO are `movc rX,zero; stc` (ctag=0)
             ldc <- subject granule        -> comes back NOT_CAP
         Arms 1-4 all place a plain store IN the subject granule. This one deliberately does
         not, so a loss here means the co-residency mechanism is NOT required and the S-07 fix
         does not cover the shape. A zero here is equally informative: it says the window alone
         is insufficient and the trigger involves state this kernel does not reproduce (stack
         history, monitor, or a domain switch).
         Arm 2 remains the positive control for the whole harness.
         A CLEAN QEMU RUN OF THIS ARM PROVES ONLY THAT IT BUILDS AND COUNTS. Only arm 2 aborts
         under emulation (see the header), so emulation cannot exhibit this defect at all and a
         zero there is not evidence about silicon. */
      wbuf_slots[i] = base;                    /* SUBJECT stc */
      wbuf_sink[0] = 0xB9B9B9B9ul + i;         /* 9 stores, all to OTHER granules */
      wbuf_sink[2] = 0xB9B9B9B9ul + i;
      wbuf_sink[4] = 0xB9B9B9B9ul + i;
      wbuf_sink[6] = 0xB9B9B9B9ul + i;
      wbuf_sink[8] = 0xB9B9B9B9ul + i;
      wbuf_sink[10] = 0xB9B9B9B9ul + i;
      wbuf_sink[12] = 0xB9B9B9B9ul + i;
      /* the two ctag=0 CAPABILITY stores -- `movc rX,zero` then `stc`, the entry class
         wbuffer_gran_clr keys on, and the feature generic traffic would miss */
      wbuf_slots[WBUF_N] = (void *)0;
      wbuf_slots[WBUF_N + 1] = (void *)0;
#elif WBUF_ARM == 14
      /* THE tval CONTROL. Not a tag test at all -- it fires the FLU trap-value instrument, which
         has NEVER been shown to produce a non-zero value on this silicon. Every latched `tval` at
         a capability wedge reads 0x00, and the one non-zero on record came from mcause 15, whose
         tval comes from the LSU rather than ex_stage.sv. So every `tval == 0` in this whole
         investigation is NO DATA, including the one a root cause was once built on and retracted.
         `cincoffsetimm` on a PLAIN INTEGER must raise mcause 25 with tval = that integer
         (ex_stage.sv:487 puts the rs1 CURSOR there for capability causes). Read tval at the wedge:
            tval == 0xBEEF -> the instrument WORKS; every previous tval==0 becomes real evidence
            tval == 0      -> the instrument is DEAD on this path; every tval reading is void
         This arm WEDGES BY DESIGN -- it must be last in any boot, and it returns nothing. */
      { unsigned long beef = 0xBEEFul, sink;
        /* funct3 = 2, DECODED from a real cincoffsetimm in this very binary
           (0x0085255b -> opcode 0x5b, funct3 2), not guessed. A first version used funct3 0 and
           the constant did not even appear in the artifact -- the arm would have wedged for an
           unrelated reason and the tval reading would have been attributed to the wrong
           instruction. */
        __asm__ volatile(".insn i 0x5b, 0x2, %0, %1, 8" : "=r"(sink) : "r"(beef));
        wbuf_slots[i] = base; }
#elif WBUF_ARM == 13
      /* THE DERIVED-SUBJECT ARM. Arms 9-12 all measured ZERO, so neither shape, granule count,
         the load->store->load chain nor the stack region reproduces it. What is left is WHAT
         KIND OF CAPABILITY the subject is:
             wbuf 0-12   `&wbuf_anchor` -- a whole-region capability straight from the cap table
             SQLite      pWInfo -- SPLIT-DERIVED, narrowed bounds, derived revnode id, and the
                         measured healthy type was 1 == NONLIN post-shift
         This arm narrows the subject with `shrink` first, so it carries derived bounds and a
         derived revnode rather than the cap table's own. It also REPORTS the subject's type in
         bits 25-27, because if wbuf's subject has been LINEAR all along then every arm above
         tested a different capability class from the one that faults. */
      { void *narrowed = base;
        narrowed = __builtin_capstone_cap_shrink(narrowed,
                     (unsigned long)(void *)&wbuf_anchor,
                     (unsigned long)(void *)&wbuf_anchor + 4ul);
        subj_type = wbuf_type(narrowed) & 7u;
        wbuf_slots[i] = narrowed;              /* SUBJECT stc, a DERIVED capability */
        wbuf_sink[0] = 0xBEBEBEBEul + i;
        wbuf_sink[2] = 0xBEBEBEBEul + i;
        wbuf_sink[4] = 0xBEBEBEBEul + i; }
#elif WBUF_ARM == 12
      /* THE STACK ARM. Arms 9, 10 and 11 all measured ZERO over 16384 trials with fired
         controls, so neither the window shape, nor granule count, nor the load->store->load
         chain reproduces it. The remaining structural difference is WHERE the slot lives:
             wbuf 0-11   a GLOBAL array (wbuf_slots), reached through the cap table
             SQLite      the monitor-carved STACK, reached through sp/s0
         Those are different capabilities over different regions -- the stack is a `split` of the
         domain data region -- and they map to different cache sets, since `wr_idx = paddr[11:4]`.
         A defect keyed to the stack region, or to a set only stack addresses reach, is invisible
         to every arm above and would be the first thing SQLite has that wbuf does not.
         Same in-arm control (bit 24). */
      /* THE HARNESS CONTRACT: every arm must leave a valid capability in wbuf_slots[i],
         because the shared check below runs FIELD queries on it and every selector except 1
         RAISES on a NOT_CAP. A first version wrote only the stack slot; the shared check then
         queried an uninitialised global and the domain TRAPPED, producing no retval at all --
         caught by the QEMU gate before it reached the board. */
      wbuf_slots[i] = base;                    /* satisfy the shared check */
      { /* 16-BYTE ALIGNED EXPLICITLY. An `stc` to a misaligned address raises
           STORE_ADDRESS_MISALIGNED (capstone_dyn_unit.anvil:418) and the domain then produces
           no output at all -- which is what the first version did. Do not rely on the
           compiler choosing capability alignment for a stack local. */
        void *volatile stack_slot[2] __attribute__((aligned(16)));
        stack_slot[0] = base;                  /* SUBJECT stc, to a stack granule */
        wbuf_sink[0] = 0xBDBDBDBDul + i;       /* the window's other-granule traffic */
        wbuf_sink[2] = 0xBDBDBDBDul + i;
        wbuf_sink[4] = 0xBDBDBDBDul + i;
        wbuf_slots[WBUF_N] = (void *)0;        /* the two ctag=0 capability stores */
        wbuf_slots[WBUF_N + 1] = (void *)0;
        if (wbuf_type(stack_slot[0]) == 7ul) lost++;   /* reload from the STACK slot */
      }
#elif WBUF_ARM == 11
      /* THE LOAD-TO-STORE-TO-LOAD CHAIN. Arms 9/10 measured ZERO loss over 16384 trials with a
         fired control, so the SQLite window's SHAPE is not sufficient. This arm adds the one
         structural difference between that window and the microbenchmark:
             wbuf arms 9/10   register -> stc -> ldc      (`base` lives in a register throughout)
             SQLite           ldc -> stc -> ldc           (pWInfo is RELOADED from the caller's
                                                           frame, passed in a2, spilled by the
                                                           callee, then reloaded again)
         So the capability being spilled is itself the result of a recent load. If a forwarding
         path carries a load's result into a store without carrying its tag, this arm sees it and
         arms 9/10 structurally cannot. Same in-arm control (bit 24). */
      wbuf_slots[i] = base;                    /* seed the source slot */
      { void *reloaded = wbuf_slots[i];        /* ldc  -- the extra link */
        wbuf_slots[(i + 1u) % WBUF_N] = reloaded;   /* stc of a just-loaded capability */
        wbuf_sink[0] = 0xBCBCBCBCul + i; }
#elif WBUF_ARM == 10
      /* ARM 9b -- THE ATTRIBUTION ARM. Identical to arm 9 except for how many distinct
         granules are in flight. COUNTED, not estimated -- wbuf_slots[] elements are 16-byte
         capabilities and wbuf_sink[] elements are 8-byte scalars, so two adjacent sink
         indices share one granule:
             arm 9   subject + 7 sink + 2 ctag = 10 granules   -> OVER  WtDcacheWbufDepth (8)
             arm 10  subject + 1 sink + 2 ctag =  4 granules   -> under the depth
         That straddles the buffer depth, which is the interesting boundary rather than an
         arbitrary pair. Arm 9 differs from arms 1-4 in TWO ways at once (no same-granule plain
         store AND far more granules), so a loss there alone could not say which mattered.
             9 vs 10          isolates GRANULE COUNT (10 over the depth vs 4 under it)
             10 vs arms 1-4   isolates the SAME-GRANULE STORE
             wb0              the paired baseline, measured 0, no plain store at all
         The "co-residency is not required" claim is the one that would be new, and this arm is
         what makes it attributable rather than merely observed. */
      wbuf_slots[i] = base;                    /* SUBJECT stc */
      wbuf_sink[0] = 0xBABABABAul + i;         /* ONE other granule, 7 stores into it */
      wbuf_sink[1] = 0xBABABABAul + i;
      wbuf_sink[0] = 0xBABABABBul + i;
      wbuf_sink[1] = 0xBABABABBul + i;
      wbuf_sink[0] = 0xBABABABCul + i;
      wbuf_sink[1] = 0xBABABABCul + i;
      wbuf_sink[0] = 0xBABABABDul + i;
      wbuf_slots[WBUF_N] = (void *)0;          /* the two ctag=0 stores, same as arm 9 */
      wbuf_slots[WBUF_N + 1] = (void *)0;
#elif WBUF_ARM == 5
      /* THE SCRUB ARM. Same store order as arm 2 -- capability first, plain store second --
         but the plain store now carries a DISTINCTIVE NON-ZERO pattern and is READ BACK.
         Scrubbing with zeros would be ambiguous: a granule whose metadata is already zero
         cannot be told from one where the store landed. */
      wbuf_slots[i] = base;            /* stc: the capability the program then tries to destroy */
      *hi = WBUF_SCRUB ^ (unsigned long)i;   /* the scrub -- YOUNGER, so it MUST clobber */
#elif WBUF_ARM == 6 || WBUF_ARM == 7
      /* THE TRANSIENT-RESIDUAL PAIR. Arms 6 and 7 differ by EXACTLY ONE THING: the delay
         between the scrub and the check. Nothing else.

         WHAT THEY TEST, which is NOT the defect the granule co-residency fix repairs. That fix
         is an ALLOCATION-time check between two write-buffer ENTRIES. The residual needs only
         ONE entry, so the check never fires and a load never consults it:

             stc  G, cap     drains to L1, cap_tag_q[G>>4] = 1
             sd   x, G+8     ONE plain entry, word 1, STILL RESIDENT
             ldc  G          granule-aligned, so it compares WORD 0, misses the word-1 entry,
                             and falls through to the STALE cap_tag_q -> LIVE CAPABILITY

         So the probe must be the TYPE QUERY taken IMMEDIATELY, in the same iteration. The
         existing arm 5 reads back in a SEPARATE LOOP after all slots are stored, which builds
         in a long delay by construction -- it therefore CANNOT distinguish "no residual" from
         "residual, checked too late", and its zero must not be read as exoneration.

         arm 6 = tight: check immediately.   arm 7 = same, with a drain delay first.
         6 non-zero and 7 zero  -> transient forwarding residual, a SEPARATE PRE-EXISTING
                                   defect that granule co-residency does not claim to repair.
         6 and 7 both zero      -> no residual visible at this scale.
         6 and 7 both non-zero  -> NOT transient. That would be serious and is not the
                                   residual described above. */
      wbuf_slots[i] = base;                  /* the capability the scrub must destroy */
      *hi = WBUF_SCRUB ^ (unsigned long)i;   /* the scrub -- younger, MUST clobber the tag */
#if WBUF_ARM == 7
      for (k = 0; k < WBUF_DRAIN; k++)       /* let the entry drain before looking */
        wbuf_sink[k & 63] = (unsigned long)k + r;
#endif
      if (wbuf_type((void *)wbuf_slots[i]) != 7ul)
        corrupt++;                           /* tag STILL LIVE after its scrub */
#elif WBUF_ARM == 8
      /* ARM 8 -- THE FORCED-EVICTION LEG, and the one arm that can fail a fix which only
         repairs L1.

         WHY IT IS NEEDED. ctag is sampled TWICE: at TX ISSUE for DRAM and at TX RETURN for L1
         (wt_dcache_wbuffer.sv:319-320 and :436-437). Any fix that mutates a resident entry can
         desynchronise the two, writing one tag value to L1 and another to DRAM. The L1 copy
         wins every immediate readback and the DRAM copy wins once the line is displaced -- so
         such a fix looks PERFECT under wr6/wr7 and leaves the capability resurrectable.

         wr6 and wr7 CANNOT see this and it is not a criticism of them: wbuf_sink is 512 bytes
         cycled k&63, which drains the write buffer -- all it was built for -- and cannot evict
         anything from a 32 KiB cache. Both arms read back while the line is still resident.

         So: scrub, then FORCE THE LINE OUT, then reload. A fix that repairs only L1 shows a
         live capability here while showing none in wr6/wr7.

         READS, NOT WRITES, for the walk. Reads create the capacity pressure that evicts and add
         no write-buffer traffic of their own -- writing would put the instrument into the very
         structure under test. */
      /* WHAT PROVES THE SCRUB HAPPENED -- the mirror of the eviction question, and the
         answer is that TWO existing arms already control for it, so wr8 does not re-measure
         it. If the scrub never executed or landed elsewhere, the granule legitimately still
         holds a capability, the reload correctly shows it, and wr8 reads as "the fix only
         repaired L1" -- a FALSE NEGATIVE that would send someone after a twice-sampled-ctag
         bug that is not there.
           * wr5 is the DIRECT control: it reads the granule's high word back as a scalar and
             compares against the written pattern, so a dropped scrub is counted outright.
           * wr7 is the INDIRECT one: wr7 == 0 says the scrub reliably destroys the tag once
             the buffer drains.
         wr7's discrimination is only valid because the scrub here is BYTE-IDENTICAL to the
         one in the arm 6/7 block -- same `hi`, declared once for the whole loop, same
         expression. If either is ever edited alone, wr7 stops being a control for wr8 and
         this comment becomes false. Keep them together. */
      wbuf_slots[i] = base;                  /* the capability the scrub must destroy */
      *hi = WBUF_SCRUB ^ (unsigned long)i;   /* the scrub */
      {
        /* Same set as the target: set index is paddr[11:4], so match bits [11:4] of the slot. */
        unsigned long setoff = ((unsigned long)(void *)&wbuf_slots[i]) & 0xFF0ul;
        unsigned long acc = 0;
        for (k = 0; k < WBUF_EVICT_WAYS; k++)
          acc += *(volatile unsigned long *)(wbuf_evict + k * 4096u + setoff);
        wbuf_sink[0] = acc;                  /* consume it so the walk cannot be dead-coded */
      }
      /* THE EVICTION PROOF, and wr8 carries no verdict without it.
         A GREEN wr8 -- no live capability -- is produced by THREE situations, and the
         disassembly proves only that the twelve loads exist, not that they DISPLACED
         anything:
           1. the fix is real and DRAM holds tag 0        <- the only one we may conclude
           2. the walk did not evict, the reload hit L1   -> wr8 silently became wr7
           3. the scrub had not drained, so the clear     -> wr8 silently became wr6, and
              fired on a resident entry                      DRAM was never consulted
         A read miss does NOT force a write-buffer drain, so 3 is not far-fetched: the twelve
         DRAM round-trips make a drain LIKELY, and "likely" is the standard this project has
         been burned by repeatedly.
         So MEASURE it, as a matched pair inside the arm rather than an absolute number that
         would need calibrating: time the post-eviction ldc, then immediately time a second
         ldc of the SAME slot, which is now certainly resident because we just loaded it. The
         RATIO is self-normalising -- no clock constant, no cross-run comparison. Cold/warm
         near 1 means nothing was evicted and the arm tested nothing; a large ratio means DRAM
         was consulted, which excludes 2 and 3 together. */
      {
        void *p8; unsigned long c0, c1;
        /* WARM THE BASE CAPABILITY FIRST, or the control can report success without the
           target ever being evicted.
           Both timed loads materialise wbuf_slots' base capability from the cap table, so
           that cost appears in cyc_cold AND cyc_warm and cancels -- UNLESS the walk also
           displaced the BASE's own line. Then cold carries a cold base plus a possibly-warm
           target while warm carries a warm base plus a warm target, the ratio comes out well
           above 16, and it reads as "eviction confirmed" when what was evicted was the base
           and not the target at all. A false positive in the very control that exists to
           prevent one, and a spuriously large ratio looks exactly like the evidence wanted.
           One touch of a NON-TARGET slot fixes it: same array so the same base capability,
           different granule so the target's residency is undisturbed. */
        wbuf_sink[2] = (unsigned long)wbuf_slots[WBUF_N + 1];
        c0 = wbuf_mcycle();
        p8 = wbuf_slots[i];                  /* THE ldc under test -- this is what must miss */
        c1 = wbuf_mcycle();
        cyc_cold += c1 - c0;
        if (wbuf_type(p8) != 7ul)
          corrupt++;                         /* tag STILL LIVE after eviction+reload */
        c0 = wbuf_mcycle();
        p8 = wbuf_slots[i];                  /* same slot, now resident: the warm control */
        c1 = wbuf_mcycle();
        cyc_warm += c1 - c0;
        wbuf_sink[1] = (unsigned long)p8;    /* consume, so neither ldc can be elided */
      }
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
#if WBUF_FIELDS && WBUF_ARM != 5 && WBUF_ARM != 6 && WBUF_ARM != 7 && WBUF_ARM != 8
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
#if WBUF_ARM == 8
  /* DISTINCT ENCODING, because arm 8 reports a measurement the other arms do not have and
     silently reusing their layout would make it unreadable:
        0xB8 | ratio[11:0] << 12 | corrupt[11:0]
     ratio = cyc_cold * 16 / cyc_warm, i.e. FIXED POINT with 16 = 1.0x, so a ratio of exactly
     16 means cold and warm were indistinguishable and THE ARM TESTED NOTHING.

     WHY A RATIO AND NOT AN ABSOLUTE, and this is the transferable part. The property that
     matters here -- that the arm cannot report a WRONG verdict, only an honest void one --
     did not come from any of the controls that were argued over. It came from the choice of
     representation. A ratio CANNOT say "evicted" when nothing was evicted, because the same
     quantity appears in both terms and cancels. A detector tells you when the instrument was
     wrong; a representation that cannot express the wrong answer leaves nothing to detect.
     Where that option exists it beats any control, and it is worth looking for FIRST rather
     than after five rounds of adding detectors, which is how it was arrived at here. Clamped, and a
     zero warm total also reports 0 rather than dividing by it. */
  {
    unsigned long ratio = cyc_warm ? (cyc_cold * 16ul) / cyc_warm : 0ul;
    /* 0xFFF is a PEG, not a measurement: a small cyc_warm saturates the 12-bit field at 255x.
       It saturates UPWARD, so the failure direction is safe -- a pegged value still reads as
       "evicted" -- but do not read 0xFFF as a ratio. */
    if (ratio > 0xFFFul) ratio = 0xFFFul;
    if (corrupt > 0xFFFul) corrupt = 0xFFFul;
    return 0xB8000000u | (unsigned)(ratio << 12) | (unsigned)corrupt;
  }
#else
#if WBUF_ARM == 9 || WBUF_ARM == 10 || WBUF_ARM == 11 || WBUF_ARM == 12 || WBUF_ARM == 13
  /* IN-ARM POSITIVE CONTROL, bit 24. A zero loss count is worthless unless the detector is
     known to fire, and arm 2 -- the harness positive control -- CANNOT be QEMU-verified: the
     native oracle is a stub that always answers WBUF_OK with zero loss (wbuf_host.c:4), and
     arm 2's whole purpose is a divergence only capability silicon exhibits. So these arms
     carry their own.
     The query is asked about a KNOWN NON-CAPABILITY. Selector 1 is total and answers 7 for
     NOT_CAP without raising, on RTL (capstone_dyn_unit.anvil:195) and now in QEMU
     (op_helper.c helper_cslcc). All three vehicles must therefore set this bit:
        bit 24 SET   -> the detector can report a loss; a zero count MEANS zero
        bit 24 CLEAR -> the query did not answer 7, and NO count in this arm carries a verdict
     Deliberately a separate bit rather than folded into `lost`, so a broken control can never
     be mistaken for a finding. */
  { unsigned long ctl_scalar = 0x5A5A5A5Aul, ctl_ty = 0ul;
    /* THE OPERAND MUST BE A PLAIN INTEGER VALUE, NOT A POINTER. A first version asked
       `wbuf_type(&ctl_scalar)` and the control correctly FAILED: the address of a local IS a
       real capability here, so the query answered its type (0/1) rather than 7, and the
       control would have reported "detector dead" on perfectly good hardware. Query the VALUE. */
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(ctl_ty) : "r"(ctl_scalar));
    if (ctl_ty == 7ul) ctl_ok = 1u; }
#endif
  return WBUF_OK | (unsigned)(subj_type << 25) | (unsigned)(ctl_ok << 24) | (unsigned)(corrupt << 12) | (unsigned)lost;
#endif
}
#endif
