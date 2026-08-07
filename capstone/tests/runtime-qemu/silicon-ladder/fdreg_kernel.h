#ifndef FDREG_KERNEL_H
#define FDREG_KERNEL_H
/* MINIMAL REPRODUCER for the SQLite silicon wedge, reconstructed off the SQLite path.
 *
 * Bisected 2026-08-06 down to ONE 13-line function. Every other step of
 * sqlite3_initialize() returns rc=0 on silicon; sqlite3AlterFunctions() alone does not
 * return (domain verified created and entered -- `SQ: A/dom-ok`, no monitor tag, so this
 * is a real in-domain wedge and not the region-share/monitor-spin infrastructure case).
 *
 * What that function is, in full (amalgamation sqlite3-capstone.c:123624):
 *
 *     static FuncDef aAlterTableFuncs[] = {          <-- STATIC, so a GLOBAL under the
 *       INTERNAL_FUNCTION(sqlite_rename_column, 9, renameColumnFunc),   gp-captable ABI
 *       ... 9 entries, each {name STRING pointer, FUNCTION pointer, 6 null pointers} ...
 *     };
 *     sqlite3InsertBuiltinFuncs(aAlterTableFuncs, ArraySize(aAlterTableFuncs));
 *
 * and InsertBuiltinFuncs links the elements into a global 23-bucket hash chain:
 *
 *     aDef[i].pNext        = sqlite3BuiltinFunctions.a[h];   <-- store a DERIVED capability
 *     sqlite3BuiltinFunctions.a[h] = &aDef[i];                   into a global
 *
 * NOTE it is `static`, NOT a local aggregate -- the CGDecl.cpp memcpy-template theory in
 * design/cap-local-aggregate-init-plan.md does not apply, and neither does locfl (which
 * probes LOCAL pointer-bearing arrays and is the C-14 vehicle). This rung is the static
 * counterpart, and it is deliberately built from the same three separable operations so
 * the first stage that fails to return IS the answer:
 *
 *   FDREG_STAGE=1  cap-init of the array + READ every zName through its capability.
 *                  No stores, no calls. Oracle 2456.
 *   FDREG_STAGE=2  stage 1 + link every element into a global bucket table by storing
 *                  &arr[i] (a derived capability) into a global, then walk the chains
 *                  back. This is InsertBuiltinFuncs, minus SQLite. Oracle 2609.
 *   FDREG_STAGE=3  stage 2 + call each entry through its xFunc field, i.e. an indirect
 *                  call through a capability loaded out of a global aggregate.
 *                  Oracle 2736.
 *
 * Every stage RETURNS a checksum, so a wrong value bisects where a hang would only ever
 * say "somewhere after the last marker".
 *
 * BOARD RESULT 2026-08-06 (caplifive_65536_nodes.bit, one boot, control k800 = 4 green):
 * all three stages RETURN their oracle -- 2456 / 2609 / 2736. So the construct alone does
 * NOT reproduce the SQLite wedge: a static pointer-bearing array, its cap-init, storing
 * derived capabilities into a global bucket table, and indirect calls through a function
 * pointer loaded out of that array are all fine on this silicon. Whatever kills
 * sqlite3AlterFunctions is a property of the SURROUNDING image, not of these operations,
 * which is what FDREG_PAD below exists to test.
 *
 * FDREG_PAD=<n> is an ORTHOGONAL knob: it emits <n> extra referenced globals ahead of
 * fdreg_defs, pushing it past gp index 128.
 *
 * Why 128 is a real boundary and not a round number: `ldc rd, imm(gp)` has a 12-bit signed
 * immediate and cap-table entries are 16 bytes, so index 128 is byte offset 2048 -- the
 * first index that immediate cannot reach. Beyond it codegen switches to a DIFFERENT
 * sequence, verified in the SQLite domain at qr13.dom:0x10510
 *
 *     lui a1, 0x1 ; addi a1, a1, -0x510 ; cincoffset a1, gp, a1 ; ldc a1, 0x0(a1)
 *
 * i.e. materialise the byte offset, cincoffset off gp, then load. qr13 uses direct
 * `imm(gp)` only up to offset 2016 (index 126) and takes the cincoffset path 846 times
 * above that. An UNPADDED rung has ~12 globals and therefore never exercises the second
 * path at all, so a clean unpadded result does not clear it. Pass FDREG_PAD=1 to emit 160
 * padding globals, which puts the rung's own array above the boundary and both paths in
 * the image. The oracle rises by 160 in every stage.
 * The padding globals must be READ (they are, in the checksum) or they are stripped and
 * never get cap-table slots -- verify with the disassembly, do not assume.
 */
#ifndef FDREG_STAGE
#define FDREG_STAGE 1
#endif
#ifndef FDREG_PAD
#define FDREG_PAD 0
#endif
/* FDREG_DRAW -- R-16 REDRAW knob, the ladder counterpart of the SQLite domain's QR_DRAW.
   The entry stall is PER-IMAGE and deterministic per binary, so retrying a stalling rung buys
   nothing; the remedy is a DIFFERENT image whose code under test is byte-identical. These nops
   sit at the top of the compute, before any probe runs, and shift layout and nothing else.
   fdreg7p stalled on its first draw and there was no way to redraw a rung -- only the SQLite
   domain had one. Vary until it enters, and sha256sum the set: two draws that hash the same
   are the same ticket. */
#ifndef FDREG_DRAW
#define FDREG_DRAW 0
#endif
#ifndef FDREG_LEAVES
#define FDREG_LEAVES 0
#endif
/* FDREG_GUARD -- makes the `if (s == 0xFFFFFFFFu) return 0;` line optional.
   It exists so the two halves of a comparison can come from ONE source revision. fdreg7
   (576, correct) and lf0 (906, wrong) are both stage 7 at LEAVES=0 and they disagree, but
   they differ in FIVE ways at once -- base VA, 60 vs 77 instructions in fdreg_compute, this
   guard, frame slot offsets, image size -- because fdreg7 predates the guard and FDREG_OUTER.
   Using it as a control was wrong. With this knob, guard-on and guard-off differ by the guard
   and nothing else. */
#ifndef FDREG_GUARD
#define FDREG_GUARD 1
#endif
/* FDREG_SHIFT -- moves the loop counters within the frame, and nothing else.
   The guard/no-guard pair (906 vs 576, base VA excluded by a 2x2) turned out NOT to differ by
   the guard's instructions: adding the guard adds a local, which shifts EVERY frame slot by 4
   bytes. The inner counter sits at s0-0x34 in the passing build and s0-0x38 in the failing one:
       -0x34 = 52,  52 mod 16 = 4   PASSES
       -0x38 = 56,  56 mod 16 = 8   FAILS
   16 bytes is exactly a capability, so the counter's position WITHIN a 16-byte granule is a
   candidate the guard was only a proxy for. This knob varies that position directly, with the
   guard and everything else held constant -- a dummy local declared ahead of the counters.

   RESULT, boots 45-49. In EVERY build the capability local `z` lands at s0-0x50 = sp+0 and is
   written by a 16-byte `stc a1, 0x0(a0)` on every inner iteration, 576 times, with the frame a
   constant 0x50 bytes. Only the counters move:

       shift  inner counter k       bytes ABOVE the stc's 16-byte end   board
         0    s0-0x34 = sp+0x1c              12                        576  CORRECT
         4    s0-0x38 = sp+0x18               8                        909  (+333)
         8    s0-0x3c = sp+0x14               4                        567  (-9)
        12    s0-0x40 = sp+0x10               0 (adjacent)             0x8000237

   *** THE PROXIMITY READING OF THIS TABLE IS RETRACTED (boot 51, 2026-08-07). ***

   It said: "severity scales with proximity and vanishes at 12 bytes, so the store damages
   memory BEYOND its nominal 16-byte footprint." That is REFUTED. Stage 13's wp0 puts the
   counters 24 bytes above the store, in a DIFFERENT 16-byte row, with a different frame size
   (0x60), and returns a byte-identical 909. Distance from the store is not the variable. The
   sweep above only looked like a distance law because it held the row fixed while moving the
   offset within it.

   WHAT ACTUALLY PREDICTS THE ANSWER is the counter's position in the CACHE GEOMETRY -- which
   8-byte bank of its 16-byte row it sits in, and its offset within that bank. And in every
   build ever measured, qc == k + 8: the accumulator is invariably the inner counter's BANK
   SIBLING.

       k bank  k off   result        builds
          1      4     576 CORRECT   shift0
          1      0     909           shift4, wp0 (different row, different frame -- same answer)
          0      4     567           shift8
          0      0     0x8000237     shift12       (= 567 with bit 27 set, i.e. metadata bits)

   That is the geometry of the D-cache write path, and it has a matching RTL mechanism, every
   line verified against the primary source rather than inferred:

       commit_stage.sv:323-325   we_gpr_o[0] = 1'b1 unconditionally, but
                                 cap_we_o[0] = commit_instr_i[0].cap_result.valid
                                 -> an ordinary instruction overwrites the integer register and
                                    LEAVES THE SHADOW CAPABILITY METADATA STALE
       wt_dcache_wbuffer.sv:602  every store captures whatever metadata is on the bus
       wt_dcache_mem.sv:138      st_wr_cap = |wr_user_i -- a store is classified as a capability
                                 store BY VALUE, NOT BY OPCODE
       wt_dcache_mem.sv:225-238  a store so classified writes BOTH 8-byte banks of its 16-byte
                                 row, bank 1 receiving the metadata (:156-158)

   and our own inner loop contains the taint sequence exactly: `ldc a0` (the volatile read-back
   of z) puts a capability in a0, the very next `lw a0` makes it the counter without clearing
   the shadow, and `sw a0` then carries stale metadata. For k in bank 0 that deposits metadata
   into k+8 == qc, the returned accumulator, which is shift8 and shift12. NOT YET EXPLAINED:
   the two k-in-bank-1 rows, where the same reading predicts corruption of k itself and shift0
   measures a clean 576. Do not write this up as the root cause until that is closed.

   This is also why every in-frame instrument destroyed the fault: a sentinel array added 32
   bytes and a `&qc` pointer added 16, each moving the counters to a different bank/offset.
   The instrument was curing the patient.

   ALSO REFUTED (same boot): the over-wide-write idea sourced from `be_gen` (ariane_pkg.sv:1045).
   `extract_transfer_size` (ariane_pkg.sv:1119-1126) fixes STC at 8 bytes/one beat and the
   metadata rides a separate sideband, so there is no wider or multi-beat DATA write to blame. */
#ifndef FDREG_SHIFT
#define FDREG_SHIFT 0
#endif
#ifndef FDREG_PEEK
#define FDREG_PEEK 0
#endif
/* FDREG_WITSEL -- selects what stage 13 returns. See stage 13. */
#ifndef FDREG_WITSEL
#define FDREG_WITSEL 0
#endif
/* FDREG_WITPAD -- moves stage 13's witnesses onto the damage window. See stage 13. */
#ifndef FDREG_PVICT
#define FDREG_PVICT 3
#endif
#ifndef FDREG_GVICT
#define FDREG_GVICT 3
#endif
#ifndef FDREG_ASMVICTIM
#define FDREG_ASMVICTIM 0x1c
#endif
#ifndef FDREG_INNER
#define FDREG_INNER FDREG_N
#endif
#ifndef FDREG_SENTINEL
#define FDREG_SENTINEL 1000000u
#endif
#ifndef FDREG_NORESET
#define FDREG_NORESET 0
#endif
#ifndef FDREG_ORDER
#define FDREG_ORDER 0
#endif
#ifndef FDREG_REREAD
#define FDREG_REREAD 0
#endif
#ifndef FDREG_DEPTH
#define FDREG_DEPTH 0
#endif
#ifndef FDREG_CAPSLOT
#define FDREG_CAPSLOT 0
#endif
#ifndef FDREG_GAP
#define FDREG_GAP 0
#endif
#ifndef FDREG_BARRIER
#define FDREG_BARRIER 0
#endif
#ifndef FDREG_WITPAD
#define FDREG_WITPAD 12
#endif
/* FDREG_OUTER -- the outer trip count, so the EXPECTED value can be varied.
   The first three leaf-count rungs all returned 906 against an expected 576. Three different
   images agreeing exactly is as consistent with a BROKEN PROBE returning a constant as with a
   real miscount, and those two must be separated before anything is claimed. Changing the
   outer count changes the expected answer: if the board still says 906, the probe is reporting
   something unrelated to the loop. */
#ifndef FDREG_OUTER
#define FDREG_OUTER 64
#endif
#define FDREG_DRAW_STR2(x) #x
#define FDREG_DRAW_STR(x) FDREG_DRAW_STR2(x)

#define FDREG_N     9
#define FDREG_HASHN 23

static volatile unsigned fdreg_gate = 1u;  /* satisfies the ldc gp[i] build gate */

#if FDREG_PAD > 0
/* One global OBJECT per cap-table slot -- an array would take a single slot and pad
   nothing. Volatile so the reads survive -O1 and the slots stay live. */
#define FDREG_P8(b)  FDREG_P1(b##0) FDREG_P1(b##1) FDREG_P1(b##2) FDREG_P1(b##3) \
                     FDREG_P1(b##4) FDREG_P1(b##5) FDREG_P1(b##6) FDREG_P1(b##7)
/* `used` so the linker KEEPS them without any code to read them. The original version kept
   them alive by summing all 160 in the compute, which cost ~3.8 KB of .text, forced
   DOMAIN_WINDOW=32k, and produced images that ENTRY-STALLED on every draw tried (fdreg7p,
   f7p8). The cap-table slot is what the probe needs, not the read. */
#define FDREG_P1(i)  static volatile unsigned fdreg_pad_##i __attribute__((used)) = 1u;
FDREG_P8(0) FDREG_P8(1) FDREG_P8(2) FDREG_P8(3) FDREG_P8(4)
FDREG_P8(5) FDREG_P8(6) FDREG_P8(7) FDREG_P8(8) FDREG_P8(9)
FDREG_P8(a) FDREG_P8(b) FDREG_P8(c) FDREG_P8(d) FDREG_P8(e)
FDREG_P8(f) FDREG_P8(g) FDREG_P8(h) FDREG_P8(i) FDREG_P8(j)
/* No read loop: `used` already keeps every padding global and its cap-table slot. */
#define FDREG_PAD_SUM() do {} while (0);
#else
#define FDREG_PAD_SUM() do {} while (0);
#endif

/* FDREG_LEAVES -- the CAP-INIT LEAF COUNT axis.
   fdreg7 has 17 .capstone_gp_initdesc entries; the SQLite domain that fails has ~176. That is
   one of only three measured differences left between the rung (returns 576) and L31 (returns
   567), now that gp index and struct layout are excluded.
   Deliberately ONE global holding many capability leaves, rather than many globals: 160
   separate globals is what the FDREG_PAD path does, and six such builds ENTRY-STALLED across
   two padding implementations, two window sizes and five base VAs. A single array adds leaves
   without adding cap-table slots, so it should not perturb entry the same way -- verify that
   in the artifact rather than assuming it. */
#if (FDREG_LEAVES) > 0
/* Each literal must be DISTINCT or the compiler merges them and the leaf count barely moves:
   a first attempt repeated one string eight times per group and produced 31 descriptor
   entries instead of 170. Verified by reading .capstone_gp_initdesc out of the artifact. */
#define FDREG_L8(b) FDREG_L1(b,0) FDREG_L1(b,1) FDREG_L1(b,2) FDREG_L1(b,3) \
                    FDREG_L1(b,4) FDREG_L1(b,5) FDREG_L1(b,6) FDREG_L1(b,7)
#define FDREG_L1(b,i) "leafpad_" #b "_" #i,
/* FDREG_LEAVES is a COUNT of 8-leaf groups, so the axis can be swept rather than switched:
   every rung built with ~170 cap-init entries has ENTRY-STALLED (six padded builds plus the
   171-leaf one), while the 10-entry rung enters reliably. Sweeping locates the boundary, and
   the largest value that still ENTERS is the most SQLite-like rung that can be measured. */
static const char *fdreg_leaves[] __attribute__((used)) = {
  FDREG_L8(0) FDREG_L8(1) FDREG_L8(2) FDREG_L8(3)
#if (FDREG_LEAVES) >= 8
  FDREG_L8(4) FDREG_L8(5) FDREG_L8(6) FDREG_L8(7)
#endif
#if (FDREG_LEAVES) >= 12
  FDREG_L8(8) FDREG_L8(9) FDREG_L8(a) FDREG_L8(b)
#endif
#if (FDREG_LEAVES) >= 20
  FDREG_L8(c) FDREG_L8(d) FDREG_L8(e) FDREG_L8(f)
  FDREG_L8(g) FDREG_L8(h) FDREG_L8(i) FDREG_L8(j)
#endif
};
#undef FDREG_L1
#undef FDREG_L8
#endif

static int fdreg_f0(void) { return 3; }
static int fdreg_f1(void) { return 5; }
static int fdreg_f2(void) { return 7; }
static int fdreg_f3(void) { return 11; }
static int fdreg_f4(void) { return 13; }
static int fdreg_f5(void) { return 17; }
static int fdreg_f6(void) { return 19; }
static int fdreg_f7(void) { return 23; }
static int fdreg_f8(void) { return 29; }

/* Same shape as FuncDef: a scalar head, then eight pointer-sized fields of which exactly
   two are non-null in the INTERNAL_FUNCTION initialiser. Keeping the null fields matters --
   they are the ones cap-init must NOT emit leaves for, and a layout with only the live
   pointers would not reproduce the same descriptor table. */
typedef struct FdregDef {
  signed char       nArg;
  unsigned          funcFlags;
  void             *pUserData;
  struct FdregDef  *pNext;
  int             (*xFunc)(void);
  void             *xFinalize;
  void             *xValue;
  void             *xInverse;
  const char       *zName;
  void             *pHash;
} FdregDef;

#define FDREG_ENTRY(zN, nA, xF) \
  { nA, 0x2820u, 0, 0, xF, 0, 0, 0, zN, 0 }

/* Non-const on purpose: FuncDef's own comment says the array "cannot be constant since
   changes are made to the pHash elements at start-time". A const array would land in
   .rodata and stop reproducing the shape. */
static FdregDef fdreg_defs[FDREG_N] = {
  FDREG_ENTRY("sqlite_rename_column",   9, fdreg_f0),
  FDREG_ENTRY("sqlite_rename_table",    7, fdreg_f1),
  FDREG_ENTRY("sqlite_rename_test",     7, fdreg_f2),
  FDREG_ENTRY("sqlite_drop_column",     3, fdreg_f3),
  FDREG_ENTRY("sqlite_rename_quotefix", 2, fdreg_f4),
  FDREG_ENTRY("sqlite_drop_constraint", 2, fdreg_f5),
  FDREG_ENTRY("sqlite_fail",            2, fdreg_f6),
  FDREG_ENTRY("sqlite_add_constraint",  3, fdreg_f7),
  FDREG_ENTRY("sqlite_find_constraint", 2, fdreg_f8),
};

/* The global bucket table InsertBuiltinFuncs links into (sqlite3BuiltinFunctions.a[]). */
static FdregDef *fdreg_buckets[FDREG_HASHN];

/* stage 33: the capability store target, in the DATA region rather than on the stack */
static const char *volatile fdreg_gz;

static unsigned fdreg_len30(const char *z) {
  unsigned n = 0;
  while (z[n]) n++;
  return n;
}

/* SQLITE_FUNC_HASH, verbatim: (c + n*3) % 23 with c the first character. */
static unsigned fdreg_hash(unsigned char c, unsigned n) {
  return ((unsigned)c + n * 3u) % FDREG_HASHN;
}

/* STAGE 4 -- the candidate MINIMAL REPRO of the SQLite wedge.
 *
 * Boot of 2026-08-06 narrowed sqlite3InsertBuiltinFuncs to one difference. Inlining its
 * exact body -- union `u.pHash` write, `pOther` branch and all -- RETURNS (qr20, rc 20).
 * Calling the real function with the array as a PARAMETER wedges (qr16n). Every other
 * candidate is cleared: the static pointer-bearing array and its cap-init (fdreg 1),
 * storing derived capabilities into a global (fdreg 2), indirect calls through the array
 * (fdreg 3), high gp index (fdreg2p), sqlite3FunctionSearch walking the real global hash
 * (qr18), and linking into that real global (qr19).
 *
 * So what is left is: `&aDef[i]` derived from an ARGUMENT capability inside a non-inlined
 * callee, rather than from `gp` in the caller. That is what this stage isolates -- stage 2's
 * work verbatim, moved behind `noinline` and reached through a pointer parameter.
 *
 * noinline is load-bearing, not a hint: if the compiler inlines it, the derivation goes back
 * through gp and the stage silently becomes stage 2. Check the disassembly for a real call,
 * do not assume.
 */
__attribute__((noinline))
static void fdreg_link_via_param(FdregDef *aDef, int nDef) {
  int i;
  for (i = 0; i < nDef; i++) {
    const char *z = aDef[i].zName;
    unsigned h = fdreg_hash((unsigned char)z[0], fdreg_len30(z));
    aDef[i].pNext = fdreg_buckets[h];
    fdreg_buckets[h] = &aDef[i];
  }
}

/* STAGE 5 -- the CONTROL for stage 4, and the half that makes stage 4 attributable.
 *
 * Stage 4 changes two things at once relative to stage 2: the work moves into a separate
 * non-inlined function, AND the array is reached through a pointer parameter. If stage 4
 * wedges, those two are not yet separated -- "a call wedges" and "an argument capability
 * wedges" are very different findings.
 *
 * This stage takes the first without the second: same noinline callee, same call, but it
 * reads the global directly through gp instead of taking it as an argument. So
 *     stage 5 returns AND stage 4 wedges  -> the ARGUMENT capability is the fault
 *     both wedge                          -> the non-inlined CALL is the fault
 *     both return                         -> stage 4's wedge, if any, is elsewhere
 * Same oracle as stages 2 and 4 (2609) -- identical work, three derivation paths.
 *
 * BOARD RESULT 2026-08-06 (control k800 = 4): the THIRD case. Stage 2 (inline), stage 5
 * (noinline, reads via gp) and stage 4 (noinline, array as a pointer parameter) ALL
 * returned 2609. So the argument-capability shape is fine here, and the reading of the
 * SQLite probe qr21 -- that this shape was the wedge -- was retracted on this run.
 *
 * That leaves a gap rather than a closed question, because the same shape inside the
 * SQLite image DOES wedge: level 19 (loop inline) returns on two independent draws, and
 * level 22 -- identical loop, same array, same declaration site, no new global, only moved
 * behind the noinline callee -- wedges, both created and entered. The images differ by
 * SCALE: this rung has 12 globals and reaches gp index 11, while the SQLite domain has 176
 * and reaches 175, so it spends 846 accesses on the lui/addi/cincoffset/ldc path that only
 * exists above index 127 -- a path this rung never touches at all.
 * FDREG_STAGE=4 with FDREG_PAD=1 is precisely that test, the parameter shape ABOVE the
 * boundary. Oracle 2769.
 */
__attribute__((noinline))
static void fdreg_link_via_global(int nDef) {
  int i;
  for (i = 0; i < nDef; i++) {
    const char *z = fdreg_defs[i].zName;
    unsigned h = fdreg_hash((unsigned char)z[0], fdreg_len30(z));
    fdreg_defs[i].pNext = fdreg_buckets[h];
    fdreg_buckets[h] = &fdreg_defs[i];
  }
}

/* STAGE 6 -- the next single-variable step, and the one the disassembly points at.
 *
 * Stage 4 RETURNS on silicon at both low and high gp index (2609 / 2769), yet the same shape
 * inside SQLite WEDGES (level 19 returns twice, level 22 wedges twice). Comparing the two
 * callees shows a structural difference that is not about capabilities at all:
 *
 *     fdreg_link_via_param  0 calls inside  -- a LEAF
 *     qr_link_via_param     1 call inside   -- NON-LEAF, and it spills/reloads the return
 *                                              capability with 2x stc/ldc on sp
 *
 * SQLite's callee calls sqlite3Strlen30; fdreg's helpers all inline away. A non-leaf callee
 * has to save `ra` -- a CAPABILITY -- across the inner call, which a leaf never does. That is
 * the one structural thing the returning rung has never exercised in this position.
 *
 * So stage 6 is stage 4 with the strlen forced out of line, making the callee non-leaf.
 * Oracle 2609, the same as stages 2, 4 and 5: identical work, and now a fourth derivation
 * path. Check the disassembly for a call INSIDE the callee, not just for the callee itself.
 */
__attribute__((noinline))
static unsigned fdreg_len30_noinline(const char *z) {
  unsigned n = 0;
  while (z[n]) n++;
  return n;
}

__attribute__((noinline))
static void fdreg_link_via_param_nonleaf(FdregDef *aDef, int nDef) {
  int i;
  for (i = 0; i < nDef; i++) {
    const char *z = aDef[i].zName;
    unsigned h = fdreg_hash((unsigned char)z[0], fdreg_len30_noinline(z));
    aDef[i].pNext = fdreg_buckets[h];
    fdreg_buckets[h] = &aDef[i];
  }
}

#if FDREG_STAGE == 23
/* STAGE 23 -- CHANGE THE CALL DEPTH. Absolute address, or sp-relative offset?
 *
 * Boot 57 localised the victim to sp+0x1c. But sp itself has been IDENTICAL in every build --
 * same domain, same entry, same call depth -- so "the poisoned slot is at sp+0x1c" and "the
 * poisoned slot is at one fixed ABSOLUTE stack address" are the same statement in all existing
 * data. Moving the store does not separate them (an array big enough to move the store also
 * swallows sp+0x1c). Changing the CALL DEPTH does: an extra frame shifts every absolute address
 * by that frame's size while leaving all sp-relative offsets inside fdreg_compute untouched.
 *
 *     same result as depth 0  -> the fault follows the sp-RELATIVE offset (frame-shaped)
 *     different result        -> the fault follows an ABSOLUTE address (one poisoned location)
 *
 * noinline is load-bearing: if the wrapper is inlined there is no extra frame and the arm
 * silently becomes depth 0. Check the disassembly for a real call, and check that the counters
 * still sit at the same sp-relative offsets.
 */
__attribute__((noinline)) static unsigned fdreg_depth_body(void);
__attribute__((noinline)) static unsigned fdreg_depth_w1(void){ return fdreg_depth_body(); }
__attribute__((noinline)) static unsigned fdreg_depth_w2(void){ return fdreg_depth_w1(); }
__attribute__((noinline)) static unsigned fdreg_depth_body(void){
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
        qc++;
      }
    return ((unsigned)(p & 0xFFF) << 20) | ((unsigned)(k & 0xF) << 16) | (qc & 0xFFFFu);
  }
}
#endif
static unsigned fdreg_compute(void) {
  unsigned i, s = 0;
#if (FDREG_DRAW) > 0
  __asm__ volatile(".rept " FDREG_DRAW_STR(FDREG_DRAW) "\n\tnop\n\t.endr" ::: "memory");
#endif

#if FDREG_STAGE == 31
  /* STAGE 31 -- IS THE MONITOR-CARVED STACK INVOLVED? Same hand-written asm, frame in a GLOBAL.
   *
   * Every failing image so far came out of our own LLVM fork, and the effect tracks where the -O0
   * allocator placed a scalar. That makes the COMPILER an untested variable: we have never run a
   * domain whose loop was written by hand. Until that is excluded, "silicon defect" is not a claim
   * we can make -- the fault could be our codegen emitting something the hardware legitimately
   * treats differently.
   *
   * This writes the whole loop in asm against a 16-byte-ALIGNED local array used as the frame, so
   * the RELATIVE geometry is ours, not the allocator's:
   *
   *     f+0x00   the 16-byte capability store target (row-aligned)
   *     f+0x14   k   -- bank 0, lanes 4-7
   *     f+0x1c   qc  -- bank 1, lanes 4-7   <-- the victim position in the c8 geometry
   *
   * FDREG_ASMVICTIM selects which offset qc uses, so the same hand-written body can be run at a
   * damaging and a non-damaging position:
   *     0x1c  the c8 victim slot (upper half)   -> expect the defect if it is not the compiler
   *     0x24  lower half of the next row        -> expect correct
   *
   * Returns qc. Oracle 576 either way; a shortfall is the finding.
   */
  {
    /* THE ONLY DIFFERENCE FROM STAGE 30: the frame lives in a GLOBAL, i.e. in the domain's data
       region, NOT on the monitor-carved stack. If the effect follows the scalars here, the stack's
       provenance and the monitor's carve are excluded as the cause. */
    static volatile unsigned char f[0x60] __attribute__((aligned(16)));
    unsigned qc_out = 0;
    unsigned i;
    for (i = 0; i < sizeof(f); i++) f[i] = 0;
    __asm__ volatile(
        /* a0 = capability over the frame array */
        "movc    a0, %1\n\t"
        /* zero the two counters: qc at ASMVICTIM, k at 0x14 */
        "sw      zero, %3(a0)\n\t"
        "sw      zero, 0x14(a0)\n\t"
        "li      t2, %4\n\t"            /* outer count */
        "1:\n\t"
        "sw      zero, 0x14(a0)\n\t"    /* k = 0 at the top of every outer pass */
        "li      t3, 9\n\t"             /* inner count */
        "2:\n\t"
        "stc     a0, 0x0(a0)\n\t"       /* 16-byte capability store at f+0 */
        "ldc     a1, 0x0(a0)\n\t"       /* the volatile read-back */
        "lw      t0, %3(a0)\n\t"        /* qc RMW */
        "addiw   t0, t0, 1\n\t"
        "sw      t0, %3(a0)\n\t"
        "lw      t1, 0x14(a0)\n\t"      /* k RMW */
        "addiw   t1, t1, 1\n\t"
        "sw      t1, 0x14(a0)\n\t"
        "addi    t3, t3, -1\n\t"
        "bnez    t3, 2b\n\t"
        "addi    t2, t2, -1\n\t"
        "bnez    t2, 1b\n\t"
        "lw      %0, %3(a0)\n\t"
        : "=r"(qc_out)
        : "r"(&f[0]), "r"(0), "i"(FDREG_ASMVICTIM), "i"(FDREG_OUTER)
        : "a0", "a1", "t0", "t1", "t2", "t3", "memory");
    return qc_out;
  }
#endif
#if FDREG_STAGE == 31
  /* STAGE 31 -- IS THE MONITOR-CARVED STACK INVOLVED? Same hand-written asm, frame in a GLOBAL.
   *
   * Every failing image so far came out of our own LLVM fork, and the effect tracks where the -O0
   * allocator placed a scalar. That makes the COMPILER an untested variable: we have never run a
   * domain whose loop was written by hand. Until that is excluded, "silicon defect" is not a claim
   * we can make -- the fault could be our codegen emitting something the hardware legitimately
   * treats differently.
   *
   * This writes the whole loop in asm against a 16-byte-ALIGNED local array used as the frame, so
   * the RELATIVE geometry is ours, not the allocator's:
   *
   *     f+0x00   the 16-byte capability store target (row-aligned)
   *     f+0x14   k   -- bank 0, lanes 4-7
   *     f+0x1c   qc  -- bank 1, lanes 4-7   <-- the victim position in the c8 geometry
   *
   * FDREG_ASMVICTIM selects which offset qc uses, so the same hand-written body can be run at a
   * damaging and a non-damaging position:
   *     0x1c  the c8 victim slot (upper half)   -> expect the defect if it is not the compiler
   *     0x24  lower half of the next row        -> expect correct
   *
   * Returns qc. Oracle 576 either way; a shortfall is the finding.
   */
  {
    /* THE ONLY DIFFERENCE FROM STAGE 30: the frame lives in a GLOBAL, i.e. in the domain's data
       region, NOT on the monitor-carved stack. If the effect follows the scalars here, the stack's
       provenance and the monitor's carve are excluded as the cause. */
    static volatile unsigned char f[0x60] __attribute__((aligned(16)));
    unsigned qc_out = 0;
    unsigned i;
    for (i = 0; i < sizeof(f); i++) f[i] = 0;
    __asm__ volatile(
        /* a0 = capability over the frame array */
        "movc    a0, %1\n\t"
        /* zero the two counters: qc at ASMVICTIM, k at 0x14 */
        "sw      zero, %3(a0)\n\t"
        "sw      zero, 0x14(a0)\n\t"
        "li      t2, %4\n\t"            /* outer count */
        "1:\n\t"
        "sw      zero, 0x14(a0)\n\t"    /* k = 0 at the top of every outer pass */
        "li      t3, 9\n\t"             /* inner count */
        "2:\n\t"
        "stc     a0, 0x0(a0)\n\t"       /* 16-byte capability store at f+0 */
        "ldc     a1, 0x0(a0)\n\t"       /* the volatile read-back */
        "lw      t0, %3(a0)\n\t"        /* qc RMW */
        "addiw   t0, t0, 1\n\t"
        "sw      t0, %3(a0)\n\t"
        "lw      t1, 0x14(a0)\n\t"      /* k RMW */
        "addiw   t1, t1, 1\n\t"
        "sw      t1, 0x14(a0)\n\t"
        "addi    t3, t3, -1\n\t"
        "bnez    t3, 2b\n\t"
        "addi    t2, t2, -1\n\t"
        "bnez    t2, 1b\n\t"
        "lw      %0, %3(a0)\n\t"
        : "=r"(qc_out)
        : "r"(&f[0]), "r"(0), "i"(FDREG_ASMVICTIM), "i"(FDREG_OUTER)
        : "a0", "a1", "t0", "t1", "t2", "t3", "memory");
    return qc_out;
  }
#endif
#if FDREG_STAGE == 34
  /* STAGE 34 -- IS IT THE REGION, OR THE ADDRESS CAPABILITY'S PROVENANCE?
   *
   * Boot 65 moved the counters to a global and cured the defect; boot 66 showed the capability
   * store's target is irrelevant. But "moved to a global" changed TWO things at once: the region,
   * and how the counter's address capability is obtained --
   *
   *     failing: address = cincoffsetimm(s0, imm)   -- from a REGISTER-RESIDENT capability
   *     passing: address = ldc gp[i], then offset   -- from a capability LOADED FROM MEMORY
   *
   * That is the same coupled-variable trap that produced four refuted "laws" earlier, so it must be
   * separated before "the stack" is treated as the cause.
   *
   * This keeps the counters ON THE STACK and changes ONLY the provenance: a `volatile` pointer
   * local holds their address, so at -O0 every access must `ldc` the capability back out of memory
   * before the `lw`/`sw` -- exactly the passing arm's pattern, in the failing arm's region.
   *
   *     still fails -> provenance is irrelevant; the REGION is the variable and "must be on the
   *                    stack" stands
   *     now clean   -> the region was never the variable; it is how the address capability is
   *                    obtained, and boot 65's reading has to be withdrawn
   *
   * FDREG_PVICT picks which slot of the on-stack array the accumulator uses, so the victim row
   * offset can be set explicitly rather than left to the allocator.
   */
  {
    volatile unsigned cnt[8] __attribute__((aligned(16)));
    volatile unsigned *volatile pc = &cnt[(FDREG_PVICT)];   /* address parked IN MEMORY */
    int p, k;
    unsigned i;
    for (i = 0; i < 8; i++) cnt[i] = 0;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* the capability store, unchanged */
        (void)z;
        (*pc)++;                                        /* ldc pc -> lw -> addi -> sw */
      }
    return cnt[(FDREG_PVICT)];
  }
#endif
#if FDREG_STAGE == 33
  /* STAGE 33 -- IS THE TRIGGER "SAME CAPABILITY FOR BOTH"? Counters stay on the stack; only the
   * capability STORE TARGET moves to a global.
   *
   * Boot 65 showed that moving the counters off the monitor-carved stack cures the defect. But that
   * build also changed the counters' ACCESS PATH (through gp instead of s0), so two things moved at
   * once. The hypothesis it suggests:
   *
   *     in the failing build the SAME frame capability (s0) derives BOTH the 16-byte `stc` target
   *     AND the scalar counter addresses; in the passing build the counters went through `gp`, a
   *     DIFFERENT capability.
   *
   * This tests that directly and in the other direction. The counters stay exactly where they were
   * -- stack locals reached via s0 -- and only the capability store's target becomes a global,
   * reached via gp.
   *
   *     still fails -> the store's provenance is irrelevant; stack counters plus ANY nearby
   *                    capability store suffice, and the "same capability" idea is dead
   *     now clean   -> the defect needs the store target and the counters to derive from the SAME
   *                    capability. That is an ABI/monitor property, not a hardware one, and it
   *                    would explain boot 65 without invoking the stack at all
   *
   * Everything else is held: same compiler, same -O0, same two per-iteration ldc's, same trip
   * counts, counters at the same frame offsets (set FDREG_SHIFT so qc lands at row offset 12).
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        fdreg_gz = fdreg_defs[k].zName;   /* the capability STORE now targets a GLOBAL, via gp */
        (void)fdreg_gz;
        qc++;
      }
    return qc;
  }
#endif
#if FDREG_STAGE == 32
  /* STAGE 32 -- CONTROLLED: stage 19's loop VERBATIM, scalars moved to a GLOBAL. One variable.
   *
   * A previous attempt (stages 30/31) put a HAND-WRITTEN loop with globals against the
   * compiler-generated stack loop and found the hand-written one clean. That proved nothing: the
   * two differed in at least FIVE ways -- hand asm vs codegen, global vs stack, half the cycle
   * count, the stored capability's provenance, and the complete absence of the two per-iteration
   * `ldc`s (one through gp, one out of the struct). Exactly the multi-variable confound that
   * killed four geometric laws in this investigation.
   *
   * This changes ONE thing from the failing c8 build: the three counters live in a GLOBAL instead
   * of on the monitor-carved stack. Same compiler, same -O0, same loop source, same capability
   * store, same two ldc's, same trip counts.
   *
   *     still fails -> the monitor-carved STACK is excluded; the fault follows the counters
   *     now clean  -> the stack (or its capability's provenance) is load-bearing, and the defect
   *                   is plausibly OURS rather than the hardware's
   *
   * FDREG_GVICT selects the accumulator's offset inside the global block so the victim geometry
   * can be set explicitly rather than left to the allocator.
   */
  /* *** BOOT 65 USING THIS STAGE IS RETRACTED. *** As originally written this block had NO
   * FDREG_SHIFT pad, so it could not be placed at the failing geometry and shipped at the SHIFT=0
   * layout -- stack slots 0x1c/0x20/0x24, byte-identical to c0/rs0/bs16/nr16, all of which the
   * dataset already lists as UNDAMAGED. gv3 was predicted correct before any global was involved,
   * so it never tested the global at all. Fifth confound of the same class.
   * It also returned only the GLOBAL accumulator while qc/p/k stayed on the stack unobserved.
   * Both are fixed here: the pad is added, and the packed return reports the STACK accumulator and
   * the GLOBAL one together so the comparison is actually made. */
  {
    static volatile unsigned gcnt[16] __attribute__((aligned(16)));
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    gcnt[(FDREG_GVICT)] = 0;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* the two ldc's, kept */
        (void)z;
        gcnt[(FDREG_GVICT)]++;                          /* accumulator, now in a GLOBAL */
        qc++;
      }
    /* bits 31..16 = STACK accumulator, bits 15..0 = GLOBAL accumulator. Correct = 0x02400240. */
    return ((qc & 0xFFFFu) << 16) | (gcnt[(FDREG_GVICT)] & 0xFFFFu);
  }
#endif
#if FDREG_STAGE == 30
  /* STAGE 30 -- IS THE COMPILER INVOLVED AT ALL? Hand-written assembly, same geometry.
   *
   * Every failing image so far came out of our own LLVM fork, and the effect tracks where the -O0
   * allocator placed a scalar. That makes the COMPILER an untested variable: we have never run a
   * domain whose loop was written by hand. Until that is excluded, "silicon defect" is not a claim
   * we can make -- the fault could be our codegen emitting something the hardware legitimately
   * treats differently.
   *
   * This writes the whole loop in asm against a 16-byte-ALIGNED local array used as the frame, so
   * the RELATIVE geometry is ours, not the allocator's:
   *
   *     f+0x00   the 16-byte capability store target (row-aligned)
   *     f+0x14   k   -- bank 0, lanes 4-7
   *     f+0x1c   qc  -- bank 1, lanes 4-7   <-- the victim position in the c8 geometry
   *
   * FDREG_ASMVICTIM selects which offset qc uses, so the same hand-written body can be run at a
   * damaging and a non-damaging position:
   *     0x1c  the c8 victim slot (upper half)   -> expect the defect if it is not the compiler
   *     0x24  lower half of the next row        -> expect correct
   *
   * Returns qc. Oracle 576 either way; a shortfall is the finding.
   */
  {
    unsigned qc_out = 0;
    /* The frame capability is derived from sp INSIDE the asm. Passing &local through an "r"
       constraint puts the address in a plain integer register and DROPS the capability, so the
       first `stc` through it faults -- that is why the earlier version of this stage produced no
       result under QEMU. sp is 16-byte aligned by ABI, so sp-0x40 is row-aligned and the offsets
       below are exact. The block reserves and restores its own space. */
    __asm__ volatile(
        "cincoffsetimm sp, sp, -0x40\n\t"
        "movc    a0, sp\n\t"
        /* zero the two counters: qc at ASMVICTIM, k at 0x14 */
        "sw      zero, %3(a0)\n\t"
        "sw      zero, 0x14(a0)\n\t"
        "li      t2, %4\n\t"            /* outer count */
        "1:\n\t"
        "sw      zero, 0x14(a0)\n\t"    /* k = 0 at the top of every outer pass */
        "li      t3, 9\n\t"             /* inner count */
        "2:\n\t"
        "stc     a0, 0x0(a0)\n\t"       /* 16-byte capability store at f+0 */
        "ldc     a1, 0x0(a0)\n\t"       /* the volatile read-back */
        "lw      t0, %3(a0)\n\t"        /* qc RMW */
        "addiw   t0, t0, 1\n\t"
        "sw      t0, %3(a0)\n\t"
        "lw      t1, 0x14(a0)\n\t"      /* k RMW */
        "addiw   t1, t1, 1\n\t"
        "sw      t1, 0x14(a0)\n\t"
        "addi    t3, t3, -1\n\t"
        "bnez    t3, 2b\n\t"
        "addi    t2, t2, -1\n\t"
        "bnez    t2, 1b\n\t"
        "lw      %0, %3(a0)\n\t"
        : "=r"(qc_out)
        : "r"(&f[0]), "r"(0), "i"(FDREG_ASMVICTIM), "i"(FDREG_OUTER)
        : "a0", "a1", "t0", "t1", "t2", "t3", "memory");
    return qc_out;
  }
#endif
#if FDREG_STAGE == 28
  /* STAGE 28 -- DOES THE DEFICIT TRACK THE INNER TRIP COUNT? The outer-pass lead.
   *
   * All three measured reset points -- 9, 72, 558 -- are multiples of 9, the inner trip count,
   * i.e. they land exactly on OUTER-PASS BOUNDARIES. Under a uniform-over-iterations null that is
   * p ~ 1/729. A per-iteration event has no reason to align that way; something that happens ONCE
   * PER OUTER PASS does. The obvious candidate is the `k = 0` re-initialisation store, which at the
   * failing geometry sits at 0x14 (bank 0, lanes 4-7) whose dual-bank sibling is exactly 0x1c --
   * the victim.
   *
   * This decouples the inner trip count from the array size so the prediction is falsifiable:
   *
   *     FDREG_INNER=9  (default)  correct = 64*9  = 576, deficits should be multiples of 9
   *     FDREG_INNER=12            correct = 64*12 = 768, deficits should be multiples of 12
   *     FDREG_INNER=6             correct = 64*6  = 384, deficits should be multiples of 6
   *
   * If the deficit tracks FDREG_INNER, the culprit is the per-outer-pass store and NOT the victim's
   * own per-iteration store -- which would also finally distinguish c8 from gp16. If the deficit
   * stays a multiple of 9 regardless, the alignment was a coincidence of three samples.
   *
   * The index is taken modulo FDREG_N so the array access stays in range for any inner count.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < (FDREG_INNER); k++) {
        const char *volatile z = fdreg_defs[k % FDREG_N].zName;
        (void)z;
        qc++;
      }
    return qc;
  }
#endif
#if FDREG_STAGE == 27
  /* STAGE 27 -- LOST INCREMENTS, or RESET TO ZERO? A sentinel start settles it.
   *
   * Every probe so far reported a counter short by an exact number of increments and called it
   * "dropped stores". There is a second reading that fits the numbers better and has an RTL path:
   *
   *   bank_be is the SAME byte-enable applied to BOTH banks, and bank_wdata[1] receives wr_user_i
   *   instead of wr_data_i when st_wr_cap fires (wt_dcache_mem.sv:156-158, :225-238). So a store
   *   whose address lies in BANK 1 has its OWN slot written with wr_user_i at those byte lanes --
   *   the real data never lands. That is exactly why the victim is ALWAYS in the upper half of its
   *   16-byte row (9/9 builds): bank 1 is the only bank that can receive something other than the
   *   store's own data. If wr_user_i is zero at those lanes, the counter is RESET TO ZERO rather
   *   than corrupted, which is why victims hold plausible counts instead of garbage.
   *
   * Read against the measurements, a single reset explains each one exactly:
   *     qc=567 = 576-9    -> reset once at iteration 9
   *     qc=504 = 576-72   -> reset at iteration 72
   *     d=18   (of 576)   -> reset at iteration 558, then 18 more increments
   *     p reset           -> the OUTER loop restarts -> EXTRA passes (+9, +333)
   *     0x8000237         -> 567 with bit 27 set: the same, with one metadata bit not zero
   *
   * THE TEST: start the accumulator at a large sentinel instead of 0.
   *     final == SENTINEL + 576  -> increments were merely LOST; the reset story is wrong
   *     final <  576             -> the slot was RESET TO ZERO and counted up from there, and the
   *                                 value names the iteration at which the last reset happened
   * These two cannot be confused, which no previous probe could manage.
   *
   * Packing must not truncate the sentinel, so this returns qc RAW and puts the loop counters in
   * a separate build. Oracle = SENTINEL + 576, computed programmatically.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = (FDREG_SENTINEL);
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
        qc++;
      }
    return qc;
  }
#endif
#if FDREG_STAGE == 26
  /* STAGE 26 -- IS k SPECIAL AT ALL, OR IS THE POISONED SLOT JUST ALONE IN ITS ROW?
   *
   * "k is immune" is confounded: -O0 packs k lowest with p=k+4 and qc=k+8, so k@0x1c can ONLY
   * occur when p and qc land at 0x20/0x24 -- i.e. only when the row 0x10..0x1f holds nothing
   * else. Every build where k sits at the poisoned slot also has that slot ALONE in its row.
   * The row-occupancy rule fits 19/20 builds against the k-rule's 17/19.
   *
   * This adds ONE extra hot scalar `d`, incremented next to qc so the two must stay equal.
   * Declared last, it lands LOWEST, which lets the row be populated independently of which
   * counter sits at +0x1c. d - qc names the victim without needing the cycle divisor.
   *
   *   FDREG_SHIFT=8  -> d@0x18, k@0x1c, p@0x20, qc@0x24: k AT the poisoned slot, sharing its row
   *                    with a hot neighbour. This is the shift0 geometry plus one scalar.
   *                      correct        -> k really is intrinsically immune, row rule dead
   *                      d == qc, both high, cycles up -> k damaged: the row rule is right and
   *                                        k's immunity dissolves
   *                      d < 576, qc ok -> the victim is d@0x18, so the +0x1c anchor is wrong
   *   FDREG_SHIFT=4  -> d@0x1c ALONE in its row (k@0x20, p@0x24, qc@0x28): a NON-k hot scalar at
   *                    the poisoned slot with an empty row. The mirror test.
   *                      correct  -> row-occupancy confirmed from the other side
   *                      d < 576  -> the slot is poisoned regardless of occupancy; k IS special
   *
   * Packing: p<<24 | d<<12 | qc. Oracle computed programmatically, never by hand -- a
   * hand-computed packed oracle already flagged a correct board result as WRONG once.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    unsigned d = 0;                 /* declared LAST -> lands LOWEST */
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
        d++;
        qc++;
      }
    return ((unsigned)(p & 0xFF) << 24) | ((d & 0xFFFu) << 12) | (qc & 0xFFFu);
  }
#endif
#if FDREG_STAGE == 25
  /* STAGE 25 -- WHY IS k SPARED? Two remaining candidates, one knob each.
   *
   * k is the only loop variable never damaged when it occupies the poisoned slot
   * (capability-store + 0x1c). The read-count explanation is REFUTED (boot 59: giving qc a second
   * read changes nothing). Two structural differences remain:
   *
   *   FDREG_NORESET=1 -- k is RE-INITIALISED to 0 at the top of every outer pass; qc and p are not.
   *       This carries the index across outer passes so it becomes MONOTONE and loses exactly that
   *       property. If k then becomes damaged at the poisoned slot, the re-initialisation is what
   *       spares it.
   *
   *   FDREG_ORDER=1 -- qc's RMW is the FIRST scalar access after the capability store, k's is the
   *       last. This swaps them. If the damage follows whichever now goes first, program-order
   *       position is the variable. (Weak prior: p is damaged although its increment lives in the
   *       OUTER tail, far from the capability store.)
   *
   * Packing: a monotone index no longer fits in 4 bits.
   *     bits 31..24 p (expect 64) | bits 23..12 k | bits 11..0 qc (expect 576)
   *     with reset   -> (64<<24)|(9<<12)|576   = 0x40009240
   *     with NORESET -> (64<<24)|(576<<12)|576 = 0x40240240
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
#if (FDREG_NORESET)
    k = 0;
    for (p = 0; p < (FDREG_OUTER); p++) {
      int kend = k + FDREG_N;
      for (; k < kend; k++) {
        const char *volatile z = fdreg_defs[k % FDREG_N].zName;
        (void)z;
        qc++;
      }
    }
#elif (FDREG_ORDER)
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; ) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
        k++;
        qc++;
      }
#else
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
        qc++;
      }
#endif
    return ((unsigned)(p & 0xFF) << 24) | ((unsigned)(k & 0xFFF) << 12) | (qc & 0xFFFu);
  }
#endif
#if FDREG_STAGE == 24
  /* STAGE 24 -- IS THE SECOND READ THE CURE? Testing k's immunity directly.
   *
   * The victim is the slot at capability-store + 0x1c, and whichever loop variable occupies it is
   * damaged -- EXCEPT k, which is spared in every build where it lands there (shift0, c0, rs0).
   * The one structural difference: k is read TWICE per iteration (once for the loop compare
   * `li a0,0x8; blt`, once for the increment), while qc and p are read once.
   *
   * This adds a second read of qc, with nothing else changed, and puts qc at the poisoned slot:
   *
   *     qc becomes CORRECT -> the second read is the cure. That is both a workaround and a strong
   *         mechanistic clue: the slot's value is RECOVERABLE, so the damage is not a lost store
   *         but something a redundant access repairs or re-fetches.
   *     qc stays 567       -> k's immunity is NOT about the read count, and the next candidate is
   *         that k is the loop's own bound-compared variable, or that its store is the LAST in the
   *         body rather than the first.
   *
   * FDREG_REREAD=1 emits the extra read. The comparison constant can never be true, so the value
   * is unchanged and the oracle stays 576 -- only the access pattern differs. Verify in the
   * artifact that an extra `lw` of qc's slot really appears, and that the frame did not move.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
#if (FDREG_REREAD)
        if (qc == 0xFFFFFFFFu) return 0;   /* a second READ of qc, mirroring k's compare */
#endif
        qc++;
      }
    return ((unsigned)(p & 0xFFF) << 20) | ((unsigned)(k & 0xF) << 16) | (qc & 0xFFFFu);
  }
#endif
#if FDREG_STAGE == 23
  {
#if (FDREG_DEPTH) >= 2
    return fdreg_depth_w2();
#elif (FDREG_DEPTH) == 1
    return fdreg_depth_w1();
#else
    return fdreg_depth_body();
#endif
  }
#endif
#if FDREG_STAGE == 22
  /* STAGE 22 -- MOVE THE CAPABILITY STORE OFF sp+0. The last confound in the localisation.
   *
   * Boot 57 localised the victim to sp+0x1c. But the 16-byte capability store has been at
   * sp+0x00 in ALL NINETEEN builds, because `z` is the only capability local and -O0 puts the
   * 16-byte-aligned object at the bottom of the frame. So "the victim is the absolute address
   * sp+0x1c" and "the victim is 0x1c above the STORE" fit the existing data equally well, and
   * they are completely different claims -- a poisoned stack address versus a store-relative
   * effect at a fixed distance.
   *
   * FDREG_CAPPAD declares extra capability locals ahead of z. They are 16-byte aligned, so each
   * one pushes z (and therefore the store) up by 16 bytes while the counters stay put:
   *
   *     CAPPAD=0   store at sp+0x00, victim slot sp+0x1c is store+0x1c   (both rules agree)
   *     CAPPAD=1   store at sp+0x10, so store+0x1c = sp+0x2c != sp+0x1c  (rules DIVERGE)
   *
   * Build it with the counters arranged so that sp+0x1c holds qc and sp+0x2c holds p, then:
   *     qc damaged -> the victim is the ABSOLUTE address sp+0x1c
   *     p  damaged -> the victim is 0x1c above the STORE, wherever the store is
   *     neither    -> the localisation is contingent on something else again
   *
   * The pads must be volatile and touched, or they are elided and the store does not move.
   * VERIFY the store's offset in the artifact -- that is the whole point of the build.
   */
  {
    /* Declaring extra capability LOCALS does not move the store: -O0 still puts the loop's own
       `z` at the frame bottom. An ARRAY does -- storing to element FDREG_CAPSLOT puts the 16-byte
       store at base + 16*FDREG_CAPSLOT while the array base stays at sp+0, so the counters do not
       move and only the store's address changes. That is exactly the variable needed. */
    const char *volatile zarr[(FDREG_CAPSLOT) + 1];
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        zarr[(FDREG_CAPSLOT)] = fdreg_defs[k].zName;   /* cap store, now at a chosen offset */
        (void)zarr[(FDREG_CAPSLOT)];
        qc++;
      }
    return ((unsigned)(p & 0xFFF) << 20) | ((unsigned)(k & 0xF) << 16) | (qc & 0xFFFFu);
  }
#endif
#if FDREG_STAGE == 20
  /* STAGE 20 -- THE ROLE SWAP. Does the fault follow the SLOT or the VARIABLE?
   *
   * Every build in this investigation has had the -O0 allocator place the three counters in
   * one fixed order: k lowest, p = k+4, qc = k+8. So "the slot at k+8" and "the accumulator"
   * have never been separated, exactly as "k's bits[3:2]" and "qc is k's bank sibling" were
   * never separated until stage 18. Two successive geometric laws died of that class of
   * confound; this is the same confound one level down.
   *
   * Swapping the declaration order puts the SAME three slots under DIFFERENT variables:
   *
   *     stage 19 (and every earlier build)      stage 20
   *     k  @ sp+0x14   inner index              k  @ sp+0x14   inner index
   *     p  @ sp+0x18   outer counter            qc @ sp+0x18   accumulator
   *     qc @ sp+0x1c   accumulator              p  @ sp+0x1c   outer counter
   *
   * k, the loop's own index, is left exactly where it was, so the geometry the old table
   * indexes on is untouched. Only which variable occupies the upper two slots changes.
   *
   *     a 909-family answer (extra outer passes) -> the fault follows the SLOT: whatever
   *         scalar sits in the upper slot is the victim, and it is now the OUTER counter, so
   *         the signature flips from "lost 9" to "ran extra passes". That would unify the 567
   *         and 909 cells as ONE fault with two victims and make the law purely geometric.
   *     a 567-family answer (qc short by 9)      -> the fault follows the VARIABLE: the
   *         accumulator loses 9 wherever it lives, and the slot geometry is not the operative
   *         thing at all.
   *     correct                                  -> neither alone; the defect needs the
   *         accumulator specifically in the upper slot.
   *
   * Returns the packed triple as stage 19 (p<<20 | k<<16 | qc), so the answer names the victim
   * instead of only reporting that a number was wrong -- and it must be read together with the
   * cycle count, which independently separates "lost an increment" from "ran extra passes".
   *
   * VERIFY THE SLOT ASSIGNMENT IN THE ARTIFACT. clang is free to reorder; the bounded-by-8
   * counter must be the low slot and the bounded-by-0x3f counter the high one. Adjusting
   * FDREG_SHIFT until k lands where intended is expected, not a workaround.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    int p;                    /* declared FIRST -> highest slot, was qc's */
    unsigned qc = 0;          /* declared SECOND -> middle slot, was p's  */
    int k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* cap field, inner resetting counter */
        (void)z;
        qc++;
      }
    return ((unsigned)(p & 0xFFF) << 20) | ((unsigned)(k & 0xF) << 16) | (qc & 0xFFFFu);
  }
#endif
#if FDREG_STAGE == 19
  /* STAGE 19 -- REPORT ALL THREE COUNTERS. The instrument fix.
   *
   * Every rung built for this defect has returned qc alone. The cycle counts then showed the
   * four-cell table conflates TWO DIFFERENT FAULTS: the 567 family executes the full 576
   * iterations and loses exactly one outer pass of accumulator increments, while the 909
   * family genuinely executes ~333 extra iterations because the OUTER counter is disturbed.
   * A probe returning qc cannot say which fault it saw, which is why two successive geometric
   * "laws" both held until the next build and then died.
   *
   * At -O0 the frame packs k, p = k+4, qc = k+8, and every one of them is a candidate victim.
   * This returns all three in one word, so a single arm says WHICH counter was damaged rather
   * than only that the answer was wrong:
   *
   *     bits 31..20  p, the outer counter at exit   (expect FDREG_OUTER = 64)
   *     bits 19..16  k, the inner index at exit     (expect FDREG_N = 9)
   *     bits 15..0   qc, the accumulator            (expect 576)
   *
   * Correct is therefore 0x04090240. Read it with the CYCLE COUNT, which is free and printed
   * on every RESULT line: cycles/76.58 gives the iterations actually executed, and that
   * separates "lost an increment" from "ran extra passes" independently of the returned value.
   *
   * The loop body is byte-identical to stage 7; only the return expression differs, so the
   * frame layout is unchanged and comparisons against the existing table stay valid. Verify
   * that in the artifact -- an added local would move every slot and cure the fault, which is
   * exactly how stages 10 and 11 destroyed what they were measuring.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    /* FDREG_GAP here too, so the frame-size variants (gp16/sep20 geometries) can be re-measured
       with the victim NAMED instead of inferred from a single number. Those two builds are what
       make "the victim is at sp+0x1c" and "the victim is at s0-0x34" each fail on one build:
       at frame 0x60 the two rules point at different slots, and only a triple report says which
       one actually took the damage. */
#if (FDREG_GAP) > 0
    volatile unsigned char fdreg_gap19[FDREG_GAP];
    fdreg_gap19[0] = 0;
#endif
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* cap field, inner resetting counter */
        (void)z;
        qc++;
      }
    return ((unsigned)(p & 0xFFF) << 20) | ((unsigned)(k & 0xF) << 16) | (qc & 0xFFFFu);
  }
#endif
#if FDREG_STAGE == 18
  /* STAGE 18 -- BREAK THE qc == k+8 INVARIANT. This is the confound in our own law.
   *
   * The bank-geometry model says the returned value is a function of the inner counter k's
   * address bits [3:2]. It predicts 7 of 7 builds. But in EVERY build ever measured,
   * qc == k + 8 exactly: the accumulator is the inner counter's 8-byte BANK SIBLING, because
   * clang always allocates them four bytes apart with p in between. So two very different
   * statements are entangled and no measurement has ever separated them:
   *
   *     (A) the answer depends on k's own position in the cache row, or
   *     (B) the answer depends on qc being k's bank sibling -- i.e. on the RELATIONSHIP
   *
   * Under (B) the "law" is not about k at all; it is about a 16-byte row that happens to
   * contain both counters, and every entry in the table is a different way of arranging that
   * one pair.
   *
   * FDREG_GAP inserts dead bytes BETWEEN qc and the counters, which moves qc away from k
   * without moving k relative to the capability store. It MUST be a multiple of 16, or k's own
   * bits [3:2] move too and the arms stop being comparable -- that is the whole point of the
   * knob and the reason it is not simply "add a local".
   *
   * Built at the SHIFT=8 geometry, which returns a wrong 567 on silicon, so a cure is visible:
   *
   *     GAP=0   -> 567   the established value; positive control that the fault is live today
   *     GAP=16  -> 567   qc moved out of k's bank and the answer did not change
   *                      => the law is about k ALONE, hypothesis (A), and the sibling
   *                         relationship was a coincidence of clang's allocator
   *     GAP=16  -> 576   moving qc out of k's bank CURED it
   *                      => hypothesis (B): the defect needs BOTH slots in one row, and every
   *                         previous reading of the table has to be redone
   *
   * Either answer is worth a boot: one promotes the law from correlation to something about k,
   * the other demolishes it. All arms RETURN a number, so none can wedge the boot.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
#if (FDREG_GAP) > 0
    volatile unsigned char fdreg_gap[FDREG_GAP];
    fdreg_gap[0] = 0;
#endif
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* cap field, inner resetting counter */
        (void)z;
        qc++;
      }
    return qc;
  }
#endif
#if FDREG_STAGE == 16
  /* STAGE 16 -- THE SHADOW-CLEARING BARRIER: a direct test of the misclassified-store
   * mechanism, and if it works, a compiler workaround.
   *
   * RTL chain, every line verified against the primary source:
   *   commit_stage.sv:323-325   we_gpr_o[0] = 1'b1 unconditionally, but
   *                             cap_we_o[0] = commit_instr_i[0].cap_result.valid
   *                             -> an ordinary instruction overwrites the integer register
   *                                and LEAVES THE SHADOW CAPABILITY METADATA STALE
   *   wt_dcache_wbuffer.sv:602  every store captures whatever metadata is on the bus
   *   wt_dcache_mem.sv:138      st_wr_cap = |wr_user_i  -- a store is classified as a
   *                             capability store BY VALUE, NOT BY OPCODE
   *   wt_dcache_mem.sv:225-238  a store so classified writes BOTH 8-byte banks of its
   *                             16-byte row, bank 1 receiving the metadata (:156-158)
   *
   * Our own inner loop contains the taint sequence exactly:
   *     ldc  a0, 0x0(a0)     <- a0 receives a CAPABILITY (the volatile read-back of z)
   *     lw   a0, 0x0(a1)     <- a0 becomes the counter; `lw` is not a capstone op, so
   *                             a0's shadow metadata is never cleared
   *     sw   a0, 0x0(a1)     <- carries the stale metadata -> st_wr_cap fires
   *
   * The value `ldc` loads is DISCARDED ((void)z), so a0 is free to clobber immediately
   * after it.
   *
   * *** THE PREMISE BELOW IS FACTUALLY WRONG -- corrected 2026-08-07. ***
   * It said: "`movc rd, zero` is a capstone op, so it sets cap_result.valid and writes ZERO into
   * the shadow." It does NOT write zero. compress_cap of a NULL capability is 0x08000000, not 0
   * (ariane_pkg.sv:753-834; the cursorless bit is bit 27). So this barrier writes 0x08000000 into
   * the shadow of a0 and a1 -- the very registers the -O0 RMW uses. Under the metadata-clobber
   * model this arm should therefore have pinned the accumulator near zero on EVERY iteration; it
   * returned 567, bit-identical to the nop control. That made bar1/bar2 a REFUTING datum for that
   * model, and it was recorded as inconclusive. Do not re-run this as a "clearing" barrier.
   *
   * FDREG_BARRIER=1 emits the clearing barrier; =2 emits the SAME NUMBER OF BYTES of nops.
   * The nop arm is the control and it is what makes this attributable: both builds have
   * identical size, identical frame and identical layout, and differ only in whether the
   * shadow is cleared. Without it, "adding two instructions cured it" is unattributable --
   * which is how the guard/no-guard pair misled this investigation once already.
   *
   *     barrier cures (576) and nop does not (567)  -> the stale-metadata misclassification
   *                                                    is CONFIRMED, and this is a workaround
   *     both 567                                    -> mechanism REFUTED for this defect
   *     both 576                                    -> the barrier's LAYOUT cured it, not the
   *                                                    clearing; attributable to nothing
   *
   * Built at SHIFT=8, i.e. the (bank 0, offset 4) geometry that measures 567 on silicon, so
   * there is a known wrong value to cure.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* cap field, inner resetting counter */
        (void)z;
#if (FDREG_BARRIER) == 1
        __asm__ volatile("movc a0, zero\n\tmovc a1, zero" ::: "a0", "a1");
#elif (FDREG_BARRIER) == 2
        __asm__ volatile("nop\n\tnop" ::: "a0", "a1");
#endif
        qc++;
      }
    return qc;
  }
#endif
#if FDREG_STAGE == 14
  /* STAGE 14 -- THE READ-ONLY WITNESS, the other half of stage 13.
   *
   * Stage 13 puts witnesses in the damage window and NEVER reads them during the loop, so it
   * asks: is memory itself damaged? This one READS all three every inner iteration and never
   * writes them, so it asks the complementary question: is a load in the store's shadow
   * answered with the wrong data even though memory is fine? The counters could never
   * separate these because a counter is read-modify-WRITTEN, so a mis-answered load is
   * immediately written back and becomes indistinguishable from real corruption.
   *
   * Together the two stages are a complete truth table:
   *
   *     stage 13 (never read)   stage 14 (read, never written)   conclusion
   *     ---------------------   ------------------------------   -------------------------
   *     damaged                 either                           OVER-WIDE WRITE
   *     intact                  wrong reads                      FALSE STORE-TO-LOAD FORWARD
   *     intact                  correct reads                    neither -- needs the RMW
   *
   * No new local: the mismatch mask goes in `s`, which already has a frame slot and is
   * already zeroed at function entry. Adding one would re-shift the frame and cure the
   * patient, exactly as it did in stages 10 and 11. The three reads are placed immediately
   * after the `stc`, the same position the counter's load occupies in stage 7.
   *
   * Returns 0xFEED000M with M the mismatch mask (bit i = wit[i] read wrong at least once),
   * or the loop's 576 when every read was correct.
   */
  {
    unsigned qc = 0;
    int p, k;
#if (FDREG_WITPAD) > 0
    volatile unsigned char witpad[FDREG_WITPAD];
    witpad[0] = 0;
#endif
    volatile unsigned wit[3];
    wit[0] = 0xA5A50000u;
    wit[1] = 0xA5A50001u;
    wit[2] = 0xA5A50002u;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* cap field, inner resetting counter */
        (void)z;
        if (wit[0] != 0xA5A50000u) s |= 1u;
        if (wit[1] != 0xA5A50001u) s |= 2u;
        if (wit[2] != 0xA5A50002u) s |= 4u;
        qc++;
      }
    (void)i;
    if (s) return 0xFEED0000u | s;
    return qc;
  }
#endif
#if FDREG_STAGE == 13
  /* STAGE 13 -- THE PASSIVE WITNESS: is MEMORY corrupted, or is only a LOAD mis-answered?
   *
   * Two mechanisms survive the shift sweep and both predict every number in it:
   *   (1) an OVER-WIDE WRITE -- the 16-byte `stc` physically writes past its footprint, so
   *       the bytes above it are damaged whether or not anyone reads them;
   *   (2) a FALSE STORE-TO-LOAD FORWARD -- memory is untouched, and only a load issued in
   *       the store's shadow with a partially-matching address is answered with the store's
   *       data. The counters are read-modify-written every iteration, so a bad forward is
   *       written back and becomes indistinguishable from real corruption after the fact.
   * The shift sweep cannot separate them because in it the damaged slot is ALSO the loaded
   * slot. This stage separates them by putting something in the damage window that is never
   * loaded while the store is in flight.
   *
   * Layout is why this is possible at zero cost. At SHIFT=0 the frame is (verified in the
   * artifact, fdreg_compute at -O0, frame 0x50, s0 = sp+0x50):
   *
   *     sp+0x00..0x0f   z, the capability -- `stc a1, 0x0(a0)`, 576 times
   *     sp+0x10..0x1b   DEAD SPACE, nothing allocated  <-- exactly the damage window
   *     sp+0x1c         k, the inner counter (12 bytes above the store: the CLEAN boundary)
   *     sp+0x20/24/28   p / qc / s
   *
   * That hole is why SHIFT=0 returns a correct 576: the overrun, if it exists, lands in
   * unallocated bytes. `wit[3]` fills the hole and nothing else moves -- wit[1] lands at
   * sp+0x14, the very slot that returns 567 when the counter sits there (SHIFT=8). So the
   * counters stay at their clean offsets, the loop must still return 576, and the only new
   * question is whether the witnesses survive.
   *
   *     qc == 576 and witnesses intact   -> memory is NOT corrupted -> (1) is REFUTED
   *     witnesses damaged                -> the store writes beyond 16 bytes -> (1)
   *
   * No new named local beyond `wit` -- `i` and `s` already have slots and are reused for the
   * check -- because any extra local re-shifts the frame and cures the patient, which is
   * what killed stages 10 and 11. VERIFY in the disassembly that k is still at s0-0x34 and
   * the frame is still 0x50 before believing any result from this stage.
   *
   * FDREG_WITSEL: 0 -> damage code (or qc when clean); 1/2/3 -> the RAW word of wit[0/1/2],
   * so the corrupting value can be read out. All four builds declare identical locals, so
   * they share one frame layout and differ only after the loop.
   */
  {
    unsigned qc = 0;
    int p, k;
    /* At -O0 clang packs the scalars DOWNWARD from s0 and then drops the 16-byte-aligned
       capability at the very bottom of the frame, so whatever is left over becomes a HOLE
       immediately above the store -- which is precisely the damage window, and precisely why
       an unpadded build is clean. A first attempt without this pad put wit at sp+0x1c, 12
       bytes up, i.e. in the clean regime, measuring nothing. This consumes the hole so wit
       drops onto sp+0x10/0x14/0x18: the three offsets the shift sweep proved corrupting.
       Declared BETWEEN the counters and wit so only wit moves. Tune against the artifact --
       the required size is a function of the frame's rounding, not a constant. */
#if (FDREG_WITPAD) > 0
    volatile unsigned char witpad[FDREG_WITPAD];
    witpad[0] = 0;
#endif
    volatile unsigned wit[3];
    wit[0] = 0xA5A50000u;
    wit[1] = 0xA5A50001u;
    wit[2] = 0xA5A50002u;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* cap field, inner resetting counter */
        (void)z;
        qc++;
      }
#if (FDREG_WITSEL) == 0
    s = 0;
    for (i = 0; i < 3u; i++)
      if (wit[i] != 0xA5A50000u + i) { s = 0xBAD00000u | (i << 16) | (qc & 0xFFFFu); break; }
    if (s) return s;
    return qc;                                   /* 576 = loop correct AND memory intact */
#else
    (void)qc; (void)i; (void)s;
    return wit[(FDREG_WITSEL) - 1];              /* raw, to name the corrupting value */
#endif
  }
#endif
#if FDREG_STAGE == 12
  /* STAGE 12 -- PEEK at the counter's neighbours WITHOUT adding storage.
     Stages 10/11 laid a sentinel array beside the counters and the fault VANISHED: the
     shift-8 build, which returns 567 without the array, returned a correct 576 with it. Of
     course it did -- 32 bytes of witnesses re-shift the frame, and the frame offset IS the
     variable. Any instrument that adds a local destroys the thing it is measuring.
     This adds NO storage. It takes the address of the accumulator, which at -O0 is already in
     memory, and reads a neighbouring word after the loop. FDREG_PEEK selects which word,
     relative to &qc, so a sweep maps what surrounds the slot:
        return value = that word, raw
     An address-like result (the 0x8000000 range is domain memory) would name a store; a small
     integer would mean one of our own values is landing in the wrong slot. */
  {
    unsigned qc = 0;
    int p, k;
    volatile unsigned *base = &qc;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
        qc++;
      }
    return base[(FDREG_PEEK)];
  }
#endif
#if FDREG_STAGE == 10 || FDREG_STAGE == 11
  /* STAGES 10 and 11 -- IDENTIFY the corrupting value instead of inferring it.
     Boot 46 moved the loop counters within the frame and got four different wrong answers from
     source QEMU computes as 576 every time: shift 0 -> 576, +4 -> 909, +8 -> 567, +12 ->
     134218295 = 0x8000237, which is +8's 567 with BIT 27 set. So a stack slot is being
     corrupted and the corruption depends on the slot's offset. What is NOT known is which
     value lands there.
     These two lay a sentinel-filled witness array beside the counters, run the nest unchanged,
     and read the array back:
        stage 10 -> 1000*(number of words corrupted) + (index of the first), so 0 means the
                    witnesses survived and the damage is confined to the counters themselves
        stage 11 -> the raw 32-bit VALUE of the first corrupted word
     If that value is address-like (the 0x8000000 range is domain memory) it names a store; if
     it is a small integer it is one of our own counters landing in the wrong slot. Split into
     two probes because the marker is one word and the value needs all 32 bits. */
  {
    volatile unsigned w[8];
    unsigned qc = 0, nbad = 0, first = 0, firstv = 0;
    int p, k;
    for (i = 0; i < 8; i++) w[i] = 0xA5A5A5A5u;
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;
        (void)z;
        qc++;
      }
    for (i = 0; i < 8; i++)
      if (w[i] != 0xA5A5A5A5u) {
        if (nbad == 0) { first = i; firstv = w[i]; }
        nbad++;
      }
#if FDREG_STAGE == 10
    if (nbad == 0) return qc;              /* witnesses clean -> report the loop result */
    return 1000u * nbad + first;
#else
    (void)qc; (void)nbad;
    return firstv;                          /* the raw corrupting value */
#endif
  }
#endif
#if FDREG_STAGE == 9
  /* STAGE 9 -- IS THE INITIALISATION ITSELF LOST?
     Boot 41 swept the outer trip count and the wrong answer tracked the expected one with a
     CONSTANT offset:
         outer 64 -> expected 576, board 906    (+330)
         outer 32 -> expected 288, board 618    (+330)
         outer 16 -> expected 144, board 474    (+330)
     A fixed surplus independent of the trip count is not an iteration miscount at all: the
     loop counts correctly and the accumulator STARTS at 330 rather than 0. That is a stale
     stack slot, and it is directly testable without any loop.
     This declares the accumulator exactly as stage 7 does and returns it IMMEDIATELY.
         0   -> the initialisation lands; the surplus comes from somewhere else
         330 -> `unsigned qc = 0` did not take effect and the slot holds prior data
     Everything else -- the leaves, the guard, the frame -- is kept identical so the slot
     lands at the same offset.

     RESULT, boot 44: init9 returns 0 -- the initialisation LANDS, so the candidate below is
     REFUTED, in agreement with an RTL audit that independently refuted x0-forwarding (the
     forward mux is closed for register 0 by `gpr_clobber_vld[0] = '0` in
     issue_read_operands.sv:577, and the FPGA regfile read port hard-zeroes address 0 at
     ariane_regfile_fpga.sv:164 with ZERO_REG_ZERO(1) on all four instantiations). The +330
     enters AFTER initialisation: the loop returns 906 where 576 is correct, i.e. the inner
     body runs ~330 times too many. Kept below because the reasoning is what the measurement
     had to rule out.

     THE CANDIDATE, from static analysis. `unsigned qc = 0` does NOT compile to a store of the
     zero register at -O0. It compiles to a CAPABILITY MOVE from x0, and that result is what
     gets stored:

         3049c: 5b 15 00 14   movc a0, zero
         304a0: 23 a0 a5 00   sw   a0, 0x0(a1)

     Counted across these rungs: `movc rd, zero` appears 2-4 times per fdreg_compute and
     `sw/sd zero` appears ZERO times. So every zero-initialisation here goes through MOVC from
     x0, and if that instruction yields a non-zero value the accumulator starts at it while
     every later increment stays correct -- precisely a constant offset independent of the
     trip count, which is what boot 41 measured.

     NOT YET A MECHANISM: the same `a0` is stored to a second slot four instructions later, and
     if that were the outer counter the loop could not run at all -- yet it runs its full
     576/288/144. Either that slot is re-initialised in the loop preamble (likely at -O0, not
     verified) or the two stores do not share the faulty value. Read that out of the
     disassembly before believing this.

     NOTE this is the opposite half of C-14 from the one that was fixed: C-14 is MOVC
     destroying its SOURCE, this would be MOVC producing the wrong DESTINATION from x0. An
     earlier audit cleared `movc rd, zero` on the source side only -- ariane_regfile_ff.sv
     forces mem[0] to 0 every cycle so the destructive write cannot take -- which says nothing
     about what rd receives. And the C-14 fix is provably INERT at -O0, which is what these
     rungs and the entire SQLite domain build at. */
  {
    unsigned qc = 0;
    FDREG_PAD_SUM()
    if (s == 0xFFFFFFFFu) return 0;
    return qc;
  }
#endif
#if FDREG_STAGE == 7
  /* STAGE 7 -- the OFF-SQLITE reproduction of the four-way conjunction.
   *
   * Every earlier fdreg stage returned correctly, and the reason is now known and is not that
   * the rung is too small: EVERY loop in this file is FLAT. Max nesting is 1. The conjunction
   * bisected inside the SQLite domain over boots 27-30 needs FOUR things at once --
   *     a NESTED loop, a CAPABILITY access in the inner body, an index that is the INNER
   *     counter, and that index RESETTING each outer pass
   * -- and condition 1 was never present here, so conditions 2-4 could never combine.
   *
   * This stage supplies all four, in 13 KB with twelve globals, against the 1.5 MB SQLite
   * image the effect has only ever been seen in. If it reproduces, the divergence has a
   * standalone testcase small enough to SIMULATE, which is the step everything is now waiting
   * on -- board bisection cannot answer what differs on an outer pass that skips its inner
   * body, and S01 has an open request for exactly this kind of waveform.
   *
   * Oracle 576 either way: a shortfall is the finding, and it is a returning number rather
   * than a hang, so it stays bisectable.
   */
  {
#if (FDREG_SHIFT) > 0
    volatile unsigned char fdreg_shift_pad[FDREG_SHIFT];
    fdreg_shift_pad[0] = 0;
#endif
    unsigned qc = 0;
    int p, k;
    /* Reads the padding globals FIRST so FDREG_PAD=1 actually reaches the cap table. Without
       this the early return leaves them unreferenced, the linker strips them, and the padded
       build silently has max gp index 9 -- i.e. it measures the SAME thing as the unpadded
       one. The guard keeps `s` live without altering the result: with PAD=1 it is 160, with
       PAD=0 it is 0, and neither is 0xFFFFFFFF. */
    FDREG_PAD_SUM()
#if (FDREG_GUARD)
    if (s == 0xFFFFFFFFu) return 0;
#endif
    for (p = 0; p < (FDREG_OUTER); p++)
      for (k = 0; k < FDREG_N; k++) {
        const char *volatile z = fdreg_defs[k].zName;   /* cap field, inner resetting counter */
        (void)z;
        qc++;
      }
    return qc;
  }
#endif


  /* Reads the padding globals so their cap-table slots stay live. No-op when FDREG_PAD=0. */
  FDREG_PAD_SUM()

  /* STAGE 1 -- read every name through its capability. */
  for (i = 0; i < FDREG_N; i++) {
    const char *z = fdreg_defs[i].zName;
    unsigned n = fdreg_len30(z);
    s += n * 8u + (unsigned)(unsigned char)z[0] + (unsigned)fdreg_defs[i].nArg;
  }

#if FDREG_STAGE >= 2
  /* STAGE 2 -- InsertBuiltinFuncs: store a DERIVED capability (&arr[i]) into a global.
     STAGE 4 does the identical work, but through a pointer PARAMETER in a noinline
     callee -- the one difference the board narrowed the SQLite wedge to. */
#if FDREG_STAGE == 4
  fdreg_link_via_param(fdreg_defs, FDREG_N);
#elif FDREG_STAGE == 5
  fdreg_link_via_global(FDREG_N);
#elif FDREG_STAGE == 6
  fdreg_link_via_param_nonleaf(fdreg_defs, FDREG_N);
#else
  for (i = 0; i < FDREG_N; i++) {
    const char *z = fdreg_defs[i].zName;
    unsigned h = fdreg_hash((unsigned char)z[0], fdreg_len30(z));
    fdreg_defs[i].pNext = fdreg_buckets[h];
    fdreg_buckets[h] = &fdreg_defs[i];
  }
#endif
  /* Walk the chains back so a broken link shows up as a wrong number, not a silent pass. */
  for (i = 0; i < FDREG_HASHN; i++) {
    FdregDef *p = fdreg_buckets[i];
    unsigned guard = 0;
    while (p && guard < 64u) { s += 17u; p = p->pNext; guard++; }
  }
#endif

/* `== 3`, not `>= 3`: stage 4 is stage 2 done through a parameter, NOT stage 3 plus
   something. Its oracle is therefore stage 2's (2609), which is also the point -- a wrong
   value would say the linking ran but produced different chains, where a hang says only
   "somewhere after the last marker". */
#if FDREG_STAGE == 3
  /* STAGE 3 -- indirect call through a capability loaded out of the global aggregate. */
  for (i = 0; i < FDREG_HASHN; i++) {
    FdregDef *p = fdreg_buckets[i];
    unsigned guard = 0;
    while (p && guard < 64u) { s += (unsigned)p->xFunc(); p = p->pNext; guard++; }
  }
#endif

  return s + fdreg_gate - 1u;
}
#endif
