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

#define FDREG_N     9
#define FDREG_HASHN 23

static volatile unsigned fdreg_gate = 1u;  /* satisfies the ldc gp[i] build gate */

#if FDREG_PAD > 0
/* One global OBJECT per cap-table slot -- an array would take a single slot and pad
   nothing. Volatile so the reads survive -O1 and the slots stay live. */
#define FDREG_P8(b)  FDREG_P1(b##0) FDREG_P1(b##1) FDREG_P1(b##2) FDREG_P1(b##3) \
                     FDREG_P1(b##4) FDREG_P1(b##5) FDREG_P1(b##6) FDREG_P1(b##7)
#define FDREG_P1(i)  static volatile unsigned fdreg_pad_##i = 1u;
FDREG_P8(0) FDREG_P8(1) FDREG_P8(2) FDREG_P8(3) FDREG_P8(4)
FDREG_P8(5) FDREG_P8(6) FDREG_P8(7) FDREG_P8(8) FDREG_P8(9)
FDREG_P8(a) FDREG_P8(b) FDREG_P8(c) FDREG_P8(d) FDREG_P8(e)
FDREG_P8(f) FDREG_P8(g) FDREG_P8(h) FDREG_P8(i) FDREG_P8(j)
#undef FDREG_P1
#define FDREG_P1(i)  s += fdreg_pad_##i;
#define FDREG_PAD_SUM() \
  FDREG_P8(0) FDREG_P8(1) FDREG_P8(2) FDREG_P8(3) FDREG_P8(4) \
  FDREG_P8(5) FDREG_P8(6) FDREG_P8(7) FDREG_P8(8) FDREG_P8(9) \
  FDREG_P8(a) FDREG_P8(b) FDREG_P8(c) FDREG_P8(d) FDREG_P8(e) \
  FDREG_P8(f) FDREG_P8(g) FDREG_P8(h) FDREG_P8(i) FDREG_P8(j)
#else
#define FDREG_PAD_SUM() do {} while (0);
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

static unsigned fdreg_compute(void) {
  unsigned i, s = 0;

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
