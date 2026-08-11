#include "sqlite3.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U
#define STR__(x) #x
#define STR_(x) STR__(x)
/* Overridable so the silicon build can shrink it. Under -capstone-gp-captable every
   global's storage is CARVED FROM dom_data, and dom_data is what is left of a
   power-of-two page allocation after the image -- so this array is charged directly
   against the domain's stack budget rather than costing only image space.
   At 1 MiB the silicon build does not fit: storage 1,120,992 against dom_data
   706,128, i.e. a stack of -510,576 (domdata-budget.py). It is the single largest
   line in that budget by an order of magnitude. */
#ifndef SQLITE_HEAP_SIZE
#define SQLITE_HEAP_SIZE (1024U * 1024U)
#endif

static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));
static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

#define CAPSTONE_DELIN(value)                                                \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(value))

static void output_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
#ifndef CAPSTONE_GP_CAPTABLE_ABI
  /* Under the gp-captable (silicon) ABI this delin is BOTH redundant AND FATAL, so it
     is compiled out. `text` points into a cap-table storage capability, and those are
     produced by `split` from `sp` -- which the entry glue already delin'd -- so they
     arrive NONLIN. The RTL's DELIN accepts CAP_TYPE_LINEAR only and raises
     UNEXPECTED_CAP_TYPE otherwise; our QEMU helper_csdelin returns early instead, which
     is why this never showed up under emulation. Same root cause as C-13 in the entry
     glue. Since `text` is already non-linear, dropping the delin is a semantic no-op.
     Kept for the non-gp-captable builds (the QEMU pure-cap row domains), where the
     string capability can still be linear.
     Compile-time rather than a runtime cap-type test simply because it is free: the ABI
     is known at build time. (An earlier version of this comment claimed a runtime test
     would be non-portable because `lcc zimm=1` returns cap_type on QEMU and cap_type-1
     on the RTL. That was WRONG: the RTL enum has NOT_CAP=0 and so is offset by one from
     QEMU's, where LIN=0, and the -1 is exactly that conversion. LINEAR(1)-1 == LIN(0).
     A runtime test via lcc zimm=1 IS portable.) */
  CAPSTONE_DELIN(text);
#endif
  char *payload = (char *)hostcall_payload;
  /* WAS "this one stays in every build", on the reasoning that `payload` is the HOST
     capability loaded fresh out of its global each call, "not a cap-table storage cap --
     it is still linear here". That reasoning is WRONG on the gp-captable ABI, and it cost
     S-02: under -capstone-gp-captable `hostcall_payload` IS reached through the cap-table,
     so it arrives NONLIN exactly like `text`, and DELIN on a non-linear capability raises
     UNEXPECTED_CAP_TYPE on the RTL -- which wedges rather than traps (R-5). QEMU's
     helper_csdelin returns early, which is why SQLite ran green under emulation and died
     on silicon.
     MEASURED 2026-08-09 on caplifive_65536_r18_fix.bit, control green in both boots:
       FS1 (return at fail() entry, before any output_text)  RETURNED in 4 s
       FS2 (return after the first output_text)              WEDGED
     and the artifact showed output_text @ 0x13addc carrying delin x1 -- this one, since
     the `text` delin above is compiled out on this ABI.
     Guarded identically to `text`. Where the capability really is linear (the QEMU
     pure-cap row domains) the delin is still emitted and still needed; where it is already
     non-linear, dropping it is a semantic no-op. Same root cause as C-13. */
#ifndef CAPSTONE_GP_CAPTABLE_ABI
  CAPSTONE_DELIN(payload);
#endif
  unsigned long offset = hostcall_metadata->length;
  while (*text && offset + 1 < SQLITE_HC_REGION_SIZE)
    payload[offset++] = *text++;
  hostcall_metadata->length = offset;
}

static void output_uint(unsigned value) {
  char digits[16];
  unsigned count = 0;
  do {
    digits[count++] = (char)('0' + value % 10U);
    value /= 10U;
  } while (value != 0);
  while (count != 0) {
    char one[2] = {digits[--count], '\0'};
    output_text(one);
  }
}

#ifdef CAPSTONE_REDRAW_PAD
/* REDRAW control for S-01 ("a ~1.6 MB domain returns, but ANY perturbation of its image makes
 * it hang"). A dead, never-called function that changes the IMAGE and nothing else -- the same
 * perturbation S-01 was characterised with, so a build carrying it is a fresh draw of an
 * otherwise byte-for-byte equivalent program.
 *
 * It exists so that "build X wedges" can be separated from "build X's CHANGE wedges". Without a
 * redraw arm, any wedge in a rebuilt domain is unattributable: S-01 says an inert perturbation
 * is enough on its own. `used` keeps --gc-sections from deleting it, which would make the
 * control silently identical to the baseline and the comparison void.
 *
 * Vary the value to get another draw. Never enable it in a measured build. */
__attribute__((used, noinline)) static int capstone_redraw_pad(void) {
  volatile int x = (int)(CAPSTONE_REDRAW_PAD);
  return x;
}
#endif

static int fail(const char *stage, int rc, sqlite3 *db) {
  output_text("SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned)rc);
  if (db) {
    output_text(" message=");
    output_text(sqlite3_errmsg(db));
  }
  output_text("\n");
  return rc ? rc : 1;
}

/* Run a statement with no result rows; fail() on error. */
static int exec_ok(sqlite3 *db, const char *sql, const char *stage) {
  char *errmsg = 0;
  int rc = sqlite3_exec(db, sql, 0, 0, &errmsg);
  if (errmsg)
    sqlite3_free(errmsg);
  if (rc != SQLITE_OK)
    return fail(stage, rc, db);
  return SQLITE_OK;
}

/* Prepare + step a query expected to return exactly one row, one integer
 * column; check it equals `want`. Exercises prepare/step/column + finalize. */
static int query_scalar_eq(sqlite3 *db, const char *sql, long want,
                           const char *stage) {
  sqlite3_stmt *st = 0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, 0);
  if (rc != SQLITE_OK)
    return fail(stage, rc, db);
  rc = sqlite3_step(st);
  if (rc != SQLITE_ROW) {
    sqlite3_finalize(st);
    return fail(stage, rc ? rc : SQLITE_MISMATCH, db);
  }
  long got = (long)sqlite3_column_int64(st, 0);
  sqlite3_finalize(st);
  if (got != want)
    return fail(stage, SQLITE_MISMATCH, db);
  return SQLITE_OK;
}

/* A richer workload: exercises transactions, a second table, a REAL column, an
 * INTEGER PRIMARY KEY (rowid btree) + a secondary INDEX (index btree cursors),
 * bound prepared inserts, UPDATE/DELETE, aggregates + the sorter, ORDER BY, a
 * JOIN, GROUP BY, and string functions (sqlite3UpperToLower). Each step is
 * validated; the first failure returns non-zero and the PASSED marker is skipped. */
static int run_sqlite_extended(sqlite3 *db) {
  int rc;

  if ((rc = exec_ok(db, "BEGIN;", "ext-begin")) != SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db,
          "CREATE TABLE nums(id INTEGER PRIMARY KEY, label TEXT, amount REAL);",
          "ext-create-nums")) != SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db, "CREATE INDEX idx_amount ON nums(amount);",
                    "ext-create-index")) != SQLITE_OK)
    return rc;

  /* Bound prepared INSERTs: int id, text label, real amount. */
  {
    sqlite3_stmt *ins = 0;
    rc = sqlite3_prepare_v2(
        db, "INSERT INTO nums(id,label,amount) VALUES(?1,?2,?3);", -1, &ins, 0);
    if (rc != SQLITE_OK)
      return fail("ext-prep-insert", rc, db);
    static const char *labels[4] = {"one", "two", "three", "four"};
    for (int i = 0; i < 4; ++i) {
      sqlite3_reset(ins);
      sqlite3_bind_int(ins, 1, i + 1);
      /* SQLITE_STATIC (not SQLITE_TRANSIENT): these labels are static storage.
       * The build patches SQLITE_TRANSIENT to a function only inside the
       * amalgamation, so the public sqlite3.h value (-1) is not recognized as
       * the transient sentinel by the patched core -- a client passing
       * SQLITE_TRANSIENT would have -1 stored as a destructor and later called.
       * Tracked as a known limitation; SQLITE_STATIC is the correct binding for
       * persistent buffers regardless. */
      sqlite3_bind_text(ins, 2, labels[i], -1, SQLITE_STATIC);
      sqlite3_bind_double(ins, 3, (double)((i + 1) * 10));
      rc = sqlite3_step(ins);
      if (rc != SQLITE_DONE) {
        sqlite3_finalize(ins);
        return fail("ext-bind-insert", rc, db);
      }
    }
    sqlite3_finalize(ins);
  }

  /* Mutations. */
  if ((rc = exec_ok(db, "UPDATE nums SET amount = amount * 2 WHERE id = 2;",
                    "ext-update")) != SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db, "DELETE FROM nums WHERE id = 4;", "ext-delete")) !=
      SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db, "COMMIT;", "ext-commit")) != SQLITE_OK)
    return rc;

  /* Aggregates over the remaining rows: id 1,2,3 with amounts 10,40,30.
   * count=3, sum=80, max=40, min=10. */
  if ((rc = query_scalar_eq(db, "SELECT COUNT(*) FROM nums;", 3,
                            "ext-count")) != SQLITE_OK)
    return rc;
  if ((rc = query_scalar_eq(db, "SELECT CAST(SUM(amount) AS INTEGER) FROM nums;",
                            80, "ext-sum")) != SQLITE_OK)
    return rc;
  if ((rc = query_scalar_eq(db, "SELECT CAST(MAX(amount) AS INTEGER) FROM nums;",
                            40, "ext-max")) != SQLITE_OK)
    return rc;

  /* ORDER BY (sorter): top amount label should be "two" (40). */
  {
    sqlite3_stmt *st = 0;
    rc = sqlite3_prepare_v2(
        db, "SELECT label FROM nums ORDER BY amount DESC;", -1, &st, 0);
    if (rc != SQLITE_OK)
      return fail("ext-prep-order", rc, db);
    rc = sqlite3_step(st);
    if (rc != SQLITE_ROW) {
      sqlite3_finalize(st);
      return fail("ext-order-step", rc, db);
    }
    const unsigned char *top = sqlite3_column_text(st, 0);
    int ok = top && sqlite3_stricmp((const char *)top, "two") == 0;
    sqlite3_finalize(st);
    if (!ok)
      return fail("ext-order-value", SQLITE_MISMATCH, db);
  }

  /* Index-driven lookup (WHERE on the indexed column), comparison ops. */
  if ((rc = query_scalar_eq(
           db, "SELECT id FROM nums WHERE amount > 35 AND amount < 45;", 2,
           "ext-index-where")) != SQLITE_OK)
    return rc;

  /* JOIN nums with the base items table (both TEXT keys). */
  if ((rc = exec_ok(db,
          "CREATE TABLE link(label TEXT, name TEXT);", "ext-create-link")) !=
      SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db,
          "INSERT INTO link VALUES('one','alpha'),('two','beta');",
          "ext-insert-link")) != SQLITE_OK)
    return rc;
  if ((rc = query_scalar_eq(
           db,
           "SELECT COUNT(*) FROM nums JOIN link ON nums.label = link.label;", 2,
           "ext-join")) != SQLITE_OK)
    return rc;

  /* GROUP BY / HAVING: two parity groups over id (1,2,3) -> odd{1,3}, even{2}. */
  if ((rc = query_scalar_eq(
           db,
           "SELECT COUNT(*) FROM (SELECT id%2 AS p FROM nums GROUP BY id%2);", 2,
           "ext-group")) != SQLITE_OK)
    return rc;

  /* String functions: exercises sqlite3UpperToLower / substr / length. */
  if ((rc = query_scalar_eq(db, "SELECT length(upper(label)) FROM nums WHERE id=3;",
                            5, "ext-strfunc")) != SQLITE_OK) /* "THREE" -> 5 */
    return rc;

  output_text("__CAPSTONE_SQLITE_EXTENDED_PASSED__\n");
  return SQLITE_OK;
}

#ifdef CAPSTONE_SQLITE_STAGE
/* The first four steps of run_sqlite(), each individually stoppable. Deliberately a
   separate function rather than #ifdefs sprinkled through run_sqlite(): the real path must
   stay byte-identical, or a bisection result says nothing about the build that matters.
     stage 0 -- return at once. Proves entry + return works and the shared region is
                writable; everything else depends on this and it has never been shown.
     stage 1 -- after sqlite3_config(SQLITE_CONFIG_HEAP, ...). First touch of sqlite_heap,
                the 256 KB global whose carve needed the granule fix.
     stage 2 -- after sqlite3_initialize().
     stage 3 -- after sqlite3_open(":memory:"). First real allocation traffic. */
#if CAPSTONE_SQLITE_STAGE == 63
/* FOUR file-scope arrays with identical contents, to test "the Nth cap-init'd array is
   broken while the first is fine" directly. Walks element 1 of each and returns a BITMAP:
   bit k set = array k's lit[1] overran. 0 = all four fine. One domain, four answers.
   (Distinct contents per array would change their cap-init leaves; identical contents keep
   everything constant except ORDER of initialisation, which is the variable under test.) */
#define CAPSTONE_LITSET { "ltrim", "rtrim", "trim", "max", "min", "typeof", "length", \
  "instr", "substr", "upper", "lower", "coalesce", "hex", "unhex", "quote", "replace" }
static const char *const capstone_lit_a[16] = CAPSTONE_LITSET;
static const char *const capstone_lit_b[16] = CAPSTONE_LITSET;
static const char *const capstone_lit_c[16] = CAPSTONE_LITSET;
static const char *const capstone_lit_d[16] = CAPSTONE_LITSET;
#endif
#if CAPSTONE_SQLITE_STAGE >= 60 && CAPSTONE_SQLITE_STAGE <= 87
/* ONE array, at FILE SCOPE, shared by stages 60-87 -- the confound remover.
   Every earlier staged block declared its OWN local `lit`, so stage 52 read the second
   cap-init'd array, stage 54 the third and stage 59 the fourth: different objects at
   different addresses, initialised by different blocks of __capstone_cap_init. The observed
   split (52 = lit[1] never terminates, 59 = lit[1] walks fine and returns 5) therefore does
   NOT isolate the access pattern -- it is equally consistent with "the Nth cap-init'd array
   is broken and the Mth is fine". Those two explanations demand completely different fixes.
   60/61/62 read THIS array and differ ONLY in how they touch it. */
static const char *const capstone_probe_lit[16] = {
  "ltrim", "rtrim", "trim", "max", "min", "typeof", "length", "instr",
  "substr", "upper", "lower", "coalesce", "hex", "unhex", "quote", "replace" };
#endif

static int run_sqlite_staged(int stage) {
  sqlite3 *db = 0;
  int rc;
  /* Stages 4-6 split what stage 2 (sqlite3_initialize) does, in dependency order. Board
     result 2026-07-31: stage 0 and stage 1 RETURN rc=0, stage 2 WEDGES -- so entry/return
     and sqlite3_config(HEAP) are fine on silicon and the fault is inside initialize().
     The first thing initialize() does that stage 1 did not is WRITE into the 256 KB heap
     (memsys5Init builds its zone headers there); stage 1 only recorded the pointer. These
     stages separate "the heap capability cannot be written across its range" from
     "something else in initialize()". They are numbered above the normal ladder so the
     ordinary stages keep their meanings. */
  if (stage == 4) {
    /* Bounds probe: touch first, middle and last byte and read each back. Returns a bitmap
       of which offsets did NOT survive, so a partial failure is distinguishable from a
       total one -- an incorrectly rounded carve would fail the LAST byte and pass the
       first, which is exactly the granule bug's signature. */
    volatile unsigned char *h = sqlite_heap;
    unsigned bad = 0;
    h[0] = 0xA5; if (h[0] != 0xA5) bad |= 1u;
    h[sizeof(sqlite_heap) / 2] = 0x5A;
    if (h[sizeof(sqlite_heap) / 2] != 0x5A) bad |= 2u;
    h[sizeof(sqlite_heap) - 1] = 0x3C;
    if (h[sizeof(sqlite_heap) - 1] != 0x3C) bad |= 4u;
    return (int)bad;
  }
  if (stage == 5) {
    /* Whole-range write. If the bounds probe passes but this wedges, the failure depends on
       the ACCESS COUNT or on crossing some interior boundary, not on the endpoints. */
    volatile unsigned char *h = sqlite_heap;
    for (unsigned long i = 0; i < sizeof(sqlite_heap); i++)
      h[i] = (unsigned char)i;
    return 0;
  }
  if (stage == 6)
    return sqlite3_os_init();   /* our VFS registration, no heap traffic */
  /* Stages 7-10 split sqlite3_initialize() at ITS OWN internal boundaries. Board result
     2026-07-31: stages 4/5/6 all returned rc=0, so the heap is writable across its whole
     256 KB range (endpoints included -- the carve-granule fix holds on hardware) and VFS
     registration works, yet initialize() still wedges. So the fault is in one of the
     remaining steps, not in raw heap access.
     Callable because sqlite_capstone_domain.c is #included into the amalgamation TU, so
     SQLITE_PRIVATE functions are in scope. Each stage re-does the CONFIG_HEAP call first
     because memsys5 is the allocator these steps depend on. */
  if (stage == 11) {
    /* S-04: config + initialize both return SQLITE_OK on silicon, yet sqlite3_open() returns
       SQLITE_NOMEM. That says the allocator is installed but hands out nothing -- so ask it
       directly rather than inferring from open()'s rc. The same build PASSES under QEMU, so
       the question is specifically what memsys5 sees on hardware.

       The staged marker carries only 8 bits (0x5A6E_ssrr), so pack:
         bit 0     sqlite3_malloc(64)    returned NULL
         bit 1     sqlite3_malloc(4096)  returned NULL
         bit 2     sqlite3_malloc(65536) returned NULL
         bit 3     sqlite3_memory_used() still 0 after those calls
         bits 4-7  sqlite3_memory_used() in 16 KiB units, saturating at 15
       0x00 means every allocation worked and was accounted for; 0x0F means nothing
       allocates at all. */
    void *a64, *a4k, *a64k;
    sqlite3_int64 used;
    unsigned r = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK)
      return rc;
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK)
      return rc;
    a64  = sqlite3_malloc(64);
    a4k  = sqlite3_malloc(4096);
    a64k = sqlite3_malloc(65536);
    used = sqlite3_memory_used();
    if (a64  == 0) r |= 1u;
    if (a4k  == 0) r |= 2u;
    if (a64k == 0) r |= 4u;
    if (used == 0) r |= 8u;
    r |= (unsigned)((used >> 14) > 15 ? 15 : (used >> 14)) << 4;
    return (int)r;
  }
  if (stage == 12) {
    /* S-04 narrowing. Stage 11 showed malloc(64/4096/65536) all SUCCEED on silicon, so the
       NOMEM from sqlite3_open is not raw allocation failure. Distinguish "the sqlite3 handle
       itself could not be allocated" from "the handle allocated and something later set the
       OOM flag": openDatabase returns a NULL db only in the former case.
         bit 0     rc != SQLITE_OK
         bit 1     db == NULL   -> the very first allocation in openDatabase failed
         bit 2     sqlite3_malloc(120000) failed  (the default lookaside is 1200*100)
         bits 4-7  rc & 0xF     -> 7 = SQLITE_NOMEM */
    sqlite3 *db12 = 0;
    void *big;
    unsigned r = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK)
      return rc;
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK)
      return rc;
    big = sqlite3_malloc(120000);
    if (big == 0) r |= 4u;
    else sqlite3_free(big);
    rc = sqlite3_open(":memory:", &db12);
    if (rc != SQLITE_OK) r |= 1u;
    if (db12 == 0)       r |= 2u;
    r |= (unsigned)(rc & 0xF) << 4;
    return (int)r;
  }
  if (stage == 13) {
    /* S-04, final narrowing. Stage 12: sqlite3_malloc(120000) SUCCEEDS while sqlite3_open
       returns NOMEM with db == NULL -- i.e. openDatabase's own
       sqlite3MallocZero(sizeof(sqlite3)) failed. A ~700-byte allocation failing while a
       120 KB one succeeds is the contradiction. Call BOTH allocators at exactly that size.
       sqlite3MallocZero is SQLITE_PRIVATE but in scope: this file is #included into the
       amalgamation TU.
         bit 0     sqlite3_malloc(sizeof(sqlite3))      == NULL
         bit 1     sqlite3MallocZero(sizeof(sqlite3))   == NULL
         bit 2     sqlite3_malloc(700)                  == NULL   (size-only control)
         bit 3     the memset inside MallocZero did not stick (first/last byte non-zero)
         bits 4-7  sizeof(sqlite3) in 128-byte units, saturating at 15 */
    void *p1, *p2, *p3;
    unsigned r = 0;
    unsigned long n = (unsigned long)sizeof(sqlite3);
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK)
      return rc;
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK)
      return rc;
    p1 = sqlite3_malloc((int)n);
    if (p1 == 0) r |= 1u; else sqlite3_free(p1);
    p3 = sqlite3_malloc(700);
    if (p3 == 0) r |= 4u; else sqlite3_free(p3);
    p2 = sqlite3MallocZero((u64)n);
    if (p2 == 0) {
      r |= 2u;
    } else {
      volatile unsigned char *q = (volatile unsigned char *)p2;
      if (q[0] != 0 || q[n - 1] != 0) r |= 8u;
      sqlite3_free(p2);
    }
    r |= (unsigned)((n >> 7) > 15 ? 15 : (n >> 7)) << 4;
    return (int)r;
  }
  if (stage == 14) {
    /* S-04, the decisive one. Stage 13 showed every allocation openDatabase needs SUCCEEDS
       when called directly, so openDatabase is failing BEFORE it allocates. Its only early
       return is its own sqlite3_initialize() (WRONG -- SQLITE_OMIT_AUTOINIT=1 IS defined,
       build-sqlite-capstone.sh:167, so openDatabase does NOT call initialize at all; this
       stage's premise was invalid, though its measurement stands), and `*ppDb = 0` is set just above it -- which yields exactly the observed
       db == NULL with rc = 7.

       A second sqlite3_initialize() should short-circuit on sqlite3GlobalConfig.isInit. If
       that global write does not persist on silicon, the second call redoes everything and
       can fail. So: call it twice and also read isInit back.
         bit 0     first  sqlite3_initialize() != SQLITE_OK
         bit 1     second sqlite3_initialize() != SQLITE_OK
         bit 2     sqlite3GlobalConfig.isInit == 0 after the FIRST call
         bit 3     sqlite3GlobalConfig.isInit == 0 after the SECOND call
         bits 4-7  the second call's rc & 0xF */
    int rc1, rc2;
    unsigned r = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK)
      return rc;
    rc1 = sqlite3_initialize();
    if (rc1 != SQLITE_OK) r |= 1u;
    if (sqlite3GlobalConfig.isInit == 0) r |= 4u;
    rc2 = sqlite3_initialize();
    if (rc2 != SQLITE_OK) r |= 2u;
    if (sqlite3GlobalConfig.isInit == 0) r |= 8u;
    r |= (unsigned)(rc2 & 0xF) << 4;
    return (int)r;
  }
  if (stage == 15) {
    /* S-04. Everything openDatabase needs allocates fine when called directly (stage 13), and
       its internal sqlite3_initialize() succeeds twice with isInit persisting (stage 14), so
       the NOMEM arises somewhere INSIDE openDatabase. Note db == NULL does NOT localise it:
       opendb_out does `if( (rc&0xff)==SQLITE_NOMEM ){ sqlite3_close(db); db = 0; }`, so the
       handle is nulled for a NOMEM raised anywhere.

       The one allocation openDatabase makes that nothing above tested in its real form is the
       per-connection LOOKASIDE buffer (default 1200 x 100). Disable it and retry: this is both
       a discriminator and, if it works, a usable workaround.
         bits 0-3  rc from sqlite3_open with lookaside DISABLED
         bit 4     db == NULL
         bit 5     the SQLITE_CONFIG_LOOKASIDE call itself failed */
    sqlite3 *db15 = 0;
    unsigned r = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK)
      return rc;
    rc = sqlite3_config(SQLITE_CONFIG_LOOKASIDE, 0, 0);
    if (rc != SQLITE_OK) r |= 32u;
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK)
      return rc;
    rc = sqlite3_open(":memory:", &db15);
    r |= (unsigned)(rc & 0xF);
    if (db15 == 0) r |= 16u;
    return (int)r;
  }
#if CAPSTONE_SQLITE_STAGE == 160
  if (stage == 160) {
    /* S-04: WHICH STEP of openDatabase raises the OOM flag?
       Established on silicon: malloc succeeds at 64/700/4096/65536/120000;
       sqlite3MallocZero(sizeof(sqlite3)) succeeds and its memset sticks; disabling
       lookaside changes nothing. And db == NULL localises nothing, because opendb_out
       does `if( (rc&0xff)==SQLITE_NOMEM ){ sqlite3_close(db); db = 0; }` for a NOMEM
       raised ANYWHERE. So re-walk openDatabase on our OWN handle and read
       db->mallocFailed and db->errCode after EACH step.
         low nibble = first step that tripped (0 = none)
         0x10 mallocFailed   0x20 errCode==NOMEM   0x40 errCode other   0x80 rc was NOMEM
         0xEr / 0xDr = the CONFIG_HEAP / initialize preamble itself failed
       0x00 is NOT a pass: it means the sequence is clean on our handle while the real
       sqlite3_open still fails, i.e. the difference is in the compiled FORM of
       openDatabase, not in the sequence -- a different finding, needing an A/B on
       codegen rather than a further split. */
    sqlite3 *d;
    unsigned int oflags = (unsigned int)(SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    char *zOpen160 = 0, *zErr160 = 0;
    unsigned f160;
    int i160;

#define S160_F(dd)                                                            \
    ( ((dd)->mallocFailed ? 0x10u : 0u)                                       \
    | ((((dd)->errCode & 0xff) == SQLITE_NOMEM) ? 0x20u : 0u)                 \
    | ((((dd)->errCode != 0) &&                                               \
        (((dd)->errCode & 0xff) != SQLITE_NOMEM)) ? 0x40u : 0u) )
#define S160_CHECK(idx)                                                       \
    do { f160 = S160_F(d); if (f160) return (int)(f160 | (unsigned)(idx)); } while (0)

    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));

    d = sqlite3MallocZero((u64)sizeof(sqlite3));      /* step 1 */
    if (d == 0) return (int)0x81u;

    d->errMask = 0xff;                                /* step 2 */
    d->nDb = 2;
    d->eOpenState = SQLITE_STATE_BUSY;
    d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1;
    d->lookaside.sz = 0;
    d->nFpDigit = 17;
    memcpy(d->aLimit, aHardLimit, sizeof(d->aLimit));
    d->aLimit[SQLITE_LIMIT_WORKER_THREADS] = SQLITE_DEFAULT_WORKER_THREADS;
    d->autoCommit = 1;
    d->nextAutovac = -1;
    d->szMmap = sqlite3GlobalConfig.szMmap;
    d->nextPagesize = 0;
    d->init.azInit = sqlite3StdType;
    d->flags |= SQLITE_ShortColNames | SQLITE_EnableTrigger | SQLITE_EnableView
              | SQLITE_CacheSpill | SQLITE_AttachCreate | SQLITE_AttachWrite
              | SQLITE_Comments | SQLITE_TrustedSchema | SQLITE_AutoIndex;
    S160_CHECK(2);   /* no error channel here: a hit means a STRAY STORE */

    sqlite3HashInit(&d->aCollSeq);                    /* step 3 */
    sqlite3HashInit(&d->aModule);
    S160_CHECK(3);

    (void)createCollation(d, sqlite3StrBINARY, SQLITE_UTF8,    0, binCollFunc, 0);
    S160_CHECK(4);
    (void)createCollation(d, sqlite3StrBINARY, SQLITE_UTF16BE, 0, binCollFunc, 0);
    S160_CHECK(5);
    (void)createCollation(d, sqlite3StrBINARY, SQLITE_UTF16LE, 0, binCollFunc, 0);
    S160_CHECK(6);
    (void)createCollation(d, "NOCASE", SQLITE_UTF8, 0, nocaseCollatingFunc, 0);
    S160_CHECK(7);
    (void)createCollation(d, "RTRIM",  SQLITE_UTF8, 0, rtrimCollFunc, 0);
    S160_CHECK(8);

    d->openFlags = oflags;                            /* step 9 */
    rc = sqlite3ParseUri(0, ":memory:", &oflags, &d->pVfs, &zOpen160, &zErr160);
    if (rc != SQLITE_OK) {
      if (rc == SQLITE_NOMEM) sqlite3OomFault(d);
      sqlite3_free(zErr160);
      f160 = S160_F(d) | ((rc == SQLITE_NOMEM) ? 0x80u : 0u);
      return (int)((f160 ? f160 : 0x40u) | 9u);
    }
    d->openFlags = oflags;
    S160_CHECK(9);

    rc = sqlite3BtreeOpen(d->pVfs, zOpen160, d, &d->aDb[0].pBt, 0,   /* step 10 */
                          (int)(oflags | SQLITE_OPEN_MAIN_DB));
    if (rc != SQLITE_OK) {
      sqlite3Error(d, rc);
      f160 = S160_F(d) | (((rc & 0xff) == SQLITE_NOMEM) ? 0x80u : 0u);
      return (int)((f160 ? f160 : 0x40u) | 10u);
    }
    S160_CHECK(10);

    d->aDb[0].pSchema = sqlite3SchemaGet(d, d->aDb[0].pBt);          /* step 11 */
    if (!d->mallocFailed && d->aDb[0].pSchema != 0)
      sqlite3SetTextEncoding(d, SCHEMA_ENC(d));
    S160_CHECK(11);

    d->aDb[1].pSchema = sqlite3SchemaGet(d, 0);                      /* step 12 */
    d->aDb[0].zDbSName = "main";
    d->aDb[1].zDbSName = "temp";
    d->eOpenState = SQLITE_STATE_OPEN;
    S160_CHECK(12);

    sqlite3Error(d, SQLITE_OK);                                      /* step 13 */
    sqlite3RegisterPerConnectionBuiltinFunctions(d);
    S160_CHECK(13);

    rc = SQLITE_OK;                                                  /* step 14 */
    for (i160 = 0; rc == SQLITE_OK && i160 < ArraySize(sqlite3BuiltinExtensions); i160++)
      rc = sqlite3BuiltinExtensions[i160](d);
    if (rc) sqlite3Error(d, rc);
    S160_CHECK(14);

    if ((sqlite3_errcode(d) & 0xff) == SQLITE_NOMEM) return (int)0x20u;
    return 0;
#undef S160_CHECK
#undef S160_F
  }
#endif /* CAPSTONE_SQLITE_STAGE == 160 */

  if (stage == 161) {
    /* S-04, after stage 160 localised the fault to step 5, createCollation(BINARY, UTF16BE),
       with mallocFailed set and no allocation having failed.
       sqlite3HashInsert (38908-38937) returns non-zero in EXACTLY two cases:
         (1) elem->data non-zero -- the key WAS found -> returns old_data  (!= pColl)
         (2) sqlite3Malloc(sizeof(HashElem)) failed    -> returns data     (== pColl)
       findCollSeqEntry calls sqlite3OomFault for either, and the assert that would have
       separated them is compiled out under NDEBUG. Separate them here, and check the key
       string itself, since sqlite3StrBINARY is a GLOBAL reached through the gp cap-table.
         bit 0  sqlite3HashFind("BINARY") returns 0 AFTER the UTF8 collation was inserted
         bit 1  the collation hash is empty (count == 0)
         bit 2  sqlite3Strlen30(sqlite3StrBINARY) != 6
         bit 3  sqlite3StrBINARY[0] != 'B'
         bits 4-7  hash element count, saturating at 15 */
    sqlite3 *d;
    const void *found;
    unsigned r = 0;
    int n161;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff;
    d->nDb = 2;
    d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1;
    d->lookaside.sz = 0;
    sqlite3HashInit(&d->aCollSeq);
    sqlite3HashInit(&d->aModule);
    /* exactly step 4: the FIRST insert of "BINARY" */
    (void)createCollation(d, sqlite3StrBINARY, SQLITE_UTF8, 0, binCollFunc, 0);
    /* now ask the table for the same key, which is what step 5 does first */
    found = sqlite3HashFind(&d->aCollSeq, sqlite3StrBINARY);
    if (found == 0) r |= 1u;
    n161 = (int)sqliteHashCount(&d->aCollSeq);
    if (n161 == 0) r |= 2u;
    if (sqlite3Strlen30(sqlite3StrBINARY) != 6) r |= 4u;
    if (sqlite3StrBINARY[0] != 0x42) r |= 8u;      /* 'B' */
    r |= (unsigned)(n161 > 15 ? 15 : n161) << 4;
    return (int)r;
  }
  if (stage == 162) {
    /* S-04 final split. Stage 161: after inserting "BINARY", the collation hash holds 1
       element, the key string is intact (len 6, first byte 'B'), and sqlite3HashFind STILL
       returns 0. findElementWithHash (38840-38847) matches only if BOTH hold:
           h == elem->h   AND   sqlite3StrICmp(elem->pKey, pKey) == 0
       Both walk a string byte by byte, which is where this platform has a documented -O0
       silicon defect. Split them.
         bit 0  strHash(sqlite3StrBINARY) != elem->h          (hash mismatch)
         bit 1  sqlite3StrICmp(elem->pKey, sqlite3StrBINARY) != 0  (compare mismatch)
         bit 2  two consecutive strHash() calls disagree      (nondeterministic)
         bit 3  memcmp(elem->pKey, sqlite3StrBINARY, 7) != 0  (the stored COPY is wrong)
         bits 4-7  low nibble of (strHash ^ elem->h) */
    sqlite3 *d;
    HashElem *e162;
    unsigned h1, h2, r = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;
    sqlite3HashInit(&d->aCollSeq);
    sqlite3HashInit(&d->aModule);
    (void)createCollation(d, sqlite3StrBINARY, SQLITE_UTF8, 0, binCollFunc, 0);
    e162 = sqliteHashFirst(&d->aCollSeq);
    if (e162 == 0) return (int)0xC0u;
    h1 = strHash(sqlite3StrBINARY);
    h2 = strHash(sqlite3StrBINARY);
    if (h1 != e162->h) r |= 1u;
    if (sqlite3StrICmp(e162->pKey, sqlite3StrBINARY) != 0) r |= 2u;
    if (h1 != h2) r |= 4u;
    if (memcmp(e162->pKey, sqlite3StrBINARY, 7) != 0) r |= 8u;
    r |= (unsigned)((h1 ^ e162->h) & 0xFu) << 4;
    return (int)r;
  }
  if (stage == 163) {
    /* S-04 root cause pinpoint. Stage 162 showed the key COPY stored in the collation hash
       differs byte-wise from "BINARY", while strHash is deterministic -- so the hash is fine
       and the DATA is wrong: memcpy(pColl[0].zName, zName, nName) at 132493 produced the
       wrong bytes. Report WHICH bytes, so the corruption has a shape.
         bits 0-6  byte i of the stored key differs from "BINARY\0"  (i = 0..6)
         bit  7    the stored key is entirely zero */
    sqlite3 *d;
    HashElem *e163;
    const unsigned char *k;
    static const unsigned char want[7] = { 0x42,0x49,0x4E,0x41,0x52,0x59,0x00 }; /* BINARY\0 */
    unsigned r = 0, zeros = 0, i;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;
    sqlite3HashInit(&d->aCollSeq);
    sqlite3HashInit(&d->aModule);
    (void)createCollation(d, sqlite3StrBINARY, SQLITE_UTF8, 0, binCollFunc, 0);
    e163 = sqliteHashFirst(&d->aCollSeq);
    if (e163 == 0) return (int)0xC0u;
    k = (const unsigned char *)e163->pKey;
    for (i = 0; i < 7; i++) {
      if (k[i] != want[i]) r |= (1u << i);
      if (k[i] == 0) zeros++;
    }
    if (zeros == 7) r |= 0x80u;
    return (int)r;
  }
  if (stage == 164) {
    /* S-04: is it memcpy, or the destination pointer? Stage 163 showed the key stored by
       findCollSeqEntry is entirely zero while the SOURCE is intact. Replicate its allocation
       and try three different ways of writing the same 7 bytes into the same kind of block.
         bit 0  pc[0].zName does not read back as &pc[3]   -> the DESTINATION POINTER is wrong
         bit 1  an explicit byte LOOP does not stick        -> memory, not memcpy
         bit 2  memcpy does not stick                       -> reproduces stage 163
         bit 3  a single direct store q[0]=0x42 does not read back
         bits 4-7  how many of the 7 bytes the byte LOOP got right (0..7)
       memcpy failing while the loop succeeds implicates memcpy; both failing implicates the
       memory or the capability the destination is reached through. */
    sqlite3 *d;
    CollSeq *pc, *pd;
    char *q;
    unsigned r = 0, i, n = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;

    /* --- exactly findCollSeqEntry's allocation and memcpy (132484-132493) --- */
    pc = (CollSeq *)sqlite3DbMallocZero(d, 3*sizeof(CollSeq) + 7);
    if (pc == 0) return (int)0x82u;
    pc[0].zName = (char *)&pc[3];
    if ((const void *)pc[0].zName != (const void *)&pc[3]) r |= 1u;
    memcpy(pc[0].zName, sqlite3StrBINARY, 7);
    if (memcmp(pc[0].zName, sqlite3StrBINARY, 7) != 0) r |= 4u;

    /* --- the same bytes, written by an explicit loop into a fresh block --- */
    pd = (CollSeq *)sqlite3DbMallocZero(d, 3*sizeof(CollSeq) + 7);
    if (pd == 0) return (int)0x83u;
    q = (char *)&pd[3];
    for (i = 0; i < 7; i++) q[i] = sqlite3StrBINARY[i];
    for (i = 0; i < 7; i++) if (q[i] == sqlite3StrBINARY[i]) n++;
    if (n != 7) r |= 2u;
    r |= (n & 0xFu) << 4;

    /* --- one single store, read straight back --- */
    q[0] = 0x42;
    if (q[0] != 0x42) r |= 8u;
    return (int)r;
  }
  if (stage == 165) {
    /* S-04: memcpy loses a 7-byte copy that an explicit byte loop performs correctly to the
       SAME address (stage 164). memcpy branches on (dest & 15) == (src & 15) and has three
       paths, so report the two alignments -- that says which path this case takes, and the
       board's heap sits at a different address from QEMU's, which is a candidate for why
       QEMU passes.
         bits 0-3  dest & 15   (dest = &pc[3], the destination findCollSeqEntry uses)
         bits 4-7  src  & 15   (src  = sqlite3StrBINARY) */
    sqlite3 *d;
    CollSeq *pc;
    unsigned long dv, sv;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;
    pc = (CollSeq *)sqlite3DbMallocZero(d, 3*sizeof(CollSeq) + 7);
    if (pc == 0) return (int)0x82u;
    __asm__ volatile("mv %0, %1" : "=r"(dv) : "r"((char *)&pc[3]));
    __asm__ volatile("mv %0, %1" : "=r"(sv) : "r"(sqlite3StrBINARY));
    return (int)(((unsigned)(dv & 0xF)) | (((unsigned)(sv & 0xF)) << 4));
  }
  if (stage == 166) {
    /* S-04: does memcpy(n=7) wrongly take its SIXTEEN-BYTE capability block path?
       Both alignments are 0 (stage 165), so memcpy skips the head loop; n=7 < 16 should then
       skip the block loop too and copy 7 bytes in the tail byte loop. If instead the block
       loop runs, it does `ldc a5,0(src); stc a5,0(dest)` -- copying a plain string AS A
       CAPABILITY -- which would explain a destination that ends up all zero.
       Poison bytes 7..15 of the destination and see whether memcpy disturbs them.
         bit 0  bytes 0..6 are NOT the copied key      (the corruption itself)
         bit 1  ANY of the poisoned bytes 7..15 changed -> the 16-byte path RAN
         bit 2  byte 0 of the destination is zero
         bits 4-7  how many of the 9 poison bytes changed, saturating at 15 */
    sqlite3 *d;
    unsigned char *buf;
    unsigned r = 0, i, changed = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;
    /* a 16-byte-aligned block with room past the 7 bytes, so overrun is observable */
    buf = (unsigned char *)sqlite3DbMallocZero(d, 64);
    if (buf == 0) return (int)0x82u;
    for (i = 7; i < 16; i++) buf[i] = 0xA5;          /* poison */
    memcpy(buf, sqlite3StrBINARY, 7);
    if (memcmp(buf, sqlite3StrBINARY, 7) != 0) r |= 1u;
    for (i = 7; i < 16; i++) if (buf[i] != 0xA5) changed++;
    if (changed) r |= 2u;
    if (buf[0] == 0) r |= 4u;
    r |= (unsigned)(changed > 15 ? 15 : changed) << 4;
    return (int)r;
  }
  if (stage == 167) {
    /* S-05: are the OTHER writing string primitives affected the same way memcpy was?
       S-04 was a store that did not commit, in a primitive compiled at -O1 whose destination
       capability came in as an argument and was used straight out of a0. memcpy is now built
       with optnone and clears; memmove, memset and strcpy have the SAME shape and are still
       at -O1. That matters here because the CREATE path writes btree pages through exactly
       those, so if any of them loses small aligned writes it is a candidate for the
       SQLITE_CORRUPT.
       Same allocation shape as stage 164 -- sqlite3DbMallocZero blocks, 16-byte aligned --
       so a nonzero result here is directly comparable to 164's.
         bit 0  a 7-byte memmove does not read back
         bit 1  a 7-byte memset does not read back
         bit 2  a 7-char strcpy does not read back
         bit 3  a 32-byte memmove does not read back  (exercises the capability block loop)
         bits 4-7  how many of the 7 memmove bytes are correct (0..7)
       0x70 means all four writers stick and the memmove copied all 7 -- i.e. clean, and the
       same encoding stage 164 uses for clean, deliberately so the two read alike. */
    sqlite3 *d;
    unsigned char *a, *b;
    char *cs;
    unsigned r = 0, i, n = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;

    a = (unsigned char *)sqlite3DbMallocZero(d, 128);
    if (a == 0) return (int)0x82u;
    b = (unsigned char *)sqlite3DbMallocZero(d, 128);
    if (b == 0) return (int)0x83u;

    /* 7-byte memmove between two distinct blocks, both 16-byte aligned */
    memcpy(b, sqlite3StrBINARY, 7);                  /* memcpy is known good now */
    memmove(a, b, 7);
    for (i = 0; i < 7; i++) if (a[i] == (unsigned char)sqlite3StrBINARY[i]) n++;
    if (n != 7) r |= 1u;
    r |= (n & 0xFu) << 4;

    /* 7-byte memset */
    memset(a + 16, 0x5A, 7);
    for (i = 0; i < 7; i++) if (a[16 + i] != 0x5A) { r |= 2u; break; }

    /* 7-char strcpy (writes 7 chars + NUL) */
    cs = (char *)(a + 32);
    strcpy(cs, "BINARY");
    for (i = 0; i < 7; i++) if (cs[i] != sqlite3StrBINARY[i]) { r |= 4u; break; }

    /* 32-byte memmove -- long enough to run the 16-byte capability block loop twice */
    for (i = 0; i < 32; i++) b[64 + i] = (unsigned char)(0xC0 + i);
    memmove(a + 64, b + 64, 32);
    for (i = 0; i < 32; i++) if (a[64 + i] != (unsigned char)(0xC0 + i)) { r |= 8u; break; }
    return (int)r;
  }
  if (stage == 168) {
    /* S-05: read the FULL error text from the failing CREATE.
       The normal run reports `stage=create rc=11 message=malforme` -- truncated, because
       output_text appends into the bounded hostcall region and by the time the error is
       emitted the region is nearly full. That matters: SQLite's message for this case is
       "malformed database schema (%s)" plus an optional " - %s" detail, and BOTH the object
       name and the detail are exactly what distinguishes the possible causes (a schema row
       whose text does not begin with CREATE, versus one that fails to re-parse). So do the
       minimum -- open, one CREATE -- and emit the message FIRST, while the region is empty.
       Returns the low byte of rc, so the marker still carries a verdict if the text is lost. */
    sqlite3 *db = 0;
    char *errmsg = 0;
    int crc;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK || db == 0) {
      output_text("E168 open rc=");
      output_uint((unsigned)rc);
      output_text("\n");
      return (int)(0xC0u | (unsigned)(rc & 0xF));
    }
    crc = sqlite3_exec(db, "CREATE TABLE t(a INTEGER, b TEXT)", 0, 0, &errmsg);
    output_text("E168 create rc=");
    output_uint((unsigned)crc);
    output_text(" errmsg=[");
    output_text(errmsg ? errmsg : "(null)");
    output_text("] errmsg2=[");
    output_text(sqlite3_errmsg(db));
    output_text("] ecode=");
    output_uint((unsigned)sqlite3_extended_errcode(db));
    output_text("\n");
    if (errmsg) sqlite3_free(errmsg);
    return (int)((unsigned)crc & 0xFFu);
  }
  if (stage == 169) {
    /* S-05: does a 16-byte ldc/stc block copy preserve all 128 bits of PLAIN data on silicon?
       Hypothesis, from two independent observations that both land on exactly 8 bytes:
         - the CREATE error text arrives as "malforme" -- the first 8 bytes of "malformed
           database schema (items)" -- with byte 8 apparently NUL, and it stays 8 bytes even
           when emitted into an empty region, so it is not output truncation;
         - stage 167 bit 3: a 32-byte memmove does not read back, while the 7-byte one does.
       Both are explained if `stc` of an UNTAGGED capability-sized word writes only the low
       64 bits, leaving the high half zero. memcpy's block loop copies via `void *` precisely
       to preserve tags, so every 16-byte chunk of plain data goes through that path. The
       source comment on memcpy already flags this as "gap 4", fixed in QEMU
       (fix/untagged-ldc-stc-128bit-preservation) -- which is exactly the kind of divergence
       QEMU would hide.
       This does NOT return a verdict byte; it PRINTS the bytes, because the shape of the
       damage is the finding and 8 bits cannot carry it. Expected if the hypothesis holds:
       bytes 0-7 correct, 8-15 zero, 16-23 correct, 24-31 zero. */
    sqlite3 *d;
    unsigned char *src, *dst;
    unsigned i, bad = 0;
    static const char hexd[] = "0123456789abcdef";
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;
    src = (unsigned char *)sqlite3DbMallocZero(d, 128);
    if (src == 0) return (int)0x82u;
    dst = (unsigned char *)sqlite3DbMallocZero(d, 128);
    if (dst == 0) return (int)0x83u;

    for (i = 0; i < 32; i++) src[i] = (unsigned char)(0xC0u + i);
    memcpy(dst, src, 32);                       /* runs the 16-byte block loop twice */

    output_text("E169 align dst=");
    output_uint((unsigned)(((unsigned long)dst) & 15u));
    output_text(" src=");
    output_uint((unsigned)(((unsigned long)src) & 15u));
    output_text(" src32=[");
    for (i = 0; i < 32; i++) {
      char two[3];
      two[0] = hexd[(src[i] >> 4) & 0xF]; two[1] = hexd[src[i] & 0xF]; two[2] = '\0';
      output_text(two);
    }
    output_text("] dst32=[");
    for (i = 0; i < 32; i++) {
      char two[3];
      two[0] = hexd[(dst[i] >> 4) & 0xF]; two[1] = hexd[dst[i] & 0xF]; two[2] = '\0';
      output_text(two);
    }
    output_text("]\n");
    for (i = 0; i < 32; i++) if (dst[i] != (unsigned char)(0xC0u + i)) bad++;
    return (int)(0x40u | (bad & 0x3Fu));        /* 0x40 = all 32 bytes survived */
  }
  if (stage >= 172 && stage <= 174) {
    /* S-05 successor: with the data corruption repaired, silicon wedges INSIDE CREATE TABLE with
       mcause 25 INVALID_CAPABILITY while QEMU passes the whole workload. `commit pc` on the board
       is the 0x2 junk sentinel, so the pc cannot be mapped the way it was under QEMU -- this
       bisects by WHAT IS EXECUTED instead, and every stage RETURNS a code.
       Stage 168 already covers open + a SHORT create, which previously succeeded on silicon.
         172  open + the WORKLOAD's own (longer) CREATE
         173  172 + the workload's INSERT
         174  173 + the workload's SELECT, stepped to completion
       Return: 0xA0 | (rc & 0xF) at the first failing step, or 0xB0 when the stage completes, so a
       clean run is distinguishable from rc=0 at a step that never ran. */
    sqlite3 *db = 0;
    int crc;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK || db == 0) return (int)(0xC0u | (unsigned)(rc & 0xF));

    crc = sqlite3_exec(db,
        "CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);", 0, 0, 0);
    if (crc != SQLITE_OK) return (int)(0xA0u | (unsigned)(crc & 0xFu));
    if (stage == 172) return (int)0xB0u;

    crc = sqlite3_exec(db,
        "INSERT INTO items VALUES('alpha',11),('beta',22),('gamma',33);", 0, 0, 0);
    if (crc != SQLITE_OK) return (int)(0xA0u | (unsigned)(crc & 0xFu));
    if (stage == 173) return (int)0xB1u;

    {
      sqlite3_stmt *st = 0;
      unsigned rows = 0;
      crc = sqlite3_prepare_v2(db, "SELECT name,value FROM items;", -1, &st, 0);
      if (crc != SQLITE_OK) return (int)(0xA0u | (unsigned)(crc & 0xFu));
      while (sqlite3_step(st) == SQLITE_ROW)
        rows++;
      sqlite3_finalize(st);
      return (int)(0xB2u | ((rows & 3u) << 2));   /* 0xBA = the expected 3 rows */
    }
  }
  if (stage == 170 || stage == 171) {
    /* Does the chunk copy preserve a CAPABILITY, not just plain data?
       Every S-06 probe so far (164/167/169) copies only plain bytes, so none of them can see
       the one thing BEEBS_LDC_HIGH_HALF_FIXUP changes for a chunk that holds a pointer: it
       plain-stores over the destination BEFORE the stc, and a plain store clears the line's
       capability tag. If the stc does not put the tag back, SQLite later dereferences an
       untagged pointer and the core wedges -- which is exactly what the full workload does
       with the fixup enabled while every primitive probe stays clean.
       Two stages on purpose, because the interesting failure CANNOT return:
         170  copy, then COMPARE BYTES only. Always returns, so it is safe to run first.
              A tag lives out of band, so matching bytes do NOT prove the tag survived --
              170 is here to show the DATA is right, and to separate "bytes wrong" from
              "tag wrong" if 171 dies.
         171  copy, then USE the copied pointer. Retiring a value proves the tag survived;
              a lost tag wedges instead of returning, so this one runs LAST in any boot.
       Layout: chunk 0 of the source holds a real capability, chunk 1 holds plain data, so a
       single 32-byte memcpy exercises both kinds in one call, as SQLite's struct copies do. */
    sqlite3 *d;
    unsigned char *src, *dst;
    int *target;
    unsigned i, r = 0;
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return (int)(0xE0u | (unsigned)(rc & 0xF));
    rc = sqlite3_initialize();
    if (rc != SQLITE_OK) return (int)(0xD0u | (unsigned)(rc & 0xF));
    d = sqlite3MallocZero((u64)sizeof(sqlite3));
    if (d == 0) return (int)0x81u;
    d->errMask = 0xff; d->nDb = 2; d->aDb = d->aDbStatic;
    d->lookaside.bDisable = 1; d->lookaside.sz = 0;
    src = (unsigned char *)sqlite3DbMallocZero(d, 128);
    if (src == 0) return (int)0x82u;
    dst = (unsigned char *)sqlite3DbMallocZero(d, 128);
    if (dst == 0) return (int)0x83u;
    target = (int *)sqlite3DbMallocZero(d, 64);
    if (target == 0) return (int)0x84u;
    target[0] = 0x1234;

    *(void **)src = (void *)target;                 /* chunk 0: a REAL capability */
    for (i = 16u; i < 32u; i++)
      src[i] = (unsigned char)(0xC0u + i);          /* chunk 1: plain data */

    memcpy(dst, src, 32);                           /* both chunks in one call */

    if (stage == 170) {
      for (i = 16u; i < 32u; i++)
        if (dst[i] != (unsigned char)(0xC0u + i)) { r |= 1u; break; }   /* plain half wrong */
      for (i = 0; i < 16u; i++)
        if (dst[i] != src[i]) { r |= 2u; break; }                       /* pointer BYTES wrong */
      return (int)(0x30u | r);                      /* 0x30 = both byte-identical */
    }
    /* stage 171 -- the load through dst is an ldc; using it as a base needs a live tag. */
    { int *p = *(int **)dst; r = (unsigned)(p[0] == 0x1234 ? 1u : 0u); }
    return (int)(0x50u | (r & 1u));                 /* 0x51 = tag survived and value correct */
  }
  if (stage >= 7 && stage <= 10) {
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK)
      return rc;
    if (stage == 7)
      return sqlite3MutexInit();          /* no-op at THREADSAFE=0; proves the call path  */
    if (stage == 8)
      return sqlite3MallocInit();         /* memsys5Init: builds zone headers IN the heap */
    if (stage == 9) {
      rc = sqlite3MallocInit();
      if (rc != SQLITE_OK) return rc;
      return sqlite3PcacheInitialize();
    }
    rc = sqlite3MallocInit();             /* stage 10 */
    if (rc != SQLITE_OK) return rc;
    sqlite3RegisterBuiltinFunctions();    /* writes the global function hash table        */
    return 0;
  }
  /* Stages 11-13 hunt a MINIMAL reproducer inside sqlite3RegisterBuiltinFunctions, which
     board-bisected as the wedge (stages 7/8/9 return rc=0, stage 10 wedges). That function
     builds a large LOCAL array of FuncDef structs -- each holding a zName string pointer
     and several function pointers -- and then hashes each name, which is how it reaches
     sqlite3Strlen30, the `ra` seen in every wedge dump today.
     Ruled out already: the frame is ~2 KB against 339 KB of stack; the array is built
     capability-preserving (593 stc vs 65 sd, no byte-copy template); and the 54
     auipc-formed values are function pointers, not zName -- auipc is untagged on BOTH
     QEMU (trans_rvi.c.inc trans_auipc uses gen_set_gpr) and RTL, and QEMU logged 380
     strlen arguments with ZERO untagged ones. */
  /* Stages 200-207: BISECT INSIDE sqlite3RegisterBuiltinFunctions (needs SQLITE_REG_BISECT=1).
     Stage 10 wedges first-position on a fresh boot in BOTH build shapes, while every
     hand-written analogue of it returns (11-16, 18), so the bisection must run the REAL
     function and stop part-way. `capstone_reg_limit` is injected into the amalgamation by the
     builder; K returns just after sub-step K-1, so:
        200 -> limit 1  nothing (entry only)      203 -> limit 4  + WindowFunctions
        201 -> limit 2  + AlterFunctions          204 -> limit 5  + DateTime
        202 -> limit 3  + the capstone strcmp loop 205 -> limit 6  + Json
        206 -> limit 7  + InsertBuiltinFuncs (== the whole function, i.e. stage 10)
     Every arm RETURNS its own limit, so a run that returns 203 and a run that returns nothing
     at 204 names the exact sub-step. Returning the limit (not a constant) also means a
     mis-selected arm cannot masquerade as a passing one. */
  /* Stages 210-219: bisect INSIDE sqlite3AlterFunctions, which is the sub-step that wedges
     (stage 202 = MallocInit + AlterFunctions only, wedges first-position on a fresh boot in
     BOTH build shapes). AlterFunctions is one sqlite3InsertBuiltinFuncs(aAlterTableFuncs, 9),
     so clamping that count to 0..9 names the exact entry at which it dies. Each arm returns
     210 + its own count, so a wrongly-selected arm cannot read as a passing one. */
  /* The sub-step bisection of sqlite3RegisterBuiltinFunctions is COMPILE-TIME
     (CAPSTONE_REG_LIMIT / CAPSTONE_ALTER_LIMIT, injected by build-sqlite-silicon.sh under
     SQLITE_REG_BISECT=1) -- one image per point, selected at BUILD time, run at stage 10.
     It was runtime once: that version added two globals, shifted the gp-captable from 179 to
     183 carves, and the image then ENTRY-STALLED 3/3 without executing one instruction of the
     domain. Instrumentation for this target must not touch the global set. */
  if (stage == 11)
    /* strlen on a plain literal: does the merged-container string path work at all? */
    return sqlite3Strlen30("capstone_probe_string");
  if (stage == 12) {
    /* Same string, but ROUND-TRIPPED through a local capability slot first -- that is what
       building the FuncDef array does. Separates "the string capability is bad" from "a
       capability stored into and reloaded from a local struct is bad". */
    struct { const char *z; int n; } local;
    local.z = "capstone_probe_string";
    local.n = 0;
    return sqlite3Strlen30(local.z);
  }
  if (stage == 13) {
    /* An ARRAY of them, indexed at run time so the compiler cannot fold it: closest thing
       to the real aBuiltinFunc construction without SQLite's machinery. */
    const char *names[4];
    int total = 0, i;
    names[0] = "alpha"; names[1] = "beta_two";
    names[2] = "gamma_three"; names[3] = "delta_four_x";
    for (i = 0; i < 4; i++)
      total += sqlite3Strlen30(names[i]);
    return total;               /* expect 5 + 8 + 11 + 12 = 36 */
  }
  if (stage == 14) {
    /* WHICH BYTES of a string literal survived? Board 2026-07-31: strlen of the 21-char
       "capstone_probe_string" returned 1, i.e. s[0] is non-zero and s[1] is zero. That is
       the signature of "a little was copied, the rest zero-filled", which is what the entry
       glue's carve loop does (copy `size` bytes from the blob, zero the tail).
       Returns a BITMAP of s[1..8] being non-zero, so one 8-bit return says exactly how far
       the good data extends. Expected 0xFF ("apstone_" all non-zero); 0x00 means only byte
       0 survived; a partial value gives the exact cut-off. */
    const char *s = "capstone_probe_string";
    unsigned m = 0, i;
    for (i = 0; i < 8; i++)
      if (s[i + 1])
        m |= 1u << i;
    return (int)m;
  }
  if (stage == 15) {
    /* STORE-BURST probe. sqlite3RegisterBuiltinFunctions issues 593 capability stores (stc)
       inside 1380 instructions -- ~62% store density, an order of magnitude denser than
       anything that has ever run on this bitstream (sqlite3_config 84 stc, memsys5Init 9,
       PcacheInitialize 1, os_init 2). It still wedges after the unaligned-copy fix, so
       density is the remaining thing that makes it unique.
       Writes 512 capabilities into a local array, reads them all back, returns how many
       survived (capped at 255). A clean 255 exonerates store bursts; a wedge reproduces the
       failure in ~20 lines with no SQLite at all -- which is the artifact to hand over. */
    const char *buf[512];
    unsigned i, ok = 0;
    for (i = 0; i < 512; i++)
      buf[i] = "capstone_probe_string" + (i & 15);
    for (i = 0; i < 512; i++)
      if (buf[i] && buf[i][0])
        ok++;
    return (int)(ok > 255 ? 255 : ok);
  }
  if (stage == 16) {
    /* SCALE probe: same shape as the FuncDef array -- many local structs, each holding a
       string pointer -- but a fraction of the size. Separates "this construct is broken"
       from "this construct breaks above some size". */
    struct { const char *z; int n; } a[128];
    unsigned i; int total = 0;
    for (i = 0; i < 128; i++) { a[i].z = "alpha"; a[i].n = (int)i; }
    for (i = 0; i < 128; i++) total += sqlite3Strlen30(a[i].z);
    return total & 0xff;                 /* expect 128*5 = 640 -> 0x80 */
  }
  if (stage == 17) {
    /* STACK -> GLOBAL struct assignment carrying a CAPABILITY. This is the one construct in
       sqlite3RegisterBuiltinFunctions that no probe has replicated. The patched amalgamation
       builds the FuncDef array as a LOCAL (build-sqlite-capstone.sh:75 strips `static`) and
       then copies it element-by-element into a real static:
           aBuiltinFunc[capstoneI] = capstoneBuiltinFunc[capstoneI];
       Each element carries a zName capability, so this moves capabilities from a stack
       object into cap-table storage. Everything else in that function is now accounted for:
       string data (fixed), 512 capability stores (pass), 128 local structs (pass), strlen on
       cap-table literals (pass), strcmp/strcpy linear-safety (fixed, still wedges).
       Returns how many survived the round trip; expect 8. */
    /* Named type: two separate anonymous struct declarations are DISTINCT types in C, so
       the element-wise assignment below would not compile against an anonymous pair. */
    struct capstone_kv { const char *z; int n; };
    static struct capstone_kv g[8];
    struct capstone_kv l[8];
    unsigned i; int ok = 0;
    for (i = 0; i < 8; i++) { l[i].z = "alpha"; l[i].n = (int)i; }
    for (i = 0; i < 8; i++) g[i] = l[i];
    for (i = 0; i < 8; i++)
      if (g[i].z && sqlite3Strlen30(g[i].z) == 5 && g[i].n == (int)i)
        ok++;
    return ok;
  }
  if (stage == 18 || stage == 19) {
    /* STRAIGHT-LINE init of a local array with DISTINCT capability constants -- the exact
       shape of capstoneBuiltinFunc[], and the one thing still untested.
       Board 2026-07-31: clamping the registration loops to ONE entry (BUILTIN_LIMIT=1) still
       WEDGES, and the clamp does not touch the array INITIALISER -- so the wedge happens
       while BUILDING the local array, before any copy, strcmp or hash insertion.
       stage 15 already stored 512 capabilities into a local array and passed, but it stored
       the SAME pointer in a LOOP. The real code is straight-line with ~72 DISTINCT constants,
       each materialised separately. That difference is what these two probe, at two sizes so
       a threshold shows up: 18 -> 16 entries, 19 -> 64 entries.
       Returns the number of entries that read back correctly (capped at 255). */
    struct kv { const char *z; const char *y; };
    struct kv a[64];
    unsigned n = (stage == 18) ? 16u : 64u, i;
    int ok = 0;
    /* Deliberately NOT a loop: each element gets its own distinct constants, so the
       compiler must materialise a separate capability per store, as it does for FuncDef. */
    a[0].z = "ltrim";      a[0].y = "aaa0";
    a[1].z = "rtrim";      a[1].y = "aaa1";
    a[2].z = "trim";       a[2].y = "aaa2";
    a[3].z = "max";        a[3].y = "aaa3";
    a[4].z = "min";        a[4].y = "aaa4";
    a[5].z = "typeof";     a[5].y = "aaa5";
    a[6].z = "length";     a[6].y = "aaa6";
    a[7].z = "instr";      a[7].y = "aaa7";
    a[8].z = "substr";     a[8].y = "aaa8";
    a[9].z = "upper";      a[9].y = "aaa9";
    a[10].z = "lower";     a[10].y = "aab0";
    a[11].z = "coalesce";  a[11].y = "aab1";
    a[12].z = "hex";       a[12].y = "aab2";
    a[13].z = "unhex";     a[13].y = "aab3";
    a[14].z = "quote";     a[14].y = "aab4";
    a[15].z = "replace";   a[15].y = "aab5";
    for (i = 16; i < 64; i++) { a[i].z = "filler"; a[i].y = "fill"; }
    for (i = 0; i < n; i++)
      if (a[i].z && a[i].y && sqlite3Strlen30(a[i].z) > 0 && sqlite3Strlen30(a[i].y) > 0)
        ok++;
    return ok > 255 ? 255 : ok;      /* expect 16 for stage 18, 64 for stage 19 */
  }
  /* Stages 20-22 split stage 18, which WEDGES, against stage 13, which PASSES. Stage 18
     changed FOUR things at once relative to stage 13 -- number of distinct literals (4 -> 16),
     element type (const char* -> a 2-pointer struct), array length (4 -> 64), and an extra
     filler loop -- so on its own it does not say which one matters. Each stage below changes
     exactly ONE of them back. Poor hygiene to have bundled them; these unbundle it. */
  if (stage >= 20 && stage <= 22) {
    struct kv2 { const char *z; const char *y; };
    struct kv2 a[64];
    unsigned i;
    int ok = 0;
    if (stage == 20) {
      /* stage 18 but only FOUR distinct literals. Isolates the COUNT of distinct constants. */
      a[0].z = "ltrim"; a[0].y = "aaa0";
      a[1].z = "rtrim"; a[1].y = "aaa1";
      a[2].z = "trim";  a[2].y = "aaa2";
      a[3].z = "max";   a[3].y = "aaa3";
      for (i = 4; i < 64; i++) { a[i].z = "filler"; a[i].y = "fill"; }
    } else if (stage == 21) {
      /* stage 18's SIXTEEN distinct literals, but assigned through a LOOP from a static
         table instead of straight-line. Isolates STRAIGHT-LINE materialisation. */
      static const char *const tbl[16] = {
        "ltrim", "rtrim", "trim", "max", "min", "typeof", "length", "instr",
        "substr", "upper", "lower", "coalesce", "hex", "unhex", "quote", "replace" };
      for (i = 0; i < 16; i++) { a[i].z = tbl[i]; a[i].y = "aaa0"; }
      for (i = 16; i < 64; i++) { a[i].z = "filler"; a[i].y = "fill"; }
    } else {
      /* stage 22: sixteen distinct literals straight-line into a FLAT pointer array, no
         struct. Isolates the 2-pointer STRUCT element type. */
      const char *f[64];
      f[0] = "ltrim"; f[1] = "rtrim"; f[2] = "trim";   f[3] = "max";
      f[4] = "min";   f[5] = "typeof"; f[6] = "length"; f[7] = "instr";
      f[8] = "substr"; f[9] = "upper"; f[10] = "lower"; f[11] = "coalesce";
      f[12] = "hex";  f[13] = "unhex"; f[14] = "quote"; f[15] = "replace";
      for (i = 16; i < 64; i++) f[i] = "filler";
      for (i = 0; i < 16; i++)
        if (f[i] && sqlite3Strlen30(f[i]) > 0) ok++;
      return ok;
    }
    for (i = 0; i < 16; i++)
      if (a[i].z && a[i].y && sqlite3Strlen30(a[i].z) > 0 && sqlite3Strlen30(a[i].y) > 0)
        ok++;
    return ok;                        /* expect 16 in every case */
  }
  /* Stages 30-34: CAP-INIT HOLDER SIZE. Board 2026-07-31: with SQLITE_STATIC_BUILTINS=1,
     stage 0 (which does nothing but return) WEDGES, and the cap-init-limit bisection puts the
     failure inside aBuiltinFunc's leaf range (leaves 223-381, holder_size=9216). The control
     passes at 406 leaves, so it is NOT the total store count. aBuiltinFunc is 4.3x larger
     than any other holder (9216 vs 2160 for aWindowFuncs, which works), and carries 159
     leaves against a maximum of 75 elsewhere.
     These are synthetic: one global array of N capability leaves, initialised by cap-init,
     then read back. No SQLite involved. Sizes bracket the working 2160 and the failing 9216.
     Returns how many entries survived, capped at 255. */
/* The trailing 580u is stage 34's size, NOT a safe default. Because `holder` is `static`,
   it is allocated and CAP-INITIALISED at compile time in EVERY staged build regardless of the
   runtime `if (stage >= 30 && stage <= 34)` -- so stages 0..6, which never touch the probe,
   were each carrying 580 extra capability leaves. Measured 2026-08-09: __capstone_cap_init
   went 558 -> 1257 stc (+699), and all 3 staged builds tried on the board were CREATED but
   never ENTERED, while 2/2 unstaged builds entered fine. A REDRAW at a different text pad did
   not help, which is what ruled out layout randomness. Stages outside the probe range now get
   a 1-element holder. */
#define CAPSTONE_HOLDER_N(st) (((st) < 30 || (st) > 34) ? 1u : \
                               (st) == 30 ? 40u : (st) == 31 ? 100u : \
                               (st) == 32 ? 160u : (st) == 33 ? 300u : 580u)
  if (stage >= 30 && stage <= 34) {
    /* Each element points into a distinct place so the leaves are not all identical; the
       holder is `static`, so every element is a cap-init leaf rather than a runtime store. */
    /* ONE holder per build, sized at COMPILE time. Two ways this probe has already been
       inert, both caught by gating on the cap-init store count before spending a board
       session:
         1. declared without initialisers -> zero-init .bss -> ZERO cap-init leaves, all
            five builds identical at 406 stores;
         2. declaring all five arrays in one translation unit -> every build carries every
            array (they are `static`, so selection at run time does not remove them), all
            five identical at 1757 stores.
       Only a compile-time-selected single array actually varies the holder under test. The
       range designator is a GNU extension clang accepts. */
    static const char *holder[CAPSTONE_HOLDER_N(CAPSTONE_SQLITE_STAGE)] =
        { [0 ... CAPSTONE_HOLDER_N(CAPSTONE_SQLITE_STAGE) - 1] = "h" };
    /* A switch, NOT a ternary chain. `cond ? ptrA : ptrB` lowers to an i128
       CapstoneISD::SELECT_CC, for which this backend has no RV64 pattern -- it aborts with
       "Cannot select" (documented in history/31-07-2026 ... i128-selectcc-gap). All five of
       these builds died on exactly that, having been written with a ternary chain. A switch
       lowers to branches and avoids the node entirely. */
    const char **p = holder;
    unsigned n = CAPSTONE_HOLDER_N(stage), i, ok = 0;
    /* Fill at run time only if cap-init left them null -- the point is to READ what cap-init
       (or its absence) produced, not to overwrite it. */
    for (i = 0; i < n; i++)
      if (p[i] == 0) p[i] = "x";
    for (i = 0; i < n; i++)
      if (p[i] && sqlite3Strlen30(p[i]) >= 0) ok++;
    return (int)(ok > 255 ? 255 : ok);
  }
  /* Stages 40-44: BLOB COPY vs ZERO-FILL in the entry glue.
     The R-14 workaround flips .capstone_gp_initdesc record 150 from blob_off = -1 (the
     zero-init sentinel) to blob_off = 52240, so the glue stops zero-filling a 9216-byte
     carve and starts COPYING 9216 bytes into it -- the largest blob copy in the domain, and
     it runs BEFORE cap-init. That build wedges at SHA5, mid FIRST SHARE ENTRY, before
     SQ: G/enter, i.e. in glue territory. Today's one confirmed root cause (an unaligned
     8-byte ld) lived in exactly that copy loop.
     An INITIALISED static forces a blob copy; an uninitialised one gets blob_off = -1 and is
     zero-filled. Stage 44 is the control: same 9216 bytes, zero-init, so same carve geometry
     with NO copy. If 42 wedges and 44 returns, the copy path is implicated and the carve size
     is exonerated. Sizes bracket the failing 9216 in both directions. */
#define CAPSTONE_BLOB_N(st) ((st) == 40 ? 1024 : (st) == 41 ? 4096 : \
                             (st) == 43 ? 16384 : 9216)
  if (stage >= 40 && stage <= 44) {
    /* Initialised -> real blob_off -> the glue COPIES it. */
    static const unsigned char blobdata[CAPSTONE_BLOB_N(CAPSTONE_SQLITE_STAGE)] =
        { [0 ... CAPSTONE_BLOB_N(CAPSTONE_SQLITE_STAGE) - 1] = 0x5A };
    /* Uninitialised -> blob_off = -1 -> the glue ZERO-FILLS it. Stage 44 reads this one. */
    static unsigned char zerodata[9216];
    unsigned n = CAPSTONE_BLOB_N(stage), i, bad = 0;
    if (stage == 44) {
      for (i = 0; i < sizeof(zerodata); i++)
        if (zerodata[i] != 0) bad++;
      return bad ? (int)(bad > 254 ? 254 : bad) : 255;   /* 255 == all zero, as expected */
    }
    /* Every byte must read back as 0x5A. Return the count of BAD bytes so a partial copy is
       distinguishable from a total failure; 255 means the whole blob arrived intact. */
    for (i = 0; i < n; i++)
      if (blobdata[i] != 0x5A) bad++;
    return bad ? (int)(bad > 254 ? 254 : bad) : 255;
  }
  if (stage == 50) {
    /* C-14 IN ISOLATION, two instructions. `movc rd, rs` writes cnull to rs unless rs is a
       NONLIN capability (capstone_flu_unit.anvil:19-24); a plain integer is NOT_CAP, so on
       silicon the first movc destroys `src` and the second copies zero. QEMU guards the same
       zeroing with rs1_v->tag (op_helper.c:580-584), so it returns 55 there.
       C-14 has only ever been demonstrated through a whole-loop checksum on gpw2; this tests
       the INSTRUCTION. 50 means every live-source movc in an image is a live hazard. */
    unsigned long src = 5, b = 0, c = 0;
    /* funct7 = 0x0A, verified by decoding a real `movc` emitted by the compiler:
       word 0x1401145b -> opcode 0x5b, funct3 0x1, funct7 (bits 31:25) = 0x0A. The 0x14
       visible in the top byte is funct7<<1, and writing 0x14 as the funct7 assembles a
       DIFFERENT instruction entirely -- which the first version of this probe did. */
    __asm__ volatile(".insn r 0x5b, 0x1, 0x0a, %0, %2, x0\n\t"
                     ".insn r 0x5b, 0x1, 0x0a, %1, %2, x0"
                     : "=&r"(b), "=&r"(c), "+r"(src));
    if (b == 5 && c == 5) return 55;   /* movc preserved a live scalar source */
    if (b == 5 && c == 0) return 50;   /* C-14 CONFIRMED on this bitstream     */
    return 51;                          /* neither -- unexpected              */
  }
/* GUARDED so this block's static arrays exist ONLY in its own build. `stage` is a function
   parameter and this is built -O0, so the compiler folds nothing: without the #if, EVERY
   staged block's arrays land in EVERY probe binary. That is not cosmetic -- it is the
   documented trap that already made three probes test nothing, and it silently grew wd51
   from 2 literal arrays to 4 when stages 54-59 were added, changing the glue's blob-copy
   workload for a domain whose result was being used as a control. */
#if CAPSTONE_SQLITE_STAGE == 51
  if (stage == 51) {
    /* WATCHDOG form of stage 18. Bounds the loop so a LIVELOCK RETURNS a marker naming the
       site instead of spinning forever. This is the instrument the campaign has lacked:
       every wedge so far produced silence, and silence cannot distinguish "the core stopped"
       from "the domain is still running a loop that never ends" -- especially now that
       ex_commit.valid is known to be the exception bit and stall_issue=1 is the steady state
       of a RAW-dependent loop.
         rc 0xB1 -> the strlen walk never terminated  => LIVELOCK, localised
         rc 16   -> stage 18 completes when bounded    => the wedge is a budget effect
         WEDGE   -> the core genuinely stops, and it is NOT a domain-code loop */
    struct kv3 { const char *z; const char *y; };
    struct kv3 a[64];
    unsigned i, guard;
    int ok = 0;
    a[0].z="ltrim";  a[0].y="aaa0";  a[1].z="rtrim";  a[1].y="aaa1";
    a[2].z="trim";   a[2].y="aaa2";  a[3].z="max";    a[3].y="aaa3";
    a[4].z="min";    a[4].y="aaa4";  a[5].z="typeof"; a[5].y="aaa5";
    a[6].z="length"; a[6].y="aaa6";  a[7].z="instr";  a[7].y="aaa7";
    a[8].z="substr"; a[8].y="aaa8";  a[9].z="upper";  a[9].y="aaa9";
    a[10].z="lower"; a[10].y="aab0"; a[11].z="coalesce"; a[11].y="aab1";
    a[12].z="hex";   a[12].y="aab2"; a[13].z="unhex"; a[13].y="aab3";
    a[14].z="quote"; a[14].y="aab4"; a[15].z="replace"; a[15].y="aab5";
    for (i = 16; i < 64; i++) { a[i].z = "filler"; a[i].y = "fill"; }
    for (i = 0; i < 16; i++) {
      const char *z = a[i].z;
      if (!z) continue;
      guard = 0;
      while (z[guard]) { if (++guard > (1u << 16)) return 0xB1; }  /* bounded strlen */
      if (guard > 0) ok++;
    }
    return ok;                          /* expect 16 */
  }
#endif
/* GUARDED so this block's static arrays exist ONLY in its own build. `stage` is a function
   parameter and this is built -O0, so the compiler folds nothing: without the #if, EVERY
   staged block's arrays land in EVERY probe binary. That is not cosmetic -- it is the
   documented trap that already made three probes test nothing, and it silently grew wd51
   from 2 literal arrays to 4 when stages 54-59 were added, changing the glue's blob-copy
   workload for a domain whose result was being used as a control. */
#if CAPSTONE_SQLITE_STAGE >= 52 && CAPSTONE_SQLITE_STAGE <= 53
  if (stage == 52 || stage == 53) {
    /* WHICH literal, and WHAT does it contain? Stage 51 returned 0xB1 -- a bounded strlen
       never terminated, so the domain was RUNNING, not hung. These two localise it.
         52: return 0xC0|i for the FIRST literal whose walk overruns; 16 if all terminate.
         53: byte-survival bitmap of that literal's first 8 bytes -- 0xFF means the bytes
             arrived intact and the fault is in the WALK, not the DATA; anything less is the
             data being wrong on silicon again, i.e. the unaligned-copy fix is incomplete.
       Same literal set as stage 51 so the two are directly comparable. */
    static const char *const lit[16] = {
      "ltrim", "rtrim", "trim", "max", "min", "typeof", "length", "instr",
      "substr", "upper", "lower", "coalesce", "hex", "unhex", "quote", "replace" };
    unsigned i, guard;
    if (stage == 52) {
      for (i = 0; i < 16; i++) {
        const char *z = lit[i];
        if (!z) return 0xD0u | i;              /* 0xDn = literal n is a NULL pointer */
        guard = 0;
        while (z[guard]) { if (++guard > (1u << 16)) return 0xC0u | i; }
      }
      return 16;                                /* all sixteen terminate */
    }
    /* stage 53: bytes 0..7 of lit[0] ("ltrim" -- 5 chars then NUL, so bits 0..4 set,
       bits 5..7 clear => expect 0x1F if the data is correct). */
    {
      const char *z = lit[0];
      unsigned m = 0;
      if (!z) return 0xD0u;
      for (i = 0; i < 8; i++)
        if (z[i]) m |= 1u << i;
      return (int)m;                            /* expect 0x1F for "ltrim" */
    }
  }
#endif
/* GUARDED so this block's static arrays exist ONLY in its own build. `stage` is a function
   parameter and this is built -O0, so the compiler folds nothing: without the #if, EVERY
   staged block's arrays land in EVERY probe binary. That is not cosmetic -- it is the
   documented trap that already made three probes test nothing, and it silently grew wd51
   from 2 literal arrays to 4 when stages 54-59 were added, changing the glue's blob-copy
   workload for a domain whose result was being used as a control. */
#if CAPSTONE_SQLITE_STAGE >= 54 && CAPSTONE_SQLITE_STAGE <= 56
  if (stage >= 54 && stage <= 56) {
    /* lit[0] walks fine and its bytes are correct (stage 53 = 0xDF = "ltrim" + NUL + "rt",
       which is right for a MERGED container -- my earlier 0x1F expectation wrongly assumed
       bytes 5..7 were zero). lit[1] never terminates (stage 52 = 0xC1) even though its bytes
       are visibly present as bits 6,7 of that same bitmap. So the DATA is there and the
       POINTER is the suspect.
         54: byte bitmap of lit[1] itself -- if it matches lit[0]'s shape the pointer is fine
             and the fault is in the walk; if it is 0xFF the pointer lands somewhere with no
             NUL nearby.
         55: (lit[1] - lit[0]) as a byte. For "ltrim\0" the correct delta is 6. Anything else
             means cap-init produced a wrong offset for the second leaf.
         56: same delta for (lit[2] - lit[1]); "rtrim\0" is also 6. Distinguishes "only leaf 1
             is wrong" from "every leaf after the first is wrong". */
    static const char *const lit[16] = {
      "ltrim", "rtrim", "trim", "max", "min", "typeof", "length", "instr",
      "substr", "upper", "lower", "coalesce", "hex", "unhex", "quote", "replace" };
    unsigned i;
    if (stage == 54) {
      const char *z = lit[1];
      unsigned m = 0;
      if (!z) return 0xD0u;
      for (i = 0; i < 8; i++)
        if (z[i]) m |= 1u << i;
      return (int)m;                 /* expect 0xDF, same shape as lit[0] */
    }
    {
      /* Pointer delta, low byte. Plain C subtraction, NOT hand-rolled asm: the compiler
         emits lcc/lcc/sub for it, which is exactly what strlen's epilogue does. A
         hand-written `.insn` got the lcc funct7 wrong once already (it is 0x04, with the
         zimm in the rs2 field, not 0x08). */
      const char *p, *q;
      if (stage == 55) { p = lit[0]; q = lit[1]; }
      else             { p = lit[1]; q = lit[2]; }
      return (int)(((unsigned long)(q - p)) & 0xff);   /* expect 6 for both */
    }
  }
#endif
/* GUARDED so this block's static arrays exist ONLY in its own build. `stage` is a function
   parameter and this is built -O0, so the compiler folds nothing: without the #if, EVERY
   staged block's arrays land in EVERY probe binary. That is not cosmetic -- it is the
   documented trap that already made three probes test nothing, and it silently grew wd51
   from 2 literal arrays to 4 when stages 54-59 were added, changing the glue's blob-copy
   workload for a domain whose result was being used as a control. */
#if CAPSTONE_SQLITE_STAGE >= 57 && CAPSTONE_SQLITE_STAGE <= 59
  if (stage >= 57 && stage <= 59) {
    /* DOES READING THE ARRAY CONSUME IT? capstone-ariane's documented LDC behaviour is
       "after an LDC that loads a linear capability, the source memory location is cleared
       to prevent aliasing". Stage 52 walks lit[i] by LOADING each element out of the array
       (ldc), whereas stages 53/54 name lit[0]/lit[1] directly and the compiler may keep the
       value in a register and never reload. That asymmetry fits the evidence exactly:
       lit[0] good, lit[1] bad, data provably present, pointers provably correct
       (31-07-2026_22-40-00_capinit-literal-leaves-codegen-is-correct.md).
       If the leaves are stored LINEAR, the first ldc of a slot empties that slot.
         57: read lit[1] TWICE through a volatile array pointer so neither read is CSE'd.
             bit0 = first read non-NULL, bit1 = second read non-NULL, bit2 = the two agree.
             7 = both reads fine (refutes consumption). 5 = second read came back NULL,
             i.e. the load CONSUMED the slot. 3 = both non-NULL but different.
         58: same for lit[0] -- the control. If 58 also reports consumption then this is
             uniform behaviour and stage 52 only *looked* like it singled out lit[1].
         59: read lit[1] once via the volatile pointer, then bounded-walk it; return the
             index of the first NUL (expect 5 for "rtrim"), or 0xB2 on overrun. Separates
             "the slot is consumed" from "the walk is broken" for the same element. */
    static const char *const lit[16] = {
      "ltrim", "rtrim", "trim", "max", "min", "typeof", "length", "instr",
      "substr", "upper", "lower", "coalesce", "hex", "unhex", "quote", "replace" };
    /* volatile so each subscript is a real load from the array, not a cached register. */
    const char *const volatile *vp = lit;
    unsigned idx = (stage == 58) ? 0u : 1u;
    const char *a = vp[idx];
    const char *b = vp[idx];
    if (stage != 59) {
      unsigned m = 0;
      if (a) m |= 1u;
      if (b) m |= 2u;
      if (a == b) m |= 4u;
      return (int)m;                 /* expect 7; 5 => the load consumed the slot */
    }
    {
      unsigned guard = 0;
      if (!a) return 0xD1u;          /* slot was already empty on the FIRST read */
      while (a[guard]) { if (++guard > 64u) return 0xB2u; }
      return (int)guard;             /* expect 5 for "rtrim" */
    }
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 60 && CAPSTONE_SQLITE_STAGE <= 62
  if (stage >= 60 && stage <= 62) {
    /* Same array (file-scope capstone_probe_lit), three access shapes. This is the only
       comparison so far in which the ARRAY is held constant, so whatever differs here is
       the access pattern and nothing else.
         60: stage-52 shape -- loop i=0..15, walk each element. Expect 16.
             0xC0|i on the first element that overruns.
         61: stage-54 shape -- z = lit[1] (plain, non-volatile), bitmap of bytes 0..7.
             Expect 0xDF ("rtrim\0" + the next literal's first byte).
         62: stage-59 shape -- volatile read of lit[1], then bounded walk. Expect 5.
       If 60 overruns while 62 returns 5, the access pattern is the mechanism and the array
       is exonerated. If all three agree, the earlier 52-vs-59 split was about WHICH array,
       and the fault is in cap-init's later blocks rather than in any walk. */
    unsigned i, guard;
    if (stage == 60) {
      for (i = 0; i < 16; i++) {
        const char *z = capstone_probe_lit[i];
        if (!z) return 0xD0u | i;
        guard = 0;
        while (z[guard]) { if (++guard > (1u << 16)) return 0xC0u | i; }
      }
      return 16;
    }
    if (stage == 61) {
      const char *z = capstone_probe_lit[1];
      unsigned m = 0;
      if (!z) return 0xD1u;
      for (i = 0; i < 8; i++)
        if (z[i]) m |= 1u << i;
      return (int)m;
    }
    {
      const char *const volatile *vp = capstone_probe_lit;
      const char *a = vp[1];
      guard = 0;
      if (!a) return 0xD1u;
      while (a[guard]) { if (++guard > 64u) return 0xB2u; }
      return (int)guard;
    }
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 63
  if (stage == 63) {
    /* bit k = array k's lit[1] overran. 0 => all four arrays fine. */
    const char *const *sets[4] = { capstone_lit_a, capstone_lit_b,
                                   capstone_lit_c, capstone_lit_d };
    unsigned k, guard, bad = 0;
    for (k = 0; k < 4; k++) {
      const char *z = sets[k][1];
      if (!z) { bad |= 1u << k; continue; }
      guard = 0;
      while (z[guard]) { if (++guard > 64u) { bad |= 1u << k; break; } }
    }
    return (int)bad;
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 64 && CAPSTONE_SQLITE_STAGE <= 65
  if (stage == 64 || stage == 65) {
    /* 64: walk lit[0] THEN lit[1] -- the stage-52 order, but only two elements.
       65: walk lit[1] alone, same array -- the stage-59 shape, as the control.
       Return 0x40 | (index of first NUL in lit[1]); expect 0x45 (5) in BOTH.
       If 64 returns 0xB3 and 65 returns 0x45, walking lit[0] first is what breaks lit[1]. */
    unsigned guard;
    const char *z;
    if (stage == 64) {
      z = capstone_probe_lit[0];
      if (!z) return 0xD0u;
      guard = 0;
      while (z[guard]) { if (++guard > 64u) return 0xB0u; }   /* lit[0] itself overran */
    }
    z = capstone_probe_lit[1];
    if (!z) return 0xD1u;
    guard = 0;
    while (z[guard]) { if (++guard > 64u) return 0xB3u; }
    return (int)(0x40u | (guard & 0xfu));                     /* expect 0x45 */
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 66
  if (stage == 66) {
    /* IS IT "THE FIRST WALK ONLY"? Stage 63 walked element 1 of four IDENTICAL file-scope
       arrays and returned 0x0E -- array 0 fine, arrays 1,2,3 all overran. Stage 60 walked
       lit[0] (fine) then lit[1] (overran). Stages 61/62 did a SINGLE walk and were correct.
       Every one of those is explained by "the first data-dependent walk succeeds and every
       later one fails", with nothing to do with which array or which index.
       This tests it with the confound gone entirely: walk THE SAME element twice.
         bit0 = first walk terminated, bit1 = second walk terminated.
         3 = both fine (refutes it). 1 = only the first terminated (confirms it). */
    unsigned guard, m = 0;
    const char *z = capstone_probe_lit[1];
    if (!z) return 0xD1u;
    guard = 0;
    while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 1u;
    guard = 0;
    while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 2u;
    return (int)m;                    /* expect 3; 1 => only the first walk works */
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 67 && CAPSTONE_SQLITE_STAGE <= 68
  if (stage == 67 || stage == 68) {
    /* Sharpening the ONE deterministic reproducer. Stage 66 walks lit[1] twice through the
       same pointer and is stable at rc=2 (0b10) across six samples: first walk overruns,
       second terminates. The two loops were verified byte-identical (23 insns each), so the
       code is not the difference.
         67: walk lit[1] THREE times -> 3-bit map. 0b110 (=6) means "only the very first walk
             fails" and the effect does not repeat. 0b010 (=2) would mean it alternates.
         68: walk a DIFFERENT element (lit[0]) FIRST as a sacrificial warm-up, then walk
             lit[1] twice -> bits 1,2 for the two lit[1] walks, bit 0 for the warm-up.
             If lit[1] then succeeds BOTH times (bits 1,2 set), the failure attaches to the
             first walk IN THE DOMAIN, not to the element -- which would finally explain why
             lit[1] looked uniquely guilty for days: it is simply what gets walked second. */
    unsigned guard, m = 0, k;
    const char *z1 = capstone_probe_lit[1];
    if (!z1) return 0xD1u;
    if (stage == 68) {
      const char *z0 = capstone_probe_lit[0];
      if (!z0) return 0xD0u;
      guard = 0;
      while (z0[guard]) { if (++guard > 64u) break; }
      if (guard <= 64u) m |= 1u;                 /* bit0 = warm-up walk terminated */
      for (k = 0; k < 2; k++) {
        guard = 0;
        while (z1[guard]) { if (++guard > 64u) break; }
        if (guard <= 64u) m |= 1u << (k + 1);    /* bits 1,2 = the two lit[1] walks */
      }
      return (int)m;                             /* 0b111 = 7 if the warm-up absorbs it */
    }
    for (k = 0; k < 3; k++) {
      guard = 0;
      while (z1[guard]) { if (++guard > 64u) break; }
      if (guard <= 64u) m |= 1u << k;
    }
    return (int)m;                               /* expect 0b110 = 6 if only the first fails */
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 69 && CAPSTONE_SQLITE_STAGE <= 71
  if (stage >= 69 && stage <= 71) {
    /* Narrowing wd66 -- the only stable failing reproducer (7 samples, rc=2: first walk of
       lit[1] overruns, second terminates, loops byte-identical).
       The contrast that matters: stage 61 reads z[0..7] in a COUNTED loop and is correct
       (0xDF, 2 samples); stage 66's walk is DATA-DEPENDENT (`while (z[guard])`) and its first
       pass fails. So the variable is either (a) the memory needs touching once before it
       reads correctly, or (b) the data-dependent loop shape itself is what fails.
         69: counted read of z[0..7] FIRST (the stage-61 shape, result discarded), THEN the
             walk. If the walk now terminates, a prior touch PRIMES the memory -> (a).
             returns 0x40|guard, expect 0x45; 0xB4 if it still overruns.
         70: same walk written as a COUNTED loop with the NUL test inside the body, i.e.
             identical semantics, different branch structure. If 70 terminates where 66's
             first walk does not, the DATA-DEPENDENT branch is the variable -> (b).
             returns 0x40|index-of-NUL, expect 0x45; 0xB5 if not found in 64.
         71: control -- the bare first walk alone, nothing before it. Expect 0xB6 (overrun),
             confirming the baseline in this same binary rather than across builds. */
    unsigned guard, i, sink = 0;
    const char *z = capstone_probe_lit[1];
    if (!z) return 0xD1u;
    if (stage == 69) {
      for (i = 0; i < 8; i++)            /* counted pre-touch, stage-61 shape */
        if (z[i]) sink |= 1u << i;
      guard = 0;
      while (z[guard]) { if (++guard > 64u) return 0xB4u; }
      return (int)(0x40u | (guard & 0xfu));
    }
    if (stage == 70) {
      for (i = 0; i < 64u; i++)          /* counted loop, NUL test in the body */
        if (z[i] == 0) return (int)(0x40u | (i & 0xfu));
      return 0xB5u;
    }
    guard = 0;                           /* stage 71: bare first walk, no pre-touch */
    while (z[guard]) { if (++guard > 64u) return 0xB6u; }
    return (int)(0x40u | (guard & 0xfu));
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 72 && CAPSTONE_SQLITE_STAGE <= 74
/* Pad in INSTRUCTION bytes before the paired walks, to move their address without changing
   anything else. 72 = 0 bytes, 73 = 32 bytes, 74 = 64 bytes. */
#if CAPSTONE_SQLITE_STAGE == 72
#define CAPSTONE_ADDR_PAD() do { } while (0)
#elif CAPSTONE_SQLITE_STAGE == 73
#define CAPSTONE_ADDR_PAD() __asm__ volatile(".rept 8\n\tnop\n\t.endr")
#else
#define CAPSTONE_ADDR_PAD() __asm__ volatile(".rept 16\n\tnop\n\t.endr")
#endif
  if (stage >= 72 && stage <= 74) {
    /* BOTH SHAPES IN ONE BINARY -- the confound every previous comparison had.
       wd71 (bare walk) passes 3/3 and wd66 (same walk, first of a pair) fails 7/7, but they
       are DIFFERENT binaries. Their data layout is identical (182 carves, same symbol vaddr,
       same carve bases) and their loop bodies are the same 21 instructions, so the only
       remaining differences are the loop's ADDRESS and its surrounding code.
       Here both shapes run in ONE image against ONE array:
         bit0 = bare walk terminated        (the wd71 shape)
         bit1 = first walk of the pair      (the wd66 shape)
         bit2 = second walk of the pair
       7 = everything works => the failure needs BOTH shapes in separate images, i.e. it is
           about the image, not the shape.
       5 = bare works, paired-first fails => reproduced WITHIN one binary; the shape/context
           is the variable and the address hypothesis is unnecessary.
       Stages 73/74 repeat it with 32 and 64 bytes of nops before the paired walks, moving
       their address and nothing else. If the outcome tracks the padding, it is INSTRUCTION
       PLACEMENT. */
    unsigned guard, m = 0, k;
    const char *z = capstone_probe_lit[1];
    if (!z) return 0xD1u;
    guard = 0;                                    /* --- bare walk (wd71 shape) --- */
    while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 1u;
    CAPSTONE_ADDR_PAD();
    for (k = 0; k < 2; k++) {                     /* --- paired walks (wd66 shape) --- */
      guard = 0;
      while (z[guard]) { if (++guard > 64u) break; }
      if (guard <= 64u) m |= 1u << (k + 1);
    }
    return (int)m;
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 75
  if (stage == 75) {
    /* CAN A DOMAIN WRITE mtvec? Route A (give the domain a trap handler so faults REPORT
       instead of vanishing) depends entirely on this, and guessing wrong means a silent
       wedge -- csrw from too low a privilege raises ILLEGAL INSTRUCTION, which is cause 2 and
       is EXCLUDED from the trap latch (cva6.sv:1077-1083), so it would look exactly like a
       hang and teach us nothing.
       So: probe it from inside a domain BEFORE touching the entry glue. Write a recognisable
       aligned value, read it back, restore 0, and return the low byte.
         0x40 -> the write took effect: the domain may set mtvec, Route A is viable.
         0x00 -> the CSR is there but the write was ignored/masked.
         WEDGE-> the csrw itself trapped: the domain may NOT write mtvec, Route A is dead and
                 the monitor (dom_seal[1]) is the only way. */
    unsigned long got = 0;
    __asm__ volatile("csrw mtvec, %1\n\t"
                     "csrr %0, mtvec\n\t"
                     "csrw mtvec, zero"
                     : "=r"(got) : "r"(0x40UL) : "memory");
    return (int)(got & 0xffUL);
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 76
  if (stage == 76) {
    /* POSITIVE CONTROL for Route A. The mtvec handler has never been shown to CATCH anything:
       mt71 returned normally, so .Ldomain_trap was never entered, and "mt10 still wedged with
       a handler installed" only means "not an exception" IF the handler works at all. This
       domain faults ON PURPOSE.
       capstone_probe_lit is 256 bytes; dereferencing ~1 MB past its base must raise
       OUT_OF_BOUNDS (capstone_dyn_unit.anvil:322-325), a cause the latch DOES record
       (mcause 23+5 = 28; the region-share family already shows 24 latching fine).
       Three distinguishable outcomes:
         RETURN WITH NO MARKER -> the handler caught the fault and took the normal return path.
             The runner reports "produced no SQ: obs= marker", which here is SUCCESS, not an
             error. Route A works and "not an exception" becomes an earned inference.
         rc = 0x77             -> no fault was raised at all; the bounds check did not fire and
             this control is invalid as written.
         WEDGE                 -> the handler does NOT work; the Route A conclusion collapses
             and every "still wedged with mtvec set" result is uninterpretable. */
    const char *const volatile *vp = capstone_probe_lit;
    const volatile char *base = (const volatile char *)(const void *)vp;
    const volatile char *far = base + (1024 * 1024);
    unsigned char v = (unsigned char)*far;      /* expected to FAULT here */
    return (int)(0x77u ^ (v & 0u));             /* only reached if no fault fired */
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 77
  if (stage == 77) {
    /* MEASURE THE CAPABILITY'S ACTUAL BOUNDS. Stage 76 read 1 MB past capstone_probe_lit (a
       256-byte cap-table global) and did NOT fault, which means either the storage capability
       is far wider than the object (cap-table carves over-granting) or bounds are not enforced
       on that path. This distinguishes them by reading the bounds directly.
       lcc field map, capstone_dyn_unit.anvil:182-191: zimm 3 = start, zimm 4 = end. NEITHER
       touches the rev-node channel -- only zimm 0 (validity) does, and that is the channel
       that hangs, so it must not be used here.
       Encoding is the in-tree one (start-gpfree-captable.S:34):
         .insn r 0x5b, 0x1, 0x4, rd, rs, x<zimm>
       Returns (class << 4) | min(15, floor(log2(len))):
         class 1 = len is exactly 256  -> bounds correct; stage 76's non-fault is a BOUNDS
                   ENFORCEMENT hole, which is a spatial-safety defect.
         class 4 = len >= 1 MiB        -> the capability really does cover the +1 MiB access,
                   so the carve OVER-GRANTS and enforcement is fine.
         class 3 = 256 < len < 1 MiB   -> over-grants, but not enough to explain stage 76.
         class 2 = len < 256           -> under-grants; different bug again. */
    const char *const volatile *vp = capstone_probe_lit;
    const void *base = (const void *)vp;
    unsigned long st = 0, en = 0;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(st) : "r"(base));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x4" : "=r"(en) : "r"(base));
    unsigned long len = (en > st) ? (en - st) : 0UL;
    unsigned cls;
    if (len == 256UL)                cls = 1u;
    else if (len >= (1UL << 20))     cls = 4u;
    else if (len > 256UL)            cls = 3u;
    else                             cls = 2u;
    unsigned lg = 0; while ((lg < 15u) && ((1UL << (lg + 1)) <= len)) lg++;
    return (int)((cls << 4) | lg);
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 78 && CAPSTONE_SQLITE_STAGE <= 79
  if (stage == 78 || stage == 79) {
    /* RESOLVE A CONFLICT between a board measurement and a source analysis.
       Stage 77 measured class 4 (len >= 1 MiB) for a 256-byte global, twice. A source-side
       trace instead predicts len == 256 EXACTLY, and explains stage 76's non-fault by
       capability COMPRESSION: register caps hold 64-bit compressed metadata whose bounds are
       reconstructed from the CURSOR (ariane_pkg.sv:692-693), so an offset that is a whole
       multiple of the 2^(E+14) alias period decodes to a window that has slid along with the
       pointer. Under that model the LENGTH is always 256 and only the position moves.
       Both cannot be right. Stage 77 clamped log2 at 15, so it could not show the magnitude.
         78: return floor(log2(end - start)) UNCLAMPED, plus bit7 set iff len == 256 exactly.
             8 => 256 bytes (source analysis right, stage 77's class was wrong)
             20 => 1 MiB, 24 => 16 MiB, etc.
         79: the DECIDER for the sliding-bounds model -- read start on the BASE pointer and
             start on the FAR pointer (base + 1 MiB), and return the difference in MiB.
             0  => bounds did NOT move: the capability genuinely spans the range.
             1  => start moved by exactly 1 MiB with the pointer: compression aliasing
                   confirmed, and the capability never really covered that memory. */
    const char *const volatile *vp = capstone_probe_lit;
    const void *base = (const void *)vp;
    unsigned long st = 0, en = 0;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(st) : "r"(base));
    if (stage == 78) {
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x4" : "=r"(en) : "r"(base));
      unsigned long len = (en > st) ? (en - st) : 0UL;
      unsigned lg = 0; while ((lg < 63u) && ((1UL << (lg + 1)) <= len)) lg++;
      return (int)((len == 256UL ? 0x80u : 0u) | (lg & 0x7fu));
    }
    {
      const volatile char *far = (const volatile char *)base + (1024 * 1024);
      unsigned long st_far = 0;
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(st_far) : "r"(far));
      unsigned long d = (st_far > st) ? (st_far - st) : 0UL;
      return (int)((d >> 20) & 0xffUL);      /* in MiB: 0 = did not move, 1 = slid 1 MiB */
    }
  }
#endif
#if defined(CAPINIT_PAD)
/* CAP-INIT STORE-COUNT BISECTION. sb0 (STATIC_BUILTINS at stage 0) wedges at ENTRY with 1257
   capability stores in __capstone_cap_init, while wd71 RETURNS with 1048. Stage 0 runs no
   SQLite code at all, so an entry-time wedge is attributable to cap-init and nothing else.
   This adds exactly CAPINIT_PAD extra capability leaves -- each element is an initialised
   pointer, so each becomes one `stc` in cap_init -- and otherwise returns immediately.
   Vary CAPINIT_PAD to walk the store count between the known-good 1048 and the known-bad
   1257 and find the threshold, if there is one. */
static const char *const capstone_pad[CAPINIT_PAD] = { [0 ... CAPINIT_PAD - 1] = "x" };
  if (stage == 80) {
    /* Touch one element so the array is certainly emitted and cap-init'd, then return fast. */
    return (int)(0x60u | ((unsigned long)capstone_pad[0] & 1u));
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 81
  if (stage == 81) {
    /* RAW GUARD VALUES for the wd66 reproducer. wd66 is deterministic at rc=2 across 7 samples,
       decoded as "first walk overran, second terminated" -- but that decode has NEVER been
       validated, and rc=2 is also what a clobbered accumulator would give. This returns the two
       walk results DIRECTLY instead of a bitmap:
         low nibble  = min(15, guard after walk 1)
         high nibble = min(15, guard after walk 2)
       0x55 => both walks found the NUL at index 5, i.e. wd66's bitmap decode was WRONG and
               there is no first-walk anomaly at all.
       0x5F => walk 1 ran past 15 (consistent with the overrun reading), walk 2 correct.
       anything else names exactly what each walk computed, which the bitmap cannot. */
    const char *z = capstone_probe_lit[1];
    unsigned g1 = 0, g2 = 0;
    if (!z) return 0xD1u;
    while (z[g1]) { if (++g1 > 64u) break; }
    while (z[g2]) { if (++g2 > 64u) break; }
    if (g1 > 15u) g1 = 15u;
    if (g2 > 15u) g2 = 15u;
    return (int)((g2 << 4) | g1);
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 82 && CAPSTONE_SQLITE_STAGE <= 83
  if (stage == 82 || stage == 83) {
    /* VALIDATE THE wd66 DECODE, one walk at a time. wd66 returns rc=2 deterministically and
       that has been read as "walk 1 overran, walk 2 terminated" -- but rc=2 is equally what a
       clobbered accumulator gives, and wd81 (which returned both guards at once) WEDGED.
       Split it so each domain reports ONE number and nothing else:
         82: do walk 1 ONLY, return 0x40 | min(15, guard).   Expect 0x45.
         83: do walk 1, DISCARD it, then walk 2, return 0x40 | min(15, guard2). Expect 0x45.
       82 = 0x45 and 83 = 0x45  -> both walks are fine and wd66's bitmap decode is WRONG;
                                   the first-walk anomaly never existed. RETRACT it.
       82 = 0x4F                -> walk 1 really does overrun; the wd66 reading HOLDS.
       Deliberately no accumulator, no bitmap, no second value in the same domain -- the whole
       point is to remove the encoding that has never been validated. */
    const char *z = capstone_probe_lit[1];
    unsigned g = 0;
    if (!z) return 0xD1u;
    while (z[g]) { if (++g > 64u) break; }
    if (stage == 83) {
      g = 0;
      while (z[g]) { if (++g > 64u) break; }
    }
    if (g > 15u) g = 15u;
    return (int)(0x40u | g);
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 84 && CAPSTONE_SQLITE_STAGE <= 85
  if (stage == 84 || stage == 85) {
    /* WHICH PART OF THE READ-MODIFY-WRITE IS LOST? Stages 82/83 proved both walks compute
       guard=5 correctly, yet wd66 returns 2 instead of 3 -- so the FIRST `m |= 1u` is lost
       while the second `m |= 2u` survives. These isolate that update.
         84: do walk 1, apply ONLY the first update, return 0x70 | m.
             0x71 => the update survives when it is the ONLY one; the loss needs the second
                     walk/update to follow it.
             0x70 => the first update is lost even alone; the RMW itself is broken and this is
                     a 3-line reproducer.
         85: the full wd66 sequence, re-encoded as 0x70 | m so the value cannot be confused
             with wd66's own encoding. Expect 0x73; 0x72 reproduces wd66 under a new encoding
             and rules out the encoding being the problem.
       One value per domain, deliberately -- wd81 returned two and wedged. */
    const char *z = capstone_probe_lit[1];
    unsigned guard = 0, m = 0;
    if (!z) return 0xD1u;
    while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 1u;
    if (stage == 85) {
      guard = 0;
      while (z[guard]) { if (++guard > 64u) break; }
      if (guard <= 64u) m |= 2u;
    }
    return (int)(0x70u | (m & 0xfu));
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 86
#ifndef BALLAST_NOPS
#define BALLAST_NOPS 0
#endif
  if (stage == 86) {
    /* CHARACTERISE THE BUILD-TO-BUILD VARIATION DIRECTLY.
       wd85 (this exact sequence) returns the correct m=3; wd66 (the same sequence, different
       binary) returns 2. Four such pairs are on record. So the variable is the BINARY, not any
       mechanism proposed so far. This measures the variation instead of chasing its symptoms.
       BALLAST_NOPS inserts N no-ops before the sequence: pure code-layout shift, no data, no
       extra globals, no semantic change whatsoever. Build several N and count how many of the
       resulting binaries get the WRONG answer.
         expected, every build: m = 3 -> rc = 0x73
         any build returning something else is an instance of the variation, and the RATE
         across N builds is the finding. */
    const char *z = capstone_probe_lit[1];
    unsigned guard = 0, m = 0;
    if (!z) return 0xD1u;
#if BALLAST_NOPS > 0
    __asm__ volatile(".rept " STR_(BALLAST_NOPS) "\n\tnop\n\t.endr");
#endif
    while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 1u;
    guard = 0;
    while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 2u;
    return (int)(0x70u | (m & 0xfu));
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 87
  if (stage == 87) {
    /* BRANCH vs STRAIGHT-LINE, tested directly.
       Disassembling the passing (wd85) and failing (bal0) binaries shows a 31-line diff whose
       ONLY semantic content is that wd85 reaches its second walk through a runtime branch:
           cincoffsetimm a0, s0, -0x38 ; lw a0, 0(a0) ; li a1, 0x55 ; bne a0, a1, ...
       while bal0 falls into it straight-line. The walk loops themselves are byte-identical
       (they do not appear in the diff at all). Everything else in the diff is stage constants
       and the offsets those shift.
       Stage 87 is stage 86 with that guard restored, in the same shape stage 85 used:
         returns 0x73 -> the BRANCH is what matters; straight-line fall-through into the
                         second walk is the failing construct, and this is a two-build,
                         one-conditional reproducer.
         WEDGES       -> the branch is not sufficient; wd85 passes for some other reason and
                         the diff's remaining content (constants/offsets) is where to look. */
    const char *z = capstone_probe_lit[1];
    unsigned guard = 0, m = 0;
    if (!z) return 0xD1u;
    while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 1u;
    if (stage == 87) {                    /* the restored guard -- always true at run time */
      guard = 0;
      while (z[guard]) { if (++guard > 64u) break; }
      if (guard <= 64u) m |= 2u;
    }
    return (int)(0x70u | (m & 0xfu));
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 88 && CAPSTONE_SQLITE_STAGE <= 90
  if (stage >= 88 && stage <= 90) {
    /* BISECT THE REMAINING SUSPECTS INSIDE sqlite3RegisterBuiltinFunctions.
       Stage 9 (MallocInit + PcacheInitialize) returns rc=0; stage 10 (MallocInit +
       RegisterBuiltinFunctions) wedges at position 2 in five separate boots, i.e. well below
       the ~6-run ceiling, so it is a real in-domain failure.
       BUILTIN_LIMIT was the wrong knob: it clamps only the strcmp loop and the
       sqlite3InsertBuiltinFuncs count. These three sub-registrations run REGARDLESS, and each
       builds and inserts its own function array:
           sqlite3WindowFunctions();
           sqlite3RegisterDateTimeFunctions();
           sqlite3RegisterJsonFunctions();
       So limit=0 wedging is fully consistent with the failure being in one of them rather than
       in the builtin array at all. One per stage; whichever wedges is the culprit.
       Each re-does CONFIG_HEAP + MallocInit first, exactly as stages 7-10 do, so the only
       delta from the passing stage 9 is the single call under test. */
    rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
    if (rc != SQLITE_OK) return rc;
    rc = sqlite3MallocInit();
    if (rc != SQLITE_OK) return rc;
    if (stage == 88) { sqlite3WindowFunctions();          return 0x88; }
    if (stage == 89) { sqlite3RegisterDateTimeFunctions(); return 0x89; }
    sqlite3RegisterJsonFunctions();
    return 0x90;
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 92
#ifndef PROBE_FD_N
#define PROBE_FD_N 72
#endif
  if (stage == 92) {
    /* SELF-CONTAINED REPRODUCER -- no SQLite in the path.
       Reached by elimination: stage 9 returns; stages 88/89/90 show all three sub-registrations
       return; BUILTIN_LIMIT=0 (strcmp loop skipped, zero entries inserted) still wedges at
       position 2. The only work left is the straight-line construction of the builtin FuncDef
       array on the STACK.
       NOTE: an earlier version of this probe was WRONG -- the initialiser list always held 96
       entries and PROBE_FD_N only bounded a verification loop, so every build constructed the
       same array and the "size" ladder measured nothing. The entries below are individually
       #if-guarded, so PROBE_FD_N now controls the CONSTRUCTION itself.
       Returns 0xC0 | (count & 0x3f). */
    struct probe_fd { const char *zName; void *p1; void *p2; unsigned char flags; };
    struct probe_fd arr[] = {
      { "fn0", (void *)0, (void *)0, (unsigned char)0 },
#if PROBE_FD_N > 1
      { "fn1", (void *)0, (void *)0, (unsigned char)1 },
#endif
#if PROBE_FD_N > 2
      { "fn2", (void *)0, (void *)0, (unsigned char)2 },
#endif
#if PROBE_FD_N > 3
      { "fn3", (void *)0, (void *)0, (unsigned char)3 },
#endif
#if PROBE_FD_N > 4
      { "fn4", (void *)0, (void *)0, (unsigned char)4 },
#endif
#if PROBE_FD_N > 5
      { "fn5", (void *)0, (void *)0, (unsigned char)5 },
#endif
#if PROBE_FD_N > 6
      { "fn6", (void *)0, (void *)0, (unsigned char)6 },
#endif
#if PROBE_FD_N > 7
      { "fn7", (void *)0, (void *)0, (unsigned char)7 },
#endif
#if PROBE_FD_N > 8
      { "fn8", (void *)0, (void *)0, (unsigned char)8 },
#endif
#if PROBE_FD_N > 9
      { "fn9", (void *)0, (void *)0, (unsigned char)9 },
#endif
#if PROBE_FD_N > 10
      { "fn10", (void *)0, (void *)0, (unsigned char)10 },
#endif
#if PROBE_FD_N > 11
      { "fn11", (void *)0, (void *)0, (unsigned char)11 },
#endif
#if PROBE_FD_N > 12
      { "fn12", (void *)0, (void *)0, (unsigned char)12 },
#endif
#if PROBE_FD_N > 13
      { "fn13", (void *)0, (void *)0, (unsigned char)13 },
#endif
#if PROBE_FD_N > 14
      { "fn14", (void *)0, (void *)0, (unsigned char)14 },
#endif
#if PROBE_FD_N > 15
      { "fn15", (void *)0, (void *)0, (unsigned char)15 },
#endif
#if PROBE_FD_N > 16
      { "fn16", (void *)0, (void *)0, (unsigned char)16 },
#endif
#if PROBE_FD_N > 17
      { "fn17", (void *)0, (void *)0, (unsigned char)17 },
#endif
#if PROBE_FD_N > 18
      { "fn18", (void *)0, (void *)0, (unsigned char)18 },
#endif
#if PROBE_FD_N > 19
      { "fn19", (void *)0, (void *)0, (unsigned char)19 },
#endif
#if PROBE_FD_N > 20
      { "fn20", (void *)0, (void *)0, (unsigned char)20 },
#endif
#if PROBE_FD_N > 21
      { "fn21", (void *)0, (void *)0, (unsigned char)21 },
#endif
#if PROBE_FD_N > 22
      { "fn22", (void *)0, (void *)0, (unsigned char)22 },
#endif
#if PROBE_FD_N > 23
      { "fn23", (void *)0, (void *)0, (unsigned char)23 },
#endif
#if PROBE_FD_N > 24
      { "fn24", (void *)0, (void *)0, (unsigned char)24 },
#endif
#if PROBE_FD_N > 25
      { "fn25", (void *)0, (void *)0, (unsigned char)25 },
#endif
#if PROBE_FD_N > 26
      { "fn26", (void *)0, (void *)0, (unsigned char)26 },
#endif
#if PROBE_FD_N > 27
      { "fn27", (void *)0, (void *)0, (unsigned char)27 },
#endif
#if PROBE_FD_N > 28
      { "fn28", (void *)0, (void *)0, (unsigned char)28 },
#endif
#if PROBE_FD_N > 29
      { "fn29", (void *)0, (void *)0, (unsigned char)29 },
#endif
#if PROBE_FD_N > 30
      { "fn30", (void *)0, (void *)0, (unsigned char)30 },
#endif
#if PROBE_FD_N > 31
      { "fn31", (void *)0, (void *)0, (unsigned char)31 },
#endif
#if PROBE_FD_N > 32
      { "fn32", (void *)0, (void *)0, (unsigned char)32 },
#endif
#if PROBE_FD_N > 33
      { "fn33", (void *)0, (void *)0, (unsigned char)33 },
#endif
#if PROBE_FD_N > 34
      { "fn34", (void *)0, (void *)0, (unsigned char)34 },
#endif
#if PROBE_FD_N > 35
      { "fn35", (void *)0, (void *)0, (unsigned char)35 },
#endif
#if PROBE_FD_N > 36
      { "fn36", (void *)0, (void *)0, (unsigned char)36 },
#endif
#if PROBE_FD_N > 37
      { "fn37", (void *)0, (void *)0, (unsigned char)37 },
#endif
#if PROBE_FD_N > 38
      { "fn38", (void *)0, (void *)0, (unsigned char)38 },
#endif
#if PROBE_FD_N > 39
      { "fn39", (void *)0, (void *)0, (unsigned char)39 },
#endif
#if PROBE_FD_N > 40
      { "fn40", (void *)0, (void *)0, (unsigned char)40 },
#endif
#if PROBE_FD_N > 41
      { "fn41", (void *)0, (void *)0, (unsigned char)41 },
#endif
#if PROBE_FD_N > 42
      { "fn42", (void *)0, (void *)0, (unsigned char)42 },
#endif
#if PROBE_FD_N > 43
      { "fn43", (void *)0, (void *)0, (unsigned char)43 },
#endif
#if PROBE_FD_N > 44
      { "fn44", (void *)0, (void *)0, (unsigned char)44 },
#endif
#if PROBE_FD_N > 45
      { "fn45", (void *)0, (void *)0, (unsigned char)45 },
#endif
#if PROBE_FD_N > 46
      { "fn46", (void *)0, (void *)0, (unsigned char)46 },
#endif
#if PROBE_FD_N > 47
      { "fn47", (void *)0, (void *)0, (unsigned char)47 },
#endif
#if PROBE_FD_N > 48
      { "fn48", (void *)0, (void *)0, (unsigned char)48 },
#endif
#if PROBE_FD_N > 49
      { "fn49", (void *)0, (void *)0, (unsigned char)49 },
#endif
#if PROBE_FD_N > 50
      { "fn50", (void *)0, (void *)0, (unsigned char)50 },
#endif
#if PROBE_FD_N > 51
      { "fn51", (void *)0, (void *)0, (unsigned char)51 },
#endif
#if PROBE_FD_N > 52
      { "fn52", (void *)0, (void *)0, (unsigned char)52 },
#endif
#if PROBE_FD_N > 53
      { "fn53", (void *)0, (void *)0, (unsigned char)53 },
#endif
#if PROBE_FD_N > 54
      { "fn54", (void *)0, (void *)0, (unsigned char)54 },
#endif
#if PROBE_FD_N > 55
      { "fn55", (void *)0, (void *)0, (unsigned char)55 },
#endif
#if PROBE_FD_N > 56
      { "fn56", (void *)0, (void *)0, (unsigned char)56 },
#endif
#if PROBE_FD_N > 57
      { "fn57", (void *)0, (void *)0, (unsigned char)57 },
#endif
#if PROBE_FD_N > 58
      { "fn58", (void *)0, (void *)0, (unsigned char)58 },
#endif
#if PROBE_FD_N > 59
      { "fn59", (void *)0, (void *)0, (unsigned char)59 },
#endif
#if PROBE_FD_N > 60
      { "fn60", (void *)0, (void *)0, (unsigned char)60 },
#endif
#if PROBE_FD_N > 61
      { "fn61", (void *)0, (void *)0, (unsigned char)61 },
#endif
#if PROBE_FD_N > 62
      { "fn62", (void *)0, (void *)0, (unsigned char)62 },
#endif
#if PROBE_FD_N > 63
      { "fn63", (void *)0, (void *)0, (unsigned char)63 },
#endif
#if PROBE_FD_N > 64
      { "fn64", (void *)0, (void *)0, (unsigned char)64 },
#endif
#if PROBE_FD_N > 65
      { "fn65", (void *)0, (void *)0, (unsigned char)65 },
#endif
#if PROBE_FD_N > 66
      { "fn66", (void *)0, (void *)0, (unsigned char)66 },
#endif
#if PROBE_FD_N > 67
      { "fn67", (void *)0, (void *)0, (unsigned char)67 },
#endif
#if PROBE_FD_N > 68
      { "fn68", (void *)0, (void *)0, (unsigned char)68 },
#endif
#if PROBE_FD_N > 69
      { "fn69", (void *)0, (void *)0, (unsigned char)69 },
#endif
#if PROBE_FD_N > 70
      { "fn70", (void *)0, (void *)0, (unsigned char)70 },
#endif
#if PROBE_FD_N > 71
      { "fn71", (void *)0, (void *)0, (unsigned char)71 },
#endif
#if PROBE_FD_N > 72
      { "fn72", (void *)0, (void *)0, (unsigned char)72 },
#endif
#if PROBE_FD_N > 73
      { "fn73", (void *)0, (void *)0, (unsigned char)73 },
#endif
#if PROBE_FD_N > 74
      { "fn74", (void *)0, (void *)0, (unsigned char)74 },
#endif
#if PROBE_FD_N > 75
      { "fn75", (void *)0, (void *)0, (unsigned char)75 },
#endif
#if PROBE_FD_N > 76
      { "fn76", (void *)0, (void *)0, (unsigned char)76 },
#endif
#if PROBE_FD_N > 77
      { "fn77", (void *)0, (void *)0, (unsigned char)77 },
#endif
#if PROBE_FD_N > 78
      { "fn78", (void *)0, (void *)0, (unsigned char)78 },
#endif
#if PROBE_FD_N > 79
      { "fn79", (void *)0, (void *)0, (unsigned char)79 },
#endif
#if PROBE_FD_N > 80
      { "fn80", (void *)0, (void *)0, (unsigned char)80 },
#endif
#if PROBE_FD_N > 81
      { "fn81", (void *)0, (void *)0, (unsigned char)81 },
#endif
#if PROBE_FD_N > 82
      { "fn82", (void *)0, (void *)0, (unsigned char)82 },
#endif
#if PROBE_FD_N > 83
      { "fn83", (void *)0, (void *)0, (unsigned char)83 },
#endif
#if PROBE_FD_N > 84
      { "fn84", (void *)0, (void *)0, (unsigned char)84 },
#endif
#if PROBE_FD_N > 85
      { "fn85", (void *)0, (void *)0, (unsigned char)85 },
#endif
#if PROBE_FD_N > 86
      { "fn86", (void *)0, (void *)0, (unsigned char)86 },
#endif
#if PROBE_FD_N > 87
      { "fn87", (void *)0, (void *)0, (unsigned char)87 },
#endif
#if PROBE_FD_N > 88
      { "fn88", (void *)0, (void *)0, (unsigned char)88 },
#endif
#if PROBE_FD_N > 89
      { "fn89", (void *)0, (void *)0, (unsigned char)89 },
#endif
#if PROBE_FD_N > 90
      { "fn90", (void *)0, (void *)0, (unsigned char)90 },
#endif
#if PROBE_FD_N > 91
      { "fn91", (void *)0, (void *)0, (unsigned char)91 },
#endif
#if PROBE_FD_N > 92
      { "fn92", (void *)0, (void *)0, (unsigned char)92 },
#endif
#if PROBE_FD_N > 93
      { "fn93", (void *)0, (void *)0, (unsigned char)93 },
#endif
#if PROBE_FD_N > 94
      { "fn94", (void *)0, (void *)0, (unsigned char)94 },
#endif
#if PROBE_FD_N > 95
      { "fn95", (void *)0, (void *)0, (unsigned char)95 },
#endif
    };
    unsigned i, n = (unsigned)(sizeof(arr) / sizeof(arr[0])), ok = 0;
    for (i = 0; i < n; i++)
      if (arr[i].zName && arr[i].zName[0] == 'f') ok++;
    /* Return the DEFICIT, not the count. `0xC0 | (count & 0x3f)` wrapped at 64 and made
       N=64 and N=72 indistinguishable from a miscount of 0 -- g64 and sh72 both returned
       0xC0 and could not be read. The deficit is 0 when correct and small when not, so it
       never wraps and a wrong answer is unmistakable. */
    return (int)(0xA0u | ((n - ok) & 0x1fu));
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 93
#ifndef PROBE_FD_N
#define PROBE_FD_N 72
#endif
  if (stage == 93) {
    /* BREAK THE CONFOUND. Stage 92 showed a straight-line struct array wedges above ~48-72
       entries, but four things scale together there: entry COUNT, the number of DISTINCT
       string literals, the stack frame size, and the `stc` count. This is stage 92 with the
       identical entry count and struct layout, but every entry points at ONE SHARED literal,
       so the number of distinct string constants is 1 regardless of N.
         N=72 RETURNS here but wedged in stage 92  -> DISTINCT LITERALS are the variable, not
                                                      entry count or frame size.
         N=72 WEDGES here too                      -> literals are irrelevant; it is the count,
                                                      the frame, or the store volume. */
    static const char kShared[] = "fnX";
    struct probe_fd { const char *zName; void *p1; void *p2; unsigned char flags; };
    struct probe_fd arr[] = {
      { kShared, (void *)0, (void *)0, (unsigned char)0 },
#if PROBE_FD_N > 1
      { kShared, (void *)0, (void *)0, (unsigned char)1 },
#endif
#if PROBE_FD_N > 2
      { kShared, (void *)0, (void *)0, (unsigned char)2 },
#endif
#if PROBE_FD_N > 3
      { kShared, (void *)0, (void *)0, (unsigned char)3 },
#endif
#if PROBE_FD_N > 4
      { kShared, (void *)0, (void *)0, (unsigned char)4 },
#endif
#if PROBE_FD_N > 5
      { kShared, (void *)0, (void *)0, (unsigned char)5 },
#endif
#if PROBE_FD_N > 6
      { kShared, (void *)0, (void *)0, (unsigned char)6 },
#endif
#if PROBE_FD_N > 7
      { kShared, (void *)0, (void *)0, (unsigned char)7 },
#endif
#if PROBE_FD_N > 8
      { kShared, (void *)0, (void *)0, (unsigned char)8 },
#endif
#if PROBE_FD_N > 9
      { kShared, (void *)0, (void *)0, (unsigned char)9 },
#endif
#if PROBE_FD_N > 10
      { kShared, (void *)0, (void *)0, (unsigned char)10 },
#endif
#if PROBE_FD_N > 11
      { kShared, (void *)0, (void *)0, (unsigned char)11 },
#endif
#if PROBE_FD_N > 12
      { kShared, (void *)0, (void *)0, (unsigned char)12 },
#endif
#if PROBE_FD_N > 13
      { kShared, (void *)0, (void *)0, (unsigned char)13 },
#endif
#if PROBE_FD_N > 14
      { kShared, (void *)0, (void *)0, (unsigned char)14 },
#endif
#if PROBE_FD_N > 15
      { kShared, (void *)0, (void *)0, (unsigned char)15 },
#endif
#if PROBE_FD_N > 16
      { kShared, (void *)0, (void *)0, (unsigned char)16 },
#endif
#if PROBE_FD_N > 17
      { kShared, (void *)0, (void *)0, (unsigned char)17 },
#endif
#if PROBE_FD_N > 18
      { kShared, (void *)0, (void *)0, (unsigned char)18 },
#endif
#if PROBE_FD_N > 19
      { kShared, (void *)0, (void *)0, (unsigned char)19 },
#endif
#if PROBE_FD_N > 20
      { kShared, (void *)0, (void *)0, (unsigned char)20 },
#endif
#if PROBE_FD_N > 21
      { kShared, (void *)0, (void *)0, (unsigned char)21 },
#endif
#if PROBE_FD_N > 22
      { kShared, (void *)0, (void *)0, (unsigned char)22 },
#endif
#if PROBE_FD_N > 23
      { kShared, (void *)0, (void *)0, (unsigned char)23 },
#endif
#if PROBE_FD_N > 24
      { kShared, (void *)0, (void *)0, (unsigned char)24 },
#endif
#if PROBE_FD_N > 25
      { kShared, (void *)0, (void *)0, (unsigned char)25 },
#endif
#if PROBE_FD_N > 26
      { kShared, (void *)0, (void *)0, (unsigned char)26 },
#endif
#if PROBE_FD_N > 27
      { kShared, (void *)0, (void *)0, (unsigned char)27 },
#endif
#if PROBE_FD_N > 28
      { kShared, (void *)0, (void *)0, (unsigned char)28 },
#endif
#if PROBE_FD_N > 29
      { kShared, (void *)0, (void *)0, (unsigned char)29 },
#endif
#if PROBE_FD_N > 30
      { kShared, (void *)0, (void *)0, (unsigned char)30 },
#endif
#if PROBE_FD_N > 31
      { kShared, (void *)0, (void *)0, (unsigned char)31 },
#endif
#if PROBE_FD_N > 32
      { kShared, (void *)0, (void *)0, (unsigned char)32 },
#endif
#if PROBE_FD_N > 33
      { kShared, (void *)0, (void *)0, (unsigned char)33 },
#endif
#if PROBE_FD_N > 34
      { kShared, (void *)0, (void *)0, (unsigned char)34 },
#endif
#if PROBE_FD_N > 35
      { kShared, (void *)0, (void *)0, (unsigned char)35 },
#endif
#if PROBE_FD_N > 36
      { kShared, (void *)0, (void *)0, (unsigned char)36 },
#endif
#if PROBE_FD_N > 37
      { kShared, (void *)0, (void *)0, (unsigned char)37 },
#endif
#if PROBE_FD_N > 38
      { kShared, (void *)0, (void *)0, (unsigned char)38 },
#endif
#if PROBE_FD_N > 39
      { kShared, (void *)0, (void *)0, (unsigned char)39 },
#endif
#if PROBE_FD_N > 40
      { kShared, (void *)0, (void *)0, (unsigned char)40 },
#endif
#if PROBE_FD_N > 41
      { kShared, (void *)0, (void *)0, (unsigned char)41 },
#endif
#if PROBE_FD_N > 42
      { kShared, (void *)0, (void *)0, (unsigned char)42 },
#endif
#if PROBE_FD_N > 43
      { kShared, (void *)0, (void *)0, (unsigned char)43 },
#endif
#if PROBE_FD_N > 44
      { kShared, (void *)0, (void *)0, (unsigned char)44 },
#endif
#if PROBE_FD_N > 45
      { kShared, (void *)0, (void *)0, (unsigned char)45 },
#endif
#if PROBE_FD_N > 46
      { kShared, (void *)0, (void *)0, (unsigned char)46 },
#endif
#if PROBE_FD_N > 47
      { kShared, (void *)0, (void *)0, (unsigned char)47 },
#endif
#if PROBE_FD_N > 48
      { kShared, (void *)0, (void *)0, (unsigned char)48 },
#endif
#if PROBE_FD_N > 49
      { kShared, (void *)0, (void *)0, (unsigned char)49 },
#endif
#if PROBE_FD_N > 50
      { kShared, (void *)0, (void *)0, (unsigned char)50 },
#endif
#if PROBE_FD_N > 51
      { kShared, (void *)0, (void *)0, (unsigned char)51 },
#endif
#if PROBE_FD_N > 52
      { kShared, (void *)0, (void *)0, (unsigned char)52 },
#endif
#if PROBE_FD_N > 53
      { kShared, (void *)0, (void *)0, (unsigned char)53 },
#endif
#if PROBE_FD_N > 54
      { kShared, (void *)0, (void *)0, (unsigned char)54 },
#endif
#if PROBE_FD_N > 55
      { kShared, (void *)0, (void *)0, (unsigned char)55 },
#endif
#if PROBE_FD_N > 56
      { kShared, (void *)0, (void *)0, (unsigned char)56 },
#endif
#if PROBE_FD_N > 57
      { kShared, (void *)0, (void *)0, (unsigned char)57 },
#endif
#if PROBE_FD_N > 58
      { kShared, (void *)0, (void *)0, (unsigned char)58 },
#endif
#if PROBE_FD_N > 59
      { kShared, (void *)0, (void *)0, (unsigned char)59 },
#endif
#if PROBE_FD_N > 60
      { kShared, (void *)0, (void *)0, (unsigned char)60 },
#endif
#if PROBE_FD_N > 61
      { kShared, (void *)0, (void *)0, (unsigned char)61 },
#endif
#if PROBE_FD_N > 62
      { kShared, (void *)0, (void *)0, (unsigned char)62 },
#endif
#if PROBE_FD_N > 63
      { kShared, (void *)0, (void *)0, (unsigned char)63 },
#endif
#if PROBE_FD_N > 64
      { kShared, (void *)0, (void *)0, (unsigned char)64 },
#endif
#if PROBE_FD_N > 65
      { kShared, (void *)0, (void *)0, (unsigned char)65 },
#endif
#if PROBE_FD_N > 66
      { kShared, (void *)0, (void *)0, (unsigned char)66 },
#endif
#if PROBE_FD_N > 67
      { kShared, (void *)0, (void *)0, (unsigned char)67 },
#endif
#if PROBE_FD_N > 68
      { kShared, (void *)0, (void *)0, (unsigned char)68 },
#endif
#if PROBE_FD_N > 69
      { kShared, (void *)0, (void *)0, (unsigned char)69 },
#endif
#if PROBE_FD_N > 70
      { kShared, (void *)0, (void *)0, (unsigned char)70 },
#endif
#if PROBE_FD_N > 71
      { kShared, (void *)0, (void *)0, (unsigned char)71 },
#endif
#if PROBE_FD_N > 72
      { kShared, (void *)0, (void *)0, (unsigned char)72 },
#endif
#if PROBE_FD_N > 73
      { kShared, (void *)0, (void *)0, (unsigned char)73 },
#endif
#if PROBE_FD_N > 74
      { kShared, (void *)0, (void *)0, (unsigned char)74 },
#endif
#if PROBE_FD_N > 75
      { kShared, (void *)0, (void *)0, (unsigned char)75 },
#endif
#if PROBE_FD_N > 76
      { kShared, (void *)0, (void *)0, (unsigned char)76 },
#endif
#if PROBE_FD_N > 77
      { kShared, (void *)0, (void *)0, (unsigned char)77 },
#endif
#if PROBE_FD_N > 78
      { kShared, (void *)0, (void *)0, (unsigned char)78 },
#endif
#if PROBE_FD_N > 79
      { kShared, (void *)0, (void *)0, (unsigned char)79 },
#endif
#if PROBE_FD_N > 80
      { kShared, (void *)0, (void *)0, (unsigned char)80 },
#endif
#if PROBE_FD_N > 81
      { kShared, (void *)0, (void *)0, (unsigned char)81 },
#endif
#if PROBE_FD_N > 82
      { kShared, (void *)0, (void *)0, (unsigned char)82 },
#endif
#if PROBE_FD_N > 83
      { kShared, (void *)0, (void *)0, (unsigned char)83 },
#endif
#if PROBE_FD_N > 84
      { kShared, (void *)0, (void *)0, (unsigned char)84 },
#endif
#if PROBE_FD_N > 85
      { kShared, (void *)0, (void *)0, (unsigned char)85 },
#endif
#if PROBE_FD_N > 86
      { kShared, (void *)0, (void *)0, (unsigned char)86 },
#endif
#if PROBE_FD_N > 87
      { kShared, (void *)0, (void *)0, (unsigned char)87 },
#endif
#if PROBE_FD_N > 88
      { kShared, (void *)0, (void *)0, (unsigned char)88 },
#endif
#if PROBE_FD_N > 89
      { kShared, (void *)0, (void *)0, (unsigned char)89 },
#endif
#if PROBE_FD_N > 90
      { kShared, (void *)0, (void *)0, (unsigned char)90 },
#endif
#if PROBE_FD_N > 91
      { kShared, (void *)0, (void *)0, (unsigned char)91 },
#endif
#if PROBE_FD_N > 92
      { kShared, (void *)0, (void *)0, (unsigned char)92 },
#endif
#if PROBE_FD_N > 93
      { kShared, (void *)0, (void *)0, (unsigned char)93 },
#endif
#if PROBE_FD_N > 94
      { kShared, (void *)0, (void *)0, (unsigned char)94 },
#endif
#if PROBE_FD_N > 95
      { kShared, (void *)0, (void *)0, (unsigned char)95 },
#endif
    };
    unsigned i, n = (unsigned)(sizeof(arr) / sizeof(arr[0])), ok = 0;
    for (i = 0; i < n; i++)
      if (arr[i].zName && arr[i].zName[0] == 'f') ok++;
    /* Return the DEFICIT, not the count. `0xC0 | (count & 0x3f)` wrapped at 64 and made
       N=64 and N=72 indistinguishable from a miscount of 0 -- g64 and sh72 both returned
       0xC0 and could not be read. The deficit is 0 when correct and small when not, so it
       never wraps and a wrong answer is unmistakable. */
    return (int)(0xA0u | ((n - ok) & 0x1fu));
  }
#endif
#if CAPSTONE_SQLITE_STAGE == 94
#ifndef PROBE_FD_N
#define PROBE_FD_N 56
#endif
  if (stage == 94) {
    /* WHICH ENTRY IS WRONG? The N=56 build returned a count of 55 of 56 -- one entry's zName
       did not read back starting with 'f'. A count says how many; this says WHICH, which
       names the failing cap-table slot directly.
       Returns: the INDEX of the first bad entry (0..95), or 0xFF if every entry is correct.
       Index and "all correct" cannot alias, unlike the earlier count encoding that wrapped
       at 64 and produced a withdrawn conclusion. */
    struct probe_fd { const char *zName; void *p1; void *p2; unsigned char flags; };
    struct probe_fd arr[] = {
      { "fn0", (void *)0, (void *)0, (unsigned char)0 },
#if PROBE_FD_N > 1
      { "fn1", (void *)0, (void *)0, (unsigned char)1 },
#endif
#if PROBE_FD_N > 2
      { "fn2", (void *)0, (void *)0, (unsigned char)2 },
#endif
#if PROBE_FD_N > 3
      { "fn3", (void *)0, (void *)0, (unsigned char)3 },
#endif
#if PROBE_FD_N > 4
      { "fn4", (void *)0, (void *)0, (unsigned char)4 },
#endif
#if PROBE_FD_N > 5
      { "fn5", (void *)0, (void *)0, (unsigned char)5 },
#endif
#if PROBE_FD_N > 6
      { "fn6", (void *)0, (void *)0, (unsigned char)6 },
#endif
#if PROBE_FD_N > 7
      { "fn7", (void *)0, (void *)0, (unsigned char)7 },
#endif
#if PROBE_FD_N > 8
      { "fn8", (void *)0, (void *)0, (unsigned char)8 },
#endif
#if PROBE_FD_N > 9
      { "fn9", (void *)0, (void *)0, (unsigned char)9 },
#endif
#if PROBE_FD_N > 10
      { "fn10", (void *)0, (void *)0, (unsigned char)10 },
#endif
#if PROBE_FD_N > 11
      { "fn11", (void *)0, (void *)0, (unsigned char)11 },
#endif
#if PROBE_FD_N > 12
      { "fn12", (void *)0, (void *)0, (unsigned char)12 },
#endif
#if PROBE_FD_N > 13
      { "fn13", (void *)0, (void *)0, (unsigned char)13 },
#endif
#if PROBE_FD_N > 14
      { "fn14", (void *)0, (void *)0, (unsigned char)14 },
#endif
#if PROBE_FD_N > 15
      { "fn15", (void *)0, (void *)0, (unsigned char)15 },
#endif
#if PROBE_FD_N > 16
      { "fn16", (void *)0, (void *)0, (unsigned char)16 },
#endif
#if PROBE_FD_N > 17
      { "fn17", (void *)0, (void *)0, (unsigned char)17 },
#endif
#if PROBE_FD_N > 18
      { "fn18", (void *)0, (void *)0, (unsigned char)18 },
#endif
#if PROBE_FD_N > 19
      { "fn19", (void *)0, (void *)0, (unsigned char)19 },
#endif
#if PROBE_FD_N > 20
      { "fn20", (void *)0, (void *)0, (unsigned char)20 },
#endif
#if PROBE_FD_N > 21
      { "fn21", (void *)0, (void *)0, (unsigned char)21 },
#endif
#if PROBE_FD_N > 22
      { "fn22", (void *)0, (void *)0, (unsigned char)22 },
#endif
#if PROBE_FD_N > 23
      { "fn23", (void *)0, (void *)0, (unsigned char)23 },
#endif
#if PROBE_FD_N > 24
      { "fn24", (void *)0, (void *)0, (unsigned char)24 },
#endif
#if PROBE_FD_N > 25
      { "fn25", (void *)0, (void *)0, (unsigned char)25 },
#endif
#if PROBE_FD_N > 26
      { "fn26", (void *)0, (void *)0, (unsigned char)26 },
#endif
#if PROBE_FD_N > 27
      { "fn27", (void *)0, (void *)0, (unsigned char)27 },
#endif
#if PROBE_FD_N > 28
      { "fn28", (void *)0, (void *)0, (unsigned char)28 },
#endif
#if PROBE_FD_N > 29
      { "fn29", (void *)0, (void *)0, (unsigned char)29 },
#endif
#if PROBE_FD_N > 30
      { "fn30", (void *)0, (void *)0, (unsigned char)30 },
#endif
#if PROBE_FD_N > 31
      { "fn31", (void *)0, (void *)0, (unsigned char)31 },
#endif
#if PROBE_FD_N > 32
      { "fn32", (void *)0, (void *)0, (unsigned char)32 },
#endif
#if PROBE_FD_N > 33
      { "fn33", (void *)0, (void *)0, (unsigned char)33 },
#endif
#if PROBE_FD_N > 34
      { "fn34", (void *)0, (void *)0, (unsigned char)34 },
#endif
#if PROBE_FD_N > 35
      { "fn35", (void *)0, (void *)0, (unsigned char)35 },
#endif
#if PROBE_FD_N > 36
      { "fn36", (void *)0, (void *)0, (unsigned char)36 },
#endif
#if PROBE_FD_N > 37
      { "fn37", (void *)0, (void *)0, (unsigned char)37 },
#endif
#if PROBE_FD_N > 38
      { "fn38", (void *)0, (void *)0, (unsigned char)38 },
#endif
#if PROBE_FD_N > 39
      { "fn39", (void *)0, (void *)0, (unsigned char)39 },
#endif
#if PROBE_FD_N > 40
      { "fn40", (void *)0, (void *)0, (unsigned char)40 },
#endif
#if PROBE_FD_N > 41
      { "fn41", (void *)0, (void *)0, (unsigned char)41 },
#endif
#if PROBE_FD_N > 42
      { "fn42", (void *)0, (void *)0, (unsigned char)42 },
#endif
#if PROBE_FD_N > 43
      { "fn43", (void *)0, (void *)0, (unsigned char)43 },
#endif
#if PROBE_FD_N > 44
      { "fn44", (void *)0, (void *)0, (unsigned char)44 },
#endif
#if PROBE_FD_N > 45
      { "fn45", (void *)0, (void *)0, (unsigned char)45 },
#endif
#if PROBE_FD_N > 46
      { "fn46", (void *)0, (void *)0, (unsigned char)46 },
#endif
#if PROBE_FD_N > 47
      { "fn47", (void *)0, (void *)0, (unsigned char)47 },
#endif
#if PROBE_FD_N > 48
      { "fn48", (void *)0, (void *)0, (unsigned char)48 },
#endif
#if PROBE_FD_N > 49
      { "fn49", (void *)0, (void *)0, (unsigned char)49 },
#endif
#if PROBE_FD_N > 50
      { "fn50", (void *)0, (void *)0, (unsigned char)50 },
#endif
#if PROBE_FD_N > 51
      { "fn51", (void *)0, (void *)0, (unsigned char)51 },
#endif
#if PROBE_FD_N > 52
      { "fn52", (void *)0, (void *)0, (unsigned char)52 },
#endif
#if PROBE_FD_N > 53
      { "fn53", (void *)0, (void *)0, (unsigned char)53 },
#endif
#if PROBE_FD_N > 54
      { "fn54", (void *)0, (void *)0, (unsigned char)54 },
#endif
#if PROBE_FD_N > 55
      { "fn55", (void *)0, (void *)0, (unsigned char)55 },
#endif
#if PROBE_FD_N > 56
      { "fn56", (void *)0, (void *)0, (unsigned char)56 },
#endif
#if PROBE_FD_N > 57
      { "fn57", (void *)0, (void *)0, (unsigned char)57 },
#endif
#if PROBE_FD_N > 58
      { "fn58", (void *)0, (void *)0, (unsigned char)58 },
#endif
#if PROBE_FD_N > 59
      { "fn59", (void *)0, (void *)0, (unsigned char)59 },
#endif
#if PROBE_FD_N > 60
      { "fn60", (void *)0, (void *)0, (unsigned char)60 },
#endif
#if PROBE_FD_N > 61
      { "fn61", (void *)0, (void *)0, (unsigned char)61 },
#endif
#if PROBE_FD_N > 62
      { "fn62", (void *)0, (void *)0, (unsigned char)62 },
#endif
#if PROBE_FD_N > 63
      { "fn63", (void *)0, (void *)0, (unsigned char)63 },
#endif
#if PROBE_FD_N > 64
      { "fn64", (void *)0, (void *)0, (unsigned char)64 },
#endif
#if PROBE_FD_N > 65
      { "fn65", (void *)0, (void *)0, (unsigned char)65 },
#endif
#if PROBE_FD_N > 66
      { "fn66", (void *)0, (void *)0, (unsigned char)66 },
#endif
#if PROBE_FD_N > 67
      { "fn67", (void *)0, (void *)0, (unsigned char)67 },
#endif
#if PROBE_FD_N > 68
      { "fn68", (void *)0, (void *)0, (unsigned char)68 },
#endif
#if PROBE_FD_N > 69
      { "fn69", (void *)0, (void *)0, (unsigned char)69 },
#endif
#if PROBE_FD_N > 70
      { "fn70", (void *)0, (void *)0, (unsigned char)70 },
#endif
#if PROBE_FD_N > 71
      { "fn71", (void *)0, (void *)0, (unsigned char)71 },
#endif
#if PROBE_FD_N > 72
      { "fn72", (void *)0, (void *)0, (unsigned char)72 },
#endif
#if PROBE_FD_N > 73
      { "fn73", (void *)0, (void *)0, (unsigned char)73 },
#endif
#if PROBE_FD_N > 74
      { "fn74", (void *)0, (void *)0, (unsigned char)74 },
#endif
#if PROBE_FD_N > 75
      { "fn75", (void *)0, (void *)0, (unsigned char)75 },
#endif
#if PROBE_FD_N > 76
      { "fn76", (void *)0, (void *)0, (unsigned char)76 },
#endif
#if PROBE_FD_N > 77
      { "fn77", (void *)0, (void *)0, (unsigned char)77 },
#endif
#if PROBE_FD_N > 78
      { "fn78", (void *)0, (void *)0, (unsigned char)78 },
#endif
#if PROBE_FD_N > 79
      { "fn79", (void *)0, (void *)0, (unsigned char)79 },
#endif
#if PROBE_FD_N > 80
      { "fn80", (void *)0, (void *)0, (unsigned char)80 },
#endif
#if PROBE_FD_N > 81
      { "fn81", (void *)0, (void *)0, (unsigned char)81 },
#endif
#if PROBE_FD_N > 82
      { "fn82", (void *)0, (void *)0, (unsigned char)82 },
#endif
#if PROBE_FD_N > 83
      { "fn83", (void *)0, (void *)0, (unsigned char)83 },
#endif
#if PROBE_FD_N > 84
      { "fn84", (void *)0, (void *)0, (unsigned char)84 },
#endif
#if PROBE_FD_N > 85
      { "fn85", (void *)0, (void *)0, (unsigned char)85 },
#endif
#if PROBE_FD_N > 86
      { "fn86", (void *)0, (void *)0, (unsigned char)86 },
#endif
#if PROBE_FD_N > 87
      { "fn87", (void *)0, (void *)0, (unsigned char)87 },
#endif
#if PROBE_FD_N > 88
      { "fn88", (void *)0, (void *)0, (unsigned char)88 },
#endif
#if PROBE_FD_N > 89
      { "fn89", (void *)0, (void *)0, (unsigned char)89 },
#endif
#if PROBE_FD_N > 90
      { "fn90", (void *)0, (void *)0, (unsigned char)90 },
#endif
#if PROBE_FD_N > 91
      { "fn91", (void *)0, (void *)0, (unsigned char)91 },
#endif
#if PROBE_FD_N > 92
      { "fn92", (void *)0, (void *)0, (unsigned char)92 },
#endif
#if PROBE_FD_N > 93
      { "fn93", (void *)0, (void *)0, (unsigned char)93 },
#endif
#if PROBE_FD_N > 94
      { "fn94", (void *)0, (void *)0, (unsigned char)94 },
#endif
#if PROBE_FD_N > 95
      { "fn95", (void *)0, (void *)0, (unsigned char)95 },
#endif
    };
    unsigned i, n = (unsigned)(sizeof(arr) / sizeof(arr[0]));
    for (i = 0; i < n; i++)
      if (!arr[i].zName || arr[i].zName[0] != 'f')
        return (int)(i & 0xffu);
    return 0xFF;
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 95 && CAPSTONE_SQLITE_STAGE <= 97
#ifndef PROBE_FD_N
#define PROBE_FD_N 56
#endif
  if (stage >= 95 && stage <= 97) {
    /* WHAT IS ACTUALLY IN THE BAD SLOT?  "arr[55].zName[0] != 'f'" conflates three faults
       that need completely different fixes:
         (a) the POINTER is wrong  -- the stc that wrote it lost its value;
         (b) the pointer is right but its TAG is gone -- it reads back as a non-capability;
         (c) pointer and tag are fine but the MEMORY it points at is wrong.
       One probe each, one value per domain (a previous two-value probe wedged):
         95: cap TYPE of arr[N-1].zName via `lcc` zimm=1. NOT_CAP means the tag was lost on
             the store or the reload -> (b). A normal type means the capability survived.
         96: low byte of its CURSOR via `lcc` zimm=2. 0x00 with a lost tag means the slot was
             never written; a plausible offset means the pointer is intact -> (c).
         97: reads the SAME string through the container instead of through arr[], i.e.
             literal "fn0" compared at the offset the last entry should hold. If this reads
             correctly while 95/96 show damage, the DATA is fine and only the stored
             capability is broken.
       lcc encoding is the in-tree one (start-gpfree-captable.S:34): funct7 0x04, zimm in rs2.
       zimm 1 and 2 are pure metadata reads -- they do NOT touch the rev-node query channel
       (only zimm 0 does, and that is the channel that can hang). */
    struct probe_fd { const char *zName; void *p1; void *p2; unsigned char flags; };
    struct probe_fd arr[] = {
      { "fn0", (void *)0, (void *)0, (unsigned char)0 },
#if PROBE_FD_N > 1
      { "fn1", (void *)0, (void *)0, (unsigned char)1 },
#endif
#if PROBE_FD_N > 2
      { "fn2", (void *)0, (void *)0, (unsigned char)2 },
#endif
#if PROBE_FD_N > 3
      { "fn3", (void *)0, (void *)0, (unsigned char)3 },
#endif
#if PROBE_FD_N > 4
      { "fn4", (void *)0, (void *)0, (unsigned char)4 },
#endif
#if PROBE_FD_N > 5
      { "fn5", (void *)0, (void *)0, (unsigned char)5 },
#endif
#if PROBE_FD_N > 6
      { "fn6", (void *)0, (void *)0, (unsigned char)6 },
#endif
#if PROBE_FD_N > 7
      { "fn7", (void *)0, (void *)0, (unsigned char)7 },
#endif
#if PROBE_FD_N > 8
      { "fn8", (void *)0, (void *)0, (unsigned char)8 },
#endif
#if PROBE_FD_N > 9
      { "fn9", (void *)0, (void *)0, (unsigned char)9 },
#endif
#if PROBE_FD_N > 10
      { "fn10", (void *)0, (void *)0, (unsigned char)10 },
#endif
#if PROBE_FD_N > 11
      { "fn11", (void *)0, (void *)0, (unsigned char)11 },
#endif
#if PROBE_FD_N > 12
      { "fn12", (void *)0, (void *)0, (unsigned char)12 },
#endif
#if PROBE_FD_N > 13
      { "fn13", (void *)0, (void *)0, (unsigned char)13 },
#endif
#if PROBE_FD_N > 14
      { "fn14", (void *)0, (void *)0, (unsigned char)14 },
#endif
#if PROBE_FD_N > 15
      { "fn15", (void *)0, (void *)0, (unsigned char)15 },
#endif
#if PROBE_FD_N > 16
      { "fn16", (void *)0, (void *)0, (unsigned char)16 },
#endif
#if PROBE_FD_N > 17
      { "fn17", (void *)0, (void *)0, (unsigned char)17 },
#endif
#if PROBE_FD_N > 18
      { "fn18", (void *)0, (void *)0, (unsigned char)18 },
#endif
#if PROBE_FD_N > 19
      { "fn19", (void *)0, (void *)0, (unsigned char)19 },
#endif
#if PROBE_FD_N > 20
      { "fn20", (void *)0, (void *)0, (unsigned char)20 },
#endif
#if PROBE_FD_N > 21
      { "fn21", (void *)0, (void *)0, (unsigned char)21 },
#endif
#if PROBE_FD_N > 22
      { "fn22", (void *)0, (void *)0, (unsigned char)22 },
#endif
#if PROBE_FD_N > 23
      { "fn23", (void *)0, (void *)0, (unsigned char)23 },
#endif
#if PROBE_FD_N > 24
      { "fn24", (void *)0, (void *)0, (unsigned char)24 },
#endif
#if PROBE_FD_N > 25
      { "fn25", (void *)0, (void *)0, (unsigned char)25 },
#endif
#if PROBE_FD_N > 26
      { "fn26", (void *)0, (void *)0, (unsigned char)26 },
#endif
#if PROBE_FD_N > 27
      { "fn27", (void *)0, (void *)0, (unsigned char)27 },
#endif
#if PROBE_FD_N > 28
      { "fn28", (void *)0, (void *)0, (unsigned char)28 },
#endif
#if PROBE_FD_N > 29
      { "fn29", (void *)0, (void *)0, (unsigned char)29 },
#endif
#if PROBE_FD_N > 30
      { "fn30", (void *)0, (void *)0, (unsigned char)30 },
#endif
#if PROBE_FD_N > 31
      { "fn31", (void *)0, (void *)0, (unsigned char)31 },
#endif
#if PROBE_FD_N > 32
      { "fn32", (void *)0, (void *)0, (unsigned char)32 },
#endif
#if PROBE_FD_N > 33
      { "fn33", (void *)0, (void *)0, (unsigned char)33 },
#endif
#if PROBE_FD_N > 34
      { "fn34", (void *)0, (void *)0, (unsigned char)34 },
#endif
#if PROBE_FD_N > 35
      { "fn35", (void *)0, (void *)0, (unsigned char)35 },
#endif
#if PROBE_FD_N > 36
      { "fn36", (void *)0, (void *)0, (unsigned char)36 },
#endif
#if PROBE_FD_N > 37
      { "fn37", (void *)0, (void *)0, (unsigned char)37 },
#endif
#if PROBE_FD_N > 38
      { "fn38", (void *)0, (void *)0, (unsigned char)38 },
#endif
#if PROBE_FD_N > 39
      { "fn39", (void *)0, (void *)0, (unsigned char)39 },
#endif
#if PROBE_FD_N > 40
      { "fn40", (void *)0, (void *)0, (unsigned char)40 },
#endif
#if PROBE_FD_N > 41
      { "fn41", (void *)0, (void *)0, (unsigned char)41 },
#endif
#if PROBE_FD_N > 42
      { "fn42", (void *)0, (void *)0, (unsigned char)42 },
#endif
#if PROBE_FD_N > 43
      { "fn43", (void *)0, (void *)0, (unsigned char)43 },
#endif
#if PROBE_FD_N > 44
      { "fn44", (void *)0, (void *)0, (unsigned char)44 },
#endif
#if PROBE_FD_N > 45
      { "fn45", (void *)0, (void *)0, (unsigned char)45 },
#endif
#if PROBE_FD_N > 46
      { "fn46", (void *)0, (void *)0, (unsigned char)46 },
#endif
#if PROBE_FD_N > 47
      { "fn47", (void *)0, (void *)0, (unsigned char)47 },
#endif
#if PROBE_FD_N > 48
      { "fn48", (void *)0, (void *)0, (unsigned char)48 },
#endif
#if PROBE_FD_N > 49
      { "fn49", (void *)0, (void *)0, (unsigned char)49 },
#endif
#if PROBE_FD_N > 50
      { "fn50", (void *)0, (void *)0, (unsigned char)50 },
#endif
#if PROBE_FD_N > 51
      { "fn51", (void *)0, (void *)0, (unsigned char)51 },
#endif
#if PROBE_FD_N > 52
      { "fn52", (void *)0, (void *)0, (unsigned char)52 },
#endif
#if PROBE_FD_N > 53
      { "fn53", (void *)0, (void *)0, (unsigned char)53 },
#endif
#if PROBE_FD_N > 54
      { "fn54", (void *)0, (void *)0, (unsigned char)54 },
#endif
#if PROBE_FD_N > 55
      { "fn55", (void *)0, (void *)0, (unsigned char)55 },
#endif
#if PROBE_FD_N > 56
      { "fn56", (void *)0, (void *)0, (unsigned char)56 },
#endif
#if PROBE_FD_N > 57
      { "fn57", (void *)0, (void *)0, (unsigned char)57 },
#endif
#if PROBE_FD_N > 58
      { "fn58", (void *)0, (void *)0, (unsigned char)58 },
#endif
#if PROBE_FD_N > 59
      { "fn59", (void *)0, (void *)0, (unsigned char)59 },
#endif
#if PROBE_FD_N > 60
      { "fn60", (void *)0, (void *)0, (unsigned char)60 },
#endif
#if PROBE_FD_N > 61
      { "fn61", (void *)0, (void *)0, (unsigned char)61 },
#endif
#if PROBE_FD_N > 62
      { "fn62", (void *)0, (void *)0, (unsigned char)62 },
#endif
#if PROBE_FD_N > 63
      { "fn63", (void *)0, (void *)0, (unsigned char)63 },
#endif
#if PROBE_FD_N > 64
      { "fn64", (void *)0, (void *)0, (unsigned char)64 },
#endif
#if PROBE_FD_N > 65
      { "fn65", (void *)0, (void *)0, (unsigned char)65 },
#endif
#if PROBE_FD_N > 66
      { "fn66", (void *)0, (void *)0, (unsigned char)66 },
#endif
#if PROBE_FD_N > 67
      { "fn67", (void *)0, (void *)0, (unsigned char)67 },
#endif
#if PROBE_FD_N > 68
      { "fn68", (void *)0, (void *)0, (unsigned char)68 },
#endif
#if PROBE_FD_N > 69
      { "fn69", (void *)0, (void *)0, (unsigned char)69 },
#endif
#if PROBE_FD_N > 70
      { "fn70", (void *)0, (void *)0, (unsigned char)70 },
#endif
#if PROBE_FD_N > 71
      { "fn71", (void *)0, (void *)0, (unsigned char)71 },
#endif
#if PROBE_FD_N > 72
      { "fn72", (void *)0, (void *)0, (unsigned char)72 },
#endif
#if PROBE_FD_N > 73
      { "fn73", (void *)0, (void *)0, (unsigned char)73 },
#endif
#if PROBE_FD_N > 74
      { "fn74", (void *)0, (void *)0, (unsigned char)74 },
#endif
#if PROBE_FD_N > 75
      { "fn75", (void *)0, (void *)0, (unsigned char)75 },
#endif
#if PROBE_FD_N > 76
      { "fn76", (void *)0, (void *)0, (unsigned char)76 },
#endif
#if PROBE_FD_N > 77
      { "fn77", (void *)0, (void *)0, (unsigned char)77 },
#endif
#if PROBE_FD_N > 78
      { "fn78", (void *)0, (void *)0, (unsigned char)78 },
#endif
#if PROBE_FD_N > 79
      { "fn79", (void *)0, (void *)0, (unsigned char)79 },
#endif
#if PROBE_FD_N > 80
      { "fn80", (void *)0, (void *)0, (unsigned char)80 },
#endif
#if PROBE_FD_N > 81
      { "fn81", (void *)0, (void *)0, (unsigned char)81 },
#endif
#if PROBE_FD_N > 82
      { "fn82", (void *)0, (void *)0, (unsigned char)82 },
#endif
#if PROBE_FD_N > 83
      { "fn83", (void *)0, (void *)0, (unsigned char)83 },
#endif
#if PROBE_FD_N > 84
      { "fn84", (void *)0, (void *)0, (unsigned char)84 },
#endif
#if PROBE_FD_N > 85
      { "fn85", (void *)0, (void *)0, (unsigned char)85 },
#endif
#if PROBE_FD_N > 86
      { "fn86", (void *)0, (void *)0, (unsigned char)86 },
#endif
#if PROBE_FD_N > 87
      { "fn87", (void *)0, (void *)0, (unsigned char)87 },
#endif
#if PROBE_FD_N > 88
      { "fn88", (void *)0, (void *)0, (unsigned char)88 },
#endif
#if PROBE_FD_N > 89
      { "fn89", (void *)0, (void *)0, (unsigned char)89 },
#endif
#if PROBE_FD_N > 90
      { "fn90", (void *)0, (void *)0, (unsigned char)90 },
#endif
#if PROBE_FD_N > 91
      { "fn91", (void *)0, (void *)0, (unsigned char)91 },
#endif
#if PROBE_FD_N > 92
      { "fn92", (void *)0, (void *)0, (unsigned char)92 },
#endif
#if PROBE_FD_N > 93
      { "fn93", (void *)0, (void *)0, (unsigned char)93 },
#endif
#if PROBE_FD_N > 94
      { "fn94", (void *)0, (void *)0, (unsigned char)94 },
#endif
#if PROBE_FD_N > 95
      { "fn95", (void *)0, (void *)0, (unsigned char)95 },
#endif
    };
    unsigned n = (unsigned)(sizeof(arr) / sizeof(arr[0]));
    const char *volatile bad = arr[n - 1].zName;
    if (stage == 97)
      return (int)(0x30u | (unsigned)(bad ? (unsigned char)bad[0] : 0) & 0xffu);
    {
      unsigned long v = 0;
      if (stage == 95)
        __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(v) : "r"(bad));
      else
        __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(v) : "r"(bad));
      return (int)(v & 0xffUL);
    }
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 98 && CAPSTONE_SQLITE_STAGE <= 99
#ifndef PROBE_FD_N
#define PROBE_FD_N 56
#endif
  if (stage == 98 || stage == 99) {
    /* ARE THE BOUNDS ALSO WRONG, OR ONLY THE CURSOR?
       Established: the bad slot holds a VALID capability (lcc did not trap, so the tag
       survived) whose cursor low byte reads 0x00 where 0x42 is expected for "fn55" at
       container offset 3906 = 0xF42.
         98: start (lcc zimm=3), low byte.
         99: end   (lcc zimm=4), low byte.
       If start/end match the merged-string container (i.e. the same values the WORKING entries
       carry) and only the cursor is wrong, the fault is in the `cincoffset` that adds this
       entry's offset -- one instruction -- rather than in how the capability was derived.
       If the bounds are wrong too, the whole derivation for this entry is bad.
       One value per domain: every two-value probe in this campaign has wedged. */
    struct probe_fd { const char *zName; void *p1; void *p2; unsigned char flags; };
    struct probe_fd arr[] = {
      { "fn0", (void *)0, (void *)0, (unsigned char)0 },
#if PROBE_FD_N > 1
      { "fn1", (void *)0, (void *)0, (unsigned char)1 },
#endif
#if PROBE_FD_N > 2
      { "fn2", (void *)0, (void *)0, (unsigned char)2 },
#endif
#if PROBE_FD_N > 3
      { "fn3", (void *)0, (void *)0, (unsigned char)3 },
#endif
#if PROBE_FD_N > 4
      { "fn4", (void *)0, (void *)0, (unsigned char)4 },
#endif
#if PROBE_FD_N > 5
      { "fn5", (void *)0, (void *)0, (unsigned char)5 },
#endif
#if PROBE_FD_N > 6
      { "fn6", (void *)0, (void *)0, (unsigned char)6 },
#endif
#if PROBE_FD_N > 7
      { "fn7", (void *)0, (void *)0, (unsigned char)7 },
#endif
#if PROBE_FD_N > 8
      { "fn8", (void *)0, (void *)0, (unsigned char)8 },
#endif
#if PROBE_FD_N > 9
      { "fn9", (void *)0, (void *)0, (unsigned char)9 },
#endif
#if PROBE_FD_N > 10
      { "fn10", (void *)0, (void *)0, (unsigned char)10 },
#endif
#if PROBE_FD_N > 11
      { "fn11", (void *)0, (void *)0, (unsigned char)11 },
#endif
#if PROBE_FD_N > 12
      { "fn12", (void *)0, (void *)0, (unsigned char)12 },
#endif
#if PROBE_FD_N > 13
      { "fn13", (void *)0, (void *)0, (unsigned char)13 },
#endif
#if PROBE_FD_N > 14
      { "fn14", (void *)0, (void *)0, (unsigned char)14 },
#endif
#if PROBE_FD_N > 15
      { "fn15", (void *)0, (void *)0, (unsigned char)15 },
#endif
#if PROBE_FD_N > 16
      { "fn16", (void *)0, (void *)0, (unsigned char)16 },
#endif
#if PROBE_FD_N > 17
      { "fn17", (void *)0, (void *)0, (unsigned char)17 },
#endif
#if PROBE_FD_N > 18
      { "fn18", (void *)0, (void *)0, (unsigned char)18 },
#endif
#if PROBE_FD_N > 19
      { "fn19", (void *)0, (void *)0, (unsigned char)19 },
#endif
#if PROBE_FD_N > 20
      { "fn20", (void *)0, (void *)0, (unsigned char)20 },
#endif
#if PROBE_FD_N > 21
      { "fn21", (void *)0, (void *)0, (unsigned char)21 },
#endif
#if PROBE_FD_N > 22
      { "fn22", (void *)0, (void *)0, (unsigned char)22 },
#endif
#if PROBE_FD_N > 23
      { "fn23", (void *)0, (void *)0, (unsigned char)23 },
#endif
#if PROBE_FD_N > 24
      { "fn24", (void *)0, (void *)0, (unsigned char)24 },
#endif
#if PROBE_FD_N > 25
      { "fn25", (void *)0, (void *)0, (unsigned char)25 },
#endif
#if PROBE_FD_N > 26
      { "fn26", (void *)0, (void *)0, (unsigned char)26 },
#endif
#if PROBE_FD_N > 27
      { "fn27", (void *)0, (void *)0, (unsigned char)27 },
#endif
#if PROBE_FD_N > 28
      { "fn28", (void *)0, (void *)0, (unsigned char)28 },
#endif
#if PROBE_FD_N > 29
      { "fn29", (void *)0, (void *)0, (unsigned char)29 },
#endif
#if PROBE_FD_N > 30
      { "fn30", (void *)0, (void *)0, (unsigned char)30 },
#endif
#if PROBE_FD_N > 31
      { "fn31", (void *)0, (void *)0, (unsigned char)31 },
#endif
#if PROBE_FD_N > 32
      { "fn32", (void *)0, (void *)0, (unsigned char)32 },
#endif
#if PROBE_FD_N > 33
      { "fn33", (void *)0, (void *)0, (unsigned char)33 },
#endif
#if PROBE_FD_N > 34
      { "fn34", (void *)0, (void *)0, (unsigned char)34 },
#endif
#if PROBE_FD_N > 35
      { "fn35", (void *)0, (void *)0, (unsigned char)35 },
#endif
#if PROBE_FD_N > 36
      { "fn36", (void *)0, (void *)0, (unsigned char)36 },
#endif
#if PROBE_FD_N > 37
      { "fn37", (void *)0, (void *)0, (unsigned char)37 },
#endif
#if PROBE_FD_N > 38
      { "fn38", (void *)0, (void *)0, (unsigned char)38 },
#endif
#if PROBE_FD_N > 39
      { "fn39", (void *)0, (void *)0, (unsigned char)39 },
#endif
#if PROBE_FD_N > 40
      { "fn40", (void *)0, (void *)0, (unsigned char)40 },
#endif
#if PROBE_FD_N > 41
      { "fn41", (void *)0, (void *)0, (unsigned char)41 },
#endif
#if PROBE_FD_N > 42
      { "fn42", (void *)0, (void *)0, (unsigned char)42 },
#endif
#if PROBE_FD_N > 43
      { "fn43", (void *)0, (void *)0, (unsigned char)43 },
#endif
#if PROBE_FD_N > 44
      { "fn44", (void *)0, (void *)0, (unsigned char)44 },
#endif
#if PROBE_FD_N > 45
      { "fn45", (void *)0, (void *)0, (unsigned char)45 },
#endif
#if PROBE_FD_N > 46
      { "fn46", (void *)0, (void *)0, (unsigned char)46 },
#endif
#if PROBE_FD_N > 47
      { "fn47", (void *)0, (void *)0, (unsigned char)47 },
#endif
#if PROBE_FD_N > 48
      { "fn48", (void *)0, (void *)0, (unsigned char)48 },
#endif
#if PROBE_FD_N > 49
      { "fn49", (void *)0, (void *)0, (unsigned char)49 },
#endif
#if PROBE_FD_N > 50
      { "fn50", (void *)0, (void *)0, (unsigned char)50 },
#endif
#if PROBE_FD_N > 51
      { "fn51", (void *)0, (void *)0, (unsigned char)51 },
#endif
#if PROBE_FD_N > 52
      { "fn52", (void *)0, (void *)0, (unsigned char)52 },
#endif
#if PROBE_FD_N > 53
      { "fn53", (void *)0, (void *)0, (unsigned char)53 },
#endif
#if PROBE_FD_N > 54
      { "fn54", (void *)0, (void *)0, (unsigned char)54 },
#endif
#if PROBE_FD_N > 55
      { "fn55", (void *)0, (void *)0, (unsigned char)55 },
#endif
#if PROBE_FD_N > 56
      { "fn56", (void *)0, (void *)0, (unsigned char)56 },
#endif
#if PROBE_FD_N > 57
      { "fn57", (void *)0, (void *)0, (unsigned char)57 },
#endif
#if PROBE_FD_N > 58
      { "fn58", (void *)0, (void *)0, (unsigned char)58 },
#endif
#if PROBE_FD_N > 59
      { "fn59", (void *)0, (void *)0, (unsigned char)59 },
#endif
#if PROBE_FD_N > 60
      { "fn60", (void *)0, (void *)0, (unsigned char)60 },
#endif
#if PROBE_FD_N > 61
      { "fn61", (void *)0, (void *)0, (unsigned char)61 },
#endif
#if PROBE_FD_N > 62
      { "fn62", (void *)0, (void *)0, (unsigned char)62 },
#endif
#if PROBE_FD_N > 63
      { "fn63", (void *)0, (void *)0, (unsigned char)63 },
#endif
#if PROBE_FD_N > 64
      { "fn64", (void *)0, (void *)0, (unsigned char)64 },
#endif
#if PROBE_FD_N > 65
      { "fn65", (void *)0, (void *)0, (unsigned char)65 },
#endif
#if PROBE_FD_N > 66
      { "fn66", (void *)0, (void *)0, (unsigned char)66 },
#endif
#if PROBE_FD_N > 67
      { "fn67", (void *)0, (void *)0, (unsigned char)67 },
#endif
#if PROBE_FD_N > 68
      { "fn68", (void *)0, (void *)0, (unsigned char)68 },
#endif
#if PROBE_FD_N > 69
      { "fn69", (void *)0, (void *)0, (unsigned char)69 },
#endif
#if PROBE_FD_N > 70
      { "fn70", (void *)0, (void *)0, (unsigned char)70 },
#endif
#if PROBE_FD_N > 71
      { "fn71", (void *)0, (void *)0, (unsigned char)71 },
#endif
#if PROBE_FD_N > 72
      { "fn72", (void *)0, (void *)0, (unsigned char)72 },
#endif
#if PROBE_FD_N > 73
      { "fn73", (void *)0, (void *)0, (unsigned char)73 },
#endif
#if PROBE_FD_N > 74
      { "fn74", (void *)0, (void *)0, (unsigned char)74 },
#endif
#if PROBE_FD_N > 75
      { "fn75", (void *)0, (void *)0, (unsigned char)75 },
#endif
#if PROBE_FD_N > 76
      { "fn76", (void *)0, (void *)0, (unsigned char)76 },
#endif
#if PROBE_FD_N > 77
      { "fn77", (void *)0, (void *)0, (unsigned char)77 },
#endif
#if PROBE_FD_N > 78
      { "fn78", (void *)0, (void *)0, (unsigned char)78 },
#endif
#if PROBE_FD_N > 79
      { "fn79", (void *)0, (void *)0, (unsigned char)79 },
#endif
#if PROBE_FD_N > 80
      { "fn80", (void *)0, (void *)0, (unsigned char)80 },
#endif
#if PROBE_FD_N > 81
      { "fn81", (void *)0, (void *)0, (unsigned char)81 },
#endif
#if PROBE_FD_N > 82
      { "fn82", (void *)0, (void *)0, (unsigned char)82 },
#endif
#if PROBE_FD_N > 83
      { "fn83", (void *)0, (void *)0, (unsigned char)83 },
#endif
#if PROBE_FD_N > 84
      { "fn84", (void *)0, (void *)0, (unsigned char)84 },
#endif
#if PROBE_FD_N > 85
      { "fn85", (void *)0, (void *)0, (unsigned char)85 },
#endif
#if PROBE_FD_N > 86
      { "fn86", (void *)0, (void *)0, (unsigned char)86 },
#endif
#if PROBE_FD_N > 87
      { "fn87", (void *)0, (void *)0, (unsigned char)87 },
#endif
#if PROBE_FD_N > 88
      { "fn88", (void *)0, (void *)0, (unsigned char)88 },
#endif
#if PROBE_FD_N > 89
      { "fn89", (void *)0, (void *)0, (unsigned char)89 },
#endif
#if PROBE_FD_N > 90
      { "fn90", (void *)0, (void *)0, (unsigned char)90 },
#endif
#if PROBE_FD_N > 91
      { "fn91", (void *)0, (void *)0, (unsigned char)91 },
#endif
#if PROBE_FD_N > 92
      { "fn92", (void *)0, (void *)0, (unsigned char)92 },
#endif
#if PROBE_FD_N > 93
      { "fn93", (void *)0, (void *)0, (unsigned char)93 },
#endif
#if PROBE_FD_N > 94
      { "fn94", (void *)0, (void *)0, (unsigned char)94 },
#endif
#if PROBE_FD_N > 95
      { "fn95", (void *)0, (void *)0, (unsigned char)95 },
#endif
    };
    unsigned n = (unsigned)(sizeof(arr) / sizeof(arr[0]));
    const char *volatile bad = arr[n - 1].zName;
    unsigned long v = 0;
    if (stage == 98)
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(v) : "r"(bad));
    else
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x4" : "=r"(v) : "r"(bad));
    return (int)(v & 0xffUL);
  }
#endif
#if (CAPSTONE_SQLITE_STAGE >= 120 && CAPSTONE_SQLITE_STAGE <= 129) || (CAPSTONE_SQLITE_STAGE >= 140 && CAPSTONE_SQLITE_STAGE <= 150)
  /* R-14 N-THRESHOLD SWEEP, with no added static data.
     Both SQLite routes are blocked: the straight-line local wedges in-domain (R-14, validated
     at stage 10 with two returning controls), and making it a static triggers R-16 (5/5).
     A third option is to keep the straight-line shape but make it SMALL. This finds the
     largest N that still returns on silicon.
       120=N4  121=N8  122=N16  123=N32  124=N48  125=N64
     Deliberately NO static table and no extra globals -- adding either is what pushed the
     static-builtins images into the entry stall. Each arm returns the count it verified, so
     every run yields a number rather than a hang. Expect N. */
  /* 126/127 -- THE MISSING CONTROL for "is it straight-line, or is it struct fields at all?"
     Variant B wedges, but B is FOUR straight-line entries PLUS sixty loop-filled ones, so its
     wedge does not say which half is at fault. These two isolate that, and neither adds a
     global (no static table), so the carve count stays at 181 -- images at >=182 have
     entry-stalled 8/8 and are not worth a boot.
       126 = ZERO straight-line: all 64 entries assigned in a LOOP from one literal.
             Returns -> loop assignment into struct fields is SAFE, and straight-line
             materialisation is the culprit; reshaping SQLite's array can work.
             Wedges  -> even loop assignment into struct capability fields wedges; no
             reshaping of aBuiltinFunc can help and SQLite is unreachable by this route.
       127 = same but a FLAT pointer array, to separate "struct field" from "capability store
             in a loop" if 126 wedges.  Both expect 64. */
  /* 128/129 -- THE MECHANISM TEST for the SQLite blocker.
     Straight-line codegen derives EVERY literal from ONE cap-table capability:
         ldc            a1, 0(gp)
         cincoffsetimm  a2, a1, 6      <- rd != rs1
         cincoffsetimm  a2, a1, 11     <- reuses a1
     helper_cscincoffset consumes rs1 ONLY when rs1 != rd:
         if (rs1 != rd) { *rd = *rs1; if (!copyable(rs1)) *rs1 = NULL; }
     If that capability is LINEAR on silicon, the FIRST derivation nulls it and every later one
     reads a nulled base. The loop form never trips this because it writes back into the same
     register (rd == rs1).
       128 = TWO derivations with rd != rs1 from ONE base, then read through the SECOND.
             Predict: garbage/wedge on silicon, correct under QEMU.
       129 = the SAME +8, but CHAINED so rd == rs1 each time. Predict: correct everywhere.
     Both return the byte at base+8, which is 'B' = 0x42. One bit of difference between the
     two probes -- that is the whole experiment. */
  /* 140-146 -- MINIMISATION LADDER for the R-14 / SQLite construct.
     The current minimal case is 4 straight-line entries into struct kv a[64] (8 capability
     stores). These strip it further, one variable at a time, so the smallest failing form can
     be named exactly. Every arm returns a small number, so a run always yields data.
     No statics, no extra globals -- carve count must stay at the entering group's value.

       140  ONE struct VARIABLE, no array at all, 2 distinct literals   (2 cap stores)
       141  array[64], ONE entry  (a[0] only), 2 distinct literals      (2 cap stores)
       142  array[64], TWO entries, distinct literals                   (4 cap stores)
       143  array[64], FOUR entries, THE SAME literal in every field    (8 stores, no distinct)
       144  array[64], FOUR entries, distinct literals  == known-wedging control
       145  array[64], FOUR entries, ONE field per entry (z only)       (4 stores, struct)
       146  FOUR plain scalar pointers, no struct, no array, distinct   (4 stores)

     Reading: the first arm that wedges names the ingredient. 143 vs 144 isolates DISTINCT
     literals; 145 isolates two-fields-per-entry; 146 isolates the struct entirely; 140 vs 141
     isolates the array. */
  if (stage >= 140 && stage <= 150) {
    struct kv5 { const char *z; const char *y; };
    unsigned i; int ok = 0;
    /* CAPSTONE_LADDER_ONLY=<n> compiles ONLY arm <n> into the image.
       Why: every image carrying the whole 140-146 block entry-stalls (R-16) -- d140 and d141
       stalled while f10.dom, built by the SAME script and differing only in which stage block
       compiles, returned in the very same boots. Selector :0 returns before any ladder code
       runs, so the block's mere PRESENCE is what blocks entry: a layout/size effect, not an
       execution one. Narrowing the compiled arm is therefore the minimisation that matters --
       it asks which INGREDIENT of the block (the 2 KB `a[64]` stack array, the struct, or
       neither) carries the property, and an arm that enters is measurable at last. */
#if !defined(CAPSTONE_LADDER_ONLY) || CAPSTONE_LADDER_ONLY == 140
    if (stage == 140) {
      struct kv5 v;
      v.z = "ltrim"; v.y = "aaa0";
      { unsigned a1=0,b1=0; const char *z=v.z,*y=v.y;
        while (z&&z[a1]) a1++; while (y&&y[b1]) b1++;
        if (z&&y&&a1&&b1) ok++; }
      return ok;                               /* expect 1 */
    }
#endif
#if !defined(CAPSTONE_LADDER_ONLY) || CAPSTONE_LADDER_ONLY == 146
    if (stage == 146) {
      const char *p0="ltrim", *p1="rtrim", *p2="trim", *p3="max";
      const char *arr[4]; arr[0]=p0; arr[1]=p1; arr[2]=p2; arr[3]=p3;
      for (i=0;i<4;i++){ unsigned n=0; const char *z=arr[i]; while(z&&z[n])n++; if(z&&n)ok++; }
      return ok;                               /* expect 4 */
    }
#endif
#if !defined(CAPSTONE_LADDER_ONLY) || \
    (CAPSTONE_LADDER_ONLY != 140 && CAPSTONE_LADDER_ONLY != 146)
    {
      struct kv5 a[64];
      unsigned n = 0;
      /* 147/148 separate STORE COUNT from OFFSET SPAN, which arms 141-145 confound: every arm
         that wedges also reaches higher offsets than the one that returns.
           141 -> 2 stores at 0x00,0x10           RETURNS 1   (measured)
           142 -> 4 stores at 0x00..0x30          WEDGES      (measured)
           143 -> 8 stores at 0x00..0x70, ONE literal  WEDGES  (measured; so distinctness is
                                                                NOT the axis)
         147 keeps the count at 2 but moves BOTH stores to high offsets; 148 keeps offsets low
         but raises the count to 3. If 147 wedges the axis is the offset/bounds; if 148 wedges
         and 147 returns, it is the count. */
      /* 149 -- PURE STORE/LOAD ROUND-TRIP. No dereference anywhere, so this arm CANNOT wedge
         on a bad pointer: it converts the failure into a number, per the project rule that a
         diagnostic should turn a hang into a wrong answer.
         Motivation (measured): arm 141 returned 1 correctly in n144.dom (3 boots), WEDGED in
         co.dom attempt 1, and returned 0 in co.dom attempt 3 -- same source, so the fault is
         NONDETERMINISTIC and the earlier "2 stores fine, 4 wedge" boundary was an artifact of
         one image. If a stored capability is sometimes unreadable, this counts exactly how
         often: `good` is how many of 8 slots round-tripped, `nulls` how many came back null.
         Returned as good*10 + nulls so one number carries both. Expect 80. */
      /* 150 -- THE CONTROL FOR 149. Dereference the cap-table literal REPEATEDLY WITHOUT ever
         storing it to the stack array. If 150 returns 8 while 141 (same deref, but through a
         store/load round-trip) faults or returns 0, the damage is in the ROUND-TRIP, not in the
         literal's capability.
         This control is needed because 149's `q == p` compares only the 64-bit cursor: a
         capability can lose its tag or bounds and still compare equal in C, so 149 returning 80
         proves the ADDRESS survived, not the capability. mcause=28 (OUT_OF_BOUNDS) on the
         wedging arms is exactly what a right-address/wrong-bounds capability produces. */
      if (stage == 150) {
        const char *p = "ltrim";
        unsigned k, n; int good = 0;
        for (k = 0; k < 8; k++) { n = 0; while (p && p[n]) n++; if (n > 0) good++; }
        return good;                             /* expect 8 */
      }
      if (stage == 149) {
        const char *p = "ltrim";
        unsigned k; int good = 0, nulls = 0;
        for (k = 0; k < 8; k++) a[k].z = p;
        for (k = 0; k < 8; k++) {
          const char *q = a[k].z;
          if (q == p) good++;
          else if (q == 0) nulls++;
        }
        return good * 10 + nulls;                /* expect 80 */
      }
      if (stage == 147) {                        /* 2 stores, HIGH offsets (0x60,0x70) */
        a[3].z = "ltrim"; a[3].y = "aaa0";
        { unsigned nz=0, ny=0; const char *z=a[3].z, *y=a[3].y;
          while (z && z[nz]) nz++;
          while (y && y[ny]) ny++;
          if (z && y && nz > 0 && ny > 0) ok++; }
        return ok;                               /* expect 1 */
      }
      if (stage == 148) {                        /* 3 stores, LOW offsets (0x00,0x10,0x20) */
        a[0].z = "ltrim"; a[0].y = "aaa0"; a[1].z = "rtrim";
        { unsigned nz=0, ny=0, n1=0; const char *z=a[0].z, *y=a[0].y, *z1=a[1].z;
          while (z && z[nz]) nz++;
          while (y && y[ny]) ny++;
          while (z1 && z1[n1]) n1++;
          if (z && y && nz > 0 && ny > 0) ok++;
          if (z1 && n1 > 0) ok++; }
        return ok;                               /* expect 2 */
      }
      if (stage == 141) { a[0].z="ltrim"; a[0].y="aaa0"; n = 1; }
      else if (stage == 142) { a[0].z="ltrim"; a[0].y="aaa0"; a[1].z="rtrim"; a[1].y="aaa1"; n = 2; }
      else if (stage == 143) { a[0].z="dup"; a[0].y="dup"; a[1].z="dup"; a[1].y="dup";
                               a[2].z="dup"; a[2].y="dup"; a[3].z="dup"; a[3].y="dup"; n = 4; }
      else if (stage == 145) { a[0].z="ltrim"; a[1].z="rtrim"; a[2].z="trim"; a[3].z="max";
                               a[0].y="s"; a[1].y="s"; a[2].y="s"; a[3].y="s"; n = 4; }
      else { a[0].z="ltrim"; a[0].y="aaa0"; a[1].z="rtrim"; a[1].y="aaa1";
             a[2].z="trim";  a[2].y="aaa2"; a[3].z="max";   a[3].y="aaa3"; n = 4; }
      for (i = 0; i < n; i++) {
        unsigned nz=0, ny=0; const char *z=a[i].z, *y=a[i].y;
        while (z && z[nz]) nz++;
        while (y && y[ny]) ny++;
        if (z && y && nz > 0 && ny > 0) ok++;
      }
      return ok;                               /* expect n */
    }
#endif
    /* Reached only when CAPSTONE_LADDER_ONLY excluded the arm the selector asked for. Return
       a DISTINCT sentinel rather than 0: 0 is stage :0's legitimate answer, so sharing it
       would make "the arm was not compiled in" indistinguishable from a successful control. */
    (void)i;
    return 99;
  }
  if (stage == 128 || stage == 129) {
    const char *base = "AxxxxxxxByyyyyyy";   /* [0]='A'  [8]='B' */
    const char *p2 = 0;
    if (stage == 128) {
      const char *p1 = 0;
      /* two INDEPENDENT derivations from `base`: each has rd != rs1 */
      /* cincoffsetimm rd, rs, off  ==  .insn i 0x5b, 0x2, rd, off(rs)
         (encoding taken from start-gp-captable-interp.S:72, not guessed) */
      __asm__ volatile(".insn i 0x5b, 0x2, %0, 0(%2)\n\t"
                       ".insn i 0x5b, 0x2, %1, 8(%2)"
                       : "=&r"(p1), "=&r"(p2) : "r"(base));
      (void)p1;
    } else {
      /* same total offset, chained: rd == rs1 both times */
      const char *q = base;
      __asm__ volatile(".insn i 0x5b, 0x2, %0, 4(%0)\n\t"
                       ".insn i 0x5b, 0x2, %0, 4(%0)"
                       : "+r"(q));
      p2 = q;
    }
    return (int)((unsigned long)(unsigned char)p2[0] & 0xff);   /* expect 0x42 'B' */
  }
  if (stage == 126 || stage == 127) {
    unsigned i; int ok = 0;
    if (stage == 126) {
      struct kv4 { const char *z; const char *y; };
      struct kv4 a[64];
      for (i = 0; i < 64; i++) { a[i].z = "filler"; a[i].y = "fill"; }
      for (i = 0; i < 64; i++) {
        unsigned nz = 0, ny = 0; const char *z = a[i].z, *y = a[i].y;
        while (z && z[nz]) nz++;
        while (y && y[ny]) ny++;
        if (z && y && nz > 0 && ny > 0) ok++;
      }
    } else {
      const char *f[64];
      for (i = 0; i < 64; i++) f[i] = "filler";
      for (i = 0; i < 64; i++) {
        unsigned n = 0; const char *z = f[i];
        while (z && z[n]) n++;
        if (z && n > 0) ok++;
      }
    }
    return ok & 0xff;                        /* expect 64 */
  }
  if (stage >= 120 && stage <= 125) {
    /* NO static table here, on purpose and this time actually: a `static const unsigned
       ns[6]` costs ONE extra cap-table global, which took the image from 181 to 182 carves,
       and every 182-carve image observed so far entry-stalls (8/8). A switch keeps the
       constants in the instruction stream and the carve count unchanged. */
    unsigned n = 4;
    switch (stage) {
      case 120: n = 4;  break;
      case 121: n = 8;  break;
      case 122: n = 16; break;
      case 123: n = 32; break;
      case 124: n = 48; break;
      default:  n = 64; break;
    }
    struct kv3 { const char *z; const char *y; };
    struct kv3 a[64];
    unsigned i; int ok = 0;
    /* straight-line materialisation of DISTINCT literals -- the R-14 variant-A shape */
    a[0].z="ltrim";    a[0].y="aaa0";   a[1].z="rtrim";    a[1].y="aaa1";
    a[2].z="trim";     a[2].y="aaa2";   a[3].z="max";      a[3].y="aaa3";
    if (n > 4) {
      a[4].z="min";    a[4].y="aaa4";   a[5].z="typeof";   a[5].y="aaa5";
      a[6].z="length"; a[6].y="aaa6";   a[7].z="instr";    a[7].y="aaa7";
    }
    if (n > 8) {
      a[8].z="substr"; a[8].y="aaa8";   a[9].z="upper";    a[9].y="aaa9";
      a[10].z="lower"; a[10].y="aab0";  a[11].z="coalesce";a[11].y="aab1";
      a[12].z="hex";   a[12].y="aab2";  a[13].z="unhex";   a[13].y="aab3";
      a[14].z="quote"; a[14].y="aab4";  a[15].z="replace"; a[15].y="aab5";
    }
    for (i = (n > 16 ? 16 : n); i < n; i++) { a[i].z = "filler"; a[i].y = "fill"; }
    for (i = 0; i < n; i++) {
      unsigned nz = 0, ny = 0;
      const char *z = a[i].z, *y = a[i].y;
      while (z && z[nz]) nz++;
      while (y && y[ny]) ny++;
      if (z && y && nz > 0 && ny > 0) ok++;
    }
    return ok & 0xff;                      /* expect n */
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 110 && CAPSTONE_SQLITE_STAGE <= 113
  /* R-14 variants A and B as STAGED probes, so they can run on the board through the same
     host/runner as everything else (the ladder .dom files are not staged in the initramfs).
     Reduced verbatim from tests/fpga-repros/ARCHIVED/R14-strline-struct/.

     Both are QEMU-CLEAN with the C-16 fix (r14a/r14b ladder rungs return 16 at -O0 and -O1),
     while the board previously WEDGED on A and returned 4 for B. These stages re-ask that
     question with the fixed compiler.

     C-16 does NOT apply here: `struct kv` is two capabilities = 32 B with no tail padding,
     and the array is uninitialised then assigned element-by-element, so there is no
     initialiser memset at all.

       110 = variant A (16 straight-line)  -- expect 16 (0x10); board previously WEDGED
       111 = variant B ( 4 straight-line)  -- expect 16 (0x10); board previously returned 4

     111 is the safe/valuable one: it RETURNS a wrong number instead of wedging, so run it
     FIRST and put 110 last (a wedge takes the core with it).  */
  /* 112/113 = the R-14 CONTROLS, which the board reports as CORRECT (16).
       112 = variant D: same straight-line materialisation as A, but a FLAT `const char *[64]`
             instead of a struct. Isolates "struct element type" as a necessary ingredient.
       113 = variant C: same struct as A, but filled in a LOOP from a static table. Isolates
             "straight-line materialisation" as the other necessary ingredient.
     Both are expected to RETURN, so they are safe to run first in a batch. If they still
     return 16 post-C-16 while A and B (110/111) wedge, the remaining fault is pinned to
     straight-line materialisation INTO STRUCT FIELDS specifically -- both ingredients
     required, which is a far sharper target than "struct array init". */
  if (stage == 112) {
    const char *f[64]; unsigned i; int ok = 0;
    f[0]="ltrim"; f[1]="rtrim"; f[2]="trim"; f[3]="max"; f[4]="min"; f[5]="typeof";
    f[6]="length"; f[7]="instr"; f[8]="substr"; f[9]="upper"; f[10]="lower";
    f[11]="coalesce"; f[12]="hex"; f[13]="unhex"; f[14]="quote"; f[15]="replace";
    for (i=16;i<64;i++) f[i]="filler";
    for (i=0;i<16;i++) { unsigned n=0; const char *z=f[i]; while (z && z[n]) n++; if (z && n>0) ok++; }
    return ok;                              /* expect 16 */
  }
  if (stage == 113) {
    static const char *const tbl[16] = {
      "ltrim","rtrim","trim","max","min","typeof","length","instr",
      "substr","upper","lower","coalesce","hex","unhex","quote","replace" };
    struct kv2 { const char *z; const char *y; };
    struct kv2 a[64]; unsigned i; int ok = 0;
    for (i=0;i<16;i++){ a[i].z=tbl[i]; a[i].y="aaa0"; }
    for (i=16;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
    for (i=0;i<16;i++) {
      unsigned nz=0, ny=0; const char *z=a[i].z, *y=a[i].y;
      while (z && z[nz]) nz++;
      while (y && y[ny]) ny++;
      if (z && y && nz>0 && ny>0) ok++;
    }
    return ok;                              /* expect 16 */
  }
  if (stage >= 110 && stage <= 111) {
    struct kv { const char *z; const char *y; };
    struct kv a[64];
    unsigned i;
    int ok = 0;
    a[0].z="ltrim"; a[0].y="aaa0"; a[1].z="rtrim"; a[1].y="aaa1";
    a[2].z="trim";  a[2].y="aaa2"; a[3].z="max";   a[3].y="aaa3";
    if (stage == 110) {
      a[4].z="min";    a[4].y="aaa4";  a[5].z="typeof";   a[5].y="aaa5";
      a[6].z="length"; a[6].y="aaa6";  a[7].z="instr";    a[7].y="aaa7";
      a[8].z="substr"; a[8].y="aaa8";  a[9].z="upper";    a[9].y="aaa9";
      a[10].z="lower"; a[10].y="aab0"; a[11].z="coalesce";a[11].y="aab1";
      a[12].z="hex";   a[12].y="aab2"; a[13].z="unhex";   a[13].y="aab3";
      a[14].z="quote"; a[14].y="aab4"; a[15].z="replace"; a[15].y="aab5";
      for (i = 16; i < 64; i++) { a[i].z = "filler"; a[i].y = "fill"; }
    } else {
      for (i = 4; i < 64; i++) { a[i].z = "filler"; a[i].y = "fill"; }
    }
    for (i = 0; i < 16; i++) {
      unsigned nz = 0, ny = 0;
      const char *z = a[i].z, *y = a[i].y;
      while (z && z[nz]) nz++;
      while (y && y[ny]) ny++;
      if (z && y && nz > 0 && ny > 0) ok++;
    }
    return ok;                            /* expect 16 = 0x10 */
  }
#endif
#if CAPSTONE_SQLITE_STAGE >= 100 && CAPSTONE_SQLITE_STAGE <= 105
#ifndef PROBE_FD_N
#define PROBE_FD_N 56
#endif
  if (stage >= 100 && stage <= 105) {
    /* THREE QUESTIONS THE CURSOR READING CANNOT ANSWER ON ITS OWN.
       Known: arr[N-1].zName is a VALID capability (lcc does not trap) whose bounds fit the
       container but whose cursor is wrong (low byte 0x00; any correct cursor ends in nibble 2).
       Unknown: whether it is stored wrong or READ wrong, and by how much.
         100: (arr[N-1].zName - arr[0].zName) low byte. Self-referencing -- needs no knowledge
              of the runtime base. "fn0" is at container offset 0 and "fn55" at 3906, so a
              correct delta is 3906 & 0xff = 0x42. This measures the ERROR directly.
         101: read the SAME slot twice and report whether the two reads AGREE
              (0xB0 = agree, 0xB1 = differ). Differing reads mean the value is corrupted on
              READ; agreeing reads mean it was stored wrong. That is the store-vs-load split,
              and nothing measured so far distinguishes them.
         102: same delta as 100 but for arr[N-2] - arr[0], i.e. the LAST GOOD entry. Expect
              the correct offset for "fn54"; confirms the neighbour is right and that the
              measurement method itself is sound. */
    struct probe_fd { const char *zName; void *p1; void *p2; unsigned char flags; };
    struct probe_fd arr[] = {
      { "fn0", (void *)0, (void *)0, (unsigned char)0 },
#if PROBE_FD_N > 1
      { "fn1", (void *)0, (void *)0, (unsigned char)1 },
#endif
#if PROBE_FD_N > 2
      { "fn2", (void *)0, (void *)0, (unsigned char)2 },
#endif
#if PROBE_FD_N > 3
      { "fn3", (void *)0, (void *)0, (unsigned char)3 },
#endif
#if PROBE_FD_N > 4
      { "fn4", (void *)0, (void *)0, (unsigned char)4 },
#endif
#if PROBE_FD_N > 5
      { "fn5", (void *)0, (void *)0, (unsigned char)5 },
#endif
#if PROBE_FD_N > 6
      { "fn6", (void *)0, (void *)0, (unsigned char)6 },
#endif
#if PROBE_FD_N > 7
      { "fn7", (void *)0, (void *)0, (unsigned char)7 },
#endif
#if PROBE_FD_N > 8
      { "fn8", (void *)0, (void *)0, (unsigned char)8 },
#endif
#if PROBE_FD_N > 9
      { "fn9", (void *)0, (void *)0, (unsigned char)9 },
#endif
#if PROBE_FD_N > 10
      { "fn10", (void *)0, (void *)0, (unsigned char)10 },
#endif
#if PROBE_FD_N > 11
      { "fn11", (void *)0, (void *)0, (unsigned char)11 },
#endif
#if PROBE_FD_N > 12
      { "fn12", (void *)0, (void *)0, (unsigned char)12 },
#endif
#if PROBE_FD_N > 13
      { "fn13", (void *)0, (void *)0, (unsigned char)13 },
#endif
#if PROBE_FD_N > 14
      { "fn14", (void *)0, (void *)0, (unsigned char)14 },
#endif
#if PROBE_FD_N > 15
      { "fn15", (void *)0, (void *)0, (unsigned char)15 },
#endif
#if PROBE_FD_N > 16
      { "fn16", (void *)0, (void *)0, (unsigned char)16 },
#endif
#if PROBE_FD_N > 17
      { "fn17", (void *)0, (void *)0, (unsigned char)17 },
#endif
#if PROBE_FD_N > 18
      { "fn18", (void *)0, (void *)0, (unsigned char)18 },
#endif
#if PROBE_FD_N > 19
      { "fn19", (void *)0, (void *)0, (unsigned char)19 },
#endif
#if PROBE_FD_N > 20
      { "fn20", (void *)0, (void *)0, (unsigned char)20 },
#endif
#if PROBE_FD_N > 21
      { "fn21", (void *)0, (void *)0, (unsigned char)21 },
#endif
#if PROBE_FD_N > 22
      { "fn22", (void *)0, (void *)0, (unsigned char)22 },
#endif
#if PROBE_FD_N > 23
      { "fn23", (void *)0, (void *)0, (unsigned char)23 },
#endif
#if PROBE_FD_N > 24
      { "fn24", (void *)0, (void *)0, (unsigned char)24 },
#endif
#if PROBE_FD_N > 25
      { "fn25", (void *)0, (void *)0, (unsigned char)25 },
#endif
#if PROBE_FD_N > 26
      { "fn26", (void *)0, (void *)0, (unsigned char)26 },
#endif
#if PROBE_FD_N > 27
      { "fn27", (void *)0, (void *)0, (unsigned char)27 },
#endif
#if PROBE_FD_N > 28
      { "fn28", (void *)0, (void *)0, (unsigned char)28 },
#endif
#if PROBE_FD_N > 29
      { "fn29", (void *)0, (void *)0, (unsigned char)29 },
#endif
#if PROBE_FD_N > 30
      { "fn30", (void *)0, (void *)0, (unsigned char)30 },
#endif
#if PROBE_FD_N > 31
      { "fn31", (void *)0, (void *)0, (unsigned char)31 },
#endif
#if PROBE_FD_N > 32
      { "fn32", (void *)0, (void *)0, (unsigned char)32 },
#endif
#if PROBE_FD_N > 33
      { "fn33", (void *)0, (void *)0, (unsigned char)33 },
#endif
#if PROBE_FD_N > 34
      { "fn34", (void *)0, (void *)0, (unsigned char)34 },
#endif
#if PROBE_FD_N > 35
      { "fn35", (void *)0, (void *)0, (unsigned char)35 },
#endif
#if PROBE_FD_N > 36
      { "fn36", (void *)0, (void *)0, (unsigned char)36 },
#endif
#if PROBE_FD_N > 37
      { "fn37", (void *)0, (void *)0, (unsigned char)37 },
#endif
#if PROBE_FD_N > 38
      { "fn38", (void *)0, (void *)0, (unsigned char)38 },
#endif
#if PROBE_FD_N > 39
      { "fn39", (void *)0, (void *)0, (unsigned char)39 },
#endif
#if PROBE_FD_N > 40
      { "fn40", (void *)0, (void *)0, (unsigned char)40 },
#endif
#if PROBE_FD_N > 41
      { "fn41", (void *)0, (void *)0, (unsigned char)41 },
#endif
#if PROBE_FD_N > 42
      { "fn42", (void *)0, (void *)0, (unsigned char)42 },
#endif
#if PROBE_FD_N > 43
      { "fn43", (void *)0, (void *)0, (unsigned char)43 },
#endif
#if PROBE_FD_N > 44
      { "fn44", (void *)0, (void *)0, (unsigned char)44 },
#endif
#if PROBE_FD_N > 45
      { "fn45", (void *)0, (void *)0, (unsigned char)45 },
#endif
#if PROBE_FD_N > 46
      { "fn46", (void *)0, (void *)0, (unsigned char)46 },
#endif
#if PROBE_FD_N > 47
      { "fn47", (void *)0, (void *)0, (unsigned char)47 },
#endif
#if PROBE_FD_N > 48
      { "fn48", (void *)0, (void *)0, (unsigned char)48 },
#endif
#if PROBE_FD_N > 49
      { "fn49", (void *)0, (void *)0, (unsigned char)49 },
#endif
#if PROBE_FD_N > 50
      { "fn50", (void *)0, (void *)0, (unsigned char)50 },
#endif
#if PROBE_FD_N > 51
      { "fn51", (void *)0, (void *)0, (unsigned char)51 },
#endif
#if PROBE_FD_N > 52
      { "fn52", (void *)0, (void *)0, (unsigned char)52 },
#endif
#if PROBE_FD_N > 53
      { "fn53", (void *)0, (void *)0, (unsigned char)53 },
#endif
#if PROBE_FD_N > 54
      { "fn54", (void *)0, (void *)0, (unsigned char)54 },
#endif
#if PROBE_FD_N > 55
      { "fn55", (void *)0, (void *)0, (unsigned char)55 },
#endif
#if PROBE_FD_N > 56
      { "fn56", (void *)0, (void *)0, (unsigned char)56 },
#endif
#if PROBE_FD_N > 57
      { "fn57", (void *)0, (void *)0, (unsigned char)57 },
#endif
#if PROBE_FD_N > 58
      { "fn58", (void *)0, (void *)0, (unsigned char)58 },
#endif
#if PROBE_FD_N > 59
      { "fn59", (void *)0, (void *)0, (unsigned char)59 },
#endif
#if PROBE_FD_N > 60
      { "fn60", (void *)0, (void *)0, (unsigned char)60 },
#endif
#if PROBE_FD_N > 61
      { "fn61", (void *)0, (void *)0, (unsigned char)61 },
#endif
#if PROBE_FD_N > 62
      { "fn62", (void *)0, (void *)0, (unsigned char)62 },
#endif
#if PROBE_FD_N > 63
      { "fn63", (void *)0, (void *)0, (unsigned char)63 },
#endif
#if PROBE_FD_N > 64
      { "fn64", (void *)0, (void *)0, (unsigned char)64 },
#endif
#if PROBE_FD_N > 65
      { "fn65", (void *)0, (void *)0, (unsigned char)65 },
#endif
#if PROBE_FD_N > 66
      { "fn66", (void *)0, (void *)0, (unsigned char)66 },
#endif
#if PROBE_FD_N > 67
      { "fn67", (void *)0, (void *)0, (unsigned char)67 },
#endif
#if PROBE_FD_N > 68
      { "fn68", (void *)0, (void *)0, (unsigned char)68 },
#endif
#if PROBE_FD_N > 69
      { "fn69", (void *)0, (void *)0, (unsigned char)69 },
#endif
#if PROBE_FD_N > 70
      { "fn70", (void *)0, (void *)0, (unsigned char)70 },
#endif
#if PROBE_FD_N > 71
      { "fn71", (void *)0, (void *)0, (unsigned char)71 },
#endif
#if PROBE_FD_N > 72
      { "fn72", (void *)0, (void *)0, (unsigned char)72 },
#endif
#if PROBE_FD_N > 73
      { "fn73", (void *)0, (void *)0, (unsigned char)73 },
#endif
#if PROBE_FD_N > 74
      { "fn74", (void *)0, (void *)0, (unsigned char)74 },
#endif
#if PROBE_FD_N > 75
      { "fn75", (void *)0, (void *)0, (unsigned char)75 },
#endif
#if PROBE_FD_N > 76
      { "fn76", (void *)0, (void *)0, (unsigned char)76 },
#endif
#if PROBE_FD_N > 77
      { "fn77", (void *)0, (void *)0, (unsigned char)77 },
#endif
#if PROBE_FD_N > 78
      { "fn78", (void *)0, (void *)0, (unsigned char)78 },
#endif
#if PROBE_FD_N > 79
      { "fn79", (void *)0, (void *)0, (unsigned char)79 },
#endif
#if PROBE_FD_N > 80
      { "fn80", (void *)0, (void *)0, (unsigned char)80 },
#endif
#if PROBE_FD_N > 81
      { "fn81", (void *)0, (void *)0, (unsigned char)81 },
#endif
#if PROBE_FD_N > 82
      { "fn82", (void *)0, (void *)0, (unsigned char)82 },
#endif
#if PROBE_FD_N > 83
      { "fn83", (void *)0, (void *)0, (unsigned char)83 },
#endif
#if PROBE_FD_N > 84
      { "fn84", (void *)0, (void *)0, (unsigned char)84 },
#endif
#if PROBE_FD_N > 85
      { "fn85", (void *)0, (void *)0, (unsigned char)85 },
#endif
#if PROBE_FD_N > 86
      { "fn86", (void *)0, (void *)0, (unsigned char)86 },
#endif
#if PROBE_FD_N > 87
      { "fn87", (void *)0, (void *)0, (unsigned char)87 },
#endif
#if PROBE_FD_N > 88
      { "fn88", (void *)0, (void *)0, (unsigned char)88 },
#endif
#if PROBE_FD_N > 89
      { "fn89", (void *)0, (void *)0, (unsigned char)89 },
#endif
#if PROBE_FD_N > 90
      { "fn90", (void *)0, (void *)0, (unsigned char)90 },
#endif
#if PROBE_FD_N > 91
      { "fn91", (void *)0, (void *)0, (unsigned char)91 },
#endif
#if PROBE_FD_N > 92
      { "fn92", (void *)0, (void *)0, (unsigned char)92 },
#endif
#if PROBE_FD_N > 93
      { "fn93", (void *)0, (void *)0, (unsigned char)93 },
#endif
#if PROBE_FD_N > 94
      { "fn94", (void *)0, (void *)0, (unsigned char)94 },
#endif
#if PROBE_FD_N > 95
      { "fn95", (void *)0, (void *)0, (unsigned char)95 },
#endif
    };
    unsigned n = (unsigned)(sizeof(arr) / sizeof(arr[0]));
    /* 103/104/105 ADDED 2026-08-02, after QEMU showed stages 100-102 do `cincoffset` on an
       UNTAGGED register (helper_cscincoffset assertion) and are therefore not a trustworthy
       instrument. These three separate "is the instrument broken" from "is the array broken",
       and none of them uses inline-asm lcc.
         103: build the array and return n. Touches NO pointer at all. If QEMU still asserts
              here, the untagged cincoffset is in the ARRAY CONSTRUCTION -- a real miscompile,
              not an instrument fault -- and that is the finding. If it runs clean, the fault
              is in the 100-102 read path and only the instrument was broken.
         104: delta arr[n-1].zName - arr[0].zName via INTEGER CASTS. Casting a capability to
              unsigned long yields its cursor without any capability arithmetic, so this is
              the 100 measurement with the broken part removed.
         105: same, but arr[n-2] -- the neighbour control for 104. */
    if (stage == 103)
      return (int)(n & 0xffu);
    if (stage == 104 || stage == 105) {
      unsigned long a0 = (unsigned long)(arr[0].zName);
      unsigned long ax = (unsigned long)(arr[(stage == 105) ? (n - 2) : (n - 1)].zName);
      return (int)((ax - a0) & 0xffUL);
    }
    if (stage == 101) {
      const char *volatile a = arr[n - 1].zName;
      const char *volatile b = arr[n - 1].zName;
      unsigned long ua = 0, ub = 0;
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(ua) : "r"(a));
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(ub) : "r"(b));
      return (int)(0xB0u | ((ua == ub) ? 0u : 1u));
    }
    {
      const char *volatile base = arr[0].zName;
      const char *volatile tgt  = arr[(stage == 102) ? (n - 2) : (n - 1)].zName;
      unsigned long ub = 0, ut = 0;
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(ub) : "r"(base));
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(ut) : "r"(tgt));
      return (int)((ut - ub) & 0xffUL);
    }
  }
#endif
  if (stage <= 0)
    return 0;
  rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK || stage <= 1)
    return rc;
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK || stage <= 2)
    return rc;
  rc = sqlite3_open(":memory:", &db);
  return rc;
}
#endif

static int run_sqlite(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *statement = 0;
#ifdef CAPSTONE_PRINT_LOAD_BASE
  /* Print the RUNTIME address of a known global symbol, so a fault pc reported by QEMU or the
   * monitor can be mapped back to an image offset and named in the disassembly.
   *
   * Without this a fault pc is unusable: the domain is loaded at a base the image does not
   * record, `SQ: self=` is an ENCODED capability rather than an address, and the only other
   * bound available is "somewhere inside 1.4 MB of .text". With it, one run gives
   *     image_VA(fault) = fault_pc - (printed - VA_of_symbol_from_readelf)
   * and the faulting instruction can be read straight out of llvm-objdump.
   *
   * Casting the function pointer to an integer yields the capability's cursor, i.e. the plain
   * address. Printed as two 32-bit halves because output_uint is 32-bit and the address is not.
   * Diagnostic only, and it perturbs the image, so the base must be read from the SAME build
   * whose fault pc is being mapped. */
  {
    unsigned long a_ = (unsigned long)(void *)&sqlite3_initialize;
    output_text("LOADBASE sqlite3_initialize=");
    output_uint((unsigned)(a_ >> 32));
    output_text(":");
    output_uint((unsigned)(a_ & 0xFFFFFFFFu));
    output_text("\n");
  }
#endif
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                          (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK)
    return fail("config-heap", rc, 0);

  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc, 0);

  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK)
    return fail("open", rc, db);

  rc = sqlite3_exec(db,
      "CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);",
      0, 0, 0);
  if (rc != SQLITE_OK)
    return fail("create", rc, db);

  rc = sqlite3_exec(db,
      "INSERT INTO items VALUES"
      "('alpha',11),('beta',22),('gamma',33);",
      0, 0, 0);
  if (rc != SQLITE_OK)
    return fail("insert", rc, db);

  rc = sqlite3_prepare_v2(
      db, "SELECT name,value FROM items;", -1, &statement, 0);
  if (rc != SQLITE_OK)
    return fail("prepare", rc, db);

  static const int expected_values[] = {11, 22, 33};
  unsigned row = 0;
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    const unsigned char *name = sqlite3_column_text(statement, 0);
    int value = sqlite3_column_int(statement, 1);
    const char *expected_name = 0;
    if (row == 0)
      expected_name = "alpha";
    else if (row == 1)
      expected_name = "beta";
    else if (row == 2)
      expected_name = "gamma";
    if (row >= 3 || !name ||
        sqlite3_stricmp((const char *)name, expected_name) != 0 ||
        value != expected_values[row])
      return fail("row-value", SQLITE_MISMATCH, db);
    output_text("row name=");
    output_text((const char *)name);
    output_text(" value=");
    output_uint((unsigned)value);
    output_text("\n");
    ++row;
  }

  if (rc != SQLITE_DONE || row != 3)
    return fail("step", rc, db);
  rc = sqlite3_finalize(statement);
  statement = 0;
  if (rc != SQLITE_OK)
    return fail("finalize", rc, db);

  /* Extended workload (indexes, joins, aggregates, mutations, string funcs) to
   * exercise more capability-sensitive SQLite machinery. */
  rc = run_sqlite_extended(db);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_close(db);
  db = 0;
  if (rc != SQLITE_OK)
    return fail("close", rc, 0);

  output_text("__CAPSTONE_SQLITE_MEMORY_PASSED__\n");
  return 0;
}

#if defined(CAPSTONE_SQLITE_QUICKRET) && (CAPSTONE_SQLITE_QUICKRET) >= 15 \
                                       && (CAPSTONE_SQLITE_QUICKRET) <= 41 \
                                       && (CAPSTONE_SQLITE_QUICKRET) != 17 \
                                       && (CAPSTONE_SQLITE_QUICKRET) != 25
/* Stand-ins for renameColumnFunc & co., used by QUICKRET levels 15-16 and 18-22. Distinct bodies so
   none of them folds into another and the array keeps nine DIFFERENT function pointers --
   the whole point of the construct. Never called; only their addresses matter. */
#define QR_CLONE_FN(n, k) \
  static void qrCloneFunc##n(sqlite3_context *c, int a, sqlite3_value **v) { \
    (void)c; (void)v; if (a == (k)) sqlite3_result_int(c, (k)); }
QR_CLONE_FN(0, 101) QR_CLONE_FN(1, 102) QR_CLONE_FN(2, 103)
QR_CLONE_FN(3, 104) QR_CLONE_FN(4, 105) QR_CLONE_FN(5, 106)
QR_CLONE_FN(6, 107) QR_CLONE_FN(7, 108) QR_CLONE_FN(8, 109)
#undef QR_CLONE_FN
#endif

#if defined(CAPSTONE_SQLITE_QUICKRET) && \
    ((CAPSTONE_SQLITE_QUICKRET) == 21 || (CAPSTONE_SQLITE_QUICKRET) == 22 || \
     (CAPSTONE_SQLITE_QUICKRET) == 23 || (CAPSTONE_SQLITE_QUICKRET) == 30)
/* Level 21's array and callee. The array is file-scope rather than block-scope only so the
   callee can take it by pointer the way sqlite3InsertBuiltinFuncs does; it is otherwise
   identical to level 19's, including being non-const. */
#if (CAPSTONE_SQLITE_QUICKRET) == 21
static FuncDef qrSplitClone21[] = {
  INTERNAL_FUNCTION(qr21_rename_column,   9, qrCloneFunc0),
  INTERNAL_FUNCTION(qr21_rename_table,    7, qrCloneFunc1),
  INTERNAL_FUNCTION(qr21_rename_test,     7, qrCloneFunc2),
  INTERNAL_FUNCTION(qr21_drop_column,     3, qrCloneFunc3),
  INTERNAL_FUNCTION(qr21_rename_quotefix, 2, qrCloneFunc4),
  INTERNAL_FUNCTION(qr21_drop_constraint, 2, qrCloneFunc5),
  INTERNAL_FUNCTION(qr21_fail,            2, qrCloneFunc6),
  INTERNAL_FUNCTION(qr21_add_constraint,  3, qrCloneFunc7),
  INTERNAL_FUNCTION(qr21_find_constraint, 2, qrCloneFunc8),
};
#endif

/* Level 19's loop, unchanged, reached through a pointer PARAMETER. This is the single
   remaining difference between qr20 (returns) and qr16n (wedges). */
__attribute__((noinline))
static void qr_link_via_param(FuncDef *aDef, int nDef) {
  int i;
  for (i = 0; i < nDef; i++) {
    const char *z = aDef[i].zName;
    int h = SQLITE_FUNC_HASH(z[0], sqlite3Strlen30(z));
    aDef[i].pNext = sqlite3BuiltinFunctions.a[h];
    sqlite3BuiltinFunctions.a[h] = &aDef[i];
  }
}

/* LEVEL 23's callee: byte-for-byte level 22's, except the strlen is a LOCAL loop instead of
   a call to sqlite3Strlen30, which makes the callee a LEAF.
   Why here and not in the fdreg rung: the rung has now failed to reproduce under FOUR
   structural hypotheses -- the construct itself, gp index > 127, an argument-derived
   capability, and a non-leaf callee -- returning its oracle every time (2456/2609/2736,
   2769, 2609, 2609). Building UP from a 12-global image is not converging. The reproduction
   lives inside the SQLite image, where the level-19/level-22 pair separates reliably (19
   returns on two draws, 22 wedges on two), so the bisection belongs THERE.
   Level 23 vs level 22 is leaf vs non-leaf with everything else held fixed:
       23 returns and 22 wedges -> the inner call is the trigger, in this image
       both wedge                -> the call itself is, and leafness is irrelevant

   BOARD 2026-08-06, boot 12, control k800 green, all four created AND entered
   (A/dom-ok=3, G/enter=3, SHA5=7, SHA6=7): the FIRST case.

       k800   returned    q19a   returned (obs 0x9E331313)
       q23a   returned (obs 0x9E331717)          q22a   WEDGED

   Inside this image, with the array, its declaration site, the search call, the padding
   and the global count all held fixed:
       loop inline              -> returns  (level 19, and on a second draw)
       noinline LEAF callee     -> returns  (level 23)
       noinline NON-LEAF callee -> WEDGES   (level 22, and on a second draw)

   The difference between 22 and 23 is ONE LINE -- sqlite3Strlen30(z) versus a local
   `while (z[n]) n++`. And the direction rules out size: the LEAF callee is BIGGER (110
   instructions against 88) and still returns, so it is the inner CALL, not the code.

   RETRACTED the same day, by returning code in this very image. `sqlite3FunctionSearch`
   is ITSELF a noinline NON-LEAF callee -- 1 inner call, 6 stc / 10 ldc capability spills --
   and the shared qi loop calls it NINE TIMES on levels 18, 19, 20 and 23, all of which
   RETURN. So "a non-leaf callee wedges" is false even inside the SQLite image, not merely
   as a general statement. Whatever separates level 22 from level 23 is narrower than "it
   has a call", and is NOT yet identified.

   The pair also is not the controlled experiment it looked like. Level 23 shifts 1791
   functions by 108 bytes relative to level 22 (domain_main grows 20 B for the added `lvl ==
   23` test, the leaf callee is +88 B). The only layout control actually run was QR_DRAW
   d4-vs-d8, a 16-byte shift -- 6.75x smaller -- and S01 documents nine image perturbations
   in this exact image class, one of them a dead never-called empty function, ALL of which
   hung. So layout is not excluded.

   What is still solid: levels 15/18/19/20/23 return and 16/21/22 wedge, every verdict with
   A/dom-ok AND G/enter present and no monitor tag, level 22 across two draws and three runs
   including first position. The OBSERVATION stands; the MECHANISM does not. */
/* A call that touches nothing: level 30 uses it to keep the jalr while dropping the two
   capability loads that level 28 removed along with the call. */
__attribute__((noinline))
static int qr_touch_nothing(void) { return 1; }

__attribute__((noinline))
static void qr_link_via_param_leaf(FuncDef *aDef, int nDef) {
  int i;
  for (i = 0; i < nDef; i++) {
    const char *z = aDef[i].zName;
    int n = 0;
    while (z[n]) n++;                      /* local, so no call: LEAF */
    aDef[i].pNext = sqlite3BuiltinFunctions.a[SQLITE_FUNC_HASH(z[0], n)];
    sqlite3BuiltinFunctions.a[SQLITE_FUNC_HASH(z[0], n)] = &aDef[i];
  }
}
#endif

/* QR_DRAW -- R-16 REDRAW knob. The entry stall is PER-IMAGE and deterministic per binary
   (x101 stalled 6/6, r112 3/3), so retrying a stalling image is futile; the remedy is to draw
   a DIFFERENT image whose code under test is byte-identical. These nops do exactly that: they
   shift layout and nothing else, they are emitted before any probe runs, and they cannot alter
   semantics. Vary QR_DRAW until the domain enters, and sha256sum the set -- two draws that
   hash the same are the same ticket. Applied EQUALLY to both halves of a level pair, or the
   pair stops being a pair. */
#ifndef QR_DRAW
#define QR_DRAW 0
#endif
#define QR_DRAW_STR2(x) #x
#define QR_DRAW_STR(x) QR_DRAW_STR2(x)

void domain_main(unsigned *res, unsigned func) {
#if (QR_DRAW) > 0
  __asm__ volatile(".rept " QR_DRAW_STR(QR_DRAW) "\n\tnop\n\t.endr" ::: "memory");
#endif
  /* DIAGNOSTIC (CAPSTONE_DIAG_FUNC): report the entry argument instead of acting on it.
     The first share entry was observed dying deep inside SQLite's VFS setup, which is only
     reachable when `func` is not REGION_SHARE -- i.e. the domain ran the whole database on
     an entry that should merely stash a capability. Writing func into the shared region
     lets the HOST print what actually arrived, rather than inferring it from where the
     domain crashed. */
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)res;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)res;
    ++shared_region_count;
    return;
  }

#ifdef CAPSTONE_DIAG_FUNC
  /* Report the entry argument on the NON-share path instead of running the database.
     Reaching here on the first entry is the anomaly: it means `func` did not arrive as
     REGION_SHARE, and the domain went on to run all of SQLite on an entry that should
     merely stash a capability (dying in VFS setup). Reporting the value distinguishes
     "argument corrupted" from "dispatch logic wrong". */
  if (res)
    *res = 0xDEAD0000u | (func & 0xffffu);
  return;
#endif
  if (hostcall_metadata)
    hostcall_metadata->length = 0;
#ifdef CAPSTONE_SQLITE_STAGE
  /* STAGED BISECTION. Run only the first N steps of run_sqlite() and RETURN, writing a
     marker the host can print.
     Why staged returns rather than more wedge probes: a wedge produces no output at all,
     so every failed board run says only "somewhere after SQ: G/enter" and costs a whole
     session to learn one bit. Six sessions were spent that way narrowing inside strlen,
     which the clamp experiment then showed was not even spinning. A build that RETURNS
     always yields a result, so the bisection converges instead of guessing.
     Marker: 0x5A6E_ssrr -- ss = stage reached, rr = the SQLite rc at that point. */
  /* RUNTIME PROBE SELECTION. run_sqlite_staged() already dispatches on a runtime `stage`;
     only this call site was a compile-time constant, which is what forced one binary per
     probe. The host may publish a selector in the shared region's `opcode` field, magic-
     guarded (0x5A6E00nn) so a zeroed/unset region falls back to the built-in stage and every
     existing build behaves byte-identically. Only stages compiled into THIS image can be
     selected -- the #if blocks still gate what exists -- so the useful grouping is a range
     that shares one block (e.g. 100-102). */
  {
    unsigned stage_sel = (unsigned)(CAPSTONE_SQLITE_STAGE);
    if (hostcall_metadata) {
      sqlite_hostcall_u64_t sel = hostcall_metadata->opcode;
      if ((sel & 0xffffff00UL) == 0x5A6E0000UL)
        stage_sel = (unsigned)(sel & 0xffUL);
    }
    *res = 0x5A6E0000u | (stage_sel << 8) |
           ((unsigned)run_sqlite_staged((int)stage_sel) & 0xffu);
  }
  return;
#endif
#if defined(CAPSTONE_SQLITE_QUICKRET) && (CAPSTONE_SQLITE_QUICKRET) > 0
  /* GRADUATED WORKLOAD LADDER. Level N runs the first N phases of run_sqlite() and returns
     0x9E33_LLrr (LL = level reached, rr = the SQLite rc). Mirrors run_sqlite()'s own order so
     the phases are the real ones, not a re-implementation:
        1 sqlite3_config(HEAP)   -- first touch of the 256 KB sqlite_heap
        2 sqlite3_initialize()
        3 sqlite3_open(":memory:")
        4 CREATE TABLE
        5 INSERT
     Run ascending in ONE boot: the first level that fails to return is the answer, and every
     level below it is a positive result. This is the bisection the blocker has never had --
     previously the only knob was "run everything" against a 90 s budget.
     Deliberately NOT the staged dispatch: that adds 2 globals and ~13 KB of .text, shifts the
     globals base 0x150000 -> 0x160000, and the resulting image dies in region-share #1. */
  {
    unsigned lvl = (unsigned)(CAPSTONE_SQLITE_QUICKRET);
    sqlite3 *qdb = 0;
    int qrc = SQLITE_OK;
    if (lvl >= 1 && lvl <= 5)
      qrc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                           (int)sizeof(sqlite_heap), 64);
    if (lvl >= 2 && lvl <= 5 && qrc == SQLITE_OK)
      qrc = sqlite3_initialize();
    if (lvl >= 3 && lvl <= 5 && qrc == SQLITE_OK)
      qrc = sqlite3_open(":memory:", &qdb);
    if (lvl >= 4 && lvl <= 5 && qrc == SQLITE_OK)
      qrc = sqlite3_exec(qdb,
          "CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);", 0, 0, 0);
    if (lvl >= 5 && lvl <= 5 && qrc == SQLITE_OK)
      qrc = sqlite3_exec(qdb,
          "INSERT INTO items VALUES('alpha',11),('beta',22),('gamma',33);", 0, 0, 0);
    /* LEVELS 6-9 split sqlite3_initialize() at ITS OWN internal boundaries, mirroring
       run_sqlite_staged()'s stages 7-10 but reachable WITHOUT the staged dispatch (which adds
       globals, shifts the globals base 64 KB, and produces an image that dies in region-share
       #1). Level 2 wedges and level 1 returns rc=0, so the fault is inside initialize(); these
       four say which step. Each re-does CONFIG_HEAP first because memsys5 is the allocator the
       later steps depend on -- same reason the staged version does.
       Callable because this file is #included into the amalgamation TU, so the SQLITE_PRIVATE
       symbols are in scope. */
    /* LEVELS 10-12 split sqlite3PcacheInitialize() itself, which level 8 showed to wedge.
       Its body is three separable things (amalgamation sqlite3-capstone.c:57063):
           read  sqlite3GlobalConfig.pcache2.xInit          -> level 10
           call  sqlite3PCacheSetDefault() if that was 0    -> level 11
           call  through the pointer: xInit(pArg)           -> level 12 re-reads it after
       sqlite3PCacheSetDefault materialises `static const sqlite3_pcache_methods2
       defaultMethods` -- an aggregate holding THIRTEEN function pointers -- and hands it to
       sqlite3_config(SQLITE_CONFIG_PCACHE2,...). On the gp-captable ABI that aggregate is a
       global whose pointer members are capability leaves initialised by __capstone_cap_init,
       which is the same shape the old notes attribute to RegisterBuiltinFunctions (a large
       array of FuncDef structs each holding a zName pointer). If pointer-bearing static
       aggregates are the failing construct, that is a far better lead than "indirect call".
       All three re-do CONFIG_HEAP + MallocInit first, matching level 7, which returns rc=0. */
    /* LEVELS 13-14 split sqlite3RegisterBuiltinFunctions itself, which is the localized wedge
       (qr9). Its body, with SQLITE_STATIC_BUILTINS=1, is: sqlite3AlterFunctions(); a copy loop
       aBuiltinFunc[i] = capstoneBuiltinFunc[i] over FuncDef structs holding pointers; a second
       loop touching zName/pUserData; sqlite3WindowFunctions/DateTime/Json; and finally
       sqlite3InsertBuiltinFuncs. These call the separable registrators directly.
       Both re-do CONFIG_HEAP + MallocInit first, matching level 7 which returns rc=0. */
    /* LEVELS 15-17 ask WHICH HALF of sqlite3AlterFunctions matters -- the construct, or the
       image it sits in. The off-SQLite reproducer (silicon-ladder rung `fdreg`) rebuilt that
       function's construct exactly -- a static array of {name string, function pointer}
       records, its cap-init, storing &arr[i] into a global bucket table, and calling through
       the function-pointer field -- and ALL THREE of its stages RETURN on this silicon
       (2456/2609/2736, control k800 = 4, 2026-08-06). So the operations are fine in a
       12-global image and something about THIS image is not.
       These three put the same construct back inside the SQLite image, ascending:
           15  a local clone: own array, own dummy functions, own bucket table. No SQLite
               call at all, so a wedge here is purely "the construct, in this image".
           16  the same clone, linked with the REAL sqlite3InsertBuiltinFuncs, so the global
               function hash and sqlite3Strlen30/sqlite3FunctionSearch join in.
           17  the real sqlite3AlterFunctions(), i.e. level 13's payload.
       None of them initialises the heap. Level 13 does CONFIG_HEAP + MallocInit first, but
       InsertBuiltinFuncs never allocates, so dropping it removes a variable rather than
       adding one -- and if 17 returns where 13 wedged, the heap init is implicated instead. */
    /* LEVELS 18-20 split sqlite3InsertBuiltinFuncs, which the qr15/qr16 pair localized.
       BOARD 2026-08-06, control k800 green, one boot: qr15 RETURNED in 4s, qr16 WEDGED
       (created and entered -- `SQ: A/dom-ok`, no monitor tag). Same clone array, same
       cap-init, same nine function pointers; the ONLY difference is that qr15 links the
       elements with a hand-written loop over its own bucket table and qr16 hands them to
       the real sqlite3InsertBuiltinFuncs. So the construct is cleared even inside the full
       SQLite image, and the fault is in what that function does differently:
           a) it calls sqlite3FunctionSearch, which walks the REAL global
              sqlite3BuiltinFunctions and compares with sqlite3StrICmp;
           b) it stores the derived capability &aDef[i] into that REAL global rather than
              into a bucket table of ours;
           c) it writes aDef[i].u.pHash -- a UNION member, where qr15 touched only pNext.
       Ascending, so the first level that fails to return is the answer:
           18  read-only: call sqlite3FunctionSearch for every name, accumulate how many
               were found, store nothing. Isolates (a).
           19  18 + link into the REAL sqlite3BuiltinFunctions.a[h], pNext only. Adds (b).
           20  the exact InsertBuiltinFuncs body, inlined, including u.pHash. Adds (c),
               and must reproduce qr16 or the split is not faithful. */
    /* LEVEL 21 -- the last difference. Board 2026-08-06: qr20 RETURNED (0x9E331414, rc 20)
       while qr16n WEDGED, and qr20 inlines the exact InsertBuiltinFuncs body -- union
       u.pHash write, pOther branch and all. So the union write, the branch, the real global
       and the search are ALL cleared, and what remains is that the real function is a CALL
       taking the array as a PARAMETER: inside it, `&aDef[i]` is derived from an argument
       capability instead of from gp in the caller.
       This level is level 19's loop verbatim, moved behind a noinline callee reached through
       a pointer parameter. noinline is load-bearing: if it inlines, the derivation goes back
       through gp and the level silently becomes level 19. Check the disassembly for a real
       call rather than trusting the attribute. */
#if defined(CAPSTONE_SQLITE_QUICKRET) && (CAPSTONE_SQLITE_QUICKRET) == 21
    if (lvl == 21) {
      qr_link_via_param(qrSplitClone21, ArraySize(qrSplitClone21));
      qrc = (int)lvl;
    }
#endif
    /* LEVEL 22 -- qr21 done properly. qr21 WEDGED where qr19 returned, but it changed THREE
       things at once: it added the noinline pointer-parameter call, it dropped the
       sqlite3FunctionSearch call, and it introduced a NEW file-scope 9-element FuncDef global.
       That third one matters here -- a new global shifts the image, and image perturbation
       moving a failure is a documented family in this project (S01).
       And the off-SQLite pair has since cleared the shape outright: fdreg stage 4 (noinline
       callee taking the array by pointer) RETURNS 2609 on silicon, as do stage 5 (noinline,
       reads via gp) and stage 2 (inline). So "an argument capability wedges" is refuted at
       low gp index in a 12-global image, and qr21's wedge is NOT yet attributed.
       Level 22 is therefore level 19's loop over level 19's OWN array, with the ONLY change
       being that the loop body moved behind the noinline callee. No new global, same search,
       same array object -- a genuine single-variable pair against level 19. */
/* LEVEL 22 rides in this same block on purpose. qr21 WEDGED where qr19 returned, but it
   changed THREE things at once: the noinline pointer-parameter call, dropping
   sqlite3FunctionSearch, and a NEW file-scope 9-element FuncDef global -- and a new global
   shifts the image, which is a documented way a failure MOVES here (S01). The off-SQLite
   pair has since cleared the shape outright: fdreg stage 4 (noinline callee taking the array
   by pointer) RETURNS 2609 on silicon, alongside stage 5 (noinline, reads via gp) and stage 2
   (inline). So "an argument capability wedges" is refuted at low gp index, and qr21's wedge
   is NOT attributed.
   CORRECTION 2026-08-06: level 22 is NOT one change from level 19; that description was
   wrong and the "single-variable pair" claim built on it is withdrawn. At lvl==22 the qi
   loop runs sqlite3FunctionSearch but takes NEITHER the lvl==19 nor the lvl==20 branch, so
   it performs NO link; qr_link_via_param then links separately, with its OWN
   sqlite3Strlen30 per element. That is THREE differences from level 19 -- link site,
   link-via-argument, and a duplicated strlen -- not one.
   The level images also shift 372-480 bytes against each other, roughly ten times the
   displacement of S01's dp0, while the only layout control run was +-16 bytes. So the
   21/22/23 verdicts are UNATTRIBUTED, and should stay that way until they can be selected
   at RUN TIME out of one byte-identical image.

   THE PAIR THAT DOES CARRY WEIGHT IS 15 vs 16, and it needs none of this machinery:

       qr15 and qr16 differ by exactly TWO BYTES, at 0x242C6-7:
           qr15   332c4: 93 05 f0 00   li a1, 0xf     -> RETURNS
           qr16   332c4: 93 05 00 01   li a1, 0x10    -> WEDGES
       identical file sizes (1550720), identical .text/.gct/gp_table offsets.

   `lvl` is a runtime variable, so BOTH code paths are compiled into BOTH images at the same
   addresses; only the selector constant differs. Layout cannot explain a two-byte immediate,
   so no statistics are needed. Level 15 links the clone array with its own inline loop and
   returns; level 16 hands the same array to the real sqlite3InsertBuiltinFuncs and wedges.
   That is the SQLite blocker's minimal repro. */
#if defined(CAPSTONE_SQLITE_QUICKRET) && (CAPSTONE_SQLITE_QUICKRET) >= 18 \
                                      && (CAPSTONE_SQLITE_QUICKRET) <= 41 \
                                      && (CAPSTONE_SQLITE_QUICKRET) != 21 \
                                      && (CAPSTONE_SQLITE_QUICKRET) != 25
    if ((lvl >= 18 && lvl <= 20) || lvl == 22 || lvl == 23 || lvl == 26 || lvl == 27 || lvl == 28 || lvl == 29 || lvl == 30 || lvl == 31 || lvl == 32 || lvl == 33 || lvl == 34 || lvl == 35 || lvl == 36 || lvl == 37 || lvl == 38 || lvl == 39 || lvl == 40 || lvl == 41) {
      static FuncDef qrSplitClone[] = {
        INTERNAL_FUNCTION(qr_split_rename_column,   9, qrCloneFunc0),
        INTERNAL_FUNCTION(qr_split_rename_table,    7, qrCloneFunc1),
        INTERNAL_FUNCTION(qr_split_rename_test,     7, qrCloneFunc2),
        INTERNAL_FUNCTION(qr_split_drop_column,     3, qrCloneFunc3),
        INTERNAL_FUNCTION(qr_split_rename_quotefix, 2, qrCloneFunc4),
        INTERNAL_FUNCTION(qr_split_drop_constraint, 2, qrCloneFunc5),
        INTERNAL_FUNCTION(qr_split_fail,            2, qrCloneFunc6),
        INTERNAL_FUNCTION(qr_split_add_constraint,  3, qrCloneFunc7),
        INTERNAL_FUNCTION(qr_split_find_constraint, 2, qrCloneFunc8),
      };
      int qi, qfound = 0;
      for (qi = 0; qi < ArraySize(qrSplitClone); qi++) {
        const char *z = qrSplitClone[qi].zName;
        int nName = sqlite3Strlen30(z);
        int h = SQLITE_FUNC_HASH(z[0], nName);
        FuncDef *pOther = sqlite3FunctionSearch(h, z);   /* level 18 does only this */
        if (pOther) qfound++;
        if (lvl == 19) {
          qrSplitClone[qi].pNext = sqlite3BuiltinFunctions.a[h];
          sqlite3BuiltinFunctions.a[h] = &qrSplitClone[qi];
        } else if (lvl == 20) {
          if (pOther) {
            qrSplitClone[qi].pNext = pOther->pNext;
            pOther->pNext = &qrSplitClone[qi];
          } else {
            qrSplitClone[qi].pNext = 0;
            qrSplitClone[qi].u.pHash = sqlite3BuiltinFunctions.a[h];
            sqlite3BuiltinFunctions.a[h] = &qrSplitClone[qi];
          }
        }
      }
#if (CAPSTONE_SQLITE_QUICKRET) == 22
      qr_link_via_param(qrSplitClone, ArraySize(qrSplitClone));
#elif (CAPSTONE_SQLITE_QUICKRET) == 23
      qr_link_via_param_leaf(qrSplitClone, ArraySize(qrSplitClone));
#elif (CAPSTONE_SQLITE_QUICKRET) == 29
      /* LEVEL 29 -- THE DISCRIMINATOR level 27 should have been.
         Level 27 reported qp=64 with an 8-bit count of 9, and that was read as "the nest ran
         576 times and the accumulator lost all but 9 adds". An audit found a simpler reading
         that fits the SAME numbers with no coincidence: the INNER loop ran 9 times TOTAL,
         because qk failed to be re-initialised to 0 on outer passes 2..64, so `qk < 9` was
         false immediately every pass after the first. That predicts qp==64, qcount==9, and
         level 26's qsum==191 as one FULL and entirely CORRECT pass -- which is also why all
         nine strlen results were right. The accumulator reading instead needs the hardware to
         drop exactly 63/64 of the adds and land precisely on the one-pass value, for a slot
         the program zeroes only once, before the loop.
         Level 27 could not tell these apart because BOTH its fields are 8-bit (lbu), so its
         "9" only ever meant qcount = 9 mod 256.
         This level returns the count in a FULL 16 bits:
             576 (0x240) -> the nest ran to completion; the ACCUMULATOR lost adds
             9   (0x009) -> the inner loop really only ran 9 times; qk's RESET is the fault
         Correct marker 0x9E290240. */
      {
        unsigned qcount = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            (void)sqlite3Strlen30(qrSplitClone[qk].zName);
            qcount++;
          }
        *res = 0x9E290000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 40
      /* LEVEL 40 -- is it the NEST, or an index that RESETS?
         The conjunction from boots 27-29 (nest AND capability AND counter-index) has a
         confound an analysis caught: in every failing level the capability index is the INNER
         counter, which is also the only value in the frame that is written to 0 in an outer
         body and then counted up past its bound. "Nest" and "resetting index" have never been
         separated. This keeps the nest and the capability access but indexes with the OUTER
         counter -- which varies, and never resets.
             576  -> a varying capability index in a nest is harmless; it must be the INNER,
                     RESETTING one, and L37 passed because of the reset, not the counter
             <576 -> any varying capability index in a nest does it
         Correct marker 0x9E410240. */
      {
        unsigned qcount = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            const char *volatile qz = qrSplitClone[qp & 7].zName;   /* outer ctr: never resets */
            (void)qz;
            qcount++;
          }
        *res = 0x9E410000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 41
      /* LEVEL 41 -- THE MISSING CELL: a RESETTING index with NO nest.
         L39 passed with a counter-derived index, but `qk % 9` comes from a MONOTONE counter,
         so it never tested a resetting index outside a nest. This is one flat loop whose
         capability index cycles 0..8 by an EXPLICIT reset -- L31's exact value history,
         without any nesting.
             <576 -> the trigger is a RESET-TO-0 index feeding an ldc, and nesting is
                     incidental. That makes it a value-history property, and it would also
                     put sqlite3InsertBuiltinFuncs back in scope, since its `i` resets per call.
             576  -> loop STRUCTURE really is required, and nothing in the RTL read so far can
                     express that -- the next step is frontend/replay, not the LSU.
         Correct marker 0x9E420240. */
      {
        unsigned qcount = 0;
        int qi, qk = 0;
        for (qi = 0; qi < 576; qi++) {
          if (qk >= (int)ArraySize(qrSplitClone)) qk = 0;         /* explicit reset, no nest */
          { const char *volatile qz = qrSplitClone[qk].zName; (void)qz; }
          qk++;
          qcount++;
        }
        *res = 0x9E420000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 39
      /* LEVEL 39 -- THE BRIDGE TEST, and the missing cell of the matrix.
         Boot 28 pinned the conjunction: the fault needs BOTH a CAPABILITY access (L38, an
         integer field indexed by the counter, is correct) AND an index that is the LOOP
         COUNTER (L37, a capability access at a dynamic non-counter index, is correct). L31
         has both and loses 9 of 576, four times running.
         Every correct single-loop control so far used a CONSTANT index (L36, and L33's
         sentinel loop). So "single loops are immune" was never tested against the failing
         shape. This is that test: ONE loop, no nest, body = a capability field indexed by the
         loop counter.
         It matters because it is EXACTLY sqlite3InsertBuiltinFuncs' shape --
         `for(i=0;i<nDef;i++){ const char *zName = aDef[i].zName; ... }` -- a single loop
         reading a capability field indexed by its own counter.
             576 -> the nest is required; the blocker is NOT this fault
             <576 -> a single loop with this shape loses iterations, and the blocker's own
                     loop has that shape
         Correct marker 0x9E3A0240 (576). */
      {
        unsigned qcount = 0;
        int qk;
        for (qk = 0; qk < 576; qk++) {
          const char *volatile qz = qrSplitClone[qk % ArraySize(qrSplitClone)].zName;
          (void)qz;
          qcount++;
        }
        *res = 0x9E3A0000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 37
      /* LEVEL 37 -- must the index be the LOOP COUNTER, or merely dynamic?
         L35 (constant index [0]) is correct; L31 (index [qk], the inner counter) loses 9 of
         576. So the surviving candidate is that the fault needs the counter to feed a
         capability address computation. This indexes with a DYNAMIC value that is NOT the
         counter -- a volatile fixed at 3, so the compiler must compute the address at runtime
         exactly as L31 does, but from a value the loop never updates.
             576 -> a dynamic index alone is harmless; it must be the COUNTER
             <576 -> any dynamic capability index does it, and the counter is incidental
         Correct marker 0x9E380240 (576). */
      {
        unsigned qcount = 0;
        int qp, qk;
        volatile int qfix = 3;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            const char *volatile qz = qrSplitClone[qfix].zName;  /* dynamic, not the counter */
            (void)qz;
            qcount++;
          }
        *res = 0x9E380000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 38
      /* LEVEL 38 -- is a capability access needed at all, or does the counter feeding ANY
         address do it? Same nest, same use of qk as an index, but into a plain INTEGER array
         reached through the same global -- no second ldc of a capability field.
             576 -> the capability field load matters
             <576 -> indexing anything by the counter is enough, capabilities are incidental
         Uses qrSplitClone[qk].nArg, an i16 already in the struct. Correct marker 0x9E390240. */
      {
        unsigned qcount = 0;
        int qp, qk, qacc = 0;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            qacc += qrSplitClone[qk].nArg;    /* integer field, indexed by the counter */
            qcount++;
          }
        *res = 0x9E390000u | (unsigned)(qcount & 0xffffu);
        (void)qacc;
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 35
      /* LEVEL 35 -- the MINIMAL POSITIVE INSTANCE the matrix never had. Every failing level so
         far carries extra baggage: L31 has two ldc, L32 adds an integer sw, L26/27/29 add a
         call. This is a nest whose inner body is ONE gp-relative capability load and nothing
         else, with a 16-bit witness. If it loses iterations, that single ldc is sufficient.
         Correct marker 0x9E360240 (576). */
      {
        unsigned qcount = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            const char *volatile qz = qrSplitClone[0].zName;   /* one gp ldc, fixed index */
            (void)qz;
            qcount++;
          }
        *res = 0x9E360000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 36
      /* LEVEL 36 -- the NEST-versus-SINGLE control, with a body identical to level 35's.
         An audit found, inside my own data, that a SINGLE non-nested loop with a gp ldc in its
         body completes 9 of 9 (L33's sentinel-init loop). That is the sharpest constraint on
         the whole matrix, and it deserves a purpose-built measurement rather than a by-product:
         576 iterations of the SAME body, in ONE loop instead of 64x9.
             576 -> single loops are immune; the fault needs a NEST
             < 576 -> the nest is irrelevant and the effect is per-iteration
         Correct marker 0x9E370240 (576). */
      {
        unsigned qcount = 0;
        int qk;
        for (qk = 0; qk < 576; qk++) {
          const char *volatile qz = qrSplitClone[0].zName;     /* same body as level 35 */
          (void)qz;
          qcount++;
        }
        *res = 0x9E370000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 34
      /* LEVEL 34 -- TEST or INCREMENT? Level 32 showed the inner body runs only during outer
         pass 0. Level 33 showed qk IS 0 at the top of every pass -- read back from the outer
         body, matching QEMU exactly (0/8) -- so the reset store and its reload are fine.
         Those two together are strange: qk == 0, `qk < 9` should be true, and yet the body
         does not run. Two candidates remain, and they differ in what qk holds when the inner
         loop gives up:
             B = highest outer pass whose inner body ran      (correct 63)
             C = qk immediately AFTER the inner loop, last pass (correct 9)
         B=0, C=0 -> in later passes the loop exited with qk still 0, i.e. `0 < 9` evaluated
                     FALSE: the COMPARE/branch is the fault
         B=0, C=9 -> qk reached 9 without the body running: the INCREMENT or the init ran away
         B=63     -> the inner body did run late, and level 32's witness is what lied
         All three witnesses live in .data on distinct elements, so no stack slot can forge
         them. Marker 0x9E35, correct 0x9E353F09. */
      {
        int qp, qk;
        for (qk = 0; qk < ArraySize(qrSplitClone); qk++)
          qrSplitClone[qk].funcFlags = 0u;
        for (qp = 0; qp < 64; qp++) {
          qk = 0;
          for (; qk < ArraySize(qrSplitClone); qk++) {
            const char *volatile qz = qrSplitClone[qk].zName;   /* the capability access */
            (void)qz;
            qrSplitClone[1].funcFlags = (u32)qp;                /* B: last pass that ran */
          }
          qrSplitClone[2].funcFlags = (u32)qk;                  /* C: qk after inner loop */
        }
        *res = 0x9E350000u | ((qrSplitClone[1].funcFlags & 0xffu) << 8)
                           | (qrSplitClone[2].funcFlags & 0xffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 33
      /* LEVEL 33 -- WHICH operation is lost? Level 32 established that the inner body runs
         only during outer pass 0 when it makes a capability access. Three things could do
         that: the store of 0 to the inner counter never lands; it lands but the reload at the
         loop test returns the stale 9; or the compare itself is mis-forwarded.
         This reads the inner counter back IMMEDIATELY after setting it to 0, from the OUTER
         body -- which is known to run all 64 passes (level 27, qp=64) -- and stamps what it
         saw into .data:
             max stamp 0 -> qk really is 0 at the top of every pass, so the store and this
                            reload are FINE and the fault is in the loop test or the increment
             max stamp 9 -> qk still reads 9, i.e. the `qk = 0` store is LOST or invisible
         The inner body keeps a capability access, since that is the trigger. Marker 0x9E34,
         not 0x9E33 -- 0x9E33 is the QUICKRET ladder's own marker family. Correct: 0x9E340009
         (max stamp 0, 9 elements sentinel-cleared and written). */
      {
        unsigned qmax = 0, qwritten = 0;
        int qp, qk;
        for (qk = 0; qk < ArraySize(qrSplitClone); qk++)
          qrSplitClone[qk].funcFlags = 0xffffffffu;
        for (qp = 0; qp < 64; qp++) {
          qk = 0;
          /* read qk back at once, from the outer body, and record it */
          qrSplitClone[qp & 7].funcFlags = (u32)qk;
          for (; qk < ArraySize(qrSplitClone); qk++) {
            const char *volatile qz = qrSplitClone[qk].zName;   /* the capability access */
            (void)qz;
          }
        }
        for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
          u32 v = qrSplitClone[qk].funcFlags;
          if (v != 0xffffffffu) { qwritten++; if (v > qmax) qmax = v; }
        }
        *res = 0x9E340000u | ((qmax & 0xffu) << 8) | (qwritten & 0xffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 32
      /* CORRECTION 2026-08-07: this level's inner store is `sw a0, 0x4(a1)` -- a plain
         INTEGER store, not an stc. Its only capability access is the `ldc 0x220(gp)` that
         fetches the array. Anywhere this level is described as "capability STORE", that is
         wrong. Related: L33's inner body already contains an stc AND a load and it returns
         correctly, so the "a load plus a store wedges" reading of L34 is withdrawn. And a
         SINGLE non-nested loop with a gp ldc in its body completes 9/9 on this silicon
         (L33's own sentinel-init loop), so the effect is about the INNER loop of a NEST, not
         about capability accesses in loops generally.
         LEVEL 32 -- the REAL discriminator. Level 29 does not separate the two readings, and
         both the audit and I got that wrong in opposite directions: qcount IS the accumulator,
         so "the nest ran 576 times and the adds were lost" and "the inner loop ran 9 times"
         BOTH predict 9. A 16-bit counter only removed level 27's mod-256 alias.
         What separates them is a witness that does NOT live on the stack. The inner body here
         stamps the OUTER pass number into a .data field of an existing global, so the answer
         is read back out of memory the frame cannot touch:
             max stamp 63 -> the inner body DID run on the last outer pass; the nest completed
                             and the stack ACCUMULATOR is what lost the adds
             max stamp 0  -> the inner body ran only during pass 0; the inner loop's induction
                             variable really is failing to re-init
         `written` counts how many of the nine elements were stamped at all, so a corrupted
         witness is distinguishable from a clean 0. Reuses qrSplitClone, adds no global, so it
         keeps the section layout of the family that enters. Correct marker 0x9E323F09. */
      {
        unsigned qwritten = 0, qmax = 0;
        int qp, qk;
        for (qk = 0; qk < ArraySize(qrSplitClone); qk++)
          qrSplitClone[qk].funcFlags = 0xffffffffu;      /* sentinel: never stamped */
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++)
            qrSplitClone[qk].funcFlags = (u32)qp;        /* stamp the OUTER pass, into .data */
        for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
          u32 v = qrSplitClone[qk].funcFlags;
          if (v != 0xffffffffu) { qwritten++; if (v > qmax) qmax = v; }
        }
        *res = 0x9E320000u | ((qmax & 0xffu) << 8) | (qwritten & 0xffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 30
      /* LEVEL 30 -- split level 28's delta. Removing the call also removed TWO capability
         loads (ldc 0x220(gp) for the array, ldc 0x70(a0) for zName) and the index
         arithmetic: eleven instructions, not one. The in-tree prior suspect is the ldc, not
         the jalr (ISSUES.md on the -O0 pointer round-trip). This keeps the CALL and drops
         both ldc -- a noinline local leaf taking no argument. Correct marker 0x9E300240. */
      {
        unsigned qcount = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            (void)qr_touch_nothing();
            qcount++;
          }
        *res = 0x9E300000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 31
      /* LEVEL 31 -- the other half of level 30's split: keep both ldc, drop the call.
         Correct marker 0x9E310240.
         BOARD 2026-08-07, run FIRST in its boot after being lost as collateral four times:
         567, where QEMU gives 576. NINE missing -- exactly one pass of sixty-four.
         That matters beyond this level: it makes the loss GRADED. L29 and L32 lose 567 of
         576 (the inner body runs only during pass 0); L31 loses 9. So "a capability access in
         the inner body means the body runs only in the first pass" is FALSE as a general
         statement and is withdrawn -- L31 executed 63 of its 64 passes. What survives is that
         a capability access in the inner body costs iterations, by an amount that varies by
         two orders of magnitude between levels, and that with BOTH a load and a store (L34)
         it stops returning at all.
         9/576 is 1.5%, the same order as the ~3% sporadic strlen rate in S01. Whether that is
         the same phenomenon is NOT established and should not be asserted from a rate
         coincidence across two different measurements.
         N=2 as of 2026-08-07: 567 twice. L32 (0/9), L33 (0/8) and L34 (wedge) also reproduce
         exactly, so the whole matrix is deterministic and the severity spread between L31 and
         L32 is a property of the builds, not variance. */
      {
        unsigned qcount = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            const char *volatile qz = qrSplitClone[qk].zName;
            (void)qz;
            qcount++;
          }
        *res = 0x9E310000u | (unsigned)(qcount & 0xffffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 28
      /* LEVEL 28 -- is it the CALL, or the loop nest? Level 27 returned qp=64 (the outer loop
         ran all 64 iterations, correctly) with the inner counter at 9 -- one pass. So loop
         CONTROL is fine and the ACCUMULATOR loses everything before the final pass. The
         obvious suspect is the call in the inner body: sqlite3Strlen30 is the only thing
         between the increments, and a callee whose frame overlapped the caller's slot would
         do exactly this.
         Level 28 is level 27 with the CALL REMOVED and nothing else changed -- same nest,
         same bounds, same counters, same accumulate:
             returns 64 (i.e. 576 & 0xff) -> the loop nest is fine, the CALL is the trigger
             returns 9  again             -> the nest alone does it, no call needed, and the
                                             fault is in the accumulator's stack slot itself
         Correct marker 0x9E284040 (qp=64, count&0xff=64). */
      {
        unsigned qcount = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++)
            qcount++;
        *res = 0x9E280000u | ((unsigned)(qp & 0xffu) << 8) | (unsigned)(qcount & 0xffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 27
      /* LEVEL 27 -- WHICH HALF of level 26 broke? Level 26 returned sum=191 on silicon where
         QEMU returns 12224 from the SAME binary: exactly ONE pass of 64, with all nine
         strlen results in that pass CORRECT. So strlen is fine and the OUTER loop is not
         iterating. Two candidates remain and this separates them by reporting the loop's own
         state instead of its product:
             low  16 bits = the number of inner iterations actually executed (max 65535)
             high 16 bits of the payload = the value of qp AFTER the loop
         qp == 64 with a low count -> the loop ran to completion but the ACCUMULATOR lost
                                      adds (a store/reload of qsum failing)
         qp >= 64 early, count 9   -> qp was corrupted to an out-of-range value and the loop
                                      exited after one pass
         qp == 1, count 9          -> the compare/branch itself misbehaved
         Counters are plain ints, no new global, same 18-26 family layout. */
      {
        unsigned qcount = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++) {
            (void)sqlite3Strlen30(qrSplitClone[qk].zName);
            qcount++;
          }
        *res = 0x9E270000u | ((unsigned)(qp & 0xffu) << 8) | (unsigned)(qcount & 0xffu);
        return;
      }
#elif (CAPSTONE_SQLITE_QUICKRET) == 26
      /* LEVEL 26 -- MEASURE the sporadic strlen instead of hunting the hang.
         S01 records strlen returning a wrong answer ~3% of the time on this image class (4 of
         128 calls returned 1), attributed to the -O0 reload: strlen re-loads the string
         capability with `ldc` from a stack slot every iteration. S01 then INFERS, without
         establishing it, that a wrong strlen corrupts the hash chain and causes this hang.
         This turns that into a NUMBER. It calls sqlite3Strlen30 over qrSplitClone's nine
         zName pointers 64 times each and returns the SUM, so a shortfall is exactly the
         count of characters lost to bad reloads.
         Deliberately reuses qrSplitClone rather than declaring its own name array: level 25
         did the latter and ENTRY-STALLED on three separate draws, because the new global
         moved .capstone_gp_table / .capstone_gp_initdesc 1344 bytes (0x151570 -> 0x151030)
         and turned the probe into a structurally different image. Reusing the existing array
         adds NO capability-bearing global, so this level keeps the 18-23 family's section
         layout -- which is the family that has actually been entering. */
      {
        unsigned long qsum = 0;
        int qp, qk;
        for (qp = 0; qp < 64; qp++)
          for (qk = 0; qk < ArraySize(qrSplitClone); qk++)
            qsum += (unsigned)sqlite3Strlen30(qrSplitClone[qk].zName);
        *res = 0x9E260000u | (unsigned)(qsum & 0xffffu);
        return;
      }
#endif
      qrc = (int)lvl;
      (void)qfound;
    }
#endif
    /* LEVEL 25 -- MEASURE the sporadic strlen failure instead of hunting the hang.
       S01-image-perturbation-hang/00-README.md records that on this image class strlen
       returns a WRONG answer sporadically: "Stage 16 calls strlen on the SAME literal
       'alpha' 128 times and totals 636, i.e. 4 of 128 calls returned 1 -- sporadic (~3%),
       not length-dependent", with the mechanism inferred as the -O0 reload -- strlen
       re-loads the string capability with `ldc` from a stack slot every iteration, and a
       result of exactly 1 is what a failed second reload produces. S01 then infers, without
       establishing it: wrong strlen -> wrong hash in sqlite3InsertBuiltinFuncs -> corrupt
       chain -> hang.
       That is the same primitive this ladder's own bisection landed on independently: level
       22 reaches strlen through sqlite3Strlen30 and wedges; level 23 replaces it with a
       local `while (z[n]) n++` and returns.
       So stop bisecting the hang and measure the primitive. This level calls sqlite3Strlen30
       over the nine names 64 times each and returns the SUM. The correct total is a fixed
       number; any shortfall is exactly the number of characters lost to bad reloads, and it
       comes back as a RETURNING wrong value rather than silence -- which is bisectable where
       a wedge is one bit.
       It also repairs a flaw in levels 18-23: they return `qrc = (int)lvl`, a CONSTANT, so a
       level can "return correctly" while having computed garbage. S01 warns exactly this --
       "stages that RETURN are already wrong". This level returns real data. */
#if defined(CAPSTONE_SQLITE_QUICKRET) && (CAPSTONE_SQLITE_QUICKRET) == 25
    if (lvl == 25) {
      static const char *const qrNames[9] = {
        "sqlite_rename_column", "sqlite_rename_table", "sqlite_rename_test",
        "sqlite_drop_column", "sqlite_rename_quotefix", "sqlite_drop_constraint",
        "sqlite_fail", "sqlite_add_constraint", "sqlite_find_constraint",
      };
      /* 20+19+18+18+22+23+11+21+22 = 174 per pass, 64 passes = 11136. */
      unsigned long qsum = 0;
      int qp, qi;
      for (qp = 0; qp < 64; qp++)
        for (qi = 0; qi < 9; qi++)
          qsum += (unsigned)sqlite3Strlen30(qrNames[qi]);
      /* Report the SHORTFALL, so 0 means every one of the 576 calls was right and a small
         positive number is the count of lost characters. Clamped into the marker byte. */
      qrc = (int)((11136UL - (qsum > 11136UL ? 11136UL : qsum)) & 0xffUL);
      *res = 0x9E250000u | ((unsigned)(qsum & 0xffffu));
      return;
    }
#endif
    if (lvl == 17) {
      /* The real thing, with no heap init -- level 13 minus CONFIG_HEAP/MallocInit. */
      sqlite3AlterFunctions();
      qrc = (int)lvl;
    }
#if defined(CAPSTONE_SQLITE_QUICKRET) && (CAPSTONE_SQLITE_QUICKRET) >= 15 \
                                      && (CAPSTONE_SQLITE_QUICKRET) <= 16
    if (lvl >= 15 && lvl <= 16) {
      static FuncDef qrAlterClone[] = {
        INTERNAL_FUNCTION(qr_clone_rename_column,   9, qrCloneFunc0),
        INTERNAL_FUNCTION(qr_clone_rename_table,    7, qrCloneFunc1),
        INTERNAL_FUNCTION(qr_clone_rename_test,     7, qrCloneFunc2),
        INTERNAL_FUNCTION(qr_clone_drop_column,     3, qrCloneFunc3),
        INTERNAL_FUNCTION(qr_clone_rename_quotefix, 2, qrCloneFunc4),
        INTERNAL_FUNCTION(qr_clone_drop_constraint, 2, qrCloneFunc5),
        INTERNAL_FUNCTION(qr_clone_fail,            2, qrCloneFunc6),
        INTERNAL_FUNCTION(qr_clone_add_constraint,  3, qrCloneFunc7),
        INTERNAL_FUNCTION(qr_clone_find_constraint, 2, qrCloneFunc8),
      };
      static FuncDef *qrCloneBuckets[SQLITE_FUNC_HASH_SZ];
      if (lvl == 15) {
        int qi;
        for (qi = 0; qi < ArraySize(qrAlterClone); qi++) {
          const char *z = qrAlterClone[qi].zName;
          int h = SQLITE_FUNC_HASH(z[0], sqlite3Strlen30(z));
          qrAlterClone[qi].pNext = qrCloneBuckets[h];
          qrCloneBuckets[h] = &qrAlterClone[qi];
        }
      } else {
        sqlite3InsertBuiltinFuncs(qrAlterClone, ArraySize(qrAlterClone));
      }
      qrc = (int)lvl;
    }
#endif
    if (lvl >= 13 && lvl <= 14) {
      qrc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                           (int)sizeof(sqlite_heap), 64);
      if (qrc == SQLITE_OK) qrc = sqlite3MallocInit();
      if (qrc == SQLITE_OK) {
        sqlite3AlterFunctions();                 /* level 13 stops here */
        if (lvl >= 14) {
          sqlite3WindowFunctions();
          sqlite3RegisterDateTimeFunctions();
          sqlite3RegisterJsonFunctions();
        }
        qrc = (int)lvl;                          /* rc echoes the level reached */
      }
    }
    if (lvl >= 10 && lvl <= 12) {
      qrc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                           (int)sizeof(sqlite_heap), 64);
      if (qrc == SQLITE_OK) qrc = sqlite3MallocInit();
      if (qrc == SQLITE_OK) {
        if (lvl == 10) {
          /* READ ONLY -- cannot wedge on a call, so this always returns a number.
             1 = xInit already non-null, 0 = null (the expected fresh state). */
          qrc = (sqlite3GlobalConfig.pcache2.xInit != 0) ? 1 : 0;
        } else if (lvl == 11) {
          /* Materialise + copy the 13-function-pointer aggregate. NO indirect call. */
          sqlite3PCacheSetDefault();
          qrc = 2;
        } else {
          /* Same, then READ BACK the installed pointer without calling it:
             3 = the copy landed and xInit is non-null, 4 = it did not. */
          sqlite3PCacheSetDefault();
          qrc = (sqlite3GlobalConfig.pcache2.xInit != 0) ? 3 : 4;
        }
      }
    }
    if (lvl >= 6 && lvl <= 9) {
      qrc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                           (int)sizeof(sqlite_heap), 64);
      if (qrc == SQLITE_OK) {
        if (lvl == 6) {
          qrc = sqlite3MutexInit();          /* no-op at THREADSAFE=0; proves the call path */
        } else if (lvl == 7) {
          qrc = sqlite3MallocInit();         /* memsys5Init: zone headers built IN the heap  */
        } else if (lvl == 8) {
          qrc = sqlite3MallocInit();
          if (qrc == SQLITE_OK) qrc = sqlite3PcacheInitialize();
        } else {
          qrc = sqlite3MallocInit();
          if (qrc == SQLITE_OK) {
            sqlite3RegisterBuiltinFunctions();  /* writes the global function hash table */
            qrc = SQLITE_OK;
          }
        }
      }
    }
    *res = 0x9E330000u | ((lvl & 0xffu) << 8) | ((unsigned)qrc & 0xffu);
    return;
  }
#endif
#ifdef CAPSTONE_SQLITE_QUICKRET
  /* MINIMAL WORKLOAD REDUCTION -- the smallest possible probe, added 2026-08-06.
     Returns a marker immediately instead of running the database. No new globals, no new
     calls, a handful of instructions.
     Purpose: the full workload is a REAL hang (900s, control green, no return), and the
     obvious tool for bisecting it -- the staged dispatch -- cannot be used, because building
     with CAPSTONE_SQLITE_STAGE adds 2 globals and ~13KB of .text and shifts the globals base
     0x150000 -> 0x160000, and THAT image dies earlier still, inside region-share #1. This is
     the S01 image-perturbation family: the instrument moves the failure.
     So this probe is deliberately the least invasive change that reduces the workload to
     nothing. If it RETURNS, the domain can enter, run C, write res and return -- and the hang
     is in the workload, which can then be grown back a piece at a time under a SHORT timeout.
     If it WEDGES with the workload removed, the fault is structural and nothing about SQLite
     itself is implicated.
     Placed AFTER the func dispatch so region shares are untouched. */
  *res = 0x9E33u;
  return;
#endif
  (void)run_sqlite();
  *res = SQLITE_HC_RET_DONE;
}
