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
  /* This one stays in every build. `payload` is the HOST capability loaded fresh out of
     its global each call, not a cap-table storage cap -- it is still linear here, and the
     copy on the line above would consume it without the delin. */
  CAPSTONE_DELIN(payload);
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
#define CAPSTONE_HOLDER_N(st) ((st) == 30 ? 40u : (st) == 31 ? 100u : \
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

void domain_main(unsigned *res, unsigned func) {
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
  *res = 0x5A6E0000u | ((unsigned)(CAPSTONE_SQLITE_STAGE) << 8) |
         ((unsigned)run_sqlite_staged(CAPSTONE_SQLITE_STAGE) & 0xffu);
  return;
#endif
  (void)run_sqlite();
  *res = SQLITE_HC_RET_DONE;
}
