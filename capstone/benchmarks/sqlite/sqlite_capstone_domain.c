#include "sqlite3.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U
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
    static const char *big30[40];
    static const char *big31[100];
    static const char *big32[160];
    static const char *big33[300];
    static const char *big34[580];
    /* A switch, NOT a ternary chain. `cond ? ptrA : ptrB` lowers to an i128
       CapstoneISD::SELECT_CC, for which this backend has no RV64 pattern -- it aborts with
       "Cannot select" (documented in history/31-07-2026 ... i128-selectcc-gap). All five of
       these builds died on exactly that, having been written with a ternary chain. A switch
       lowers to branches and avoids the node entirely. */
    const char **p;
    switch (stage) {
      case 30: p = big30; break;
      case 31: p = big31; break;
      case 32: p = big32; break;
      case 33: p = big33; break;
      default: p = big34; break;
    }
    unsigned n = CAPSTONE_HOLDER_N(stage), i, ok = 0;
    /* Fill at run time only if cap-init left them null -- the point is to READ what cap-init
       (or its absence) produced, not to overwrite it. */
    for (i = 0; i < n; i++)
      if (p[i] == 0) p[i] = "x";
    for (i = 0; i < n; i++)
      if (p[i] && sqlite3Strlen30(p[i]) >= 0) ok++;
    return (int)(ok > 255 ? 255 : ok);
  }
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
