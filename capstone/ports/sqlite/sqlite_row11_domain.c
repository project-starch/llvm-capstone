/* row11 -- the LITERAL matched pair for
 * cve-repros/row11_go_double_finalize (LINEAR / double-free).
 *
 * The SAME row11 program as before.c, real SQLite C API, one Capstone domain:
 *
 *     open :memory: -> prepare "SELECT 1" -> finalize(stmt) -> finalize(stmt)
 *
 * before.c double-finalizes a statement handle. On the host that is a double
 * free (ASan SEGV, see cve-repros/row11.../NOTE.md). Here SQLite's ENTIRE heap
 * is the revoke-on-free linear allocator (revoke_on_free_alloc.h) installed via
 * SQLITE_CONFIG_MALLOC, exactly as in row3 fork B2. The statement handle SQLite
 * returns from sqlite3_prepare_v2 is a pointer into an rof_malloc allocation
 * (the Vdbe block). The FIRST sqlite3_finalize frees that block -- the allocator
 * REVOKEs the allocation's node -- so `stmt` now dangles. The SECOND
 * sqlite3_finalize dereferences `stmt` (loads v->db and friends) through the
 * revoked capability and FAULTS.
 *
 *   host    : the 2nd finalize is a use-after-free of the stmt block (before.c).
 *   Capstone: the 2nd finalize is a deterministic capability fault.
 *
 * The pointer that faults is the exact value sqlite3_prepare_v2 handed back --
 * SQLite's own statement handle, revoked by SQLite's own first finalize. No
 * wrapper, no carved copy, no driver-fired revoke. This is the LINEAR
 * "move-only handle consumed by the first finalize" shape expressed literally:
 * a revoke-on-free allocation gives the consumed handle no surviving authority.
 *
 * Cause is opt-level dependent (task-007/008): -O0 spills `stmt` across the
 * first finalize call, so the reload before the second call comes back untagged
 * (revoked node) -> cause 24; -O1/-O2 keep `stmt` register-held across the first
 * finalize -> the second call derefs a tagged-but-revoked cap -> cause 25,
 * self-proving. See run-sqlite-row11.sh.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "revoke_on_free_alloc.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* ---- host-call text output (mirrors sqlite_row3_b2_domain.c) ---- */
static void output_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
  const char *src = (const char *)__builtin_capstone_cap_delin((void *)text);
  char *payload = (char *)__builtin_capstone_cap_delin((void *)hostcall_payload);
  unsigned long offset = hostcall_metadata->length;
  while (*src && offset + 1 < SQLITE_HC_REGION_SIZE)
    payload[offset++] = *src++;
  hostcall_metadata->length = offset;
}

static void output_uint(unsigned long v) {
  char buf[21];
  unsigned i = 21;
  buf[--i] = '\0';
  if (v == 0)
    buf[--i] = '0';
  while (v && i) {
    buf[--i] = (char)('0' + (v % 10));
    v /= 10;
  }
  output_text(&buf[i]);
}

/* ---- SQLite memory methods backed by the revoke-on-free allocator ---- */
static void *rof_xMalloc(int n) { return rof_malloc((unsigned long)n); }
static void rof_xFree(void *p) { rof_free(p); }
static int rof_xSize(void *p) { return (int)rof_size(p); }
static int rof_xRoundup(int n) { return (int)rof_roundup((unsigned long)n); }
static int rof_xInit(void *unused) {
  (void)unused;
  return SQLITE_OK;
}
static void rof_xShutdown(void *unused) { (void)unused; }

static void *rof_xRealloc(void *p, int n) {
  if (!p)
    return rof_malloc((unsigned long)n);
  if (n <= 0) {
    rof_free(p);
    return (void *)0;
  }
  unsigned long oldsz = rof_size(p);
  unsigned long newsz = rof_roundup((unsigned long)n);
  void *np = rof_malloc((unsigned long)n);
  if (!np)
    return (void *)0;
  unsigned long copy = oldsz < newsz ? oldsz : newsz;
  rof_copy_caps(np, p, copy);
  rof_free(p);
  return np;
}

static const sqlite3_mem_methods rof_mem_methods = {
    rof_xMalloc, rof_xFree,  rof_xRealloc,  rof_xSize,
    rof_xRoundup, rof_xInit, rof_xShutdown, (void *)0};

static int fail(const char *stage, int rc) {
  output_text("row11 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

/* The literal row11 sequence. Returns only if the second finalize does NOT
 * fault (the double-free we demonstrate is caught). */
static int run_row11(int revoke_at_finalize) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;

  /* SQLite's WHOLE heap is the revoke-on-free allocator. */
  int rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &rof_mem_methods);
  if (rc != SQLITE_OK)
    return fail("config-malloc", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);
  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK)
    return fail("open", rc);

  /* before.c:  sqlite3_prepare_v2(db, "SELECT 1", -1, &stmt, 0);
   * `stmt` is SQLite's own handle -- a pointer into an rof_malloc allocation
   * (the Vdbe block). No wrapper, no copy. */
  rc = sqlite3_prepare_v2(db, "SELECT 1", -1, &stmt, 0);
  if (rc != SQLITE_OK)
    return fail("prepare", rc);
  if (!stmt)
    return fail("prepare-null", 1);

  output_text("row11 prepared stmt ok\n");

  /* before.c:  sqlite3_finalize(stmt);
   * The first finalize frees the Vdbe block -- the allocator REVOKEs that
   * allocation's node. `stmt` now dangles. */
  rc = sqlite3_finalize(stmt);
  if (rc != SQLITE_OK)
    return fail("finalize-1", rc);

  output_text("row11 first finalize ok\n");

  if (!revoke_at_finalize) {
    /* Control: identical program and allocator, but the free path recycles the
     * slot WITHOUT revoking (ROW11_NO_REVOKE). `stmt` is still a live tagged
     * capability into intact memory, so the second finalize dereferences it
     * successfully and we RETURN. This disambiguates the -O0 cause-24 fault
     * (tag gone, which a plain spill reload also yields) and proves SQLite runs
     * correctly on the allocator. */
    rc = sqlite3_finalize(stmt);
    output_text("row11 NOTRAP second finalize rc=");
    output_uint((unsigned long)(rc < 0 ? -rc : rc));
    output_text("\n");
    (void)sqlite3_close(db);
    return 0;
  }

  /* before.c:  sqlite3_finalize(stmt);  -- DOUBLE FINALIZE of SQLite's own
   * handle. With revoke-on-free, this dereference of the revoked stmt FAULTS. */
  rc = sqlite3_finalize(stmt);
  output_text("row11 NOTRAP second finalize rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  (void)sqlite3_close(db);
  return 0;
}

void domain_main(void *arg, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)arg;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)arg;
    else if (shared_region_count == 2)
      rof_init(arg); /* the LINEAR grant becomes the allocator's arena */
    ++shared_region_count;
    return;
  }

  if (hostcall_metadata)
    hostcall_metadata->length = 0;

#if defined(ROW11_NO_REVOKE)
  rof_no_revoke = 1;
  (void)run_row11(0);
#else
  (void)run_row11(1);
#endif

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
