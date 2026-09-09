/* row3 fork B2 -- the LITERAL matched pair for
 * cve-repros/row3_diesel_colname_cached.
 *
 * The SAME row3 program as before.c, real SQLite C API, one Capstone domain:
 *
 *     open :memory: -> CREATE t(a) -> INSERT 1 -> prepare "SELECT a AS colname"
 *     -> step -> name = column_name(stmt,0) -> finalize -> read name[0]
 *
 * The difference from fork B1 (sqlite_row3_domain.c) is the whole point of B2:
 * there is NO wrapper. B1 carved a revocable COPY of the column name and revoked
 * that copy, because memsys5 allocations are not independently revocable. Here
 * SQLite's ENTIRE heap is a revoke-on-free linear allocator
 * (revoke_on_free_alloc.h) installed via SQLITE_CONFIG_MALLOC: every SQLite
 * allocation is its own SPLIT sub-capability with its own revocation node, and
 * sqlite3_finalize's own xFree of the column-name buffer REVOKEs it. So the
 * pointer that faults post-finalize is the exact value sqlite3_column_name
 * returned -- SQLite's own pointer, revoked by SQLite's own free path.
 *
 *   host   : name[0] after finalize is ASan heap-use-after-free (before.c).
 *   Capstone: name[0] after finalize is a deterministic capability fault.
 *
 * The arena is a REAL monitor-delivered REV_TRANSFERRED linear capability
 * (region #2); the domain owns it, mints its own MREVs per allocation, and
 * revokes intra-domain. No start.S / monitor change (task-007). The allocator
 * NEVER coalesces (SPLIT is one-way), so it fragments; region #2 is sized for
 * row3's cumulative churn. See run-sqlite-row3-b2.sh and the Phase-2 note.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "revoke_on_free_alloc.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* ---- host-call text output (mirrors sqlite_row3_domain.c) ---- */
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

/* realloc = malloc + copy + free. The free REVOKEs the old node, which models
 * realloc-invalidates-old exactly: a cached pointer to the old block faults. */
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
  /* Copy the smaller of the two ALLOCATION sizes (both 16-multiples), tag-
   * preserving. Copying whole allocations rather than min(old,n) is safe -- the
   * extra bytes are within bounds -- and keeps the copy capability-aligned. */
  unsigned long copy = oldsz < newsz ? oldsz : newsz;
  rof_copy_caps(np, p, copy);
  rof_free(p);
  return np;
}

static const sqlite3_mem_methods rof_mem_methods = {
    rof_xMalloc, rof_xFree,  rof_xRealloc,  rof_xSize,
    rof_xRoundup, rof_xInit, rof_xShutdown, (void *)0};

static int fail(const char *stage, int rc) {
  output_text("row3-b2 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

/* The literal row3 sequence. Returns only if the post-finalize read does NOT
 * fault (the bug we demonstrate is caught). */
static int run_row3_b2(int revoke_at_finalize) {
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
  rc = sqlite3_exec(db, "CREATE TABLE t(a); INSERT INTO t VALUES(1)", 0, 0, 0);
  if (rc != SQLITE_OK)
    return fail("exec", rc);
  rc = sqlite3_prepare_v2(db, "SELECT a AS colname FROM t", -1, &stmt, 0);
  if (rc != SQLITE_OK)
    return fail("prepare", rc);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW)
    return fail("step", rc);

  /* before.c:  name = sqlite3_column_name(stmt, 0);
   * SQLite's OWN pointer, into an rof_malloc allocation. No wrapper, no copy. */
  const char *name = sqlite3_column_name(stmt, 0);
  if (!name)
    return fail("column_name", 1);

  output_text("row3-b2 live name=");
  output_text(name);
  {
    char one[2] = {name[0], '\0'};
    output_text("\nrow3-b2 live name[0]=");
    output_text(one);
    output_text("\n");
  }

  /* before.c:  sqlite3_finalize(stmt);
   * finalize frees the statement, including the column-name buffer -- the
   * allocator REVOKEs that allocation's node. `name` now dangles. */
  rc = sqlite3_finalize(stmt);
  stmt = 0;
  if (rc != SQLITE_OK)
    return fail("finalize", rc);

  if (!revoke_at_finalize) {
    /* Control: identical program, but the allocator's revoke is suppressed
     * (ROW3_B2_NO_REVOKE), so the post-finalize read succeeds and we RETURN. */
    volatile char c = name[0];
    output_text("row3-b2 post-finalize NOTRAP name[0]=");
    {
      char one[2] = {(char)c, '\0'};
      output_text(one);
      output_text("\n");
    }
    (void)sqlite3_close(db);
    return 0;
  }

  /* before.c:  name[0]  -- USE AFTER FINALIZE of SQLite's own pointer.
   * With revoke-on-free, this cached pointer FAULTS. Kept volatile so the
   * optimiser cannot fold it; at -O1+ it stays register-held across finalize so
   * the fault is cause 25 (self-proving). */
  volatile char c = name[0];
  output_text("row3-b2 post-finalize NOTRAP name[0]=");
  {
    char one[2] = {(char)c, '\0'};
    output_text(one);
    output_text("\n");
  }
  (void)sqlite3_close(db);
  return 0;
}

#if defined(ROW3_B2_CHURN)
/* Phase-2 generality probe: how far does a NON-COALESCING revoke-on-free heap
 * get before the arena depletes? Run the row3 prepare/step/column_name/finalize
 * cycle in a loop (revoke ON at every finalize) and report how many iterations
 * completed and the cumulative bytes carved vs the peak live, quantifying the
 * fragmentation cost. Returns cleanly (no cached-pointer UAF here) so the host
 * flushes the report. */
static int run_row3_b2_churn(unsigned iters) {
  sqlite3 *db = 0;
  int rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &rof_mem_methods);
  if (rc != SQLITE_OK)
    return fail("config-malloc", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);
  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK)
    return fail("open", rc);
  rc = sqlite3_exec(db, "CREATE TABLE t(a); INSERT INTO t VALUES(1)", 0, 0, 0);
  if (rc != SQLITE_OK)
    return fail("exec", rc);

  unsigned long peak_live = 0;
  unsigned done = 0;
  for (unsigned i = 0; i < iters; ++i) {
    sqlite3_stmt *stmt = 0;
    rc = sqlite3_prepare_v2(db, "SELECT a AS colname FROM t", -1, &stmt, 0);
    if (rc != SQLITE_OK)
      break; /* NOMEM: arena depleted (non-coalescing) */
    (void)sqlite3_step(stmt);
    (void)sqlite3_column_name(stmt, 0);
    if (rof_live_bytes > peak_live)
      peak_live = rof_live_bytes;
    sqlite3_finalize(stmt);
    ++done;
  }

  output_text("row3-b2 churn: completed ");
  output_uint(done);
  output_text(" of ");
  output_uint(iters);
  output_text(" iters; carved_total=");
  output_uint(rof_carved_total);
  output_text(" peak_live=");
  output_uint(peak_live);
  output_text(" arena_left=");
  output_uint(__builtin_capstone_cap_get_end(rof_arena) -
              __builtin_capstone_cap_get_base(rof_arena));
  output_text("\n");
  (void)sqlite3_close(db);
  return 0;
}
#endif

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

#if defined(ROW3_B2_CHURN)
  (void)run_row3_b2_churn((unsigned)(ROW3_B2_CHURN));
#elif defined(ROW3_B2_NO_REVOKE)
  /* Control: suppress the allocator's revoke so free is malloc-slot-recycle only
   * and the post-finalize read returns. Disambiguates the -O0 cause-24 fault. */
  rof_no_revoke = 1;
  (void)run_row3_b2(0);
#else
  (void)run_row3_b2(1);
#endif

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
