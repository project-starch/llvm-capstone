/* row14 -- the LITERAL matched pair for
 * cve-repros/row14_cpython_uninit_connection (UNINIT / use-before-init).
 *
 * before.c allocates a connection wrapper (calloc) and reads through its
 * `sqlite3 *db` field BEFORE sqlite3_open assigns it:
 *
 *     struct connection *c = calloc(1, sizeof(*c));
 *     byte = *(unsigned char *)c->db;   // db is not a usable handle yet
 *
 * On the host that is a NULL/uninitialised-pointer deref (ASan SEGV at 0x0).
 *
 * Here the connection handle `db` is modelled as a real UNINIT capability: it
 * has a tag and a revocation node but NO read/use authority, exactly the state
 * "allocated wrapper, connection not yet opened". A read through it before
 * sqlite3_open FAULTS with cause 26 (Unexpected capability type -- a load
 * through an UNINIT capability, the disclosure the type exists to prevent). This
 * is strictly stronger than the host's NULL deref: it catches a non-NULL but
 * uninitialised handle too.
 *
 *   host    : pre-open read of c->db is a null/uninit deref (before.c).
 *   Capstone: pre-open read of db is a deterministic UNINIT capability fault.
 *
 * The matched pair is the ORDER of the two operations, which is exactly the
 * before.c defect (db used before open assigns it):
 *   - fault variant : read db, THEN sqlite3_open  -> the pre-open read faults.
 *   - correct control: sqlite3_open FIRST (real SQLite overwrites db with a
 *     valid connection handle), THEN read db  -> succeeds and RETURNS.
 * Real SQLite is linked and, in the correct control, actually opens the
 * connection; the read that faults is the literal `*(unsigned char *)db`.
 *
 * UNINIT derivation (task-009): revoke a STILL-LINEAR lineage and the retained
 * handle comes back UNINIT (cursor==end) rather than LIN. We do it on a small
 * SPLIT sub-capability carved off the arena tail, so the rest of the arena stays
 * LINEAR and backs SQLite's real heap for the control's sqlite3_open. No monitor
 * op, no start.S change, no csinit stand-in for open.
 *
 * RESIDUAL (documented, not hidden): real sqlite3_open does not itself PRODUCE
 * the UNINIT capability -- SQLite allocates a fresh connection object and writes
 * its pointer into &db, it does not initialise a caller-provided UNINIT region.
 * So we MINT the UNINIT db handle (from the arena) to model "uninitialised
 * connection", and the real sqlite3_open supplies the valid handle in the
 * correct path. The fault is on a genuine UNINIT capability; the correct path
 * runs real SQLite. See run-sqlite-row14.sh and the history note.
 *
 * Cause 26 does NOT move with -O (task-009): the UNINIT handle keeps its tag and
 * a valid rev node, so it faults on TYPE, not tag-gone, at every opt level.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "revoke_on_free_alloc.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* ---- host-call text output ---- */
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
  output_text("row14 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

/* Carve a small (64-byte) SPLIT sub-capability off the arena tail and turn it
 * into an UNINIT capability -- the model of `db` before open. The remainder of
 * the arena stays LINEAR and is SQLite's heap. NO csdelin on the sub-cap:
 * revoking a still-linear lineage is exactly what yields UNINIT (task-009). */
static void *uninit_db_handle(void) {
  void *arena = rof_arena; /* LIN grant, parked by rof_init */
  unsigned long end = __builtin_capstone_cap_get_end(arena);
  void *slot = rof_split(&arena, end - 64); /* slot=[end-64,end), fresh node, LIN */
  rof_arena = arena;                        /* SQLite heap = remainder */
  void *rev = __builtin_capstone_cap_mrev(slot); /* senior to slot, still LIN */
  return __builtin_capstone_cap_revoke(rev);     /* -> UNINIT over [end-64,end) */
}

/* The literal row14 sequence. `open_first` selects the correct order (control)
 * vs the buggy order (fault). Returns only if the pre-open read did NOT fault. */
static int run_row14(int open_first) {
  int rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &rof_mem_methods);
  if (rc != SQLITE_OK)
    return fail("config-malloc", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);

  /* db starts as a genuine UNINIT capability: allocated wrapper, connection not
   * usable yet. This is before.c's `c->db` in the calloc'd, not-yet-opened
   * wrapper -- but as an UNINIT cap, so even a non-NULL stale handle would trap. */
  sqlite3 *db = (sqlite3 *)uninit_db_handle();

  if (open_first) {
    /* Correct order (control): open the connection FIRST. Real SQLite allocates
     * the sqlite3 object and overwrites db with a valid handle. Then the read
     * through db succeeds and we RETURN. */
    rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK)
      return fail("open", rc);
    output_text("row14 opened connection ok\n");
    volatile unsigned char byte = *(volatile unsigned char *)db;
    output_text("row14 NOTRAP post-open read byte=");
    output_uint((unsigned long)byte);
    output_text("\n");
    (void)sqlite3_close(db);
    return 0;
  }

  /* Buggy order (fault variant): before.c's `byte = *(unsigned char *)c->db`
   * BEFORE open. db is UNINIT -> this load FAULTS with cause 26. */
  volatile unsigned char byte = *(volatile unsigned char *)db;
  output_text("row14 NOTRAP pre-open read byte=");
  output_uint((unsigned long)byte);
  output_text("\n");
  /* Never reached in the fault variant. */
  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK)
    return fail("open", rc);
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

#if defined(ROW14_OPEN_FIRST)
  (void)run_row14(1); /* correct control */
#else
  (void)run_row14(0); /* buggy: pre-open read faults */
#endif

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
