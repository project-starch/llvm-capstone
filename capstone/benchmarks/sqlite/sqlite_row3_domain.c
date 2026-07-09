/* row3 matched pair -- the "after" for cve-repros/row3_diesel_colname_cached.
 *
 * The SAME row3 program as before.c, running the real SQLite C API inside ONE
 * Capstone domain:
 *
 *     open :memory: -> CREATE t(a) -> INSERT 1 -> prepare "SELECT a AS colname"
 *     -> step -> name = column_name(stmt,0) -> finalize -> read name[0]
 *
 * before.c's post-finalize `name[0]` read is a use-after-free: on the host it is
 * ASan heap-use-after-free. Here it is a deterministic CAPABILITY FAULT.
 *
 * WHAT IS AND ISN'T LITERAL (read before citing in the paper):
 *
 *   SQLite hands out `column_name`'s buffer from its memsys5 heap, and a memsys5
 *   allocation is NOT an independently revocable capability: `&zPool[i]` lowers to
 *   cincoffset, which inherits the pool's rev_node_id/type/bounds, so an MREV of
 *   it would mint a node senior to the whole pool and a revoke would sweep the
 *   entire heap. (SPLIT is the only fresh-node derivation and is one-way; memsys5
 *   coalesces. See intra-domain-mrev-revoke-probe/probe_linear_arena.h and
 *   history/09-07-2026_23-05-00_option-b-held-cap-probe-steps-1-3.md, "Step 3".)
 *
 *   So this "after" is the PRAGMATIC single-domain shape (row3 fork B1): a thin
 *   wrapper around column_name carves an independently revocable sub-capability
 *   out of a monitor-granted linear arena, copies the real column-name bytes into
 *   it, and hands THAT alias back as `name`; the finalize wrapper REVOKEs it. The
 *   post-finalize read of the cached `name` faults. It is a matched pair for the
 *   engine + real value + finalize lifecycle + real intra-domain cap fault; the
 *   residual is that the revoked pointer is a carved COPY, not SQLite's own heap
 *   pointer. The literal form (MREV of SQLite's own pointer) is fork B2 and needs
 *   emulator work (a merge op, or a non-coalescing SPLIT-per-allocation heap).
 *
 * The arena is a REAL monitor-delivered REV_TRANSFERRED linear capability -- the
 * domain owns it outright, mints its own MREV, and revokes intra-domain; no
 * start.S/monitor change (task-007). No second domain, no magic sentinel.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U
#define SQLITE_HEAP_SIZE (1024U * 1024U)

/* memsys5-backed SQLite heap. Ordinary .bss; its allocations are deliberately
 * NOT the thing we revoke (see the header comment). */
static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* The monitor-granted LINEAR arena (region #2), parked LIN across the
 * REGION_SHARE and CALL entries. 16-byte aligned by the i128 pointer ABI. */
static void *row3_arena;

/* ---- host-call text output (mirrors sqlite_capstone_domain.c) ----
 *
 * Uses the modelled __builtin_capstone_cap_delin, not a raw `.insn`+"+r" delin:
 * at -O2 the hand-rolled inline-asm form let the register allocator feed delin a
 * value it did not model as a capability, tripping helper_csdelin's tag assert.
 * The builtin is what the -O2-green probes use. */
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

/* ---- revocable-carve glue over the granted linear arena ----
 *
 * cssplit rd, rs1, rs2 -- split rs1 at address rs2: rs1 keeps [base,mid), rd
 * gets [mid,end) with a FRESH revocation-tree node (independently revocable).
 * No builtin/mnemonic exists, hence the raw R-type encoding. See
 * intra-domain-mrev-revoke-probe/probe_domain.h. */
static inline void *row3_cap_split(void **lo, unsigned long mid) {
  void *hi;
  void *l = *lo;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x06, %0, %1, %2"
                   : "=&r"(hi), "+r"(l)
                   : "r"(mid));
  *lo = l;
  return hi;
}

typedef struct {
  void *alias; /* NONLIN copyable alias handed to the caller as `name` */
  void *rev;   /* revocation handle kept by the owner, fired at finalize */
} row3_protected_buf;

/* Carve `len` bytes off the tail of the remaining arena as an independently
 * revocable buffer. One-way (no merge op); `len` must be < what remains. */
static inline row3_protected_buf row3_carve(unsigned long len) {
  row3_protected_buf b;
  void *cur = row3_arena; /* LIN */
  unsigned long end = __builtin_capstone_cap_get_end(cur);
  void *hi = row3_cap_split(&cur, end - len); /* fresh rev node, still LIN */
  row3_arena = cur;                           /* remainder [base, end-len) */
  b.rev = __builtin_capstone_cap_mrev(hi);    /* senior to hi's node only */
  b.alias = __builtin_capstone_cap_delin(hi); /* copyable, still revocable */
  return b;
}

static int fail(const char *stage, int rc) {
  output_text("row3 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=?\n");
  return rc ? rc : 1;
}

/* The row3 sequence, real SQLite, one domain. Returns only if the
 * post-finalize read does NOT fault (which is the bug we are demonstrating is
 * caught). A capability fault halts the domain and the evidence is the monitor's
 * fault line -- see run-sqlite-row3.sh. */
static int run_row3(int revoke_at_finalize) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                          (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK)
    return fail("config-heap", rc);
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
   * The wrapper: carve a revocable copy of the real column name. */
  const char *raw = sqlite3_column_name(stmt, 0); /* "colname", memsys5 ptr */
  if (!raw)
    return fail("column_name", 1);
  row3_protected_buf b = row3_carve(256);
  char *name = (char *)b.alias;
  unsigned i = 0;
  while (raw[i] && i < 255) {
    name[i] = raw[i];
    ++i;
  }
  name[i] = '\0';

  /* Control: the carved alias reads correctly while it is live. */
  output_text("row3 live name=");
  output_text(name);
  {
    char one[2] = {name[0], '\0'};
    output_text("\nrow3 live name[0]=");
    output_text(one);
    output_text("\n");
  }

  /* before.c:  sqlite3_finalize(stmt);  -- the revoke lands here. */
  rc = sqlite3_finalize(stmt);
  stmt = 0;
  if (rc != SQLITE_OK)
    return fail("finalize", rc);
  if (revoke_at_finalize)
    __builtin_capstone_cap_revoke(b.rev); /* lifecycle point */

  /* before.c:  name[0]  -- USE AFTER FINALIZE.
   * With the revoke, this cached alias FAULTS. Kept volatile so the optimiser
   * cannot fold it away; at -O1+ the alias stays in a register across the revoke
   * so the fault is cause 25 (self-proving). */
  volatile char c = name[0];
  output_text("row3 post-finalize NOTRAP name[0]=");
  {
    char one[2] = {(char)c, '\0'};
    output_text(one);
    output_text("\n");
  }

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
      row3_arena = arg; /* the LINEAR grant, tag intact */
    ++shared_region_count;
    return;
  }

  if (hostcall_metadata)
    hostcall_metadata->length = 0;

  /* ROW3_NO_REVOKE builds the control .dom: identical program, no revoke at
   * finalize, so the post-finalize read succeeds and the domain RETURNS. This
   * is the disambiguation control for the -O0 fault (cause 24 = "tag gone",
   * which a plain reload also produces); the fault .dom faults, this one does
   * not. See run-sqlite-row3.sh. */
#if defined(ROW3_NO_REVOKE)
  (void)run_row3(0);
#else
  (void)run_row3(1);
#endif

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
