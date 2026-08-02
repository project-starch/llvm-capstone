/* The xlang Capstone column: one domain for all 14 rows.
 *
 * The shims are #define blocks over a shared template, so unlike the sqlite
 * corpus (seven hand-written sqlite_row*_domain.c, because its rows genuinely
 * differ) one parameterised domain covers every row here. Select with
 *   -DROW_SRC='"../cheri/shims/mruby_range_uaf.c"'
 *
 * Topology mirrors sqlite_row3_b2_domain.c: three shared regions arrive via
 * domain_main(func == CAPSTONE_DPI_REGION_SHARE) -- hostcall metadata, the text
 * payload, and the LINEAR arena grant that becomes the allocator's arena.
 *
 * Build -O0. At -O1+ the compiler hoists the cached pointer's load above the
 * free or elides the dangling access entirely, so the access the mechanism must
 * police is never emitted and the row "passes" for the wrong reason. The CHERI
 * column builds -O0 for the same reason; see ALLOCATOR-CONTRACT.md §4.
 */
/* NOT revoke_on_free_alloc.h: it is all-static, so including it here would
 * give this TU a private rof_arena and the mock's allocator would read its own
 * still-null copy (QEMU aborts in helper_cslcc). mock_mruby_capstone.c owns the
 * allocator; we reach it through these two. */
extern void xlang_arena_init(void *grant);
extern void xlang_set_no_revoke(void);
extern void *xlang_probe_alloc_and_revoke(void);

/* The REAL layout, shared with the host. Do not re-declare it locally: the
 * struct is {phase, opcode, offset, length}, so `length` sits at offset 24. A
 * local one-field version would silently write the cursor into `phase` and the
 * host would read a length of 0 -- output lost, and the row scored as a fault
 * that never happened. */
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U
#define XLANG_HC_REGION_SIZE SQLITE_HC_REGION_SIZE

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

static void output_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
  const char *src = (const char *)__builtin_capstone_cap_delin((void *)text);
  char *payload = (char *)__builtin_capstone_cap_delin((void *)hostcall_payload);
  unsigned long offset = hostcall_metadata->length;
  while (*src && offset + 1 < XLANG_HC_REGION_SIZE)
    payload[offset++] = *src++;
  hostcall_metadata->length = offset;
}

/* The shim template calls this on the path where the stale access RETURNED --
 * i.e. the mechanism did not catch it. A row that faults never reaches here,
 * which is what makes the two outcomes distinguishable from the host side. */
void mock_report(const char *row, const char *what) {
  output_text("XLANG ");
  output_text(row);
  output_text(" ");
  output_text(what);
  output_text("\n");
}

/* Harness failure, NOT a verdict: arena depleted, or realloc grew in place so
 * there is no stale pointer to measure. Must be loud -- a silent one would be
 * indistinguishable from a MISS. */
void xlang_fail(const char *msg, int code) {
  output_text("XLANG-INVALID rc=");
  /* NOT a ternary: `code == 66 ? "66 " : "67 "` selects between two capabilities
   * and crashes the backend with `Cannot select: i128 = CapstoneISD::SELECT_CC`
   * (the C-2 family). Branch instead of selecting. */
  if (code == 66)
    output_text("66 ");
  else
    output_text("67 ");
  output_text(msg);
  output_text("\n");
  for (;;)
    ; /* freestanding: no exit(); the host sees the marker and no MOCK line */
}

/* memcpy for the ACCESS_SIZE == 16 rows. Byte-wise on purpose: the stale READ
 * is a plain data read of 16 bytes at a possibly 8-aligned offset (ACCESS_OFF
 * is 72 on some rows), exactly as ASan reported it. A capability-wide load
 * would fault on alignment and be scored as a catch that never happened. */
void *memcpy(void *dst, const void *src, unsigned long n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  for (unsigned long i = 0; i < n; ++i)
    d[i] = s[i];
  return dst;
}

/* memset, for the shims that clear a region. */
void *memset(void *dst, int c, unsigned long n) {
  unsigned char *d = (unsigned char *)dst;
  for (unsigned long i = 0; i < n; ++i)
    d[i] = (unsigned char)c;
  return dst;
}

/* A shim calling abort() means its own precondition failed -- not a verdict.
 * Route it to the loud path so it can never be read as a MISS. */
void abort(void) { xlang_fail("shim called abort(): precondition failed", 66); }

/* The row itself. Brings in main() from the shim template; renamed so the
 * domain owns the entry point. */
#define main xlang_row_main
#include ROW_SRC
#undef main

void domain_main(void *arg, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)arg;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)arg;
    else if (shared_region_count == 2)
      xlang_arena_init(arg); /* the LINEAR grant becomes the arena */
    ++shared_region_count;
    return;
  }

  if (hostcall_metadata)
    hostcall_metadata->length = 0;

#ifdef XLANG_NO_REVOKE
  /* ALLOCATOR-CONTRACT §4: the control. Identical program, allocator and free
   * path, minus the one REVOKE. The stale access must then RETURN and the row
   * must report. Without this a fault cannot be told apart from an unrelated
   * -O0 spill reload, so a verdict without its control is not a verdict. */
  xlang_set_no_revoke();
#endif

#ifdef XLANG_PROBE_SHARES
  /* Diagnostic only: prove the three regions arrived, WITHOUT touching the
   * allocator. Splits "share plumbing broken" from "allocator faults", which a
   * bare cslcc assertion cannot tell apart.
   *
   * Deliberately reads NO capability field: cap_get_base/_end are cslcc, which
   * is the very thing asserting, and there is no cap_get_tag builtin. A plain
   * != 0 compares the address only and cannot trap. Branches, never ternaries
   * over string capabilities (SELECT_CC, see xlang_fail). */
  output_text("XLANG-PROBE shares=");
  if (shared_region_count == 3)
    output_text("3");
  else
    output_text("NOT3");
  output_text(" arena=");
  output_text("(owned by the allocator TU)");
  output_text("\n");
  return;
#endif

#ifdef XLANG_PROBE_DELIVERY
  /* Does OUR allocator's revoke reach the modelled fault path?
   *
   * The rows die on `regs + ACCESS_OFF`, a cscincoffsetimm on an untagged
   * capability, where QEMU has a bare assert(rs1_v->tag) and no exception path
   * -- an emulator gap (13 such asserts vs 46 real raises), not evidence about
   * the architecture. So the arithmetic route cannot answer "is a fault
   * DELIVERED".
   *
   * This dereferences at offset 0 instead: no arithmetic, straight to a load
   * through the revoked capability. sqlite row3_b2 shows that route yields a
   * clean `Cap mem access on revoked capability` (cause 25). If this does too,
   * the revoke reaches the delivery path with OUR allocator and the rows'
   * aborts are the emulator gap rather than a dead mechanism.
   *
   * NOTE what this still does NOT settle: what real hardware does on the
   * ARITHMETIC. That needs the RTL or silicon. */
  {
    void *p = xlang_probe_alloc_and_revoke();
    output_text("XLANG-DELIVERY about-to-deref\n");
    volatile unsigned long v = *(volatile unsigned long *)p; /* offset 0 */
    (void)v;
    output_text("XLANG-DELIVERY NOTRAP deref returned\n");
    return;
  }
#endif

  (void)xlang_row_main();
}
