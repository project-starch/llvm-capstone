/* repro322_common.h -- shared control-arm scaffolding for the SQLite 3.22.0
 * temporal-bug corpus on UNPROTECTED Capstone.
 *
 * Every case file does:
 *     #include "repro322_common.h"
 *     static int run_case(void) { ...drive SQLite to the freed-then-used path... }
 *     REPRO322_MAIN("<tag>")
 *
 * CONTROL arm: SQLite's own memsys5 heap + lookaside, NOTHING revoked. On
 * unprotected Capstone the post-free access is not caught, so run_case() must
 * reach its "<tag> NOTRAP done" print and RETURN. The matching host ASan build
 * is what flags the heap-use-after-free / double-free. All bugs are public and
 * already fixed upstream; collected for the Capstone/Sublet temporal-safety study.
 */
#ifndef REPRO322_COMMON_H
#define REPRO322_COMMON_H

#include "sqlite3.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

#ifndef SQLITE_HEAP_SIZE
#define SQLITE_HEAP_SIZE (1024U * 1024U)
#endif
static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

static void out_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload) return;
  /* Under the gp-captable (silicon) ABI both capabilities arrive NON-LINEAR (string literals from
     cap-table storage, the payload through the cap-table too), and the RTL's DELIN raises
     UNEXPECTED_CAPABILITY_TYPE on any non-linear operand, a wedge on this RTL, where QEMU's
     helper returns early. Same fix as output_text in sqlite_capstone_domain.c (ISSUES S-02,
     S-15). The QEMU corpus build defines no CAPSTONE_GP_CAPTABLE_ABI and is unchanged. */
#ifdef CAPSTONE_GP_CAPTABLE_ABI
  const char *src = text;
  char *payload = (char *)hostcall_payload;
#else
  const char *src = (const char *)__builtin_capstone_cap_delin((void *)text);
  char *payload = (char *)__builtin_capstone_cap_delin((void *)hostcall_payload);
#endif
  unsigned long offset = hostcall_metadata->length;
  while (*src && offset + 1 < SQLITE_HC_REGION_SIZE) payload[offset++] = *src++;
  hostcall_metadata->length = offset;
}
static void out_uint(unsigned long v) {
  char buf[21]; unsigned i = 21; buf[--i] = '\0';
  if (v == 0) buf[--i] = '0';
  while (v && i) { buf[--i] = (char)('0' + (v % 10)); v /= 10; }
  out_text(&buf[i]);
}
static int run_case(void);

/* config memsys5 + init; return 0 on success */
static int repro_init(void) {
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK) { out_text("repro ERROR config-heap rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK) { out_text("repro ERROR initialize rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  return 0;
}

#define REPRO322_MAIN(TAG)                                               \
  void domain_main(unsigned *res, unsigned func) {                      \
    if (func == CAPSTONE_DPI_REGION_SHARE) {                            \
      if (shared_region_count == 0)                                     \
        hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)res;  \
      else if (shared_region_count == 1)                                \
        hostcall_payload = (volatile char *)res;                       \
      ++shared_region_count;                                            \
      return;                                                           \
    }                                                                   \
    if (hostcall_metadata) hostcall_metadata->length = 0;              \
    (void)run_case();                                                   \
    if (res) *res = SQLITE_HC_RET_DONE;                                 \
  }

/* helper: a fail() that prints "<stage> rc=<n>" and returns rc (nonzero) */
#define FAILRC(stage, rc) (out_text(stage), out_text(" rc="), out_uint((unsigned long)((rc)<0?-(rc):(rc))), out_text("\n"), (rc)?(rc):1)

#endif
