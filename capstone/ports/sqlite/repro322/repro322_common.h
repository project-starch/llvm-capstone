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

#ifdef REPRO322_VIRTUAL
/* THE VIRTUAL ARM. Same cases, same file, an ordinary program.
 *
 * The scaffolding above is a freestanding domain's: output goes through shared
 * hostcall regions, the pool arrives as a granted region, and the entry point
 * is domain_main(), which the physical monitor calls. None of that exists in
 * the virtual address space, where a case is a Linux process under
 * capstone-vexec with a Capstone musl. So this branch supplies the SAME five
 * things over libc -- out_text, out_uint, repro_init, REPRO322_MAIN, FAILRC --
 * and the cases are not touched.
 *
 * WHICH ALLOCATOR IS UNDER THE CASE, which is the whole point of the arm:
 *
 *   default            SQLite's own sqlite3MemMalloc, i.e. the platform's
 *                      malloc. On this platform that is the virtual runtime's
 *                      allocator: every SQLite allocation is its own object,
 *                      bounded to the request, and freeing it revokes every
 *                      copy of the alias. The system allocator therefore sees
 *                      each engine lifetime individually.
 *   REPRO322_VIRTUAL_MEMSYS5
 *                      SQLITE_CONFIG_HEAP over one static array, so memsys5
 *                      sub-allocates inside a single object exactly as the
 *                      physical control arm does. The system allocator sees
 *                      one allocation and no frees at all.
 *   REPRO322_VIRTUAL_SUBLET
 *                      the same heap, with memsys5 under its Sublet port
 *                      (ports/sqlite/sublet/sublet-3220000-memsys5.patch):
 *                      every block goes out as a child lifetime of the pool,
 *                      bounded to the block, and every sqlite3_free revokes it.
 *
 * The three are the measurement: the same defect and the same binary modulo
 * one configure call, with the nested allocator absent, present, and present
 * and protected. Requires
 * SQLITE_ENABLE_MEMSYS5 in the amalgamation; without it sqlite3_config returns
 * SQLITE_ERROR and repro_init says so rather than running the case.
 */
#include "sqlite3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void out_text(const char *text) { fputs(text, stdout); }
static void out_uint(unsigned long v) { printf("%lu", v); }
static int run_case(void);

#ifndef SQLITE_HEAP_SIZE
#define SQLITE_HEAP_SIZE (1024U * 1024U)
#endif
#ifdef REPRO322_VIRTUAL_MEMSYS5
static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));
#endif

static int repro_init(void) {
  int rc;
#ifdef REPRO322_VIRTUAL_MEMSYS5
#ifdef REPRO322_NO_LOOKASIDE
  /* THE PROBE. SQLite has TWO nested allocators and the Sublet arm's patch
   * (ports/sqlite/sublet/sublet-3220000-memsys5.patch) covers one. The
   * lookaside takes a connection's small allocations -- up to LOOKASIDE_SMALL
   * -- and its free is `pBuf->pNext = db->lookaside.pFree`, a push onto a free
   * list with no revoke and no per-object bound.
   *
   * Turning the lookaside off sends those allocations to memsys5. If a case
   * silent on the Sublet arm then faults, the silence was the unprotected
   * layer and not memsys5. */
  rc = sqlite3_config(SQLITE_CONFIG_LOOKASIDE, 0, 0);
  out_text("repro lookaside=off rc="); out_uint((unsigned)(rc<0?-rc:rc));
  out_text("\n");
  if (rc != SQLITE_OK) {
    out_text("CONTROL-FAILED could not disable the lookaside\n");
    return rc;
  }
#endif
  rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof sqlite_heap, 64);
  if (rc != SQLITE_OK) {
    /* Never fall through to the platform allocator here: that would silently
     * turn the nested arm into the system-allocator arm and the two columns
     * would stop differing for a reason nobody could see. */
    out_text("CONTROL-FAILED config-heap rc="); out_uint((unsigned)rc);
    out_text(" (build the amalgamation with SQLITE_ENABLE_MEMSYS5)\n");
    return rc;
  }
#ifdef REPRO322_VIRTUAL_SUBLET
  out_text("repro allocator=memsys5-sublet heap_bytes=");
#else
  out_text("repro allocator=memsys5 heap_bytes=");
#endif
  out_uint(sizeof sqlite_heap);
  out_text("\n");
#else
  out_text("repro allocator=platform\n");
#endif
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK) {
    out_text("CONTROL-FAILED initialize rc="); out_uint((unsigned)rc); out_text("\n");
    return rc;
  }
  return 0;
}

/* The tag on the command line is the fixture check the contract asks for: a
 * program told to run another case refuses instead of running this one and
 * reporting under the wrong name. BEGIN is printed before anything else, so
 * its absence means the image did not run rather than that the case was
 * silent, and RETURNED is printed only on the way out, so a fault is visible
 * as the missing line as well as through the launcher's own record. */
#define REPRO322_MAIN(TAG)                                               \
  int main(int argc, char **argv) {                                      \
    setvbuf(stdout, NULL, _IONBF, 0);                                    \
    if (argc > 1 && strcmp(argv[1], (TAG))) {                            \
      fprintf(stderr, "CONTROL-FAILED fixture is %s, run asked for %s\n", \
              (TAG), argv[1]);                                           \
      return 75;                                                         \
    }                                                                    \
    printf("%s BEGIN\n", (TAG));                                         \
    (void)run_case();                                                    \
    printf("%s RETURNED\n", (TAG));                                      \
    return 0;                                                            \
  }

#define FAILRC(stage, rc) (out_text(stage), out_text(" rc="), \
  out_uint((unsigned long)((rc) < 0 ? -(rc) : (rc))), out_text("\n"), (rc) ? (rc) : 1)

#else /* the freestanding domain scaffolding */

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

#endif /* REPRO322_VIRTUAL */

#endif
