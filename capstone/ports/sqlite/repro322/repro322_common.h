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

#ifdef REPRO322_SUBLET
/* SUBLET MODE.
 *
 * Sublet's discipline is that the pool arrives from the level below as a LINEAR
 * capability and never sits in a C variable. The static sqlite_heap[] above is
 * domain-local storage, not a grant, so it cannot carry the discipline: with it,
 * memsys5's Sublet port reads an empty slot and faults inside capstone_cap_base
 * on `lcc rs2=3` (base). That is the port failing, not a bug being caught, and it
 * is what three domains did before this was added.
 *
 * So in this mode the host supplies the pool as shared region 2 -- run the domain
 * under `h.user --arena <bytes>`, which shares it LINEAR (`--pool` is the
 * non-linear variant for the unprotected arm) -- and the domain hands it straight
 * to the port's slot and nowhere else. sqlite3_config(SQLITE_CONFIG_HEAP, ...) is
 * NOT called: memsys5 reaches its memory through the Sublet primitives instead. */
void sqlite3_sublet_grant(void *pLinear);
/* Region 3. Under Sublet, memsys5's bookkeeping moves OUT OF BAND -- aLink, aCap and
 * aPar live beside the pool, not inside it -- and SQLITE_CONFIG_HEAP is given THAT
 * region, not the pool. The pool arrives separately as the linear grant. Sizing is
 * speedtest1_domain.c's, for `atoms` atoms of szAtom bytes:
 *     ((atoms+15)&~15)        aCtrl, one byte per atom
 *   + atoms*8                 aLink, the free-list links
 *   + atoms*16                aCap,  one capability slot per atom
 *   + (atoms+32)*16           aPar,  the handle taken before each split
 *   + 64                      slack
 * Run with `h.user <dom> --tail --arena <pool> --tables <this>`. */
static volatile unsigned char *repro_tables_region;
static unsigned long repro_tables_size;
#define REPRO322_SUBLET_TABLES(atoms) \
  ((unsigned long)(((atoms)+15)&~15UL) + (unsigned long)(atoms)*8 \
   + (unsigned long)(atoms)*16 + ((unsigned long)(atoms)+32)*16 + 64)
#endif

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
#ifdef REPRO322_SUBLET
  /* Three things must line up, and getting any of them wrong looks like a clean run.
   *
   * 1. CONFIG_HEAP must still be called: it is what INSTALLS memsys5, and with
   *    -DSQLITE_ZERO_MALLOC skipping it leaves no allocator at all, so
   *    sqlite3_initialize() fails and every case returns before doing anything.
   * 2. The buffer it is given is the TABLES region, not the pool and not the static
   *    array: the Sublet port keeps aLink/aCap/aPar out of band. Passing the static
   *    sqlite_heap[] here makes the side tables describe a pool that is not the one
   *    memsys5Init reads from the grant -- the domain then hangs, or faults on
   *    `lcc` with an untagged operand inside Sublet's own primitives.
   * 3. The pool itself arrives as the linear grant (region 2). */
  if (!repro_tables_region) {
    out_text("repro ERROR no tables region -- run with --tail --arena N --tables M\n");
    return 1;
  }
  {
    unsigned long atoms = (unsigned long)SQLITE_HEAP_SIZE / 64UL;
    int rc = sqlite3_config(SQLITE_CONFIG_HEAP, (void *)repro_tables_region,
                            (int)REPRO322_SUBLET_TABLES(atoms), 64);
    out_text("repro allocator=sublet-grant tables_rc=");
    out_uint((unsigned long)(rc < 0 ? -rc : rc));
    out_text(" tables_bytes="); out_uint(REPRO322_SUBLET_TABLES(atoms)); out_text("\n");
    if (rc != SQLITE_OK) return rc;
    rc = sqlite3_initialize();
    out_text("repro sublet init_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
    return rc;
  }
#endif
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK) { out_text("repro ERROR config-heap rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK) { out_text("repro ERROR initialize rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  return 0;
}

#ifdef REPRO322_SUBLET
/* the pool, linear, into the port's slot and nowhere else */
#define REPRO322_SUBLET_GRANT                                            \
      else if (shared_region_count == 2)                                 \
        sqlite3_sublet_grant((void *)res); /* the pool, linear */        \
      else if (shared_region_count == 3)                                 \
        repro_tables_region = (volatile unsigned char *)res;
#else
#define REPRO322_SUBLET_GRANT
#endif

#define REPRO322_MAIN(TAG)                                               \
  void domain_main(unsigned *res, unsigned func) {                      \
    if (func == CAPSTONE_DPI_REGION_SHARE) {                            \
      if (shared_region_count == 0)                                     \
        hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)res;  \
      else if (shared_region_count == 1)                                \
        hostcall_payload = (volatile char *)res;                       \
      REPRO322_SUBLET_GRANT                                             \
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
