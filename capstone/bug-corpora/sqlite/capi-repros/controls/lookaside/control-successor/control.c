/* la-control-successor -- the successor of a released lookaside slot is usable, its stale
 * reference is not.
 *
 * Pre-registration: docs/plans/2026-10-12-nested-lifetime-controls-extended.md, control 3 at
 * the lookaside. Take slot A from a connection's lookaside (sqlite3DbMallocRaw), fill it, keep a
 * copy of A's reference, release A (sqlite3DbFree); then take C of A's size until the lookaside
 * hands out A's start address again (at most MAX_TRIES requests, each one kept, so the loop
 * cannot see the same slot twice), print both addresses and require them equal; write C and
 * read it back. Every C is checked to be a lookaside slot, as A is.
 *
 * Built twice from this file (build-controls.sh):
 *   la-control-successor        stops there. Expected on both arms: completes.
 *   la-control-successor-stale  CONTROL_STALE: the same sequence, then ONE read through the
 *                               kept reference to A at the probe `stale-A-read`. Expected: the
 *                               protected arm faults there with cause 25 (revoked lifetime);
 *                               the release arm completes and the read returns C's bytes.
 */
#include "sqlite3-capstone.c"
#include "repro322_common.h"
#include "la_common.h"

#define N 128
#define MAX_TRIES 64

#ifdef CONTROL_STALE
#define CONTROL_TAG "la-control-successor-stale"
#else
#define CONTROL_TAG "la-control-successor"
#endif

static int run_case(void) {
  unsigned char pata[N], patc[N], got[N];
  unsigned char *held[MAX_TRIES];
  int nheld = 0, tries;
  if (repro_init()) return 1;
  sqlite3 *db = la_open();
  if (!db) return FAILRC("open", SQLITE_ERROR);

  unsigned char *a = la_alloc(db, "A", N);
  if (!a) return FAILRC("alloc", SQLITE_NOMEM);
  unsigned long a_addr = ctl_addr(a);
  ctl_fill(pata, N, 0xA0);
  PROBE_WRITE("A-fill", a, N, pata);

  unsigned char *volatile stale = a;      /* the old reference, kept in memory past the free */
  sqlite3DbFree(db, a);
  printf("FREED A addr=0x%lx\n", a_addr);

  unsigned char *c = 0;
  for (tries = 1; tries <= MAX_TRIES; tries++) {
    unsigned char *x = la_alloc(db, "C-try", N);
    if (!x) break;
    printf("TRY %d addr=0x%lx\n", tries, ctl_addr(x));
    if (ctl_addr(x) == a_addr) { c = x; break; }
    held[nheld++] = x;
  }
  printf("SUCCESSOR A=0x%lx C=0x%lx tries=%d\n", a_addr, c ? ctl_addr(c) : 0UL, c ? tries : -1);
  ctl_check("C-start-equals-A-start", c != 0);
  if (!c) {
    printf("CONTROL %s RESULT CHECKS-FAILED failures=%d\n", CONTROL_TAG, ctl_failures);
    return 1;
  }
  ctl_bounds("C", c);

  ctl_fill(patc, N, 0xC0);
  PROBE_WRITE("C-write", c, N, patc);
  PROBE_READ("C-read", c, N, got);
  ctl_check("C-write-read-back", !memcmp(got, patc, N));

#ifdef CONTROL_STALE
  PROBE_READ("stale-A-read", stale, N, got);
  ctl_check("stale-read-returns-C-bytes", !memcmp(got, patc, N));
#endif

  for (int i = 0; i < nheld; i++) sqlite3DbFree(db, held[i]);
  sqlite3DbFree(db, c);
  la_close(db);
  printf("CONTROL %s RESULT %s failures=%d\n", CONTROL_TAG, ctl_failures ? "CHECKS-FAILED" : "all-checks-ok",
         ctl_failures);
  return 0;
}

REPRO322_MAIN(CONTROL_TAG)
