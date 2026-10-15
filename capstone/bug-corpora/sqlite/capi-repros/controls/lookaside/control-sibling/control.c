/* la-control-sibling -- a lookaside sibling stays usable, and intact, when its neighbour is
 * released.
 *
 * Pre-registration: docs/plans/2026-10-12-nested-lifetime-controls-extended.md, control 2 at
 * the lookaside. Take slots A and B from one connection's lookaside (sqlite3DbMallocRaw), fill
 * both, release A (sqlite3DbFree), then read B through B's own reference and check its bytes,
 * write it, and read the write back. On the protected arm A and B are sibling CDERIVE children
 * of lookaside.pLife, and sqlite3DbFree(db, A) is CREVOKE(pLife, A).
 *
 * Expected on both arms: every access completes and B's contents are intact. The control
 * reports whether A and B are adjacent slots.
 */
#include "sqlite3-capstone.c"
#include "repro322_common.h"
#include "la_common.h"

#define N 128

static int run_case(void) {
  unsigned char pata[N], patb[N], patb2[N], got[N];
  if (repro_init()) return 1;
  sqlite3 *db = la_open();
  if (!db) return FAILRC("open", SQLITE_ERROR);

  unsigned char *a = la_alloc(db, "A", N);
  unsigned char *b = la_alloc(db, "B", N);
  if (!a || !b) return FAILRC("alloc", SQLITE_NOMEM);
  unsigned long a_addr = ctl_addr(a), b_addr = ctl_addr(b);
  printf("SIBLING B-A=%ld slot_size=%u adjacent=%d\n", (long)(b_addr - a_addr),
         (unsigned)db->lookaside.sz,
         (long)(b_addr - a_addr) == (long)db->lookaside.sz ||
             (long)(a_addr - b_addr) == (long)db->lookaside.sz);

  ctl_fill(pata, N, 0xA0);
  ctl_fill(patb, N, 0xB0);
  PROBE_WRITE("A-fill", a, N, pata);
  PROBE_WRITE("B-fill", b, N, patb);

  sqlite3DbFree(db, a);
  printf("FREED A addr=0x%lx pFree=0x%lx\n", a_addr, ctl_addr(db->lookaside.pFree));

  PROBE_READ("sibling-read", b, N, got);
  ctl_check("sibling-B-intact", !memcmp(got, patb, N));
  ctl_fill(patb2, N, 0xB5);
  PROBE_WRITE("sibling-write", b, N, patb2);
  PROBE_READ("sibling-readback", b, N, got);
  ctl_check("sibling-write-read-back", !memcmp(got, patb2, N));

  sqlite3DbFree(db, b);
  la_close(db);
  printf("CONTROL la-control-sibling RESULT %s failures=%d\n",
         ctl_failures ? "CHECKS-FAILED" : "all-checks-ok", ctl_failures);
  return 0;
}

REPRO322_MAIN("la-control-sibling")
