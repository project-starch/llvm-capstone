/* control-sibling -- a sibling stays usable, and intact, when its neighbour is released.
 *
 * Pre-registration: docs/plans/2026-10-11-nested-lifetime-controls.md, control 2. Allocate A and
 * B of the same size from memsys5's pool, fill both, release A with sqlite3_free, then read B
 * through B's own reference and check its bytes, write it, and read the write back. On the
 * protected arm (sublet-3220000-memsys5.patch) A and B are sibling CDERIVE children of
 * mem5.zPool, and sqlite3_free(A) is CREVOKE(pool, A).
 *
 * Expected on both arms: every access completes and B's contents are intact. The control
 * reports whether A and B are buddies (adjacent halves of one memsys5 block), and A is the
 * lower address of the two either way.
 *
 * Built into one translation unit with the amalgamation, as the other controls are
 * (build-controls.sh), though this one needs no static of sqlite3.c.
 */
#include "sqlite3-capstone.c"
#include "repro322_common.h"
#include "control_common.h"

#define N 128

static int run_case(void) {
  unsigned char pata[N], patb[N], patb2[N], got[N];
  if (repro_init()) return 1;

  unsigned char *a = sqlite3_malloc(N);
  unsigned char *b = sqlite3_malloc(N);
  if (!a || !b) return FAILRC("malloc", SQLITE_NOMEM);
  ctl_bounds("pool", mem5.zPool);
  ctl_bounds("A", a);
  ctl_bounds("B", b);
  unsigned long a_addr = ctl_addr(a), b_addr = ctl_addr(b);
  long ia = (long)((a_addr - ctl_addr(mem5.zPool)) / (unsigned long)mem5.szAtom);
  long ib = (long)((b_addr - ctl_addr(mem5.zPool)) / (unsigned long)mem5.szAtom);
  int lg = mem5.aCtrl[ia] & CTRL_LOGSIZE;
  printf("MEM5 A.index=%ld B.index=%ld logsize=%d buddies=%d B-A=%ld\n", ia, ib, lg,
         (ia ^ (1L << lg)) == ib, (long)(b_addr - a_addr));

  ctl_fill(pata, N, 0xA0);
  ctl_fill(patb, N, 0xB0);
  PROBE_WRITE("A-fill", a, N, pata);
  PROBE_WRITE("B-fill", b, N, patb);

  sqlite3_free(a);
  printf("FREED A addr=0x%lx\n", a_addr);

  PROBE_READ("sibling-read", b, N, got);
  ctl_check("sibling-B-intact", !memcmp(got, patb, N));
  ctl_fill(patb2, N, 0xB5);
  PROBE_WRITE("sibling-write", b, N, patb2);
  PROBE_READ("sibling-readback", b, N, got);
  ctl_check("sibling-write-read-back", !memcmp(got, patb2, N));

  sqlite3_free(b);
  printf("CONTROL control-sibling RESULT %s failures=%d\n", ctl_failures ? "CHECKS-FAILED" : "all-checks-ok",
         ctl_failures);
  return 0;
}

REPRO322_MAIN("control-sibling")
