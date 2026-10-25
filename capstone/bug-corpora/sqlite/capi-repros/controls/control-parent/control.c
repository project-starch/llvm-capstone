/* control-parent -- the parent reference stays usable after a child is released.
 *
 * Pre-registration: docs/plans/2026-10-11-nested-lifetime-controls.md, control 1. Allocate A
 * through memsys5 (sqlite3_malloc over SQLITE_CONFIG_HEAP), release it with sqlite3_free, then
 * read and write A's former range through mem5.zPool + offset: the reference memsys5 itself
 * reaches every block through, its in-band free-list link included (MEM5LINK). On the protected
 * arm (sublet-3220000-memsys5.patch) A was a CDERIVE child of mem5.zPool and sqlite3_free
 * revoked it (CREVOKE); the pool is the parent and is not revoked.
 *
 * Expected on both arms: every access completes, and the read returns what memsys5 itself left
 * there -- its Mem5Link {next = the free list's previous head, prev = -1} in the first 8 bytes,
 * A's own bytes after them. The write is undone before memsys5 runs again (the link is live
 * allocator state), and the next request of A's size must get A's block back.
 *
 * The amalgamation is compiled INTO this translation unit (build-controls.sh): mem5 is a static
 * of sqlite3.c, and no other file can name the pool memsys5 uses.
 */
#include "sqlite3-capstone.c"
#include "repro322_common.h"
#include "control_common.h"

#define N 128

static int run_case(void) {
  unsigned char pat[N], got[N], saved[N], scratch[N];
  if (repro_init()) return 1;

  unsigned char *a = sqlite3_malloc(N);
  unsigned char *g = sqlite3_malloc(N);   /* stays checked out for the whole control */
  if (!a || !g) return FAILRC("malloc", SQLITE_NOMEM);
  ctl_bounds("pool", mem5.zPool);
  ctl_bounds("A", a);
  ctl_bounds("G", g);
  unsigned long a_addr = ctl_addr(a);     /* taken while A is live; A is never touched after its free */
  unsigned long off = a_addr - ctl_addr(mem5.zPool);
  int ia = (int)(off / (unsigned long)mem5.szAtom);
  int lg = mem5.aCtrl[ia] & CTRL_LOGSIZE;
  int ib = ia ^ (1 << lg);                /* A's buddy at A's size */
  printf("MEM5 szAtom=%d A.index=%d A.logsize=%d buddy.index=%d G.index=%d\n", mem5.szAtom, ia, lg,
         ib, (int)((ctl_addr(g) - ctl_addr(mem5.zPool)) / (unsigned long)mem5.szAtom));

  ctl_fill(pat, N, 0xA0);
  PROBE_WRITE("A-fill", a, N, pat);

  int head_before = mem5.aiFreelist[lg];
  int buddy_free = (mem5.aCtrl[ib] & CTRL_FREE) && (mem5.aCtrl[ib] & CTRL_LOGSIZE) == lg;
  sqlite3_free(a);
  printf("FREED A addr=0x%lx freelist[%d] head before=%d after=%d\n", a_addr, lg, head_before,
         mem5.aiFreelist[lg]);
  /* The precondition of the expected value: A did not coalesce, so it heads its free list. */
  ctl_check("setup-A-not-coalesced", !buddy_free);
  ctl_check("setup-A-heads-free-list",
            (mem5.aCtrl[ia] & CTRL_FREE) && (mem5.aCtrl[ia] & CTRL_LOGSIZE) == lg &&
            mem5.aiFreelist[lg] == ia);

  u8 *former = &mem5.zPool[off];          /* the parent reference, at A's former start */
  printf("PARENT ref=0x%lx (mem5.zPool + %lu)\n", ctl_addr(former), off);
  PROBE_READ("parent-read", former, N, got);
  Mem5Link link;
  memcpy(&link, got, sizeof link);
  printf("LINK read next=%d prev=%d, memsys5 wrote next=%d prev=-1\n", link.next, link.prev,
         head_before);
  ctl_check("parent-read-is-memsys5-link", link.next == head_before && link.prev == -1);
  ctl_check("parent-read-rest-is-A-bytes",
            !memcmp(got + sizeof link, pat + sizeof link, N - sizeof link));

  memcpy(saved, got, N);
  ctl_fill(scratch, N, 0x5A);
  PROBE_WRITE("parent-write", former, N, scratch);
  PROBE_READ("parent-readback", former, N, got);
  ctl_check("parent-write-read-back", !memcmp(got, scratch, N));
  PROBE_WRITE("parent-restore", former, N, saved);

  unsigned char *again = sqlite3_malloc(N);
  printf("AGAIN addr=0x%lx\n", ctl_addr(again));
  ctl_check("memsys5-reissues-A-block", again && ctl_addr(again) == a_addr);
  sqlite3_free(again);
  sqlite3_free(g);
  printf("CONTROL control-parent RESULT %s failures=%d\n", ctl_failures ? "CHECKS-FAILED" : "all-checks-ok",
         ctl_failures);
  return 0;
}

REPRO322_MAIN("control-parent")
