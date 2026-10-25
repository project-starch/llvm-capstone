/* la-control-parent -- the lookaside's own reference to its buffer stays usable after a slot
 * is released.
 *
 * Pre-registration: docs/plans/2026-10-12-nested-lifetime-controls-extended.md, control 1 at
 * the lookaside. Take slots A and D from a connection's lookaside (sqlite3DbMallocRaw), release
 * D, then A (sqlite3DbFree), and read and write A's former range through
 * db->lookaside.pStart + offset: the reference through which the lookaside reaches every slot,
 * its in-band free-list link included. On the protected arm (sublet-3220000-memsys5.patch,
 * lookasideRelease) A was a CDERIVE child of lookaside.pLife and sqlite3DbFree revoked it
 * (CREVOKE); pStart is not revoked.
 *
 * Expected on both arms: every access completes, and the read returns what the lookaside itself
 * left there: its LookasideSlot.pNext link = D (the free list's previous head) in the first
 * pointer-sized bytes, A's own bytes after them. The scratch write is undone before the lookaside
 * runs again -- the link is live allocator state and a POINTER, so it is restored with a typed
 * store -- and the next two requests must get A, then D, back.
 */
#include "sqlite3-capstone.c"
#include "repro322_common.h"
#include "la_common.h"

#define N 128

static int run_case(void) {
  unsigned char pat[N], got[N], saved[N], scratch[N];
  if (repro_init()) return 1;
  sqlite3 *db = la_open();
  if (!db) return FAILRC("open", SQLITE_ERROR);
  Lookaside *l = &db->lookaside;
  const unsigned P = (unsigned)sizeof(LookasideSlot *);

  unsigned char *a = la_alloc(db, "A", N);
  unsigned char *d = la_alloc(db, "D", N);
  if (!a || !d) return FAILRC("alloc", SQLITE_NOMEM);
  unsigned long a_addr = ctl_addr(a), d_addr = ctl_addr(d);  /* taken while live */
  unsigned long off = a_addr - ctl_addr(l->pStart);

  ctl_fill(pat, N, 0xA0);
  PROBE_WRITE("A-fill", a, N, pat);

  sqlite3DbFree(db, d);
  unsigned long head_before = ctl_addr(l->pFree);
  sqlite3DbFree(db, a);
  printf("FREED D addr=0x%lx, then A addr=0x%lx; pFree before A=0x%lx after=0x%lx\n", d_addr,
         a_addr, head_before, ctl_addr(l->pFree));
  ctl_check("setup-D-heads-free-list-before-A", head_before == d_addr);
  ctl_check("setup-A-heads-free-list", ctl_addr(l->pFree) == a_addr);

  u8 *former = &((u8 *)l->pStart)[off];   /* the parent reference, at A's former start */
  printf("PARENT ref=0x%lx (lookaside.pStart + %lu)\n", ctl_addr(former), off);
  ctl_bounds("parent", former);
  PROBE_READ("parent-read", former, N, got);
  LookasideSlot *link = 0;
  PROBE_LOAD_PTR("parent-link-read", former, link);
  printf("LINK read next=0x%lx tag=%d, the lookaside wrote next=0x%lx (D)\n", ctl_addr(link),
         (int)__builtin_capstone_cap_get_tag((void *)link), d_addr);
  ctl_check("parent-read-is-lookaside-link", ctl_addr(link) == d_addr);
  ctl_check("parent-read-link-bytes-not-A-bytes", memcmp(got, pat, P) != 0);
  ctl_check("parent-read-rest-is-A-bytes", !memcmp(got + P, pat + P, N - P));

  memcpy(saved, got, N);
  ctl_fill(scratch, N, 0x5A);
  PROBE_WRITE("parent-write", former, N, scratch);
  PROBE_READ("parent-readback", former, N, got);
  ctl_check("parent-write-read-back", !memcmp(got, scratch, N));
  PROBE_WRITE("parent-restore", former, N, saved);
  PROBE_STORE_PTR("parent-restore-link", former, link);

  unsigned char *again = sqlite3DbMallocRaw(db, N);
  unsigned char *again2 = sqlite3DbMallocRaw(db, N);
  printf("AGAIN addr=0x%lx then 0x%lx\n", ctl_addr(again), ctl_addr(again2));
  ctl_check("lookaside-reissues-A-then-D",
            again && again2 && ctl_addr(again) == a_addr && ctl_addr(again2) == d_addr);
  if (again) sqlite3DbFree(db, again);
  if (again2) sqlite3DbFree(db, again2);
  la_close(db);
  printf("CONTROL la-control-parent RESULT %s failures=%d\n",
         ctl_failures ? "CHECKS-FAILED" : "all-checks-ok", ctl_failures);
  return 0;
}

REPRO322_MAIN("la-control-parent")
