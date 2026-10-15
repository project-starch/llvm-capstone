/* la_common.h -- what the lookaside controls (README.md here) add to ../control_common.h.
 *
 * The three controls drive SQLite's LOOKASIDE, not memsys5: a connection opened with the
 * default lookaside (SQLITE_DEFAULT_LOOKASIDE, 1200-byte slots), allocations by
 * sqlite3DbMallocRaw(db, n) and releases by sqlite3DbFree(db, p) -- the two internal entry
 * points the lifetime patch instruments (lookasideIssue / lookasideRelease). Both are static to
 * the amalgamation, so each control #includes the staged sqlite3-capstone.c, as the memsys5
 * controls do.
 *
 * la_open() opens the connection and prints the lookaside's state. la_alloc() is the only way a
 * control takes a block: it records the lookaside hit counter around the request and checks
 * that the block is a lookaside slot -- inside [pStart, pEnd), on a slot boundary, counted as
 * one lookaside hit, and reported by sqlite3DbMallocSize as the slot size. A block that came from
 * the backing allocator instead fails those checks, and the runner then reports the control as
 * not held.
 *
 * Two probes beyond control_common.h's byte probes: the lookaside's free-list link is a POINTER
 * stored in the freed slot (LookasideSlot.pNext). A byte copy carries no capability tag, so the
 * parent control reads and restores that link with one typed load and one typed store, each on
 * one source line, printed the same way as the byte probes.
 */
#ifndef LA_COMMON_H
#define LA_COMMON_H

#include "control_common.h"

#define PROBE_LOAD_PTR(label, slot, out) do { LookasideSlot *volatile *ctl_q_ = (LookasideSlot *volatile *)(slot); ctl_say((label), "read", __FILE__, __LINE__, (const void *)ctl_q_, (unsigned)sizeof(LookasideSlot *)); (out) = *ctl_q_; printf("PROBE %s done\n", (label)); } while (0)

#define PROBE_STORE_PTR(label, slot, val) do { LookasideSlot *volatile *ctl_q_ = (LookasideSlot *volatile *)(slot); ctl_say((label), "write", __FILE__, __LINE__, (const void *)ctl_q_, (unsigned)sizeof(LookasideSlot *)); *ctl_q_ = (val); printf("PROBE %s done\n", (label)); } while (0)

static sqlite3 *la_open(void) {
  sqlite3 *db = 0;
  int rc = sqlite3_open(":memory:", &db);
  printf("LA open rc=%d\n", rc);
  if (rc != SQLITE_OK) return 0;
  Lookaside *l = &db->lookaside;
  printf("LA sz=%u nSlot=%u bDisable=%u bMalloced=%u pStart=0x%lx pEnd=0x%lx hits=%u "
         "ptr_bytes=%u\n", (unsigned)l->sz, (unsigned)l->nSlot, (unsigned)l->bDisable,
         (unsigned)l->bMalloced, ctl_addr(l->pStart), ctl_addr(l->pEnd), (unsigned)l->anStat[0],
         (unsigned)sizeof(LookasideSlot *));
  ctl_bounds("lookaside-pStart", l->pStart);
#ifdef REPRO322_VIRTUAL_SUBLET
  /* The patch's two lifetimes over the buffer: pBlock (from pStart), pLife (parent of every slot). */
  ctl_bounds("lookaside-pBlock", l->pBlock);
  ctl_bounds("lookaside-pLife", l->pLife);
#endif
  ctl_check("lookaside-enabled", l->bDisable == 0 && l->sz > 0 && l->pStart != (void *)db);
  return db;
}

/* One block from the lookaside, with the evidence that it is a lookaside slot. */
static unsigned char *la_alloc(sqlite3 *db, const char *name, unsigned n) {
  Lookaside *l = &db->lookaside;
  unsigned hits = l->anStat[0];
  unsigned char *p = sqlite3DbMallocRaw(db, n);
  if (!p) { printf("SLOT %s NULL\n", name); ctl_check("alloc-not-null", 0); return 0; }
  unsigned long off = ctl_addr(p) - ctl_addr(l->pStart);
  int within = ctl_addr(p) >= ctl_addr(l->pStart) && ctl_addr(p) < ctl_addr(l->pEnd);
  int dbsize = sqlite3DbMallocSize(db, p);
  printf("SLOT %s addr=0x%lx offset=%lu index=%lu rem=%lu within=%d hit_delta=%u dbmallocsize=%d\n",
         name, ctl_addr(p), off, within ? off / l->sz : 0UL, within ? off % l->sz : 0UL, within,
         l->anStat[0] - hits, dbsize);
  ctl_bounds(name, p);
  char what[64];
  snprintf(what, sizeof what, "%s-is-lookaside-slot", name);
  ctl_check(what, within && off % l->sz == 0 && l->anStat[0] - hits == 1 && dbsize == l->sz);
  return p;
}

static void la_close(sqlite3 *db) {
  int rc = sqlite3_close(db);
  printf("LA close rc=%d\n", rc);
  ctl_check("close-ok", rc == SQLITE_OK);
}

#endif
