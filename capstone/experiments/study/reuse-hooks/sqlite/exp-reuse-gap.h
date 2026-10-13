/* Reuse-distance hooks for SQLite 3.22.0's nested allocators, both arms.
 * Pre-registration: capstone/docs/plans/2026-10-11-nested-reuse-distance.md.
 *
 * Included once from the amalgamation, under EXP_REUSE_GAP, by
 * hooks-{stock,protected}.patch; build.sh copies this file and
 * ../../reuse-gap-observer.h next to the patched sqlite3-capstone.c.
 *
 * Two boundaries, one observer each, tables in static storage:
 *   memsys5    memsys5MallocUnsafe hands out a block, memsys5FreeUnsafe
 *              takes it back. Start = &mem5.zPool[i*szAtom], the address the
 *              buddy code itself computes; size = the power-of-two block.
 *   lookaside  sqlite3DbMallocRawNN hands out a slot, sqlite3DbFreeNN pushes
 *              it back on the free list. Start = the slot; size = lookaside.sz.
 * Only integers are recorded. The report is printed at exit, one line per
 * boundary on stderr:
 *   REUSE-GAP <boundary> attempts=.. issues=.. releases=.. reuses=..
 *     distinct=.. capacity=.. error=.. bins=b0,..,b31
 * bins[b] holds distances of bit length b + 1. */
#ifndef EXP_REUSE_GAP_SQLITE_H
#define EXP_REUSE_GAP_SQLITE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "reuse-gap-observer.h"

#define EXP_GAP_MEMSYS5_SLOTS (1u << 17)
#define EXP_GAP_LOOKASIDE_SLOTS (1u << 12)

static struct reuse_gap_slot exp_gap_memsys5_slots[EXP_GAP_MEMSYS5_SLOTS];
static struct reuse_gap_slot exp_gap_lookaside_slots[EXP_GAP_LOOKASIDE_SLOTS];
static struct reuse_gap_observer exp_gap_memsys5, exp_gap_lookaside;
static int exp_gap_ready, exp_gap_atexit_rc;

static void exp_gap_print(const char *name, const struct reuse_gap_observer *o,
                          unsigned capacity) {
  unsigned i;
  fprintf(stderr, "REUSE-GAP %s attempts=%llu issues=%llu releases=%llu "
          "reuses=%llu distinct=%llu capacity=%u error=%u bins=",
          name, (unsigned long long)o->attempts,
          (unsigned long long)o->issues, (unsigned long long)o->releases,
          (unsigned long long)o->reuses,
          (unsigned long long)o->distinct_starts, capacity, o->error);
  for (i = 0; i < 32; ++i)
    fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long)o->bins[i]);
  fprintf(stderr, "\n");
}

static void exp_gap_report(void) {
  exp_gap_print("memsys5", &exp_gap_memsys5, EXP_GAP_MEMSYS5_SLOTS);
  exp_gap_print("lookaside", &exp_gap_lookaside, EXP_GAP_LOOKASIDE_SLOTS);
  fflush(stderr);
}

static void exp_gap_init(void) {
  if (exp_gap_ready) return;
  exp_gap_ready = 1;
  reuse_gap_init(&exp_gap_memsys5, exp_gap_memsys5_slots, EXP_GAP_MEMSYS5_SLOTS);
  reuse_gap_init(&exp_gap_lookaside, exp_gap_lookaside_slots,
                 EXP_GAP_LOOKASIDE_SLOTS);
  exp_gap_atexit_rc = atexit(exp_gap_report);
  if (exp_gap_atexit_rc) fprintf(stderr, "REUSE-GAP atexit failed\n");
}

/* One hand-out: advance the boundary's event index, then record the start. */
static void exp_gap_issue(struct reuse_gap_observer *o, uint64_t start,
                          uint64_t size) {
  exp_gap_init();
  reuse_gap_attempt(o);
  reuse_gap_issue(o, start, size ? size : 1);
}

static void exp_gap_release(struct reuse_gap_observer *o, uint64_t start) {
  exp_gap_init();
  reuse_gap_release(o, start);
}

#endif
