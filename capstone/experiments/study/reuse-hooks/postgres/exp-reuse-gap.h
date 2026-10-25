/* Reuse-distance hooks for PostgreSQL 17.5's memory contexts, both arms.
 * Pre-registration: capstone/docs/plans/2026-10-11-nested-reuse-distance.md.
 *
 * Included once from src/backend/utils/mmgr/mcxt.c, under EXP_REUSE_GAP, by
 * hooks-{stock,protected}.patch, after GetMemoryChunkMethodID is defined;
 * build.sh copies this file and ../../reuse-gap-observer.h into that directory.
 * The bookkeeping is ports/postgres/app/toolchain/spatial-reuse-gap.c (patch
 * 0017's observer), unchanged in logic: all of it is static, outside memory
 * contexts, so a reset or delete cannot erase it.
 *
 * One boundary, "contexts": every chunk mcxt.c gets from a manager's alloc
 * method (aset, generation, slab, bump) is an issue at the chunk's address,
 * the address the manager computes (MemoryChunkGetPointer); pfree and a
 * moving repalloc release it; a reset or delete of a context releases every
 * chunk still live in it (bump's only release). An aligned allocation is the
 * ordinary chunk below it, counted once. Only integers are recorded. The
 * report is printed at exit on stderr:
 *   REUSE-GAP contexts attempts=.. issues=.. releases=.. reuses=..
 *     distinct=.. capacity=.. error=.. reset_releases=.. resizes=..
 *     bins=b0,..,b31
 * bins[b] holds distances of bit length b + 1. */
#ifndef EXP_REUSE_GAP_PG_H
#define EXP_REUSE_GAP_PG_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "reuse-gap-observer.h"

#define EXP_GAP_SLOTS (1u << 17)
#define EXP_GAP_CONTEXTS (1u << 15)
static struct reuse_gap_slot exp_gap_slots[EXP_GAP_SLOTS];
static struct reuse_gap_observer exp_gap;
static uintptr_t exp_gap_context_key[EXP_GAP_CONTEXTS];
static uint32_t exp_gap_context_head[EXP_GAP_CONTEXTS];
static uint32_t exp_gap_next[EXP_GAP_SLOTS], exp_gap_prev[EXP_GAP_SLOTS];
static uint32_t exp_gap_owner[EXP_GAP_SLOTS]; /* context index + 1; 0 none */
static unsigned exp_gap_ready, exp_gap_reported;
static uint64_t exp_gap_reset_releases, exp_gap_resizes;

static void exp_gap_report(void) {
  if (exp_gap_reported++) return;
  fprintf(stderr, "REUSE-GAP contexts attempts=%llu issues=%llu releases=%llu "
          "reuses=%llu distinct=%llu capacity=%u error=%u reset_releases=%llu "
          "resizes=%llu bins=",
          (unsigned long long)exp_gap.attempts,
          (unsigned long long)exp_gap.issues,
          (unsigned long long)exp_gap.releases,
          (unsigned long long)exp_gap.reuses,
          (unsigned long long)exp_gap.distinct_starts,
          EXP_GAP_SLOTS, exp_gap.error,
          (unsigned long long)exp_gap_reset_releases,
          (unsigned long long)exp_gap_resizes);
  for (unsigned i = 0; i < 32; ++i)
    fprintf(stderr, "%s%llu", i ? "," : "",
            (unsigned long long)exp_gap.bins[i]);
  fprintf(stderr, "\n");
  fflush(stderr);
}

static void exp_gap_init(void) {
  if (exp_gap_ready) return;
  exp_gap_ready = 1;
  reuse_gap_init(&exp_gap, exp_gap_slots, EXP_GAP_SLOTS);
  if (atexit(exp_gap_report)) fprintf(stderr, "REUSE-GAP atexit failed\n");
}

static uint32_t exp_gap_owner_index(uintptr_t context) {
  uint32_t i = (uint32_t)reuse_gap_hash((uint64_t)context) & (EXP_GAP_CONTEXTS - 1);
  for (uint32_t n = 0; n < EXP_GAP_CONTEXTS; ++n) {
    if (!exp_gap_context_key[i]) { exp_gap_context_key[i] = context; return i; }
    if (exp_gap_context_key[i] == context) return i;
    i = (i + 1) & (EXP_GAP_CONTEXTS - 1);
  }
  exp_gap.error = 1;
  return 0;
}

static uint32_t exp_gap_slot_index(uintptr_t start) {
  struct reuse_gap_slot *slot = reuse_gap_find(&exp_gap, (uint64_t)start);
  if (!slot) return 0;
  return (uint32_t)(slot - exp_gap_slots) + 1;
}

static void exp_gap_unlink(uint32_t id) {
  uint32_t owner = exp_gap_owner[id - 1];
  if (!owner) { exp_gap.error = 3; return; }
  uint32_t prev = exp_gap_prev[id - 1], next = exp_gap_next[id - 1];
  if (prev) exp_gap_next[prev - 1] = next;
  else exp_gap_context_head[owner - 1] = next;
  if (next) exp_gap_prev[next - 1] = prev;
  exp_gap_next[id - 1] = exp_gap_prev[id - 1] = exp_gap_owner[id - 1] = 0;
}

static void exp_gap_issue(uintptr_t context, uintptr_t start, size_t size) {
  if (!start) return;
  exp_gap_init();
  uint32_t owner = exp_gap_owner_index(context), id = exp_gap_slot_index(start);
  if (!id || exp_gap.error) return;
  if (exp_gap_owner[id - 1]) { exp_gap.error = 2; return; }
  reuse_gap_attempt(&exp_gap);
  reuse_gap_issue(&exp_gap, start, size ? size : 1);
  if (exp_gap.error) return;
  exp_gap_owner[id - 1] = owner + 1;
  exp_gap_next[id - 1] = exp_gap_context_head[owner];
  exp_gap_prev[id - 1] = 0;
  if (exp_gap_context_head[owner]) exp_gap_prev[exp_gap_context_head[owner] - 1] = id;
  exp_gap_context_head[owner] = id;
}

static void exp_gap_release(uintptr_t start) {
  exp_gap_init();
  uint32_t id = exp_gap_slot_index(start);
  if (!id || exp_gap.error) return;
  reuse_gap_release(&exp_gap, start);
  if (!exp_gap.error) exp_gap_unlink(id);
}

static void exp_gap_resize(uintptr_t start, size_t size) {
  exp_gap_init();
  ++exp_gap_resizes;
  reuse_gap_resize(&exp_gap, start, size ? size : 1);
}

/* A reset or delete: every chunk still live in the context ends. */
static void exp_gap_reset(uintptr_t context) {
  exp_gap_init();
  uint32_t owner = exp_gap_owner_index(context);
  if (exp_gap.error) return;
  while (exp_gap_context_head[owner] && !exp_gap.error) {
    uint32_t id = exp_gap_context_head[owner];
    ++exp_gap_reset_releases;
    exp_gap_release((uintptr_t)exp_gap_slots[id - 1].start);
  }
}

/* The chunk a pointer belongs to: an aligned redirect chunk is a view of the
 * ordinary chunk below it. Reads the chunk header, as pfree itself does. */
static uintptr_t exp_gap_base(void *pointer) {
  if (GetMemoryChunkMethodID(pointer) == MCTX_ALIGNED_REDIRECT_ID)
    return (uintptr_t)MemoryChunkGetBlock(PointerGetMemoryChunk(pointer));
  return (uintptr_t)pointer;
}

#endif
