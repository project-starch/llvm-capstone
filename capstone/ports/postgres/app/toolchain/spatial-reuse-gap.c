/* Original PostgreSQL memory-context chunk lifetimes. All bookkeeping is
 * static, outside memory contexts, so reset/delete cannot erase it. */
#include "spatial-reuse-gap.h"
#include "reuse-gap-observer.h"
#include <stdio.h>

#define GAP_SLOTS (1u << 17)
#define GAP_CONTEXTS (1u << 15)
static struct reuse_gap_slot slots[GAP_SLOTS];
static struct reuse_gap_observer observer;
static uintptr_t context_key[GAP_CONTEXTS];
static uint32_t context_head[GAP_CONTEXTS];
static uint32_t next_slot[GAP_SLOTS], prev_slot[GAP_SLOTS];
static uint32_t slot_owner[GAP_SLOTS]; /* context index + 1; zero means none */
static unsigned initialized, reported;

static void init(void) {
  if (initialized) return;
  reuse_gap_init(&observer, slots, GAP_SLOTS);
  initialized = 1;
}

static uint32_t owner_index(uintptr_t context) {
  uint32_t i = (uint32_t)reuse_gap_hash((uint64_t)context) & (GAP_CONTEXTS - 1);
  for (uint32_t n = 0; n < GAP_CONTEXTS; ++n) {
    if (!context_key[i]) { context_key[i] = context; return i; }
    if (context_key[i] == context) return i;
    i = (i + 1) & (GAP_CONTEXTS - 1);
  }
  observer.error = 1;
  return 0;
}

static uint32_t slot_index(uintptr_t start) {
  struct reuse_gap_slot *slot = reuse_gap_find(&observer, (uint64_t)start);
  if (!slot) return 0;
  return (uint32_t)(slot - slots) + 1;
}

static void unlink_slot(uint32_t id) {
  uint32_t owner = slot_owner[id - 1];
  if (!owner) { observer.error = 3; return; }
  uint32_t prev = prev_slot[id - 1], next = next_slot[id - 1];
  if (prev) next_slot[prev - 1] = next;
  else context_head[owner - 1] = next;
  if (next) prev_slot[next - 1] = prev;
  next_slot[id - 1] = prev_slot[id - 1] = slot_owner[id - 1] = 0;
}

void pg_spatial_gap_issue(uintptr_t context, uintptr_t start, size_t size) {
  if (!start) return;
  init();
  uint32_t owner = owner_index(context), id = slot_index(start);
  if (!id || observer.error) return;
  if (slot_owner[id - 1]) { observer.error = 2; return; }
  reuse_gap_attempt(&observer);
  reuse_gap_issue(&observer, start, size ? size : 1);
  if (observer.error) return;
  slot_owner[id - 1] = owner + 1;
  next_slot[id - 1] = context_head[owner];
  prev_slot[id - 1] = 0;
  if (context_head[owner]) prev_slot[context_head[owner] - 1] = id;
  context_head[owner] = id;
}

void pg_spatial_gap_release(uintptr_t start) {
  init();
  uint32_t id = slot_index(start);
  if (!id || observer.error) return;
  reuse_gap_release(&observer, start);
  if (!observer.error) unlink_slot(id);
}

void pg_spatial_gap_resize(uintptr_t start, size_t size) {
  init();
  reuse_gap_resize(&observer, start, size ? size : 1);
}

void pg_spatial_gap_reset(uintptr_t context) {
  init();
  uint32_t owner = owner_index(context);
  if (observer.error) return;
  while (context_head[owner] && !observer.error) {
    uint32_t id = context_head[owner];
    pg_spatial_gap_release((uintptr_t)slots[id - 1].start);
  }
}

void pg_reuse_gap_report(void) {
  if (reported++) return;
  init();
  fprintf(stderr, "PG_REUSE_GAP attempts=%llu issues=%llu releases=%llu "
          "reuses=%llu distinct=%llu capacity=%u error=%u bins=",
          (unsigned long long)observer.attempts,
          (unsigned long long)observer.issues,
          (unsigned long long)observer.releases,
          (unsigned long long)observer.reuses,
          (unsigned long long)observer.distinct_starts,
          GAP_SLOTS, observer.error);
  for (unsigned i = 0; i < 32; ++i)
    fprintf(stderr, "%s%llu", i ? "," : "",
            (unsigned long long)observer.bins[i]);
  fprintf(stderr, "\n");
}
