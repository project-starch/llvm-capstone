/* Integer-only observer for allocation starts at an application's inner
 * allocator boundary. The caller owns the fixed table outside revocable
 * storage and must reject a run if error is nonzero. No guest capability is
 * retained by this observer. */
#ifndef CAPSTONE_REUSE_GAP_OBSERVER_H
#define CAPSTONE_REUSE_GAP_OBSERVER_H

#include <stddef.h>
#include <stdint.h>

struct reuse_gap_slot {
  uint64_t start;
  uint64_t released_at;
  uint64_t size;
  unsigned char state; /* 0 empty, 1 live, 2 retired */
};

struct reuse_gap_observer {
  struct reuse_gap_slot *slots;
  size_t capacity; /* power of two */
  uint64_t attempts, issues, releases, reuses, distinct_starts;
  uint64_t bins[32];
  unsigned error; /* 1 table full, 2 overlapping issue, 3 unknown release */
};

static inline void reuse_gap_init(struct reuse_gap_observer *o,
                                  struct reuse_gap_slot *slots,
                                  size_t capacity) {
  *o = (struct reuse_gap_observer){0};
  o->slots = slots;
  o->capacity = capacity;
  if (!capacity || (capacity & (capacity - 1))) o->error = 1;
  for (size_t i = 0; i < capacity; ++i) slots[i] = (struct reuse_gap_slot){0};
}

static inline size_t reuse_gap_hash(uint64_t start) {
  start ^= start >> 33;
  start *= UINT64_C(0xff51afd7ed558ccd);
  start ^= start >> 33;
  start *= UINT64_C(0xc4ceb9fe1a85ec53);
  return (size_t)(start ^ (start >> 33));
}

static inline struct reuse_gap_slot *reuse_gap_find(struct reuse_gap_observer *o,
                                                     uint64_t start) {
  if (o->error) return NULL;
  size_t i = reuse_gap_hash(start) & (o->capacity - 1);
  for (size_t n = 0; n < o->capacity; ++n) {
    struct reuse_gap_slot *slot = &o->slots[i];
    if (!slot->state || slot->start == start) return slot;
    i = (i + 1) & (o->capacity - 1);
  }
  o->error = 1;
  return NULL;
}

/* Advance the event index at the chosen boundary. Current application
 * adapters call this on each successful new lifetime, so their index is
 * successful issues; a future fixed-follow-up retirement metric must also
 * call it for failed allocation/reallocation attempts. An in-place realloc
 * does not issue a new start. */
static inline void reuse_gap_attempt(struct reuse_gap_observer *o) {
  ++o->attempts;
}

static inline void reuse_gap_issue(struct reuse_gap_observer *o,
                                    uint64_t start, uint64_t size) {
  struct reuse_gap_slot *slot = reuse_gap_find(o, start);
  if (!slot) return;
  if (slot->state == 1 || !size || !o->attempts) {
    o->error = 2;
    return;
  }
  if (slot->state == 2) {
    uint64_t gap = o->attempts - slot->released_at;
    unsigned bin = 0;
    if (!gap) { o->error = 2; return; }
    while (bin < 31 && gap > (UINT64_C(1) << (bin + 1)) - 1) ++bin;
    ++o->bins[bin];
    ++o->reuses;
  } else {
    slot->start = start;
    ++o->distinct_starts;
  }
  slot->state = 1;
  slot->size = size;
  ++o->issues;
}

static inline void reuse_gap_release(struct reuse_gap_observer *o,
                                      uint64_t start) {
  struct reuse_gap_slot *slot = reuse_gap_find(o, start);
  if (!slot) return;
  if (slot->state != 1) { o->error = 3; return; }
  slot->state = 2;
  slot->released_at = o->attempts;
  ++o->releases;
}

/* An in-place realloc preserves the lifetime and does not create a reuse. */
static inline void reuse_gap_resize(struct reuse_gap_observer *o,
                                     uint64_t start, uint64_t size) {
  struct reuse_gap_slot *slot = reuse_gap_find(o, start);
  if (!slot) return;
  if (slot->state != 1 || !size) { o->error = 2; return; }
  slot->size = size;
}

#endif
