/* One nested-allocator boundary's reuse observer and its exit-time report,
 * for the reuse-distance measurement (docs/plans/2026-10-11-nested-reuse-
 * distance.md). Compiled only under EXP_REUSE_GAP; the hook patches include
 * it next to experiments/study/reuse-gap-observer.h, which the build wrapper
 * copies into the prepared source beside it.
 *
 * The observer keeps integers only: a start address, its size and the event
 * index at its release. The table is static, its capacity a power of two.
 * Every hand-out is one reuse_gap_attempt then one reuse_gap_issue with the
 * address the allocator itself works with; a return is reuse_gap_release.
 * A boundary whose allocator frees idle entries to the system allocator
 * (tracks_entries = 1) also counts, outside the observer and without changing
 * the histogram: `drops`, entries freed that way, and `fresh_reuses`, reuses
 * at a hand-out of a NEWLY allocated entry, i.e. the system allocator placed
 * a new entry at a start address an earlier, dropped entry had. Both are
 * printed only for such a boundary.
 *
 * At exit one line goes to stderr:
 *   REUSE-GAP <boundary> attempts=.. issues=.. releases=.. reuses=..
 *     distinct_starts=.. [drops=.. fresh_reuses=..] capacity=.. error=..
 *     bins=b0,..,b31
 * bins[b] holds distances of bit length b + 1 (bucket k = bins[k-1]). The
 * numbers are formatted here, without printf, so no variadic call is
 * involved (ISSUES.md C-48). */
#ifndef REUSE_GAP_HOOK_H
#define REUSE_GAP_HOOK_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "reuse-gap-observer.h"

struct reuse_gap_hook {
  const char *boundary;
  struct reuse_gap_slot *slots;
  size_t capacity;
  void (*report)(void);
  int ready, tracks_entries;
  uint64_t drops, fresh_reuses;
  struct reuse_gap_observer o;
};

static void reuse_gap_hook_report(struct reuse_gap_hook *h);

/* A static hook for one boundary, with its own table and atexit report. */
#define REUSE_GAP_HOOK(name, boundary_name, log2_capacity, tracks)           \
  static struct reuse_gap_slot name##_slots[(size_t)1 << (log2_capacity)];  \
  static void name##_report(void);                                          \
  static struct reuse_gap_hook name = {                                     \
      boundary_name, name##_slots, (size_t)1 << (log2_capacity),            \
      name##_report, 0, tracks, 0, 0, {0}};                                 \
  static void name##_report(void) { reuse_gap_hook_report(&name); }

static inline void reuse_gap_hook_ready(struct reuse_gap_hook *h) {
  if (h->ready)
    return;
  h->ready = 1;
  reuse_gap_init(&h->o, h->slots, h->capacity);
  if (atexit(h->report))
    h->o.error = 4; /* no report would be written; never seen as a result */
}

/* A hand-out of the allocator's entry at start, size bytes. */
static inline void reuse_gap_hook_issue(struct reuse_gap_hook *h,
                                        uint64_t start, uint64_t size) {
  reuse_gap_hook_ready(h);
  reuse_gap_attempt(&h->o);
  reuse_gap_issue(&h->o, start, size);
}

/* The same, telling whether the entry was newly allocated (fresh) or
 * reissued from the allocator's free list; fresh is counted only. */
static inline void reuse_gap_hook_issue_entry(struct reuse_gap_hook *h,
                                              uint64_t start, uint64_t size,
                                              int fresh) {
  reuse_gap_hook_ready(h);
  reuse_gap_attempt(&h->o);
  if (fresh) {
    struct reuse_gap_slot *slot = reuse_gap_find(&h->o, start);
    if (slot && slot->state == 2)
      ++h->fresh_reuses;
  }
  reuse_gap_issue(&h->o, start, size);
}

/* The entry at start is back with the allocator. */
static inline void reuse_gap_hook_release(struct reuse_gap_hook *h,
                                          uint64_t start) {
  reuse_gap_hook_ready(h);
  reuse_gap_release(&h->o, start);
}

/* The allocator freed an idle entry to the system allocator. */
static inline void reuse_gap_hook_drop(struct reuse_gap_hook *h) {
  reuse_gap_hook_ready(h);
  ++h->drops;
}

struct reuse_gap_line {
  char text[1024];
  size_t n;
  int overflow;
};

static void reuse_gap_put(struct reuse_gap_line *l, const char *s) {
  while (*s) {
    if (l->n + 1 >= sizeof l->text) {
      l->overflow = 1;
      return;
    }
    l->text[l->n++] = *s++;
  }
}

static void reuse_gap_put_u64(struct reuse_gap_line *l, uint64_t v) {
  char digits[24];
  size_t i = sizeof digits;
  digits[--i] = 0;
  do {
    digits[--i] = (char)('0' + v % 10);
    v /= 10;
  } while (v);
  reuse_gap_put(l, digits + i);
}

static void reuse_gap_put_field(struct reuse_gap_line *l, const char *name,
                                uint64_t v) {
  reuse_gap_put(l, " ");
  reuse_gap_put(l, name);
  reuse_gap_put(l, "=");
  reuse_gap_put_u64(l, v);
}

static void reuse_gap_hook_report(struct reuse_gap_hook *h) {
  struct reuse_gap_line l;
  l.n = 0;
  l.overflow = 0;
  reuse_gap_put(&l, "REUSE-GAP ");
  reuse_gap_put(&l, h->boundary);
  reuse_gap_put_field(&l, "attempts", h->o.attempts);
  reuse_gap_put_field(&l, "issues", h->o.issues);
  reuse_gap_put_field(&l, "releases", h->o.releases);
  reuse_gap_put_field(&l, "reuses", h->o.reuses);
  reuse_gap_put_field(&l, "distinct_starts", h->o.distinct_starts);
  if (h->tracks_entries) {
    reuse_gap_put_field(&l, "drops", h->drops);
    reuse_gap_put_field(&l, "fresh_reuses", h->fresh_reuses);
  }
  reuse_gap_put_field(&l, "capacity", h->capacity);
  reuse_gap_put_field(&l, "error", h->o.error);
  reuse_gap_put(&l, " bins=");
  for (unsigned i = 0; i < 32; ++i) {
    if (i)
      reuse_gap_put(&l, ",");
    reuse_gap_put_u64(&l, h->o.bins[i]);
  }
  reuse_gap_put(&l, l.overflow ? " TRUNCATED\n" : "\n");
  if (write(2, l.text, l.n) < 0)
    l.n = 0; /* nothing else can report it */
}

#endif
