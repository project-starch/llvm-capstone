/* Reuse-distance hooks for memcached 1.6.45 (pre-registration 16d2171a3edc), compiled in only with
 * -DEXP_REUSE_GAP and identical in both arms: stock (patches 0001-0006, MC_CAPSTONE_SUBLET unset) and
 * protected (-DMC_CAPSTONE_SUBLET). One observer per boundary, each a static object in the file that
 * owns the boundary (slabs.c, cache.c), with a static table of MC_RG_CAPACITY slots: no heap, no
 * capability retained, only integer addresses.
 *
 * mc_rg_issue    the allocator hands out a unit: reuse_gap_attempt, then reuse_gap_issue with the
 *                unit's start address (the allocator's own pointer, read as an integer) and size;
 * mc_rg_release  the allocator takes a unit back: reuse_gap_release at the same start address.
 *
 * The report is one line per boundary on stderr at process exit (atexit, registered at the
 * boundary's first event): memcached's SIGTERM handler ends the event loop and main returns.
 *   REUSE-GAP <boundary> arm=<stock|protected> attempts= issues= releases= reuses= distinct_starts=
 *             error= capacity= b1= ... b32=
 * bK is the observer's bins[K-1]: distances d with 2^(K-1) <= d < 2^K. */
#ifndef MC_REUSE_GAP_H
#define MC_REUSE_GAP_H

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "reuse-gap-observer.h"

#define MC_RG_CAPACITY ((size_t)1 << 15)

struct mc_rg {
  const char *name;
  pthread_mutex_t *lock; /* NULL where the allocator's own lock already serialises the boundary */
  int ready;
  struct reuse_gap_observer o;
  struct reuse_gap_slot slots[MC_RG_CAPACITY];
};

static void mc_rg_print(struct mc_rg *g) {
  char line[1024];
  if (g->lock) pthread_mutex_lock(g->lock);
  int n = snprintf(line, sizeof line,
      "REUSE-GAP %s arm=%s attempts=%llu issues=%llu releases=%llu reuses=%llu distinct_starts=%llu "
      "error=%u capacity=%zu", g->name,
#ifdef MC_CAPSTONE_SUBLET
      "protected",
#else
      "stock",
#endif
      (unsigned long long)g->o.attempts, (unsigned long long)g->o.issues,
      (unsigned long long)g->o.releases, (unsigned long long)g->o.reuses,
      (unsigned long long)g->o.distinct_starts, g->o.error, g->o.capacity);
  for (unsigned k = 1; k <= 32 && n > 0 && (size_t)n < sizeof line; ++k)
    n += snprintf(line + n, sizeof line - (size_t)n, " b%u=%llu", k,
                  (unsigned long long)g->o.bins[k - 1]);
  if (g->lock) pthread_mutex_unlock(g->lock);
  if (n > 0 && (size_t)n + 1 < sizeof line) {
    line[n++] = '\n';
    (void)!write(2, line, (size_t)n);
  }
}

/* The atexit handler of the file that includes this header reports its boundary. */
static void mc_rg_report(void);

static inline void mc_rg_ready(struct mc_rg *g) {
  if (g->ready) return;
  reuse_gap_init(&g->o, g->slots, MC_RG_CAPACITY);
  g->ready = 1;
  atexit(mc_rg_report);
}

static inline void mc_rg_issue(struct mc_rg *g, const void *start, size_t size) {
  if (g->lock) pthread_mutex_lock(g->lock);
  mc_rg_ready(g);
  reuse_gap_attempt(&g->o);
  reuse_gap_issue(&g->o, (uint64_t)(uintptr_t)start, (uint64_t)size);
  if (g->lock) pthread_mutex_unlock(g->lock);
}

static inline void mc_rg_release(struct mc_rg *g, const void *start) {
  if (g->lock) pthread_mutex_lock(g->lock);
  mc_rg_ready(g);
  reuse_gap_release(&g->o, (uint64_t)(uintptr_t)start);
  if (g->lock) pthread_mutex_unlock(g->lock);
}

#endif
