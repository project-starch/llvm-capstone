/* Reuse-distance hooks for Wireshark 4.6.8's wmem (pre-registration 16d2171a3edc), compiled in
 * only with -DEXP_REUSE_GAP and identical in both arms (the build adds -DEXP_REUSE_GAP_PROTECTED to
 * the protected arm's two allocator objects, which only labels the report): stock (wmem as released) and protected
 * (wmem patch 0001, TSAPP_SUBLET=1). The hooks sit in upstream's consumer functions of the block
 * and block_fast allocators, which patch 0001 leaves in place and calls from its wrappers, so in
 * both arms they see the allocator's own pointer, never the caller's capability.
 *
 * One observer per boundary, a static object in the file that owns it (wmem-block in
 * wmem_allocator_block.c, wmem-block-fast in wmem_allocator_block_fast.c), with a static table
 * of WMEM_RG_CAPACITY slots: no heap, no capability retained, only integer addresses.
 *
 * wmem_rg_issue          a hand-out: reuse_gap_attempt, then reuse_gap_issue(start, size);
 * wmem_rg_release        a release at start: reuse_gap_release;
 * wmem_rg_resize         an in-place realloc: reuse_gap_resize, no new start;
 * wmem_rg_moved          a realloc that may move (jumbo objects, realloc'd through the system
 *                        allocator): resize if the start is unchanged, else release + issue;
 * wmem_rg_release_range  a reset (free_all) that ends every object in [lo, lo + len): each live
 *                        start in the range is released. block_fast's free is a no-op upstream,
 *                        so its objects are released only here, as the allocator reuses them.
 *
 * The report is one line per boundary on stderr at process exit (atexit, registered at the
 * boundary's first event; tshark's main returns through exit):
 *   REUSE-GAP <boundary> arm=<stock|protected> attempts= issues= releases= reuses=
 *             distinct_starts= error= capacity= b1= ... b32=
 * bK is the observer's bins[K-1]: distances d with 2^(K-1) <= d < 2^K. */
#ifndef WMEM_REUSE_GAP_H
#define WMEM_REUSE_GAP_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "reuse-gap-observer.h"

#ifdef EXP_REUSE_GAP_PROTECTED
#define EXP_REUSE_GAP_ARM "protected"
#else
#define EXP_REUSE_GAP_ARM "stock"
#endif
#define WMEM_RG_CAPACITY ((size_t)1 << 16)

struct wmem_rg {
  const char *name;
  int ready;
  struct reuse_gap_observer o;
  struct reuse_gap_slot slots[WMEM_RG_CAPACITY];
};

static void wmem_rg_print(struct wmem_rg *g) {
  char line[1024];
  int n = snprintf(line, sizeof line,
      "REUSE-GAP %s arm=%s attempts=%llu issues=%llu releases=%llu reuses=%llu distinct_starts=%llu "
      "error=%u capacity=%zu", g->name, EXP_REUSE_GAP_ARM,
      (unsigned long long)g->o.attempts, (unsigned long long)g->o.issues,
      (unsigned long long)g->o.releases, (unsigned long long)g->o.reuses,
      (unsigned long long)g->o.distinct_starts, g->o.error, g->o.capacity);
  for (unsigned k = 1; k <= 32 && n > 0 && (size_t)n < sizeof line; ++k)
    n += snprintf(line + n, sizeof line - (size_t)n, " b%u=%llu", k,
                  (unsigned long long)g->o.bins[k - 1]);
  if (n > 0 && (size_t)n + 1 < sizeof line) {
    line[n++] = '\n';
    (void)!write(2, line, (size_t)n);
  }
}

/* The atexit handler of the file that includes this header reports its boundary. */
static void wmem_rg_report(void);

static inline void wmem_rg_ready(struct wmem_rg *g) {
  if (g->ready) return;
  reuse_gap_init(&g->o, g->slots, WMEM_RG_CAPACITY);
  g->ready = 1;
  atexit(wmem_rg_report);
}

static inline void wmem_rg_issue(struct wmem_rg *g, const void *start, size_t size) {
  wmem_rg_ready(g);
  reuse_gap_attempt(&g->o);
  reuse_gap_issue(&g->o, (uint64_t)(uintptr_t)start, (uint64_t)size);
}

static inline void wmem_rg_release(struct wmem_rg *g, const void *start) {
  wmem_rg_ready(g);
  reuse_gap_release(&g->o, (uint64_t)(uintptr_t)start);
}

static inline void wmem_rg_resize(struct wmem_rg *g, const void *start, size_t size) {
  wmem_rg_ready(g);
  reuse_gap_resize(&g->o, (uint64_t)(uintptr_t)start, (uint64_t)size);
}

static inline void wmem_rg_moved(struct wmem_rg *g, const void *old, const void *start, size_t size) {
  if ((uintptr_t)old == (uintptr_t)start) {
    wmem_rg_resize(g, start, size);
    return;
  }
  wmem_rg_release(g, old);
  wmem_rg_issue(g, start, size);
}

static inline void wmem_rg_release_range(struct wmem_rg *g, const void *lo, size_t len) {
  uint64_t from = (uint64_t)(uintptr_t)lo, to = from + len;
  wmem_rg_ready(g);
  for (size_t i = 0; i < g->o.capacity && !g->o.error; ++i) {
    struct reuse_gap_slot *slot = &g->slots[i];
    if (slot->state == 1 && slot->start >= from && slot->start < to)
      reuse_gap_release(&g->o, slot->start);
  }
}

#endif
