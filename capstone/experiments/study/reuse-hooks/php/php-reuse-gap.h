/* The reuse-distance observers of PHP 5.0.0's Zend allocator
 * (docs/plans/2026-11-15-php-reuse-distance.md). Included by the hooked Zend/zend_alloc.c, after
 * zend_alloc.h; without EXP_REUSE_GAP it defines the five hook macros as nothing and the file
 * compiles as before. The build wrapper copies this file, reuse-gap-hook.h and
 * experiments/study/reuse-gap-observer.h into Zend/.
 *
 * Two boundaries, told apart by the size class Zend itself computes (REAL_SIZE, cache_index):
 *
 *   zend-small  blocks whose cache_index is below MAX_CACHED_MEMORY (REAL_SIZE up to 80
 *               bytes): _emalloc hands them out from AG(cache) or, when that bin is empty,
 *               from malloc; _efree parks them in AG(cache) or, when the bin is full, frees
 *               them to malloc (a drop).
 *   zend-large  every other block: _emalloc takes it from malloc and _efree frees it.
 *
 * The address recorded is the block header (zend_mem_header *), the pointer Zend itself works
 * with, in both arms; never the caller's payload pointer. The size given to the observer is the
 * block Zend asks for, header plus REAL_SIZE, so emalloc(0) is a valid entry.
 *
 * _erealloc: a block that keeps its address and its boundary is resized; any other outcome is
 * a release of the old header at its boundary and a fresh hand-out of the new header at its
 * boundary, and a drop when the address changed (realloc freed the old block). Request shutdown frees AG(cache)'s blocks (drops of released entries) and every
 * block PHP never freed (a release, then a drop), so a later hand-out at such an address is a
 * reuse rather than an overlapping issue.
 *
 * Besides each boundary's REUSE-GAP line (reuse-gap-hook.h, tracks_entries = 1), one line per
 * boundary at exit:
 *   REUSE-GAP-PHP <boundary> fresh=.. cache_hits=.. moves=.. resizes=.. shutdown_releases=..
 * fresh: hand-outs of a block newly taken from malloc (emalloc misses and moved reallocs);
 * cache_hits: hand-outs from AG(cache); moves and resizes: _erealloc outcomes, counted at the
 * new block's boundary; shutdown_releases: blocks request shutdown freed without an _efree. */
#ifndef PHP_REUSE_GAP_H
#define PHP_REUSE_GAP_H

#ifndef EXP_REUSE_GAP
# define REUSE_GAP_PHP_ISSUE(p, size, fresh)
# define REUSE_GAP_PHP_RELEASE(p, size)
# define REUSE_GAP_PHP_DROP(size)
# define REUSE_GAP_PHP_SHUTDOWN_FREE(p, size)
# define REUSE_GAP_PHP_REALLOC(old, p, oldsize, newsize)
#else

#include "reuse-gap-hook.h"

REUSE_GAP_HOOK(php_small_reuse_gap, "zend-small", 20, 1)
REUSE_GAP_HOOK(php_large_reuse_gap, "zend-large", 20, 1)

struct php_reuse_gap_counts {
  uint64_t fresh, cache_hits, moves, resizes, shutdown_releases;
};
static struct php_reuse_gap_counts php_reuse_gap_small, php_reuse_gap_large;
static int php_reuse_gap_counts_armed;

#define PHP_REUSE_GAP_INDEX(size) ((((size) + 7) & ~(size_t)7) >> 3)
#define PHP_REUSE_GAP_BLOCK(size) \
  (sizeof(zend_mem_header) + MEM_HEADER_PADDING + (((size) + 7) & ~(size_t)7))
#define PHP_REUSE_GAP_ADDRESS(p) ((uint64_t)(uintptr_t)(p))

static void php_reuse_gap_counts_line(const char *boundary, const struct php_reuse_gap_counts *c) {
  struct reuse_gap_line l;
  l.n = 0;
  l.overflow = 0;
  reuse_gap_put(&l, "REUSE-GAP-PHP ");
  reuse_gap_put(&l, boundary);
  reuse_gap_put_field(&l, "fresh", c->fresh);
  reuse_gap_put_field(&l, "cache_hits", c->cache_hits);
  reuse_gap_put_field(&l, "moves", c->moves);
  reuse_gap_put_field(&l, "resizes", c->resizes);
  reuse_gap_put_field(&l, "shutdown_releases", c->shutdown_releases);
  reuse_gap_put(&l, l.overflow ? " TRUNCATED\n" : "\n");
  if (write(2, l.text, l.n) < 0)
    l.n = 0;
}

static void php_reuse_gap_counts_report(void) {
  php_reuse_gap_counts_line("zend-small", &php_reuse_gap_small);
  php_reuse_gap_counts_line("zend-large", &php_reuse_gap_large);
}

static int php_reuse_gap_is_small(size_t size) {
  if (!php_reuse_gap_counts_armed) {
    php_reuse_gap_counts_armed = 1;
    if (atexit(php_reuse_gap_counts_report)) {
      php_small_reuse_gap.o.error = 4;
      php_large_reuse_gap.o.error = 4;
    }
  }
  return PHP_REUSE_GAP_INDEX(size) < MAX_CACHED_MEMORY;
}

/* A hand-out by _emalloc: from AG(cache) (fresh = 0) or from malloc (fresh = 1). */
static __attribute__((noinline)) void php_reuse_gap_issue(const void *p, size_t size, int fresh) {
  int small = php_reuse_gap_is_small(size);
  struct reuse_gap_hook *h = small ? &php_small_reuse_gap : &php_large_reuse_gap;
  struct php_reuse_gap_counts *c = small ? &php_reuse_gap_small : &php_reuse_gap_large;

  reuse_gap_hook_issue_entry(h, PHP_REUSE_GAP_ADDRESS(p), PHP_REUSE_GAP_BLOCK(size), fresh);
  if (fresh)
    ++c->fresh;
  else
    ++c->cache_hits;
}

/* _efree took the block back: parked in AG(cache) or about to be freed to malloc. */
static __attribute__((noinline)) void php_reuse_gap_release(const void *p, size_t size) {
  int small = php_reuse_gap_is_small(size);

  reuse_gap_hook_release(small ? &php_small_reuse_gap : &php_large_reuse_gap,
                         PHP_REUSE_GAP_ADDRESS(p));
}

/* The block goes back to malloc (_efree past a full bin, a large block, shutdown). */
static __attribute__((noinline)) void php_reuse_gap_drop(size_t size) {
  int small = php_reuse_gap_is_small(size);

  reuse_gap_hook_drop(small ? &php_small_reuse_gap : &php_large_reuse_gap);
}

/* Request shutdown frees a block PHP never freed: a release, then a drop. */
static __attribute__((noinline)) void php_reuse_gap_shutdown_free(const void *p, size_t size) {
  int small = php_reuse_gap_is_small(size);

  php_reuse_gap_release(p, size);
  php_reuse_gap_drop(size);
  ++(small ? &php_reuse_gap_small : &php_reuse_gap_large)->shutdown_releases;
}

/* A successful _erealloc from old (oldsize bytes requested) to p (newsize bytes). */
static __attribute__((noinline)) void php_reuse_gap_realloc(const void *old, const void *p,
                                                            size_t oldsize, size_t newsize) {
  int old_small = php_reuse_gap_is_small(oldsize);
  int new_small = php_reuse_gap_is_small(newsize);
  struct reuse_gap_hook *h = new_small ? &php_small_reuse_gap : &php_large_reuse_gap;
  struct php_reuse_gap_counts *c = new_small ? &php_reuse_gap_small : &php_reuse_gap_large;

  if (PHP_REUSE_GAP_ADDRESS(old) == PHP_REUSE_GAP_ADDRESS(p) && old_small == new_small) {
    reuse_gap_hook_ready(h);
    reuse_gap_resize(&h->o, PHP_REUSE_GAP_ADDRESS(p), PHP_REUSE_GAP_BLOCK(newsize));
    ++c->resizes;
    return;
  }
  php_reuse_gap_release(old, oldsize);
  if (PHP_REUSE_GAP_ADDRESS(old) != PHP_REUSE_GAP_ADDRESS(p))
    php_reuse_gap_drop(oldsize); /* realloc moved the block and freed the old one */
  reuse_gap_hook_issue_entry(h, PHP_REUSE_GAP_ADDRESS(p), PHP_REUSE_GAP_BLOCK(newsize), 1);
  ++c->fresh;
  ++c->moves;
}

#define REUSE_GAP_PHP_ISSUE(p, size, fresh) php_reuse_gap_issue((p), (size), (fresh))
#define REUSE_GAP_PHP_RELEASE(p, size) php_reuse_gap_release((p), (size))
#define REUSE_GAP_PHP_DROP(size) php_reuse_gap_drop(size)
#define REUSE_GAP_PHP_SHUTDOWN_FREE(p, size) php_reuse_gap_shutdown_free((p), (size))
#define REUSE_GAP_PHP_REALLOC(old, p, oldsize, newsize) \
  php_reuse_gap_realloc((old), (p), (oldsize), (newsize))

#endif /* EXP_REUSE_GAP */
#endif
