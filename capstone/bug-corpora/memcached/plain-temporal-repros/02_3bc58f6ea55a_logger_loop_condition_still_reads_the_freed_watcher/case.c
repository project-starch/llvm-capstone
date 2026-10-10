#include "corpus.h"

/* logger.c, logger_thread_write_entry's request loop, fix 3bc58f6ea55a.
 * The object is ONE direct malloc-family allocation -- neither slabs.c nor
 * cache.c is in the path -- which is what puts this row in the plain-temporal
 * corpus rather than ../allocator-repros. */

MCT_CASE(2) {
  /* Case 2 -- the request loop's own condition, fix 3bc58f6ea55a. This is the
   * RESIDUAL of case 01's fix: guarding the WRITE left the loop's READS of the
   * freed watcher unguarded.
   *
   * At this fix's parent -- which is case 01's fix:
   *
   *     if (watchers[x] != NULL) {
   *         w->failed_flush = true;
   *     }
   *
   * so with the watcher closed the flag is never set, and the while condition
   *
   *     while (!w->failed_flush &&
   *            (skip_scr = bipbuf_request(w->buf, scratch_len + 128)) == NULL)
   *
   * is re-evaluated against freed memory. The fix skips the iteration instead:
   *
   *     if (watchers[x] == NULL) {
   *         continue;
   *     }
   *     w->failed_flush = true;
   */
  const unsigned long n = 48;
  unsigned char *watcher = malloc((size_t)n);
  CHECK(watcher, 821);
  memset(watcher, 0x11, (size_t)n);

  volatile unsigned char *w = watcher;
  unsigned char *watchers_x = watcher;
  watchers_x = NULL;                              /* the close nulls the global slot ... */
  free(watcher);                                  /* ... and frees the struct */
  o->freed = 1;

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 822);
  CHECK_REUSE(fresh == watcher, 823);
  memset(fresh, 0xAA, (size_t)n);

  /* The loop condition's re-read of w->failed_flush. The predecessor's guard
   * stops the WRITE and not this; the fix's `continue` leaves the loop. */
  int reads_stale = 1;
  if (fixed && watchers_x == NULL)
    reads_stale = 0;                              /* the fix's continue */

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = reads_stale ? read_probe(w) : 0u; /* the labelled crossing */
  o->aliased = reads_stale && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "guarding only the write left the loop condition re-reading w->failed_flush and "
                   "w->buf out of the freed watcher, so the loop cannot terminate";
  o->fixed_text = "the fix skips the whole iteration with continue, so no field of the freed "
                  "watcher is read";
  free(fresh);
}
