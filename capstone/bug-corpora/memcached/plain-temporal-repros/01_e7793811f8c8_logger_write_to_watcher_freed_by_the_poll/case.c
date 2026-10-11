#include "corpus.h"

/* logger.c, logger_thread_write_entry, fix e7793811f8c8.
 * The object is ONE direct malloc-family allocation -- neither slabs.c nor
 * cache.c is in the path -- which is what puts this row in the plain-temporal
 * corpus rather than ../allocator-repros. */

MCT_CASE(1) {
  /* Case 1 -- logger_thread_write_entry's flag store, fix e7793811f8c8. The
   * freed object is the watcher; the poll the function itself called may close
   * and free it, and the function then WRITES into it through its local `w`.
   *
   * The fix rechecks the global slot before the write:
   *
   *     // Oddity; poll_watchers can free *w, recheck it.
   *     if (watchers[x] != NULL) {
   *         w->failed_flush = true;
   *     }
   *
   * logger_thread_close_watcher nulls watchers[w->id] but cannot null the
   * caller's local, which is what leaves the dangling pointer.
   *
   * THIS ONE WRITES through the stale pointer. */
  const unsigned long n = 48;
  unsigned char *watcher = malloc((size_t)n);
  CHECK(watcher, 811);
  memset(watcher, 0x11, (size_t)n);

  volatile unsigned char *w = watcher;            /* the caller's local */
  unsigned char *watchers_x = watcher;            /* the global slot */

  /* logger_thread_poll_watchers finds the socket gone and closes the watcher. */
  watchers_x = NULL;                              /* close nulls the GLOBAL slot ... */
  free(watcher);                                  /* ... and frees the struct */
  o->freed = 1;

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 812);
  CHECK_REUSE(fresh == watcher, 813);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  /* w->failed_flush = true, guarded in the fixed arm by the global recheck. */
  if (!fixed || watchers_x != NULL)
    write_probe(w, (unsigned char)0x5A);          /* the labelled crossing */
  o->observed = fresh[0];                         /* did the store land in the NEW object? */
  o->aliased = o->observed == 0x5A;
  o->damage = o->aliased;
  o->defect_text = "the poll the function called may close and free the watcher, and the flag store "
                   "that follows writes into storage that now belongs to another object";
  o->fixed_text = "the fix rechecks the global watchers slot before writing through the local";
  free(fresh);
}
