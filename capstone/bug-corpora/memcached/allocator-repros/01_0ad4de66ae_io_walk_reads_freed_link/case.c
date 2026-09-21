/* Case 1: 0ad4de66ae -- resetting a bad proxy backend walks its pending-IO
 * list with STAILQ_FOREACH while the body returns, and frees, the current IO
 *
 * Shape: stale object pointer / cache.c reuse / list link read through the dead pointer
 * Consumer: proxy_network.c, _reset_bad_backend()
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

/* io_pending_proxy_t reduced to its lifetime-bearing parts: the pointers a
 * request carries, the list entry the backend's io_head threads through, and
 * the inline payload. Same kinds of fields as upstream's, so the object is
 * the same order of size; the values are not consulted. */
struct io_pending {
  void *thread, *conn, *client_resp;
  void (*return_cb)(void *);
  void (*finalize_cb)(void *);
  int status;
  STAILQ_ENTRY(io_pending) io_next; /* queue chain */
  char data[120];
};
STAILQ_HEAD(io_head_s, io_pending);

static cache_t *io_cache;
static unsigned returned;

/* return_io_pending: the IO goes back to its worker thread, which finishes
 * the request and returns the object to that thread's io cache. Upstream's
 * race is that the worker can do this, and take the object again for another
 * request, before the event thread's loop has read io->io_next; the
 * reduction performs that interleaving in order. */
static struct io_pending *return_io(struct io_pending *io, int model_reuse) {
  ++returned;
  cache_free(io_cache, io);
  if (!model_reuse)
    return NULL;
  struct io_pending *fresh = cache_alloc(io_cache); /* the worker's next request */
  CHECK(fresh, 720);
  memset(fresh, 0, sizeof *fresh);
  return fresh;
}

MC_CASE(1) {
  o->defect_text = "the loop read the next link out of an IO that had been returned and reissued";
  o->fixed_text = "each IO was unlinked before it was returned; the walk finished";

  io_cache = cache_create("io", sizeof(struct io_pending), sizeof(char *));
  CHECK(io_cache, 710);
  struct io_head_s io_head = STAILQ_HEAD_INITIALIZER(io_head);
  STAILQ_INIT(&io_head);
  struct io_pending *pending[3];
  for (int i = 0; i < 3; i++) {
    pending[i] = cache_alloc(io_cache);
    CHECK(pending[i], 711);
    memset(pending[i], 0, sizeof *pending[i]);
    STAILQ_INSERT_TAIL(&io_head, pending[i], io_next);
  }

  struct io_pending *io, *fresh = NULL;
  /* Addresses, not pointers, for everything compared after a return: in the
   * protected arm the pointer itself is dead, and even arithmetic on it
   * faults -- so the probe's address is taken while the IO is still live. */
  uintptr_t first = (uintptr_t)pending[0];
  returned = 0;
  if (!fixed) {
    /* _reset_bad_backend before the fix: STAILQ_FOREACH, whose step reads
     * io->io_next after the body has returned io. */
    for (io = STAILQ_FIRST(&io_head); io;) {
      uintptr_t address = (uintptr_t)io;
      held = (volatile unsigned char *)&io->io_next;
      io->status = -1; /* MCMC_ERR */
      fresh = return_io(io, 1);
      if (address == first) {
        o->unit_reissued = (uintptr_t)fresh == first;
        mark(1);
        (void)read_probe(held);
        o->accessed_through_stale = 1;
      }
      io = STAILQ_NEXT(io, io_next); /* the FOREACH step, through the dead pointer */
    }
  } else {
    /* upstream 0ad4de66ae: unlink first, return afterwards */
    while (!STAILQ_EMPTY(&io_head)) {
      io = STAILQ_FIRST(&io_head);
      STAILQ_REMOVE_HEAD(&io_head, io_next);
      uintptr_t address = (uintptr_t)io;
      io->status = -1;
      fresh = return_io(io, 1);
      if (address == first)
        o->unit_reissued = (uintptr_t)fresh == first;
    }
  }
  /* Upstream reports a crash. The reduction's reissued object is zeroed by
   * its new owner, so the stale link reads as the end of the list and the
   * walk ends early instead: pending requests are never returned. */
  o->damage = returned != 3;
  cache_destroy(io_cache);
}
