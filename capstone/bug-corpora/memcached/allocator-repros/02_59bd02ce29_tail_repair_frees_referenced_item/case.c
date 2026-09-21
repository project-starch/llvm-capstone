/* Case 2: tail repair throws away a live reference and frees the item
 *
 * Shape: allocator-forced free of a referenced item / slabs reuse / read through the dead pointer
 * Consumer: items.c, do_item_alloc()'s LRU-tail branch under settings.tail_repair_time
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

/* The item layer, reduced to the three functions that decide whether a chunk
 * goes back to slabs. Upstream's are items.c:item_free, :do_item_remove and
 * :do_item_unlink_nolock; the hash table, the LRU queues, the stats, the
 * storage hooks and the locks are left out because none of them takes part in
 * that decision. What is kept is the refcount arithmetic and the one call to
 * the real allocator. */
static void item_free_reduced(item *it) {
  /* items.c:item_free -- the only path from an item to slabs */
  unsigned int clsid = ITEM_clsid(it);
  slabs_free(it, clsid);
}
static void do_item_remove_reduced(item *it) {
  /* items.c:do_item_remove */
  CHECK(it->refcount > 0, 720);
  if (--it->refcount == 0)
    item_free_reduced(it);
}
static void do_item_unlink_nolock_reduced(item *it) {
  /* items.c:do_item_unlink_nolock -- unlink drops the reference that being
   * linked is worth, which is the one that reaches zero here */
  if ((it->it_flags & ITEM_LINKED) != 0) {
    it->it_flags &= ~ITEM_LINKED;
    do_item_remove_reduced(it);
  }
}

#define STORED 0xA7 /* what the holder wrote and expects to read back */
#define OTHER 0x5C  /* what the item that takes the chunk next writes */

MC_CASE(2) {
  o->defect_text = "tail repair discarded a live reference; the chunk came back as another item";
  o->fixed_text = "the eviction left the referenced item alone, as the default does";

  unsigned id = slabs_clsid(sizeof(item) + 64);
  CHECK(id, 710);

  /* The LRU tail: an item a client stored. do_slabs_alloc hands it out with
   * refcount 1; do_item_link adds the reference that being linked is worth
   * and sets ITEM_LINKED, and the storing client then drops its own. */
  item *search = slabs_alloc(id, 0);
  CHECK(search, 711);
  search->slabs_clsid = (uint8_t)id;
  search->nkey = 0;
  search->nbytes = 16;
  search->it_flags = ITEM_LINKED;
  search->refcount++; /* 2: linked, plus the client that stored it */
  do_item_remove_reduced(search); /* the storing client lets go -> 1 */
  CHECK(search->refcount == 1, 712);
  unsigned char *payload = (unsigned char *)search + sizeof(item);
  memset(payload, STORED, 16);

  /* Another client fetches it and keeps its reference and its pointer, which
   * is what do_item_get leaves behind (items.c: refcount_incr). */
  search->refcount++; /* 2: linked, plus this holder */
  uintptr_t tail_at = (uintptr_t)search;
  held = (volatile unsigned char *)search;

  /* do_item_alloc needs a chunk and walks to the LRU tail (items.c:1111).
   * Its probe takes a reference of its own, so anything other than 2 means
   * somebody else is holding this item. */
  if (++search->refcount != 2) {
    if (!fixed) {
      /* items.c:1115-1125, reached when -o tail_repair_time is set and the
       * item is older than it. Upstream's own comment above these lines is
       * "WARNING: This can cause terrible corruption".
       *
       *     search->refcount = 1;
       *     do_item_unlink_nolock(search, hv);
       *
       * The assignment discards every outstanding reference, including the
       * holder's, and the unlink then takes the count to zero. */
      search->refcount = 1;
      do_item_unlink_nolock_reduced(search); /* -> item_free -> slabs_free */
    } else {
      /* The shipped default, tail_repair_time == 0: the branch is not taken,
       * the probe's reference is given back and the eviction leaves this item
       * alone (upstream then keeps walking the tail). */
      search->refcount--;
    }
  }

  /* The allocation the eviction was making in the first place. With the chunk
   * on its class's free list, do_slabs_alloc pops it straight back. */
  item *fresh = slabs_alloc(id, 0);
  CHECK(fresh, 713);
  fresh->slabs_clsid = (uint8_t)id;
  fresh->nkey = 0;
  fresh->nbytes = 16;
  fresh->it_flags = ITEM_LINKED;
  memset((unsigned char *)fresh + sizeof(item), OTHER, 16);
  o->unit_reissued = (uintptr_t)fresh == tail_at;

  if (!fixed) {
    /* The holder still believes it owns an item. The setup that makes this a
     * stale access -- the chunk came back as `fresh` -- is checked BEFORE the
     * marker, so the marker's presence is itself evidence that it did. */
    CHECK(o->unit_reissued, 714);
    mark(2);
    (void)read_probe(held);
    o->accessed_through_stale = 1;
    /* What the holder reads back where it wrote STORED. */
    const volatile unsigned char *theirs = held + sizeof(item);
    for (int i = 0; i < 16; i++) {
      if (theirs[i] != STORED) {
        o->damage = 1;
        break;
      }
    }
  }
}
