/* Case 4: 152ddb68f7 -- an unlocked reference-count decrement drifts below the
 * true number of holders, and the next release frees a held item
 *
 * Shape: unlocked refcount update / count drifts below the holders / slabs reuse / read through the dead pointer
 * Consumer: proto_parser.c, process_mget_cmd()'s error path
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

static void item_free_reduced(item *it) {
  /* items.c:item_free */
  unsigned int clsid = ITEM_clsid(it);
  slabs_free(it, clsid);
}
static int do_item_remove_reduced(item *it) {
  /* items.c:do_item_remove, under the item lock */
  if (--it->refcount == 0) {
    item_free_reduced(it);
    return 1;
  }
  return 0;
}

#define STORED 0x6B
#define OTHER 0x91

MC_CASE(4) {
  o->defect_text = "an unlocked decrement lost another client's get; the next release freed an item still held";
  o->fixed_text = "the decrement took the item lock, so the concurrent get was not lost";

  unsigned id = slabs_clsid(sizeof(item) + 64);
  CHECK(id, 710);

  item *it = slabs_alloc(id, 0);
  CHECK(it, 711);
  it->slabs_clsid = (uint8_t)id;
  it->nkey = 0;
  it->nbytes = 16;
  it->it_flags = ITEM_LINKED;
  it->refcount++;
  CHECK(do_item_remove_reduced(it) == 0, 712); /* linked only: 1 */
  memset((unsigned char *)it + sizeof(item), STORED, 16);

  /* Client A fetches the key and holds its reference. */
  it->refcount++; /* 2: linked, plus A */

  /* Client B's `mg` carries an overlarge opaque token and takes the error
   * path. Before 152ddb68f7 that path called do_item_remove -- the variant
   * that assumes the item lock is held -- whether or not it was:
   *
   *     error:
   *         if (it) {
   *             do_item_remove(it);
   *             if (of.locked) { item_unlock(hv); }
   *         }
   *
   * refcount_decr is --(it->refcount) (memcached.h:1041): a read, a subtract
   * and a store. Under the lock nothing can land between them. Without it,
   * client C's get does, and its increment is lost -- the drift the fix's
   * message describes. The three steps are written out here because the
   * interleaving IS the defect; a single -- would hide it. */
  if (!fixed) {
    unsigned short seen = it->refcount; /* B reads 2 */
    it->refcount++;                     /* C's do_item_get lands here: 3 */
    held = (volatile unsigned char *)it;
    it->refcount = (unsigned short)(seen - 1); /* B stores 1; C's get is lost */
  } else {
    /* 152ddb68f7: when the lock is not held, take it -- item_remove, not
     * do_item_remove. C's increment then cannot land inside the update. */
    it->refcount++;                 /* C's get, serialised by the lock */
    held = (volatile unsigned char *)it;
    (void)do_item_remove_reduced(it); /* B's release, serialised: 3 -> 2 */
  }
  uintptr_t item_at = (uintptr_t)it;

  /* Client A is done and releases. The count says what B left behind, not
   * what is really held: with the drift it reaches zero here, while A is
   * gone but C and the hash table still hold the item. */
  int freed = do_item_remove_reduced(it);
  CHECK(freed == !fixed, 713);

  /* The next item of this class takes the chunk back. */
  item *fresh = slabs_alloc(id, 0);
  CHECK(fresh, 714);
  fresh->slabs_clsid = (uint8_t)id;
  fresh->nkey = 0;
  fresh->nbytes = 16;
  fresh->it_flags = ITEM_LINKED;
  memset((unsigned char *)fresh + sizeof(item), OTHER, 16);
  o->unit_reissued = (uintptr_t)fresh == item_at;

  if (!fixed) {
    /* Client C, whose get was lost, still holds the item it was given. */
    CHECK(o->unit_reissued, 715);
    mark(4);
    (void)read_probe(held);
    o->accessed_through_stale = 1;
    const volatile unsigned char *theirs = held + sizeof(item);
    for (int i = 0; i < 16; i++)
      if (theirs[i] != STORED) {
        o->damage = 1;
        break;
      }
  }
}
