/* Case 3: a8c4a82787 / CVE-2018-1000127 -- the item reference count overflows
 * and the item is freed with its holders still holding it
 *
 * Shape: reference count overflow / item freed with holders remaining / slabs reuse / read through the dead pointer
 * Consumer: memcached.c, process_get_command()'s multiget loop, via items.c do_item_get
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

/* The item layer, reduced to the functions that decide whether a chunk goes
 * back to slabs. Upstream's are items.c:item_free and :do_item_remove. */
static void item_free_reduced(item *it) {
  /* items.c:item_free */
  unsigned int clsid = ITEM_clsid(it);
  slabs_free(it, clsid);
}
static int do_item_remove_reduced(item *it) {
  /* items.c:do_item_remove -- refcount_decr is --(it->refcount),
   * memcached.h:1041, and reaching zero frees the item */
  if (--it->refcount == 0) {
    item_free_reduced(it);
    return 1;
  }
  return 0;
}

/* memcached.c:2185. The fix this case pairs against: a get that would push the
 * count this high is answered as a miss instead. */
#define IT_REFCOUNT_LIMIT 60000
/* An unsigned short holds the count (memcached.h:622), so it wraps here. */
#define REFCOUNT_WRAP 65536UL

#define STORED 0xC3
#define OTHER 0x2E

MC_CASE(3) {
  o->defect_text = "the count wrapped past the holders; the first release freed an item 65536 clients were holding";
  o->fixed_text = "the get was answered as a miss before the count could wrap";

  unsigned id = slabs_clsid(sizeof(item) + 64);
  CHECK(id, 710);

  /* An item stored under a key, linked into the hash table. do_slabs_alloc
   * hands it out with refcount 1; do_item_link adds the reference that being
   * linked is worth, and the storing client then drops its own. */
  item *it = slabs_alloc(id, 0);
  CHECK(it, 711);
  it->slabs_clsid = (uint8_t)id;
  it->nkey = 0;
  it->nbytes = 16;
  it->it_flags = ITEM_LINKED;
  it->refcount++;
  CHECK(do_item_remove_reduced(it) == 0, 712); /* back to 1 */
  memset((unsigned char *)it + sizeof(item), STORED, 16);

  /* A client fetches the key and keeps its reference and its pointer. */
  it->refcount++; /* 2: linked, plus this holder */
  uintptr_t item_at = (uintptr_t)it;
  held = (volatile unsigned char *)it;

  /* The multiget: `get k k k k ...`, one do_item_get per occurrence, each
   * taking a reference that is held until the whole response is written. */
  unsigned long taken = 0;
  int refused = 0;
  for (unsigned long i = 0; i < REFCOUNT_WRAP; i++) {
    if (fixed && it->refcount > IT_REFCOUNT_LIMIT) {
      /* memcached.c:limited_get -- a8c4a82787 answers a miss here */
      refused = 1;
      break;
    }
    it->refcount++; /* items.c: refcount_incr(it) in do_item_get */
    taken++;
  }
  /* 2 + 65536 does not fit in the count: it reads 2 while 65538 references
   * exist. The fixed arm stopped at the limit and the count still counts. */
  int wrapped = !fixed && it->refcount == 2 && taken == REFCOUNT_WRAP;
  CHECK(fixed ? refused : wrapped, 713);

  /* The response is written and the multiget lets go of its references. The
   * second one takes the count to zero, and the item is freed while this
   * client, the other 65534 occurrences and the hash table still hold it. */
  int freed = 0;
  for (int i = 0; i < 2 && !freed; i++)
    freed = do_item_remove_reduced(it);
  CHECK(freed == !fixed, 714);

  /* The next item to need a chunk of this class gets that one back. */
  item *fresh = slabs_alloc(id, 0);
  CHECK(fresh, 715);
  fresh->slabs_clsid = (uint8_t)id;
  fresh->nkey = 0;
  fresh->nbytes = 16;
  fresh->it_flags = ITEM_LINKED;
  memset((unsigned char *)fresh + sizeof(item), OTHER, 16);
  o->unit_reissued = (uintptr_t)fresh == item_at;

  if (!fixed) {
    /* The client that fetched the key before the wrap still believes it owns
     * the item. The reuse is checked BEFORE the marker, so the marker's
     * presence is itself evidence that the chunk came back. */
    CHECK(o->unit_reissued, 716);
    mark(3);
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
