#include "corpus.h"

/* slab_rebalance_finish's page-list shuffle, fix fa51ad8452d5. The crossed
 * array is `slab_list`, the slab class's own bookkeeping, grown by a plain
 * realloc in grow_slab_list -- not storage the slab allocator hands out. That
 * is what makes this row PLAIN rather than nested. */

MCH_CASE(8) {
  /* Case 8 -- the slab page-list shuffle, fix fa51ad8452d5. PLAIN HEAP: the
   * backwards shuffle ran over the live region and read one element past,
   * because the count was decremented AFTER the loop.
   *
   * At the fix's parent:
   *
   *     for (x = 0; x < s_cls->slabs; x++) {
   *         s_cls->slab_list[x] = s_cls->slab_list[x+1];
   *     }
   *     s_cls->slabs--;
   *
   * and the fix moves the decrement above the loop:
   *
   *     s_cls->slabs--;
   *     for (x = 0; x < s_cls->slabs; x++) {
   *         s_cls->slab_list[x] = s_cls->slab_list[x+1];
   *     }
   *
   * The last iteration has x == slabs-1 and reads index `slabs`. grow_slab_list
   * doubles only when `slabs == list_size`, so slabs == list_size is a normal
   * steady state and that read is one element past the allocation.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long list_size = 2;                       /* 2 * 8 == 16 B: a size class */
  const unsigned long bytes = list_size * sizeof(void *);
  CHECK(bytes % 16 == 0, 961);

  void **slab_list = calloc((size_t)bytes, 1);
  CHECK(slab_list, 962);
  for (unsigned long i = 0; i < list_size; i++)
    slab_list[i] = slab_list;                              /* any non-null marker */

  /* The steady state grow_slab_list permits: the list is exactly full. */
  unsigned long slabs = list_size;
  CHECK(slabs == list_size, 963);                          /* the premise, asserted */

  if (fixed)
    slabs--;                                               /* the fix's order */
  unsigned long touched = 0;
  for (unsigned long x = 0; x < slabs; x++) {
    const unsigned long src = x + 1;
    touched = src;
    if (src >= list_size)
      (void)read_probe((const volatile unsigned char *)(slab_list + src));  /* the crossing */
    else
      slab_list[x] = slab_list[src];
  }
  if (!fixed)
    slabs--;

  o->cap = bytes;
  o->touched = touched * sizeof(void *);
  o->crossed = touched >= list_size;
  o->damage = o->crossed;

  o->defect_text = "the backwards shuffle read slab_list[slabs] because the count was decremented "
                   "after the loop, one element past a list that grow_slab_list leaves exactly full";
  o->fixed_text = "the fix decrements before the loop, so the last source index is the last live "
                  "element";
  free(slab_list);
}
