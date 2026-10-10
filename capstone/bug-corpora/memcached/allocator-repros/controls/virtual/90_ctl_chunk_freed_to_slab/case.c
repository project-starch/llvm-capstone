/* CONTROL, not a defect: a slab chunk handed back to its class with slabs_free, then read through
 * the pointer that held it. slabs.c keeps the chunk on its class's free list inside the page, so no
 * system free happens: on `virtual-malloc` (MCP_STOCK_MALLOC, the page one virtual-mallocng object)
 * the read must COMPLETE; on `virtual-nested-pools` (MCP_SUBLET: the application's patch 0006)
 * slabs_free revokes the chunk (CREVOKE) and the read must FAULT at read_probe. It is case 2's free
 * without case 2's refcount story, run as `buggy 90`. */
#include "../shared/corpus.h"

MC_CASE(90) {
  o->defect_text = "control: the freed chunk was read through its old pointer";
  o->fixed_text = "control: no fixed sequence";
  unsigned id = slabs_clsid(sizeof(item) + 64);
  CHECK(id, 710);
  item *it = slabs_alloc(id, 0);
  CHECK(it, 711);
  it->slabs_clsid = (uint8_t)id;
  memset((unsigned char *)it + sizeof(item), 0xA7, 16);
  held = (volatile unsigned char *)it + sizeof(item);
  if (fixed)
    return;
  slabs_free(it, id);
  (void)read_probe(held);
  o->accessed_through_stale = 1;
  o->damage = 1;
}
