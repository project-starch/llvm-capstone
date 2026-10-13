/* CONTROL 1, parent: A's chunk, after slabs_free, read and written through the reference slabs.c
 * itself keeps for A's page.
 *
 * WHICH REFERENCE. slabs.c keeps each page in its class's slab_list[]; slabs_peek_page returns
 * slab_list[0] (it is the page mover's way in, and answers once the class has two pages, which
 * ctl_page_zero arranges). Under patch 0006 that is the page's lifetime: memory_allocate derives it
 * (CDERIVE) from the memory virtual mallocng returned, bounded to the 1 MiB page, and keeps it in
 * slab_list. It is not A's direct parent. split_slab_page_into_freelist derives from it the
 * generation for the class (bounded to perslab * chunk size) and writes that generation into byte 0
 * of every chunk (item.sublet_parent); do_slabs_alloc hands A out as a CDERIVE child of the
 * generation, and slabs_free revokes A with CREVOKE(generation, A). So the chain is
 *     virtual-mallocng object -> page lifetime (slab_list[0]) -> generation (chunk byte 0) -> A
 * and variant `page` accesses A's chunk through the page lifetime (A's grandparent), variant
 * `generation` through the generation (A's direct parent, the CREVOKE parent), which it loads from
 * A's former chunk, slot 0, through the page lifetime. In the release build (virtual-malloc,
 * MCP_STOCK_MALLOC) slab_list[0] is virtual mallocng's own pointer to the page and there is no
 * generation, so only `page` runs there.
 *
 * WHAT IS READ AND WRITTEN, through the parent reference, at A's former chunk:
 *   read  it_flags (2 bytes) and slabs_clsid: what do_slabs_free itself wrote there -- ITEM_SLABBED
 *         and the class -- next to the free-list link;
 *   read  data[0..16): the 0xA1 the owner wrote through A before the free, which slabs_free leaves;
 *   write data[0..16) = 0x7E, then read it back.
 * Not touched: byte 0 (the generation under patch 0006), next and prev (the free-list link), the
 * rest of the header. Afterwards slabs_alloc must hand A's chunk out again (the free list is LIFO
 * and intact) holding the 0x7E the parent wrote. Expected on both arms: completes. */
#include "../controls.h"

const int ctl_number = 91;

void ctl_body(const char *variant) {
  int via_generation = !strcmp(variant, "generation");
  CHECK(via_generation || !strcmp(variant, "page"), 820);
#ifndef MC_CAPSTONE_SUBLET
  CHECK(!via_generation, 821); /* no generation without patch 0006 */
#endif
  unsigned size, perslab;
  unsigned id = ctl_class(&size, &perslab);
  item *a = slabs_alloc(id, 0);
  CHECK(a, 822);
  ctl_print_chunk("A", a);
  unsigned long addr_a = ctl_addr(a);
  ctl_fill_span("A-data-by-owner", "A", (unsigned char *)a + CTL_DATA, 0xA1);
  void *held_chunks[1] = {a};
  unsigned char *page = ctl_page_zero(id, size, perslab, held_chunks, 1);
  unsigned long off = addr_a - ctl_addr(page);
  printf("A offset_in_page=%#lx chunk_index=%lu\n", off, off / size);

  slabs_free(a, id);
  printf("FREED A by slabs_free(A, %u)\n", id);

  unsigned char *ref = page + off;
  const char *via = "slab_list0";
#ifdef MC_CAPSTONE_SUBLET
  if (via_generation) {
    printf("ACCESS-BEGIN name=A-slot0-generation kind=read via=slab_list0 addr=%#lx site=%s:%d "
           "probe=ctl_load_ptr@controls.h:%d\n", ctl_addr(ref), __FILE__, __LINE__ + 1, ctl_load_ptr_line);
    void *gen = ctl_load_ptr((void *volatile *)ref);
    printf("ACCESS-DONE name=A-slot0-generation value=%#lx\n", ctl_addr(gen));
    printf("GENERATION base=%#llx end=%#llx tag=%llu\n",
           (unsigned long long)__builtin_capstone_cap_get_base(gen),
           (unsigned long long)__builtin_capstone_cap_get_end(gen),
           (unsigned long long)__builtin_capstone_cap_get_tag(gen));
    ref = (unsigned char *)gen + (addr_a - (unsigned long)__builtin_capstone_cap_get_base(gen));
    via = "generation";
  }
#endif
  item *view = (item *)ref;
  unsigned char f0 = CTL_READ("A-it_flags-byte0", via, (unsigned char *)&view->it_flags);
  unsigned char f1 = CTL_READ("A-it_flags-byte1", via, (unsigned char *)&view->it_flags + 1);
  unsigned char cls = CTL_READ("A-slabs_clsid", via, &view->slabs_clsid);
  printf("CHECK name=A-header-left-by-slabs_free it_flags=%#x slabs_clsid=%u want it_flags=%#x slabs_clsid=%u\n",
         (unsigned)(f0 | f1 << 8), cls, ITEM_SLABBED, id);
  CHECK((unsigned)(f0 | f1 << 8) == ITEM_SLABBED && cls == id, 823);
  ctl_expect_span("A-data-left-by-slabs_free", via, ref + CTL_DATA, 0xA1, 824);
  ctl_fill_span("A-data-by-parent", via, ref + CTL_DATA, 0x7E);
  ctl_expect_span("A-data-by-parent-readback", via, ref + CTL_DATA, 0x7E, 825);

  item *again = slabs_alloc(id, 0);
  CHECK(again, 826);
  ctl_print_chunk("A-reissued", again);
  printf("CHECK name=free-list-intact reissued=%#lx A=%#lx equal=%d\n", ctl_addr(again), addr_a,
         ctl_addr(again) == addr_a);
  CHECK(ctl_addr(again) == addr_a, 827);
  ctl_expect_span("A-reissued-holds-parent-write", "A-reissued", (unsigned char *)again + CTL_DATA, 0x7E, 828);
  slabs_free(again, id);
  printf("FREED A-reissued by slabs_free\n");
}
