/* CONTROL 1, the cache's own reference: A's storage, after cache_free(A), read and written through
 * the reference the cache itself keeps to it: cache->head.stqh_first, the head of cache.c's free list.
 * (The pre-registration calls it the parent control.)
 *
 * WHICH REFERENCE. Under patch 0006 cache_free renews A's lifetime through the caller's pointer
 * (CRENEW) and pushes the renewed pointer on the list, so the list head names the new lifetime of
 * A's object: the reference cache_alloc hands out next. Before the free the control records A's
 * bounds; after it, it requires the list head to have A's address and those bounds, tagged. On
 * the release arm the list head is the object pointer the ledger handed back (mcp_object_release:
 * the virtual-mallocng object's own pointer).
 *
 * WHAT IS READ AND WRITTEN, through the list head, at A's former storage:
 *   load  the head's first pointer: the free-list link cache_free itself wrote (NULL: the list held
 *         only A), in the object's first 16 bytes;
 *   read  object bytes [16, 32): the 0xA1 the owner wrote through A before the free;
 *   write the same bytes = 0x7E, then read them back.
 * Afterwards cache_alloc must hand A's storage out again (the list is LIFO and intact) holding that
 * 0x7E. Expected on both arms: completes. */
#include "../cache-controls.h"

const int ctl_number = 94;

void ctl_body(const char *variant) {
  CHECK(!strcmp(variant, "list-head"), 920);
  cache_t *cache = cc_create("ctl94");
  unsigned long primed[1];
  cc_prime(cache, 1, primed);
  void *a = cc_take_from_list(cache, "A");
  unsigned long addr_a = ctl_addr(a);
  ctl_fill_span("A-data-by-owner", "A", (unsigned char *)a + CC_DATA, 0xA1);
#ifdef MC_CAPSTONE_SUBLET
  unsigned long a_base = (unsigned long)__builtin_capstone_cap_get_base(a);
  unsigned long a_end = (unsigned long)__builtin_capstone_cap_get_end(a);
#endif

  cache_free(cache, a);
  printf("FREED A by cache_free(cache, A)\n");
  cc_state(cache, "after-free-A");
  CHECK(cache->freecurr == 1, 921);

  unsigned char *ref = (unsigned char *)STAILQ_FIRST(&cache->head);
  cc_print_cap("PARENT-REF", "list-head", ref);
  CHECK(ctl_addr(ref) == addr_a, 922);
#ifdef MC_CAPSTONE_SUBLET
  CHECK((unsigned long)__builtin_capstone_cap_get_base(ref) == a_base &&
            (unsigned long)__builtin_capstone_cap_get_end(ref) == a_end &&
            __builtin_capstone_cap_get_tag(ref), 923);
  printf("CHECK name=list-head-is-A-renewed addr=%#lx bounds=[%#lx,%#lx) ok\n", ctl_addr(ref), a_base, a_end);
#endif
  const char *via = "list-head";
  printf("ACCESS-BEGIN name=A-free-list-link kind=read via=%s addr=%#lx site=%s:%d probe=ctl_load_ptr@controls.h:%d\n",
         via, ctl_addr(ref), __FILE__, __LINE__ + 1, ctl_load_ptr_line);
  void *link = ctl_load_ptr((void *volatile *)ref);
  printf("ACCESS-DONE name=A-free-list-link value=%#lx\n", ctl_addr(link));
  printf("CHECK name=A-link-left-by-cache_free link=%#lx want=0\n", ctl_addr(link));
  CHECK(ctl_addr(link) == 0, 924);
  unsigned char *data = ref + CC_DATA;
  ctl_expect_span("A-data-left-by-cache_free", via, data, 0xA1, 925);
  ctl_fill_span("A-data-by-list-head", via, data, 0x7E);
  ctl_expect_span("A-data-by-list-head-readback", via, data, 0x7E, 926);

  void *again = cache_alloc(cache);
  CHECK(again, 927);
  ctl_print_chunk("A-reissued", again);
  printf("CHECK name=free-list-intact reissued=%#lx A=%#lx equal=%d\n", ctl_addr(again), addr_a,
         ctl_addr(again) == addr_a);
  CHECK(ctl_addr(again) == addr_a, 928);
  ctl_expect_span("A-reissued-holds-list-head-write", "A-reissued", (unsigned char *)again + CC_DATA, 0x7E, 929);
  cache_free(cache, again);
  printf("FREED A-reissued by cache_free\n");
  cc_state(cache, "end");
}
