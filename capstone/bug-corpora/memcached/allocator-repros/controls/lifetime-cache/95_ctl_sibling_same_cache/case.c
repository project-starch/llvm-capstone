/* CONTROL 2, sibling: A and B, both live objects of the same cache, both handed out from its free
 * list; A released with cache_free; B used afterwards and found intact.
 *
 * B's object bytes [16, 32) are written 0xB2 through B before A's release. After it, through B:
 *   read  the slot in front of B (patch 0006 only): B's lifetime, the reference cache_free(B) needs,
 *         required unchanged (same address, bounds, tag);
 *   read  B's bytes [16, 32), required 0xB2;
 *   write them 0xB5 and read them back.
 * Then cache_free(B) must complete (under patch 0006 it CREVOKEs B against that lifetime, so this
 * also shows B was still a live child after A's revocation). Expected on both arms: completes. */
#include "../cache-controls.h"

const int ctl_number = 95;

void ctl_body(const char *variant) {
  CHECK(!strcmp(variant, "sibling"), 930);
  cache_t *cache = cc_create("ctl95");
  unsigned long primed[2];
  cc_prime(cache, 2, primed);
  void *a = cc_take_from_list(cache, "A");
  void *b = cc_take_from_list(cache, "B");
  CHECK(ctl_addr(a) == primed[0] && ctl_addr(b) == primed[1], 931); /* primed[] holds object addresses */
  unsigned long addr_a = ctl_addr(a), addr_b = ctl_addr(b);
  printf("SIBLING A=%#lx B=%#lx distance=%ld\n", addr_a, addr_b, (long)(addr_b - addr_a));
  ctl_fill_span("A-data-by-owner", "A", (unsigned char *)a + CC_DATA, 0xA1);
  ctl_fill_span("B-data-by-owner", "B", (unsigned char *)b + CC_DATA, 0xB2);
#ifdef MC_CAPSTONE_SUBLET
  void *life_before = cc_read_slot("B-slot-before", "B", b);
#endif

  cache_free(cache, a);
  printf("FREED A by cache_free(cache, A)\n");
  cc_state(cache, "after-free-A");
  CHECK(cache->freecurr == 1, 932);

#ifdef MC_CAPSTONE_SUBLET
  void *life_after = cc_read_slot("B-slot-after", "B", b);
  CHECK(ctl_addr(life_after) == ctl_addr(life_before) &&
            __builtin_capstone_cap_get_base(life_after) == __builtin_capstone_cap_get_base(life_before) &&
            __builtin_capstone_cap_get_end(life_after) == __builtin_capstone_cap_get_end(life_before) &&
            __builtin_capstone_cap_get_tag(life_after), 933);
  printf("CHECK name=B-slot-intact ok\n");
#endif
  ctl_expect_span("B-data-after-A-free", "B", (unsigned char *)b + CC_DATA, 0xB2, 934);
  ctl_fill_span("B-data-rewrite", "B", (unsigned char *)b + CC_DATA, 0xB5);
  ctl_expect_span("B-data-rewrite-readback", "B", (unsigned char *)b + CC_DATA, 0xB5, 935);

  cache_free(cache, b);
  printf("FREED B by cache_free(cache, B)\n");
  cc_state(cache, "end");
  CHECK(cache->freecurr == 2, 936);
}
