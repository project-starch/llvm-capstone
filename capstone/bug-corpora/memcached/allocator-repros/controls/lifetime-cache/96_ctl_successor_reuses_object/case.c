/* CONTROL 3, successor: A released with cache_free, then cache_alloc on the same cache until A's
 * address comes back as C (cache.c's free list is LIFO: cache_free pushes A at the head and
 * cache_alloc pops the head, so the first allocation is expected to be it). Both addresses are
 * printed and required equal.
 *
 *   variant `through-c`: write C's bytes [16, 32) = 0xC3 through C and read them back. Expected on
 *     both arms: completes.
 *   variant `stale`: the same, then ONE read of A's former byte 16 through the old A reference, at
 *     the corpus's labelled probe read_probe (shared/corpus.h). Expected: protected (patch 0006)
 *     rejects it with cause 25 (revoked lifetime) at read_probe; release completes and reads 0xC3,
 *     C's data, not A's 0xA1.
 *
 * The stale reference (`held`, corpus.h) is A + 16, formed while A is live; its address is recorded
 * before the free, because arithmetic on a revoked reference faults in this emulator. */
#include "../cache-controls.h"

const int ctl_number = 96;

void ctl_body(const char *variant) {
  int stale = !strcmp(variant, "stale");
  CHECK(stale || !strcmp(variant, "through-c"), 940);
  cache_t *cache = cc_create("ctl96");
  unsigned long primed[1];
  cc_prime(cache, 1, primed);
  void *a = cc_take_from_list(cache, "A");
  unsigned long addr_a = ctl_addr(a);
  ctl_fill_span("A-data-by-owner", "A", (unsigned char *)a + CC_DATA, 0xA1);
  held = (volatile unsigned char *)a + CC_DATA;
  unsigned long held_addr = ctl_addr(held);

  cache_free(cache, a);
  printf("FREED A by cache_free(cache, A)\n");
  cc_state(cache, "after-free-A");

  void *c = NULL;
  unsigned tries = 0;
  while (tries < 8) {
    void *got = cache_alloc(cache);
    CHECK(got, 941);
    ++tries;
    printf("ALLOC try=%u addr=%#lx\n", tries, ctl_addr(got));
    if (ctl_addr(got) == addr_a) {
      c = got;
      break;
    }
  }
  CHECK(c, 942);
  ctl_print_chunk("C", c);
  cc_state(cache, "after-alloc-C");
  printf("SUCCESSOR A=%#lx C=%#lx equal=%d allocations=%u\n", addr_a, ctl_addr(c), ctl_addr(c) == addr_a,
         tries);

  ctl_fill_span("C-data", "C", (unsigned char *)c + CC_DATA, 0xC3);
  ctl_expect_span("C-data-readback", "C", (unsigned char *)c + CC_DATA, 0xC3, 943);

  if (stale) {
    printf("ACCESS-BEGIN name=stale-A-read kind=read via=A(stale) addr=%#lx site=%s:%d "
           "probe=read_probe@shared/corpus.h\n", held_addr, __FILE__, __LINE__ + 1);
    unsigned v = read_probe(held);
    printf("ACCESS-DONE name=stale-A-read value=%#x\n", v);
    printf("STALE-READ value=%#x reads=%s\n", v,
           v == 0xC3 ? "C-data" : v == 0xA1 ? "A-data" : "other");
  }
}
