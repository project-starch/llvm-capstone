/* CONTROL 3, successor: A released with slabs_free, then slabs_alloc on the same class until A's
 * chunk address comes back as C (the class's free list is LIFO: do_slabs_free pushes A at the head,
 * do_slabs_alloc pops the head, so the first allocation is expected to be it). Both addresses are
 * printed and required equal.
 *
 *   variant `through-c`: write C's data[0..16) = 0xC3 through C and read it back. Expected on both
 *     arms: completes.
 *   variant `stale`: the same, then ONE read of A's former data[0] through the old A reference, at
 *     the corpus's labelled probe read_probe (shared/corpus.h). Expected: protected (patch 0006)
 *     rejects it with cause 25 (revoked lifetime) at read_probe; release completes and reads 0xC3,
 *     C's data, not A's 0xA1.
 *
 * The stale reference (`held`, corpus.h) is A + offsetof(data), formed while A is live; its address
 * is recorded before the free, because arithmetic on a revoked reference faults in this emulator.
 * Byte 0 and the header fields are not written. */
#include "../controls.h"

const int ctl_number = 93;

void ctl_body(const char *variant) {
  int stale = !strcmp(variant, "stale");
  CHECK(stale || !strcmp(variant, "through-c"), 840);
  unsigned size, perslab;
  unsigned id = ctl_class(&size, &perslab);
  item *a = slabs_alloc(id, 0);
  CHECK(a, 841);
  ctl_print_chunk("A", a);
  unsigned long addr_a = ctl_addr(a);
  ctl_fill_span("A-data-by-owner", "A", (unsigned char *)a + CTL_DATA, 0xA1);
  held = (volatile unsigned char *)a + CTL_DATA;
  unsigned long held_addr = ctl_addr(held);

  slabs_free(a, id);
  printf("FREED A by slabs_free(A, %u)\n", id);

  item *c = NULL;
  unsigned tries = 0;
  while (tries < 2 * perslab + 2) {
    item *got = slabs_alloc(id, 0);
    CHECK(got, 842);
    ++tries;
    printf("ALLOC try=%u addr=%#lx\n", tries, ctl_addr(got));
    if (ctl_addr(got) == addr_a) {
      c = got;
      break;
    }
  }
  CHECK(c, 843);
  ctl_print_chunk("C", c);
  printf("SUCCESSOR A=%#lx C=%#lx equal=%d allocations=%u\n", addr_a, ctl_addr(c), ctl_addr(c) == addr_a,
         tries);

  ctl_fill_span("C-data", "C", (unsigned char *)c + CTL_DATA, 0xC3);
  ctl_expect_span("C-data-readback", "C", (unsigned char *)c + CTL_DATA, 0xC3, 844);

  if (stale) {
    printf("ACCESS-BEGIN name=stale-A-read kind=read via=A(stale) addr=%#lx site=%s:%d "
           "probe=read_probe@shared/corpus.h\n", held_addr, __FILE__, __LINE__ + 1);
    unsigned v = read_probe(held);
    printf("ACCESS-DONE name=stale-A-read value=%#x\n", v);
    printf("STALE-READ value=%#x reads=%s\n", v,
           v == 0xC3 ? "C-data" : v == 0xA1 ? "A-data" : "other");
  }
}
