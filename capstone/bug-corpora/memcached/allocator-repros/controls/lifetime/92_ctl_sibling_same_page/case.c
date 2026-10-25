/* CONTROL 2, sibling: A and B from the same slab page, A released with slabs_free, then B read and
 * written through B's own reference, and B's contents required intact.
 *
 * A and B are the first two chunks the class hands out; ctl_page_zero then fills the class's first
 * page and opens a second so that slabs_peek_page returns slab_list[0], and requires both A and B
 * to lie inside it, at chunk boundaries. Under patch 0006 both are CDERIVE children of the same
 * generation, and slabs_free(A) is CREVOKE(generation, A).
 *
 * WHAT IS READ AND WRITTEN, all through B:
 *   before the free: write data[0..16) = 0xB2; read B's it_flags, slabs_clsid and refcount;
 *   after the free:  read the same header fields and data[0..16) and require them unchanged
 *                    (it_flags 0, slabs_clsid = class, refcount 1, data 0xB2), then write
 *                    data[0..16) = 0xB3 and read it back.
 * Byte 0 (B's generation under patch 0006) and the link fields are not touched.
 * Expected on both arms: completes, B intact. */
#include "../controls.h"

const int ctl_number = 92;

struct ctl_header {
  unsigned it_flags, slabs_clsid, refcount;
};

static struct ctl_header read_header(const char *phase, item *b) {
  char n[64];
  struct ctl_header h;
  snprintf(n, sizeof n, "B-it_flags-byte0-%s", phase);
  unsigned f0 = CTL_READ(n, "B", (unsigned char *)&b->it_flags);
  snprintf(n, sizeof n, "B-it_flags-byte1-%s", phase);
  unsigned f1 = CTL_READ(n, "B", (unsigned char *)&b->it_flags + 1);
  snprintf(n, sizeof n, "B-slabs_clsid-%s", phase);
  h.slabs_clsid = CTL_READ(n, "B", &b->slabs_clsid);
  snprintf(n, sizeof n, "B-refcount-byte0-%s", phase);
  unsigned r0 = CTL_READ(n, "B", (unsigned char *)&b->refcount);
  snprintf(n, sizeof n, "B-refcount-byte1-%s", phase);
  unsigned r1 = CTL_READ(n, "B", (unsigned char *)&b->refcount + 1);
  h.it_flags = f0 | f1 << 8;
  h.refcount = r0 | r1 << 8;
  printf("HEADER B %s it_flags=%#x slabs_clsid=%u refcount=%u\n", phase, h.it_flags, h.slabs_clsid,
         h.refcount);
  return h;
}

void ctl_body(const char *variant) {
  CHECK(!strcmp(variant, "sibling"), 830);
  unsigned size, perslab;
  unsigned id = ctl_class(&size, &perslab);
  item *a = slabs_alloc(id, 0);
  CHECK(a, 831);
  item *b = slabs_alloc(id, 0);
  CHECK(b, 832);
  ctl_print_chunk("A", a);
  ctl_print_chunk("B", b);
  void *held_chunks[2] = {a, b};
  (void)ctl_page_zero(id, size, perslab, held_chunks, 2);
  printf("SAME-PAGE A=%#lx B=%#lx distance=%ld chunk_size=%u\n", ctl_addr(a), ctl_addr(b),
         (long)(ctl_addr(a) - ctl_addr(b)), size);

  ctl_fill_span("B-data-before", "B", (unsigned char *)b + CTL_DATA, 0xB2);
  struct ctl_header before = read_header("before", b);
  CHECK(before.it_flags == 0 && before.slabs_clsid == id && before.refcount == 1, 833);

  slabs_free(a, id);
  printf("FREED A by slabs_free(A, %u)\n", id);

  struct ctl_header after = read_header("after", b);
  printf("CHECK name=B-header-intact equal=%d\n",
         after.it_flags == before.it_flags && after.slabs_clsid == before.slabs_clsid &&
             after.refcount == before.refcount);
  CHECK(after.it_flags == before.it_flags && after.slabs_clsid == before.slabs_clsid &&
            after.refcount == before.refcount, 834);
  ctl_expect_span("B-data-intact", "B", (unsigned char *)b + CTL_DATA, 0xB2, 835);
  ctl_fill_span("B-data-after", "B", (unsigned char *)b + CTL_DATA, 0xB3);
  ctl_expect_span("B-data-after-readback", "B", (unsigned char *)b + CTL_DATA, 0xB3, 836);
  slabs_free(b, id);
  printf("FREED B by slabs_free\n");
}
