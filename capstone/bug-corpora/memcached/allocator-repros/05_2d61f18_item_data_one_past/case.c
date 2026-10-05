/* Case 5: 2d61f18 -- the incr/decr rewrite copies three bytes of a two-byte
 * terminator onto an item, one byte past the item's data.
 *
 * Shape: one byte written past an item's data into the next chunk of the page
 * Consumer: memcached.c, the incr/decr value rewrite
 * SPATIAL -- the corpus's first. No lifetime ends; the item is alive and the
 * copy simply runs one byte long.
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

MC_CASE(5) {
  o->defect_text = "the three-byte copy of a two-byte terminator wrote one byte "
                   "past the item's data, into the next chunk of the slab page";
  o->fixed_text = "the two-byte copy ends exactly at the item's data end";

  unsigned id = slabs_clsid(sizeof(item) + 64);
  CHECK(id, 750);
  /* Two chunks of the same class. Their distance IS the chunk stride, so the
   * case needs no knowledge of memcached's class sizes: sizing the item's data
   * to end exactly at the stride makes the one-byte overflow cross the CHUNK
   * bound rather than land in the class's rounding slack. That is the reachable
   * worst case, and it is derived rather than assumed. */
  item *a = slabs_alloc(id, 0);
  CHECK(a, 751);
  item *b = slabs_alloc(id, 0);
  CHECK(b, 752);
  /* The free list does not promise ascending addresses, so the LOWER chunk is
   * the one the overflow runs out of. Ordered here rather than assumed: the
   * first version of this case asserted b > a and the control refused the run,
   * which is the harness doing its job. */
  item *it = (char *)a < (char *)b ? a : b;
  item *next = (char *)a < (char *)b ? b : a;
  CHECK((char *)next > (char *)it, 753);
  size_t stride = (size_t)((char *)next - (char *)it);

  it->slabs_clsid = (uint8_t)id;
  it->nkey = 0;
  it->it_flags = 0;
  /* data starts after the item header plus the key's NUL; no suffix here. */
  unsigned char *data = (unsigned char *)it + sizeof(item) + 1;
  CHECK((char *)data > (char *)it && (size_t)((char *)data - (char *)it) < stride, 754);
  size_t room = stride - (size_t)((char *)data - (char *)it);
  CHECK(room >= 4, 755);
  it->nbytes = (int)room;              /* the data field fills the chunk exactly */
  memset(data, 0, room);

  /* The next chunk's first byte is what a one-byte overflow reaches. */
  memset((unsigned char *)next, 0x5a, 1);
  unsigned char before = *(volatile unsigned char *)next;
  CHECK(before == 0x5a, 756);

  /* memcached.c: memcpy(ITEM_data(new_it), buf, res) then
   *              memcpy(ITEM_data(new_it) + res, "\r\n", N)
   * with N = 3 before the fix and 2 after. res leaves exactly two bytes, so
   * N = 3 writes the terminator's NUL one byte past. */
  size_t res = room - 2;
  size_t n = fixed ? 2 : 3;
  o->unit_reissued = 0;                /* nothing is freed or reissued here */
  held = data + res;                   /* set while live, as the seam requires */
  mark(5); /* LAST thing before the access: its presence is the setup's evidence */
  CHECK((char *)(data + res + n) > (char *)next || fixed, 757);
  for (size_t i = 0; i < n; i++)
    write_probe((volatile unsigned char *)(data + res + i));

  o->accessed_through_stale = !fixed;  /* the crossing happened */
  o->damage = *(volatile unsigned char *)next != 0x5a;
  slabs_free(b, id);
  slabs_free(a, id);
}
