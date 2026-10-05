/* Case 6: 78eb770 -- the flags are memcpy'd into suffix space the item was
 * never allocated.
 *
 * Shape: four bytes written into an item field that has no room, inside the chunk
 * Consumer: items.c, do_item_alloc()'s header fill
 * SPATIAL, and a SUB-OBJECT crossing: when nsuffix is 0, ITEM_suffix(it) IS
 * ITEM_data(it), so the four-byte flags copy lands in the value's storage. The
 * chunk bound is never crossed, which is the point -- a chunk-granular bound
 * cannot see it.
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

MC_CASE(6) {
  o->defect_text = "the four-byte flags copy overwrote the value's storage, because "
                   "with nsuffix = 0 the suffix field has no room of its own";
  o->fixed_text = "the guard skipped the copy when no suffix space was allocated";

  unsigned id = slabs_clsid(sizeof(item) + 64);
  CHECK(id, 760);
  item *it = slabs_alloc(id, 0);
  CHECK(it, 761);
  it->slabs_clsid = (uint8_t)id;
  it->nkey = 0;
  it->it_flags = 0;
  it->nbytes = 2;             /* a zero-length value: just the CRLF */

  /* With nsuffix = 0 these two are the same address, which is the defect.
   * Asserted rather than assumed. */
  unsigned char *suffix = (unsigned char *)it + sizeof(item) + 1;
  unsigned char *data = suffix;        /* ITEM_data == ITEM_suffix when nsuffix is 0 */
  CHECK(suffix == data, 762);

  /* The value, and a sentinel in the two bytes just past it: those belong to
   * the chunk but NOT to the value, and a four-byte copy at `suffix` reaches
   * them. */
  data[0] = '\r'; data[1] = '\n';
  data[2] = 0x5a; data[3] = 0x5a;
  CHECK(*(volatile unsigned char *)(data + 2) == 0x5a, 763);

  const uint32_t flags = 0x41424344u;
  o->unit_reissued = 0;
  held = suffix;
  mark(6); /* LAST thing before the access */
  if (!fixed) {
    /* items.c before the fix: memcpy(ITEM_suffix(it), &flags, sizeof(flags)) */
    for (size_t i = 0; i < sizeof flags; i++)
      write_probe((volatile unsigned char *)(suffix + i));
  }
  /* The fix is "if (nsuffix > 0)", and nsuffix is 0 here, so the fixed arm
   * writes nothing at all. */

  o->accessed_through_stale = !fixed;
  o->damage = (*(volatile unsigned char *)(data + 2) != 0x5a) ||
              (*(volatile unsigned char *)(data + 3) != 0x5a);
  slabs_free(it, id);
}
