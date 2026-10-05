/* Case 7: ecdb011 -- an item key that is not NUL-terminated is handed to a
 * string formatter, which reads past the key.
 *
 * Shape: unbounded read forward from an item key with no terminator
 * Consumer: items.c, do_item_cachedump()
 * SPATIAL. The read runs from the key into the item's own later fields and on
 * through the chunk; how far depends on where the next zero byte happens to
 * be, which is why the fix copies a bounded nkey into a local buffer and
 * terminates it itself.
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

#define KEY_LEN 8
#define SCAN_LIMIT 64   /* how far the case is willing to follow the read */

MC_CASE(7) {
  o->defect_text = "the formatter read past the key, because nothing terminated it";
  o->fixed_text = "the bounded copy terminated at nkey, so the read stopped there";

  unsigned id = slabs_clsid(sizeof(item) + 64);
  CHECK(id, 770);
  item *it = slabs_alloc(id, 0);
  CHECK(it, 771);
  it->slabs_clsid = (uint8_t)id;
  it->nkey = KEY_LEN;
  it->it_flags = 0;
  it->nbytes = 2;

  unsigned char *key = (unsigned char *)it + sizeof(item);
  /* The key, deliberately WITHOUT the terminator upstream relies on, and the
   * bytes after it made non-zero so the read has somewhere to run. */
  memset(key, 'k', KEY_LEN);
  memset(key + KEY_LEN, 0x5a, SCAN_LIMIT - KEY_LEN);
  CHECK(*(volatile unsigned char *)(key + KEY_LEN) != 0, 772);

  /* The fix: strncpy(key_temp, ITEM_key(it), it->nkey) into a local buffer,
   * then terminate. Modelled as a bounded copy the scan then reads. */
  unsigned char local[KEY_LEN + 1];
  const volatile unsigned char *scan;
  if (fixed) {
    memcpy(local, key, KEY_LEN);
    local[KEY_LEN] = 0;
    scan = local;
  } else {
    scan = key;                       /* passed straight to the formatter */
  }

  o->unit_reissued = 0;
  held = (volatile unsigned char *)scan;
  mark(7); /* LAST thing before the access */
  /* How far a %s-style read gets before it finds a zero. */
  unsigned n = 0;
  while (n < SCAN_LIMIT && read_probe(scan + n) != 0)
    n++;

  o->accessed_through_stale = !fixed && n > KEY_LEN;
  o->damage = n > KEY_LEN;            /* the read left the key field */
  slabs_free(it, id);
}
