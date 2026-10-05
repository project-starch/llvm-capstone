#include "corpus.h"

WM_CASE(13) {
/* Row 13 -- HTTP Range, fix 0261fd7da6 ("Fix buffer overflow, use after free
 * in HTTP Range"). The one SPATIAL case in this corpus: every other row ends a
 * lifetime, this one never does. The object is alive throughout and the cursor
 * simply leaves it.
 *
 * dissect_http_message copies the Range header value into FILE scope and then
 * advances the cursor by a fixed 6 ("past bytes=") at :3740 or by a fixed 8 at
 * :3789, with no check that the value is that long. A value shorter than the
 * skip puts the cursor past the end of the wmem chunk, and strtok/strtoul read
 * on from there -- into the next chunk of the SAME block, which is one
 * g_malloc. That is why malloc-granular bounds cannot see it: the read is
 * inside the region the system allocator handed wmem.
 *
 * Modelled on the :3788-3790 site (the "+= 8" one), because with a 6-byte
 * value its cursor lands at or past the neighbour's first byte rather than in
 * alignment padding, so the read has something to find. */
  const char *value = "bytes"; /* 6 bytes with the NUL: shorter than the skip */
  size_t len = strlen(value) + 1;
  CHECK(len == 6, 1);
  /* wmem_strdup(wmem_file_scope(), value) at :3788, as alloc + copy: the
   * corpus seam includes wmem_core only, and wmem_strdup is exactly this. */
  char *str = wmem_alloc(wm_file_scope(), len);
  CHECK(str, 2);
  memcpy(str, value, len);
  /* The storage the overread lands in: the next chunk carved from the same
   * block. Filled with a pattern, and its position asserted BEFORE the marker,
   * so the marker's presence is evidence that the crossing was really created
   * rather than merely written down. */
  unsigned char *neighbour = wmem_alloc(wm_file_scope(), 64);
  CHECK(neighbour, 3);
  CHECK((uintptr_t)neighbour > (uintptr_t)str, 4);
  memset(neighbour, 0x5a, 64);
  /* The two halves of the claim, both asserted rather than assumed:
   *   the cursor is PAST the chunk ..... 8 >= len
   *   and still INSIDE the block ....... below the neighbour's end */
  CHECK(8 >= len, 5);
  CHECK((uintptr_t)str + 8 < (uintptr_t)neighbour + 64, 6);
  /* "str += 8", :3789, unconditional. Then strtoul reads from the cursor. */
  wm_held = (unsigned char *)str + 8;
  wm_mark();
  (void)wm_probe(wm_held);
}
