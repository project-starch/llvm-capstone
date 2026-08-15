/* Split out of libc-ext/string.c, one function per file, for the reason musl
 * splits its own: an archive member is pulled whole. Programs that predate
 * this libc pass benchmarks/beebs/adapted/beebs_freestanding_string.c ahead of
 * the archive and define memcpy, memmove and strlen themselves; if those three
 * shared a member with memchr, the first stdio call would pull the member for
 * memchr and collide on the other three. See string.c for why the STRING
 * functions are byte at a time; this one must not be, and cap-copy.h says why. */
#include <stddef.h>

#include "cap-copy.h"

void *memcpy(void *restrict dest, const void *restrict src, size_t n) {
  __capstone_cap_copy_fwd(dest, src, n);
  return dest;
}
