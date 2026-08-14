/* Split out of libc-ext/string.c, one function per file, for the reason musl
 * splits its own: an archive member is pulled whole. Programs that predate
 * this libc pass benchmarks/beebs/adapted/beebs_freestanding_string.c ahead of
 * the archive and define memcpy, memmove and strlen themselves; if those three
 * shared a member with memchr, the first stdio call would pull the member for
 * memchr and collide on the other three. See string.c for why these are byte at
 * a time. */
#include <stddef.h>

void *memcpy(void *restrict dest, const void *restrict src, size_t n) {
  unsigned char *d = dest;
  const unsigned char *s = src;
  for (size_t i = 0; i < n; i++)
    d[i] = s[i];
  return dest;
}
