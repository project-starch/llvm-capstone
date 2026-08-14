/* Split out of libc-ext/string.c, one function per file, for the reason musl
 * splits its own: an archive member is pulled whole. Programs that predate
 * this libc pass benchmarks/beebs/adapted/beebs_freestanding_string.c ahead of
 * the archive and define memcpy, memmove and strlen themselves; if those three
 * shared a member with memchr, the first stdio call would pull the member for
 * memchr and collide on the other three. See string.c for why these are byte at
 * a time. */
#include <stddef.h>

void *memmove(void *dest, const void *src, size_t n) {
  unsigned char *d = dest;
  const unsigned char *s = src;
  if (d == s || n == 0)
    return dest;
  /* Backwards when the regions overlap with dest above src; forwards otherwise.
     Compared as pointers, not as integers: the integer cast is the thing this
     file exists to avoid. */
  if (d < s || d >= s + n) {
    for (size_t i = 0; i < n; i++)
      d[i] = s[i];
  } else {
    for (size_t i = n; i > 0; i--)
      d[i - 1] = s[i - 1];
  }
  return dest;
}
