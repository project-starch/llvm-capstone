/* Six of the nine src/string files musl cannot compile for capstone64, byte at a time.
 *
 * They all fail the same way: musl reads memory a word at a time and decides
 * whether it may by testing `(uintptr_t)s % ALIGN`. Casting a capability to an
 * integer to inspect its low bits is precisely what this target refuses, so the
 * fast paths are unavailable and the portable ones are what is left.
 *
 * WHY IN THE LIBC. Every domain so far linked
 * benchmarks/beebs/adapted/beebs_freestanding_string.c ahead of the archive for
 * memcpy/memmove/strlen, which is a per-program dependency on a benchmark
 * directory, and it still did not cover memchr -- so strnlen, which musl's own
 * printf path uses, failed to link the moment stdio was pulled in. That
 * dependency is what this file removes.
 *
 * Programs that keep passing beebs_freestanding_string.c ahead of the archive
 * are unaffected: their definitions resolve first and these are never pulled.
 *
 * THE CEILING: byte at a time is slower than musl's word-at-a-time original by
 * roughly the word size on long buffers. The upgrade path is not to reinstate
 * the integer cast but to test alignment with the capability query builtins
 * (__builtin_capstone_cap_get_base and friends, as revoke_on_free_alloc.h
 * does), which is a real port of these routines rather than a fallback.
 */
#include <stddef.h>

void *memchr(const void *s, int c, size_t n) {
  const unsigned char *p = s;
  unsigned char want = (unsigned char)c;
  for (size_t i = 0; i < n; i++)
    if (p[i] == want)
      return (void *)(p + i);
  return 0;
}

void *memccpy(void *restrict dest, const void *restrict src, int c, size_t n) {
  unsigned char *d = dest;
  const unsigned char *s = src;
  unsigned char want = (unsigned char)c;
  for (size_t i = 0; i < n; i++) {
    d[i] = s[i];
    if (s[i] == want)
      return d + i + 1;
  }
  return 0;
}

char *stpcpy(char *restrict d, const char *restrict s) {
  size_t i = 0;
  for (; s[i]; i++)
    d[i] = s[i];
  d[i] = 0;
  return d + i; /* the NUL, not one past it */
}

char *stpncpy(char *restrict d, const char *restrict s, size_t n) {
  size_t i = 0;
  for (; i < n && s[i]; i++)
    d[i] = s[i];
  char *end = d + i;
  for (; i < n; i++) /* pad, as the standard requires */
    d[i] = 0;
  return end;
}

char *strchrnul(const char *s, int c) {
  char want = (char)c;
  size_t i = 0;
  for (; s[i] && s[i] != want; i++)
    ;
  return (char *)(s + i); /* the match, or the terminator */
}

size_t strlcpy(char *d, const char *s, size_t n) {
  size_t len = 0;
  while (s[len])
    len++;
  if (n) {
    size_t copy = len < n - 1 ? len : n - 1;
    for (size_t i = 0; i < copy; i++)
      d[i] = s[i];
    d[copy] = 0;
  }
  return len; /* the length it WANTED, so truncation is detectable */
}
