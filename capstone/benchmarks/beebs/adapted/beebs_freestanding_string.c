/*
 * Compact, self-contained, freestanding string/memory routines shared by the
 * Capstone PureCap libc-frontier BEEBS benchmarks (fasta, and later dtoa/trio).
 *
 * This is the "pure computation" slice of libc — memcpy/memmove/memset/strlen/
 * strcmp/strcpy touch no OS and make no syscalls, so they are implemented
 * locally (the design-sanctioned approach for fine-grained helpers; HostCall is
 * reserved for true OS/IO boundary crossings, and there is no hosted libc on the
 * bare-metal domain path).  It is the string counterpart to the shared
 * double-precision libm in beebs_softfloat_libm.c.
 *
 * Compiled and linked like the libm object; -ffunction-sections/--gc-sections in
 * the per-benchmark build drops whatever a given benchmark does not reference, so
 * pulling in this whole file costs nothing in the final domain image.
 *
 * Build the native self-test with:
 *   cc -DBEEBS_FREESTANDING_STRING_TEST -O2 beebs_freestanding_string.c \
 *      -o /tmp/str_test && /tmp/str_test
 */

typedef __SIZE_TYPE__ bsize_t;

void *memcpy(void *dst, const void *src, bsize_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  for (bsize_t i = 0; i < n; i++)
    d[i] = s[i];
  return dst;
}

void *memmove(void *dst, const void *src, bsize_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  if (d == s || n == 0)
    return dst;
  if (d < s) {
    for (bsize_t i = 0; i < n; i++)
      d[i] = s[i];
  } else {
    for (bsize_t i = n; i != 0; i--)
      d[i - 1] = s[i - 1];
  }
  return dst;
}

void *memset(void *dst, int c, bsize_t n) {
  unsigned char *d = (unsigned char *)dst;
  for (bsize_t i = 0; i < n; i++)
    d[i] = (unsigned char)c;
  return dst;
}

bsize_t strlen(const char *s) {
  const char *p = s;
  while (*p)
    p++;
  return (bsize_t)(p - s);
}

int strcmp(const char *a, const char *b) {
  while (*a && (*a == *b)) {
    a++;
    b++;
  }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

char *strcpy(char *dst, const char *src) {
  char *d = dst;
  while ((*d++ = *src++))
    ;
  return dst;
}

#ifdef BEEBS_FREESTANDING_STRING_TEST
#include <stdio.h>
/* Our routines use the standard libc names, so we cannot include <string.h>
   here (it would redeclare/clash).  Instead exercise them and sanity-check the
   results directly. */
int main(void) {
  char buf[32], buf2[32];
  int fail = 0;

  for (int i = 0; i < 32; i++)
    buf[i] = (char)0xAA;
  memset(buf, 'x', 10);
  for (int i = 0; i < 10; i++)
    if (buf[i] != 'x')
      fail = 1;
  if ((unsigned char)buf[10] != 0xAA)
    fail = 1;

  const char *msg = "hello world";
  if (strlen(msg) != 11)
    fail = 1;

  memcpy(buf2, msg, 12);
  if (strcmp(buf2, "hello world") != 0)
    fail = 1;

  strcpy(buf, "abc");
  if (strcmp(buf, "abc") != 0 || strcmp(buf, "abd") >= 0)
    fail = 1;

  /* overlapping move: shift "0123456789" right by 2 */
  char ov[16] = "0123456789";
  memmove(ov + 2, ov, 8);
  if (ov[2] != '0' || ov[9] != '7')
    fail = 1;

  printf("freestanding string self-test: %s\n", fail ? "FAIL" : "ok");
  return fail;
}
#endif
