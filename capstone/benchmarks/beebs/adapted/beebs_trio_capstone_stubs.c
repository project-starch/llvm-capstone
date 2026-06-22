#include "beebs_trio_capstone_preamble.h"

int errno;

void *memset(void *s, int c, size_t n) {
  unsigned char *p = s;
  while (n--)
    *p++ = (unsigned char)c;
  return s;
}

void *memcpy(void *d, const void *s, size_t n) {
  char *dd = d;
  const char *ss = s;
  while (n--)
    *dd++ = *ss++;
  return d;
}

void *memmove(void *d, const void *s, size_t n) {
  char *dd = d;
  const char *ss = s;
  if (dd < ss) {
    while (n--)
      *dd++ = *ss++;
  } else {
    dd += n;
    ss += n;
    while (n--)
      *--dd = *--ss;
  }
  return d;
}

int memcmp(const void *a, const void *b, size_t n) {
  const unsigned char *x = a;
  const unsigned char *y = b;
  while (n--) {
    if (*x != *y)
      return *x < *y ? -1 : 1;
    ++x;
    ++y;
  }
  return 0;
}

size_t strlen(const char *s) {
  size_t n = 0;
  while (s[n])
    ++n;
  return n;
}

int strcmp(const char *a, const char *b) {
  while (*a && *a == *b) {
    ++a;
    ++b;
  }
  return (unsigned char)*a - (unsigned char)*b;
}

int strncmp(const char *a, const char *b, size_t n) {
  while (n && *a && *a == *b) {
    ++a;
    ++b;
    --n;
  }
  return n ? (unsigned char)*a - (unsigned char)*b : 0;
}

char *strchr(const char *s, int c) {
  while (*s) {
    if (*s == (char)c)
      return (char *)s;
    ++s;
  }
  return c == 0 ? (char *)s : 0;
}

size_t strspn(const char *s, const char *accept) {
  size_t n = 0;
  while (s[n] && strchr(accept, s[n]))
    ++n;
  return n;
}

size_t strcspn(const char *s, const char *reject) {
  size_t n = 0;
  while (s[n] && !strchr(reject, s[n]))
    ++n;
  return n;
}

char *strcat(char *d, const char *s) {
  char *r = d;
  while (*d)
    ++d;
  while ((*d++ = *s++))
    ;
  return r;
}

char *strstr(const char *s, const char *needle) {
  if (!*needle)
    return (char *)s;
  for (; *s; ++s) {
    const char *a = s;
    const char *b = needle;
    while (*a && *b && *a == *b) {
      ++a;
      ++b;
    }
    if (!*b)
      return (char *)s;
  }
  return 0;
}

char *strtok(char *s, const char *delim) {
  static char *next;
  if (!s)
    s = next;
  if (!s)
    return 0;
  s += strspn(s, delim);
  if (!*s) {
    next = 0;
    return 0;
  }
  char *end = s + strcspn(s, delim);
  if (*end)
    *end++ = 0;
  next = end;
  return s;
}

char *strncpy(char *d, const char *s, size_t n) {
  char *r = d;
  size_t i = 0;
  for (; i < n && s[i]; ++i)
    d[i] = s[i];
  for (; i < n; ++i)
    d[i] = 0;
  return r;
}

static int digit_value(int c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  c |= 32;
  if (c >= 'a' && c <= 'z')
    return c - 'a' + 10;
  return -1;
}

unsigned long strtoul(const char *nptr, char **endptr, int base) {
  const char *s = nptr;
  unsigned long v = 0;
  while (isspace(*s))
    ++s;
  if (*s == '+')
    ++s;
  if ((base == 0 || base == 16) && s[0] == '0' &&
      (s[1] == 'x' || s[1] == 'X')) {
    base = 16;
    s += 2;
  } else if (base == 0) {
    base = (*s == '0') ? 8 : 10;
  }
  const char *start = s;
  for (;;) {
    int d = digit_value(*s);
    if (d < 0 || d >= base)
      break;
    v = v * (unsigned)base + (unsigned)d;
    ++s;
  }
  if (endptr)
    *endptr = (char *)(s == start ? nptr : s);
  return v;
}

long strtol(const char *nptr, char **endptr, int base) {
  const char *s = nptr;
  while (isspace(*s))
    ++s;
  int neg = (*s == '-');
  if (*s == '-' || *s == '+')
    ++s;
  unsigned long v = strtoul(s, endptr, base);
  return neg ? -(long)v : (long)v;
}
