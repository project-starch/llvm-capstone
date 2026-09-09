#include "capstone_sqlite_libc.h"

void *memchr(const void *ptr, int value, size_t count) {
  const unsigned char *bytes = (const unsigned char *)ptr;
  for (size_t i = 0; i < count; ++i) {
    if (bytes[i] == (unsigned char)value)
      return (void *)&bytes[i];
  }
  return NULL;
}

int strncmp(const char *lhs, const char *rhs, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    unsigned char a = (unsigned char)lhs[i];
    unsigned char b = (unsigned char)rhs[i];
    if (a != b)
      return (int)a - (int)b;
    if (a == 0)
      return 0;
  }
  return 0;
}

char *strncpy(char *dst, const char *src, size_t count) {
  size_t i = 0;
  while (i < count && src[i] != '\0') {
    dst[i] = src[i];
    ++i;
  }
  while (i < count)
    dst[i++] = '\0';
  return dst;
}

char *strchr(const char *text, int value) {
  char needle = (char)value;
  do {
    if (*text == needle)
      return (char *)text;
  } while (*text++);
  return NULL;
}

char *strrchr(const char *text, int value) {
  const char *last = NULL;
  char needle = (char)value;
  do {
    if (*text == needle)
      last = text;
  } while (*text++);
  return (char *)last;
}

char *strstr(const char *haystack, const char *needle) {
  if (!*needle)
    return (char *)haystack;
  for (; *haystack; ++haystack) {
    size_t i = 0;
    while (needle[i] && haystack[i] == needle[i])
      ++i;
    if (!needle[i])
      return (char *)haystack;
  }
  return NULL;
}

size_t strspn(const char *text, const char *accept) {
  size_t count = 0;
  while (text[count]) {
    if (!strchr(accept, text[count]))
      break;
    ++count;
  }
  return count;
}

size_t strcspn(const char *text, const char *reject) {
  size_t count = 0;
  while (text[count] && !strchr(reject, text[count]))
    ++count;
  return count;
}

int isspace(int value) {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
         value == '\f' || value == '\v';
}

int isalpha(int value) {
  return (value >= 'a' && value <= 'z') ||
         (value >= 'A' && value <= 'Z');
}

int isdigit(int value) { return value >= '0' && value <= '9'; }

int isalnum(int value) { return isalpha(value) || isdigit(value); }

int isxdigit(int value) {
  return isdigit(value) || (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

int toupper(int value) {
  return value >= 'a' && value <= 'z' ? value - ('a' - 'A') : value;
}

int tolower(int value) {
  return value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value;
}

void abort(void) {
  for (;;)
    ;
}

void *malloc(size_t size) {
  (void)size;
  return NULL;
}

void *realloc(void *ptr, size_t size) {
  (void)ptr;
  (void)size;
  return NULL;
}

void free(void *ptr) { (void)ptr; }

char *getenv(const char *name) {
  (void)name;
  return NULL;
}

struct tm *gmtime(const time_t *time) {
  (void)time;
  return NULL;
}

size_t strftime(char *dst, size_t size, const char *format,
                const struct tm *time) {
  (void)dst;
  (void)size;
  (void)format;
  (void)time;
  return 0;
}
