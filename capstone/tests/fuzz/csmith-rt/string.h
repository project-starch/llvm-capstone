/* Freestanding string.h for csmith programs in a domain: declarations only; the
 * definitions come from benchmarks/beebs/adapted/beebs_freestanding_string.c, which
 * is linked into every fuzz domain at the same optimisation level. */
#ifndef CAPSTONE_FUZZ_STRING_H
#define CAPSTONE_FUZZ_STRING_H
#include <stddef.h>
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);
int strcmp(const char *a, const char *b);
#endif
