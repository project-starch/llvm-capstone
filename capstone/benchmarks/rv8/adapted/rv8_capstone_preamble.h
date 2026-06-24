/*
 * Freestanding preamble for Capstone PureCap RV8 (rv8-bench) domain builds.
 *
 * rv8-bench programs are written for hosted Linux (stdio/stdlib/string/sys-time).
 * On the bare-metal Capstone domain there is no hosted libc, so the per-benchmark
 * build strips those includes and this preamble supplies the minimal declarations
 * the upstream sources need. Definitions come from the shared adapted runtime:
 *   - memcpy/memset/strlen/strcpy/strcmp -> beebs adapted/beebs_freestanding_string.c
 *   - malloc/free                        -> adapted/rv8_malloc.c (16-aligned arena)
 *   - gettimeofday/printf/exit           -> adapted/rv8_stubs.c (no-ops; timing and
 *                                           reporting are irrelevant to correctness)
 * The benchmark result is validated by an adapted oracle, not by stdout.
 */
#ifndef RV8_CAPSTONE_PREAMBLE_H
#define RV8_CAPSTONE_PREAMBLE_H

typedef unsigned long size_t;

#ifndef NULL
#define NULL ((void *)0)
#endif

#ifndef assert
#define assert(x) ((void)0)
#endif

#ifndef bzero
#define bzero(s, n) ((void)memset((s), 0, (n)))
#endif

struct timeval {
  long tv_sec;
  long tv_usec;
};

int gettimeofday(struct timeval *tv, void *tz);
int printf(const char *fmt, ...);
void exit(int code);

void *malloc(size_t n);
void free(void *p);

void *memcpy(void *d, const void *s, size_t n);
void *memset(void *s, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);
char *strcpy(char *d, const char *s);
int strcmp(const char *a, const char *b);

double sqrt(double x);

#endif /* RV8_CAPSTONE_PREAMBLE_H */
