/* Freestanding <stdlib.h> for WAMR. malloc/free/realloc are NOT declared as the
   domain's: WAMR routes its own allocations through mem-alloc/ems over the buffer
   os_mmap returns, and the platform layer backs os_malloc with the same arena. A
   second malloc here would put an allocator between the measurement and its
   subject. */
#ifndef CAPSTONE_WAMR_STDLIB_H
#define CAPSTONE_WAMR_STDLIB_H
#include <stddef.h>
void abort(void);
void exit(int);
long labs(long);
int abs(int);
void *bsearch(const void *, const void *, size_t, size_t,
              int (*)(const void *, const void *));
void qsort(void *, size_t, size_t, int (*)(const void *, const void *));
long strtol(const char *, char **, int);
unsigned long strtoul(const char *, char **, int);
unsigned long long strtoull(const char *, char **, int);
long long strtoll(const char *, char **, int);
double strtod(const char *, char **);
float strtof(const char *, char **);
int atoi(const char *);
#ifndef NULL
#define NULL ((void *)0)
#endif
#endif
