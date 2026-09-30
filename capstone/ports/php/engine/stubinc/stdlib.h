#ifndef _STDLIB_H
#define _STDLIB_H 1
#include <stddef.h>
/* malloc/free/realloc are supplied by the Capstone allocator, NOT by a libc. Keeping
 * ZEND_MM undefined means zend_alloc.c's ZEND_DO_MALLOC is exactly these, which is the
 * seam that gives every emalloc block its own bounded capability. */
void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void free(void *);
void exit(int) __attribute__((noreturn));
void abort(void) __attribute__((noreturn));
int atoi(const char *);
long atol(const char *);
double atof(const char *);
long strtol(const char *, char **, int);
unsigned long strtoul(const char *, char **, int);
double strtod(const char *, char **);
char *getenv(const char *);
void qsort(void *, size_t, size_t, int (*)(const void *, const void *));
int abs(int);
#define RAND_MAX 2147483647
#endif
