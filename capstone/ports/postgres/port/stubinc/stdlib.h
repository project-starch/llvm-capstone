#ifndef A11_STDLIB_H
#define A11_STDLIB_H
#include <stddef.h>
void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void free(void *);
void abort(void);
void exit(int);
char *getenv(const char *);
long strtol(const char *, char **, int);
#endif
