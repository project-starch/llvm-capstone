#pragma once
#include <stddef.h>
void *malloc(size_t);
void *realloc(void *, size_t);
void free(void *);
void abort(void);
void exit(int);
long strtol(const char *, char **, int);
unsigned long strtoul(const char *, char **, int);
double strtod(const char *, char **);

/* Math.random() reaches rand() through ecma-builtin-math.c:278-281. RAND_MAX is
   2^31-1 so the ratio rand()/RAND_MAX covers [0,1] the way that code assumes. */
#define RAND_MAX 2147483647
int rand(void);
void srand(unsigned int);
