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
