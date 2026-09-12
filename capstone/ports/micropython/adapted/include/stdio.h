#pragma once
#include <stddef.h>
#include <stdarg.h>
typedef struct _FILE FILE;
extern FILE *stderr;
extern FILE *stdout;
int printf(const char *, ...);
int fprintf(FILE *, const char *, ...);
int snprintf(char *, size_t, const char *, ...);
int vsnprintf(char *, size_t, const char *, va_list);
int putchar(int);
int puts(const char *);
