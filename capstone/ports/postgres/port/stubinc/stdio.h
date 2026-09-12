#ifndef A11_STDIO_H
#define A11_STDIO_H
#include <stddef.h>
#include <stdarg.h>
typedef struct _A11_FILE FILE;
extern FILE *stderr;
extern FILE *stdout;
int fprintf(FILE *, const char *, ...);
int printf(const char *, ...);
int snprintf(char *, size_t, const char *, ...);
int vsnprintf(char *, size_t, const char *, va_list);
int vfprintf(FILE *, const char *, va_list);
int vprintf(const char *, va_list);
int fflush(FILE *);
int fputc(int, FILE *);
size_t fwrite(const void *, size_t, size_t, FILE *);
#endif
