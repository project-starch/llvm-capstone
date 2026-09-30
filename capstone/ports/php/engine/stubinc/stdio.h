/* Freestanding stub. PHP reaches for stdio mostly on paths we compile out (logging,
 * php.ini, the CLI SAPI). What the required Zend set actually needs is the FILE type
 * to exist and a few prototypes to resolve; the domain never calls them. */
#ifndef _STDIO_H
#define _STDIO_H 1
#include <stddef.h>
#include <stdarg.h>
typedef struct _IO_FILE FILE;
extern FILE *stdin, *stdout, *stderr;
#define EOF (-1)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define BUFSIZ 8192
int printf(const char *, ...);
int fprintf(FILE *, const char *, ...);
int sprintf(char *, const char *, ...);
int snprintf(char *, size_t, const char *, ...);
int vsnprintf(char *, size_t, const char *, va_list);
int vsprintf(char *, const char *, va_list);
int vfprintf(FILE *, const char *, va_list);
int fputs(const char *, FILE *);
int fputc(int, FILE *);
int fflush(FILE *);
int fclose(FILE *);
FILE *fopen(const char *, const char *);
FILE *fdopen(int, const char *);
size_t fread(void *, size_t, size_t, FILE *);
size_t fwrite(const void *, size_t, size_t, FILE *);
int fileno(FILE *);
int puts(const char *);
#endif
