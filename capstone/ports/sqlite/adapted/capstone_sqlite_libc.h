#ifndef CAPSTONE_SQLITE_LIBC_H
#define CAPSTONE_SQLITE_LIBC_H

#include <stdarg.h>
#include <stddef.h>

/*
 * The Capstone bare-metal target has no hosted sysroot. Prevent SQLite's
 * standard includes from pulling in host glibc declarations, whose FILE and
 * pointer layouts assume 64-bit pointers rather than 128-bit capabilities.
 */
#define _STDIO_H 1
#define _STDLIB_H 1
#define _STRING_H 1
#define _ASSERT_H 1
#define _CTYPE_H 1
#define _TIME_H 1
#define _MATH_H 1

typedef struct capstone_sqlite_file FILE;
typedef long time_t;

struct tm {
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
};

#ifndef NULL
#define NULL ((void *)0)
#endif
#define EOF (-1)
#define FILENAME_MAX 256

#define assert(condition) ((void)0)

void *memcpy(void *dst, const void *src, size_t count);
void *memmove(void *dst, const void *src, size_t count);
void *memset(void *dst, int value, size_t count);
int memcmp(const void *lhs, const void *rhs, size_t count);
void *memchr(const void *ptr, int value, size_t count);

size_t strlen(const char *text);
int strcmp(const char *lhs, const char *rhs);
int strncmp(const char *lhs, const char *rhs, size_t count);
char *strcpy(char *dst, const char *src);
char *strncpy(char *dst, const char *src, size_t count);
char *strchr(const char *text, int value);
char *strrchr(const char *text, int value);
char *strstr(const char *haystack, const char *needle);
size_t strspn(const char *text, const char *accept);
size_t strcspn(const char *text, const char *reject);

int isspace(int value);
int isalpha(int value);
int isalnum(int value);
int isdigit(int value);
int isxdigit(int value);
int toupper(int value);
int tolower(int value);

void abort(void);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);
char *getenv(const char *name);
struct tm *gmtime(const time_t *time);
size_t strftime(char *dst, size_t size, const char *format,
                const struct tm *time);

#endif
