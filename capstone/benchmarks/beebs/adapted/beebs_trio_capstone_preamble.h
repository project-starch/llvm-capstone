#ifndef BEEBS_TRIO_CAPSTONE_PREAMBLE_H
#define BEEBS_TRIO_CAPSTONE_PREAMBLE_H

typedef unsigned long size_t;
typedef long ptrdiff_t;
typedef long intmax_t;
typedef unsigned long uintmax_t;
typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

typedef struct FILE FILE;
struct tm;

#ifndef NULL
#define NULL ((void *)0)
#endif
#ifndef EOF
#define EOF (-1)
#endif

#define INT_MAX 2147483647
#define INT_MIN (-2147483647 - 1)
#define UINT_MAX 4294967295U
#define LONG_MAX 9223372036854775807L
#define LONG_MIN (-9223372036854775807L - 1L)
#define ULONG_MAX 18446744073709551615UL
#define UCHAR_MAX 255U
#define CHAR_MAX 127
#define CHAR_BIT 8
#define MB_LEN_MAX 6

extern int errno;

#define assert(x) ((void)0)
#define isascii(c) (((unsigned)(c) & ~0x7fU) == 0)
#define isdigit(c) ((unsigned)((c) - '0') <= 9U)
#define isxdigit(c) (isdigit(c) || ((unsigned)(((c) | 32) - 'a') <= 5U))
#define isalpha(c) ((unsigned)(((c) | 32) - 'a') <= 25U)
#define isalnum(c) (isalpha(c) || isdigit(c))
#define isspace(c) ((c) == ' ' || (unsigned)((c) - 9) <= 4U)
#define isprint(c) ((unsigned)((c) - 32) < 95U)
#define toupper(c) ((char)((((c) >= 'a') && ((c) <= 'z')) ? ((c) - 32) : (c)))
#define tolower(c) ((char)((((c) >= 'A') && ((c) <= 'Z')) ? ((c) + 32) : (c)))

void *memset(void *s, int c, size_t n);
void *memcpy(void *d, const void *s, size_t n);
void *memmove(void *d, const void *s, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);
int strcmp(const char *a, const char *b);
int strncmp(const char *a, const char *b, size_t n);
char *strchr(const char *s, int c);
size_t strspn(const char *s, const char *accept);
size_t strcspn(const char *s, const char *reject);
char *strcat(char *d, const char *s);
char *strstr(const char *s, const char *needle);
char *strtok(char *s, const char *delim);
char *strncpy(char *d, const char *s, size_t n);
long strtol(const char *nptr, char **endptr, int base);
unsigned long strtoul(const char *nptr, char **endptr, int base);

void *malloc_beebs(size_t size);
void free_beebs(void *ptr);
void *realloc_beebs(void *ptr, size_t size);

#endif
