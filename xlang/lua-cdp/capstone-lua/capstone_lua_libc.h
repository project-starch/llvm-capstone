/* Freestanding libc surface for a reference-Lua-5.4.7 Capstone domain.
 *
 * Force-included (-include) ahead of every Lua TU. It predefines the glibc
 * include-guards of the HOSTED headers so `#include <string.h>` (etc.) pulls in
 * NOTHING from the host glibc — whose FILE/pointer layouts assume 8-byte
 * pointers and do not survive the 16-byte-capability target (bits/floatn.h
 * __float128, struct _IO_FILE negative padding, __int64_t long-vs-long-long).
 * Freestanding clang still supplies stddef/stdarg/stdint/limits/float itself.
 *
 * Same mechanism as capstone/benchmarks/sqlite/adapted/capstone_sqlite_libc.h,
 * extended with the surface the Lua core + base/string/table/math libs need
 * (math, setjmp, locale, snprintf, strtod, the full ctype set). Only
 * DECLARATIONS live here; implementations are separate TUs (most reused from the
 * SQLite/beebs libc + libm; the gaps — snprintf, setjmp/longjmp, fmod/modf/
 * frexp/ldexp, __divdf3 — are added by the port).
 */
#ifndef CAPSTONE_LUA_LIBC_H
#define CAPSTONE_LUA_LIBC_H

#include <stddef.h>
#include <stdarg.h>

/* Predefine the guards of every hosted header Lua reaches. */
#define _STDIO_H 1
#define _STDLIB_H 1
#define _STRING_H 1
#define _STRINGS_H 1
#define _ASSERT_H 1
#define _CTYPE_H 1
#define _TIME_H 1
#define _MATH_H 1
#define _SETJMP_H 1
#define _LOCALE_H 1
#define _ERRNO_H 1

#ifndef NULL
#define NULL ((void *)0)
#endif
#define EOF (-1)
#define BUFSIZ 512

/* Lua builds without internal asserts by default (lua_assert = ((void)0)); make
 * a bare assert() a no-op in case any TU includes <assert.h>. */
#define assert(condition) ((void)0)

/* ---- string.h ---------------------------------------------------------- */
void  *memcpy(void *, const void *, size_t);
void  *memmove(void *, const void *, size_t);
void  *memset(void *, int, size_t);
int    memcmp(const void *, const void *, size_t);
void  *memchr(const void *, int, size_t);
size_t strlen(const char *);
int    strcmp(const char *, const char *);
int    strncmp(const char *, const char *, size_t);
char  *strcpy(char *, const char *);
char  *strchr(const char *, int);
char  *strpbrk(const char *, const char *);
size_t strspn(const char *, const char *);
char  *strstr(const char *, const char *);
int    strcoll(const char *, const char *);

/* ---- ctype.h ----------------------------------------------------------- */
int isdigit(int), isalpha(int), isalnum(int), isspace(int), iscntrl(int),
    isupper(int), islower(int), isxdigit(int), ispunct(int), isgraph(int),
    isprint(int);
int toupper(int), tolower(int);

/* ---- stdlib.h ---------------------------------------------------------- */
void  abort(void);
void *malloc(size_t);
void  free(void *);
void *realloc(void *, size_t);
double strtod(const char *, char **);
int    abs(int);

/* ---- stdio.h  (buffer formatting only; no FILE I/O in the domain) ------- */
typedef struct capstone_lua_FILE FILE;
int snprintf(char *, size_t, const char *, ...);
int vsnprintf(char *, size_t, const char *, va_list);

/* ---- math.h ------------------------------------------------------------ */
double floor(double), ceil(double), fabs(double), sqrt(double);
double fmod(double, double), pow(double, double), atan2(double, double);
double exp(double), log(double), log2(double), log10(double);
double sin(double), cos(double), tan(double), asin(double), acos(double);
double sinh(double), cosh(double), tanh(double);
double frexp(double, int *), ldexp(double, int), modf(double, double *);
#define HUGE_VAL (__builtin_huge_val())

/* ---- locale.h ---------------------------------------------------------- */
struct lconv { char *decimal_point; };
struct lconv *localeconv(void);

/* ---- time.h  (only luai_makeseed's hash seed; override luai_makeseed to a
 * cycle counter in the real domain to drop this) ------------------------- */
typedef long time_t;
time_t time(time_t *);

/* ---- hosted stubs: declared so lauxlib/lbaselib/ltablib link. These paths are
 * NEVER taken in the domain -- scripts load via luaL_loadbufferx (not the fopen
 * file loader), print/error route to the domain writer, and the sort-pivot clock
 * is stubbed. Implementations (all error/no-op stubs) live in the libc TU. ---- */
extern int errno;
char  *strerror(int);
typedef long clock_t;
clock_t clock(void);
extern FILE *stdin, *stdout, *stderr;
FILE  *fopen(const char *, const char *);
FILE  *freopen(const char *, const char *, FILE *);
int    fclose(FILE *);
unsigned long fread(void *, unsigned long, unsigned long, FILE *);
unsigned long fwrite(const void *, unsigned long, unsigned long, FILE *);
int    fflush(FILE *);
int    fprintf(FILE *, const char *, ...);
int    getc(FILE *);
int    ungetc(int, FILE *);
int    feof(FILE *);
int    ferror(FILE *);

/* ---- setjmp.h  (capability-wide slots: capstone64 saved regs are 16 B) ---
 * 14 saved capabilities (ra, sp, s0-s11) x 16 B = 224 B, 16-byte aligned --
 * stc/ldc DROP the capability tag on a non-16-aligned address. See
 * capstone-lua/capstone_setjmp.S. */
typedef struct __attribute__((aligned(16))) { long long __c[28]; } __capstone_jmpbuf[1];
#define jmp_buf __capstone_jmpbuf
int  setjmp(jmp_buf);
void longjmp(jmp_buf, int);

#endif /* CAPSTONE_LUA_LIBC_H */
