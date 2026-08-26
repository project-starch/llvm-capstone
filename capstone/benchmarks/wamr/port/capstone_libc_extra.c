/* The libc gap the census named, and nothing more.
 *
 * Deliberately small. snprintf/vsnprintf come from
 * xlang/lua-cdp/capstone-lua/lua_libc.c and the string routines from
 * beebs_freestanding_string.c; duplicating either would collide at link, which is
 * the mistake lua_libc.c's own header warns about.
 *
 * ALREADY PROVIDED elsewhere, do NOT add them here -- they collide at link:
 *   abs, snprintf, vsnprintf, strtod, strchr, strncmp, strstr, memchr,
 *   isalpha and the rest of ctype   -> xlang/lua-cdp/capstone-lua/lua_libc.c
 *   memcpy/memmove/memset/memcmp/strlen/strcmp/strcpy -> beebs_freestanding_string.c
 *   floor/ceil/fabs/sqrt/trunc/rint -> beebs_softfloat_libm.c
 *
 * Self-check: build native and run the asserts --
 *   cc -DCAPSTONE_WAMR_LIBC_SELFTEST -O2 capstone_libc_extra.c -o /tmp/t && /tmp/t
 */
#ifdef CAPSTONE_WAMR_LIBC_SELFTEST
/* No <stdlib.h> here on purpose: it declares labs/abs/bsearch itself, and on a
   current glibc `abs` is a _Generic macro, so including it turns the definitions
   below into syntax errors rather than redefinitions. <stdarg.h> and <string.h>
   ARE needed: va_list for vsnprintf, strcmp for the asserts. */
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#define snprintf  capstone_snprintf_selftest
#define vsnprintf capstone_vsnprintf_selftest
#else
#include <stddef.h>
#include "platform_internal.h"
#define ABORT_IMPL abort
#endif

long
labs(long v)
{
    return v < 0 ? -v : v;
}


/* Binary search over a sorted array. WAMR uses it for opcode and export tables,
   where nmemb is small; the loop is written for correctness on an EMPTY array and
   on nmemb == 1, which is where hand-rolled versions usually get it wrong. */
void *
bsearch(const void *key, const void *base, size_t nmemb, size_t size,
        int (*cmp)(const void *, const void *))
{
    const unsigned char *b = (const unsigned char *)base;
    size_t lo = 0, hi = nmemb;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int r = cmp(key, b + mid * size);
        if (r == 0)
            return (void *)(b + mid * size);
        if (r < 0)
            hi = mid;
        else
            lo = mid + 1;
    }
    return NULL;
}

#ifndef CAPSTONE_WAMR_LIBC_SELFTEST
/* A domain cannot exit(); the loader reads a marker out of the return value.
   Spinning would look exactly like a wedge, and this project spends board time on
   telling those apart, so abort RETURNS through the trap the caller can classify.
   ponytail: the trap is an unreachable, which the backend turns into an illegal
   instruction. Ceiling: no message reaches the host. Upgrade path is a marker
   write into the result word, which needs the result pointer threaded here. */
void
abort(void)
{
    __builtin_trap();
}

void
exit(int code)
{
    (void)code;
    __builtin_trap();
}
#endif

/* Insertion sort. WAMR sorts export and opcode tables, which are tens of entries,
   so O(n^2) is the right trade against the code size of a quicksort.
   ponytail: the ceiling is n^2; a module with thousands of exports would notice,
   and the upgrade path is heapsort, which needs no extra storage either. */
void
qsort(void *base, size_t nmemb, size_t size,
      int (*cmp)(const void *, const void *))
{
    unsigned char *b = (unsigned char *)base;
    for (size_t i = 1; i < nmemb; i++) {
        for (size_t j = i; j > 0 && cmp(b + (j - 1) * size, b + j * size) > 0; j--) {
            unsigned char *x = b + (j - 1) * size, *y = b + j * size;
            for (size_t k = 0; k < size; k++) {
                unsigned char t = x[k];
                x[k] = y[k];
                y[k] = t;
            }
        }
    }
}

/* libm gaps. beebs_softfloat_libm.c has the double ceil/floor/fabs/sqrt; these are
   what wasm's f32/f64 nearest/trunc/ceil/floor/sqrt opcodes additionally need.

   The float variants go through the double ones on purpose and that is EXACT, not
   an approximation: every float is representable as a double, so the rounding
   happens once and the result converts back without loss. */
double floor(double), ceil(double), fabs(double), sqrt(double);

double
trunc(double x)
{
    return x >= 0.0 ? floor(x) : ceil(x);
}

/* Round to nearest, TIES TO EVEN -- which is what wasm f64.nearest specifies and
   what a naive floor(x + 0.5) gets wrong for exactly the values a test never
   covers. 2^52 is where a double has no fractional bits left, so anything at or
   above it (and NaN and infinity, since the comparison is false for them) is
   already integral and returned unchanged. */
double
rint(double x)
{
    if (!(fabs(x) < 4503599627370496.0))
        return x;
    double f = floor(x), d = x - f;
    if (d > 0.5)
        return f + 1.0;
    if (d < 0.5)
        return f;
    double h = f * 0.5;
    return (h == floor(h)) ? f : f + 1.0;
}

float ceilf(float x)  { return (float)ceil((double)x); }
float floorf(float x) { return (float)floor((double)x); }
float fabsf(float x)  { return (float)fabs((double)x); }
float sqrtf(float x)  { return (float)sqrt((double)x); }
float truncf(float x) { return (float)trunc((double)x); }
float rintf(float x)  { return (float)rint((double)x); }

/* snprintf/vsnprintf.
 *
 * WRITTEN HERE RATHER THAN REUSED, and the reason is structural rather than
 * preference: xlang/lua-cdp/capstone-lua/lua_libc.c already has both, but it owns
 * nineteen file-scope globals and needs its own force-included header. The
 * gp-captable ABI allows exactly ONE translation unit to own globals, so linking
 * it alongside the amalgamation trips that gate. Reuse was tried first and the ABI
 * refused it.
 *
 * Supports what WAMR actually emits, counted from its own format strings:
 * %s %d %i %u %x %X %p %c %%, the l/ll/z length modifiers, zero- and space-padded
 * widths. Everything else is copied through verbatim so a format this does not
 * know is visible as itself rather than silently dropped.
 *
 * ponytail: %f prints "<f>". Ceiling: WAMR uses it in eight places, all in
 * memory-profiling output which this build disables. Upgrade path is a
 * fixed-point printer, and it should land before any float is quoted from a log.
 */
typedef struct {
    char *buf;
    size_t cap;      /* buffer size INCLUDING the terminator */
    size_t len;      /* characters that WOULD have been written, C99 semantics */
} wfmt;

static void
wput(wfmt *w, char c)
{
    if (w->len + 1 < w->cap)
        w->buf[w->len] = c;
    w->len++;
}

static void
wpad(wfmt *w, const char *digits, int n, int width, int zero, int neg)
{
    int total = n + (neg ? 1 : 0);
    if (zero && neg)
        wput(w, '-');
    for (int i = total; i < width; i++)
        wput(w, zero ? '0' : ' ');
    if (!zero && neg)
        wput(w, '-');
    for (int i = 0; i < n; i++)
        wput(w, digits[i]);
}

static void
wnum(wfmt *w, unsigned long long v, unsigned base, int upper, int width, int zero, int neg)
{
    static const char lo[] = "0123456789abcdef", up[] = "0123456789ABCDEF";
    const char *d = upper ? up : lo;
    char tmp[24];
    int n = 0;
    if (v == 0)
        tmp[n++] = '0';
    while (v) {
        tmp[n++] = d[v % base];
        v /= base;
    }
    char rev[24];
    for (int i = 0; i < n; i++)
        rev[i] = tmp[n - 1 - i];
    wpad(w, rev, n, width, zero, neg);
}

int
vsnprintf(char *out, size_t size, const char *fmt, va_list ap)
{
    wfmt w = { out, size, 0 };
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') { wput(&w, *p); continue; }
        const char *start = p++;
        int zero = 0, width = 0, lng = 0;
        while (*p == '0') { zero = 1; p++; }
        while (*p >= '0' && *p <= '9') { width = width * 10 + (*p - '0'); p++; }
        while (*p == 'l' || *p == 'z' || *p == 'h') { if (*p == 'l' || *p == 'z') lng++; p++; }
        switch (*p) {
        case 'd': case 'i': {
            long long v = lng ? va_arg(ap, long long) : (long long)va_arg(ap, int);
            int neg = v < 0;
            unsigned long long m = neg ? (unsigned long long)(-(v + 1)) + 1ull
                                       : (unsigned long long)v;
            wnum(&w, m, 10, 0, width, zero, neg);
            break;
        }
        case 'u':
            wnum(&w, lng ? va_arg(ap, unsigned long long)
                         : (unsigned long long)va_arg(ap, unsigned), 10, 0, width, zero, 0);
            break;
        case 'x': case 'X':
            wnum(&w, lng ? va_arg(ap, unsigned long long)
                         : (unsigned long long)va_arg(ap, unsigned),
                 16, *p == 'X', width, zero, 0);
            break;
        case 'p': {
            /* The ADDRESS, which is what a diagnostic wants and all a 64-bit slot
               can hold. Deliberately not an attempt to render a capability. */
            void *v = va_arg(ap, void *);
            wput(&w, '0'); wput(&w, 'x');
            wnum(&w, (unsigned long long)(uintptr_t)v, 16, 0, 0, 0, 0);
            break;
        }
        case 'c':
            wput(&w, (char)va_arg(ap, int));
            break;
        case 's': {
            const char *v = va_arg(ap, const char *);
            if (!v) v = "(null)";
            size_t n = 0;
            while (v[n]) n++;
            wpad(&w, v, (int)n, width, 0, 0);
            break;
        }
        case 'f': case 'e': case 'g':
            (void)va_arg(ap, double);
            wput(&w, '<'); wput(&w, 'f'); wput(&w, '>');
            break;
        case '%':
            wput(&w, '%');
            break;
        default:
            /* Unknown: copy the whole directive through, so it shows up rather
               than disappearing. */
            for (const char *q = start; q <= p; q++)
                wput(&w, *q);
            break;
        }
    }
    if (w.cap)
        w.buf[w.len < w.cap ? w.len : w.cap - 1] = 0;
    return (int)w.len;
}

int
snprintf(char *out, size_t size, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(out, size, fmt, ap);
    va_end(ap);
    return n;
}

#ifdef CAPSTONE_WAMR_LIBC_SELFTEST
#include <assert.h>
static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}
int main(void)
{
    assert(labs(-5) == 5 && labs(5) == 5 && labs(0) == 0);

    int arr[] = { 1, 3, 5, 7, 9 };
    int k;
    for (k = 1; k <= 9; k += 2)
        assert(bsearch(&k, arr, 5, sizeof(int), cmp_int) != NULL);
    for (k = 0; k <= 10; k += 2)
        assert(bsearch(&k, arr, 5, sizeof(int), cmp_int) == NULL);

    /* The two cases a hand-rolled bsearch gets wrong. */
    k = 1;
    assert(bsearch(&k, arr, 0, sizeof(int), cmp_int) == NULL);   /* empty */
    assert(bsearch(&k, arr, 1, sizeof(int), cmp_int) == &arr[0]); /* single, hit */
    k = 2;
    assert(bsearch(&k, arr, 1, sizeof(int), cmp_int) == NULL);    /* single, miss */

    /* qsort, including the two sizes where an insertion sort is usually wrong. */
    int q0[] = { 0 }, q1[] = { 2, 1 }, q[] = { 5, 1, 4, 1, 9, 2, 6 };
    qsort(q0, 0, sizeof(int), cmp_int);
    qsort(q1, 2, sizeof(int), cmp_int);
    assert(q1[0] == 1 && q1[1] == 2);
    qsort(q, 7, sizeof(int), cmp_int);
    for (int i = 1; i < 7; i++)
        assert(q[i - 1] <= q[i]);
    assert(q[0] == 1 && q[1] == 1 && q[6] == 9);   /* stable over duplicates */

    /* rint's ties, which are the whole reason it is not floor(x + 0.5). */
    assert(rint(0.5) == 0.0);    /* to even, DOWN */
    assert(rint(1.5) == 2.0);    /* to even, up */
    assert(rint(2.5) == 2.0);    /* to even, DOWN -- floor(x+0.5) gives 3 */
    assert(rint(-0.5) == 0.0);
    assert(rint(-1.5) == -2.0);
    assert(rint(-2.5) == -2.0);
    assert(rint(1.4) == 1.0 && rint(1.6) == 2.0);
    assert(rint(1e300) == 1e300);            /* past 2^52, unchanged */

    assert(trunc(1.9) == 1.0 && trunc(-1.9) == -1.0);
    assert(trunc(-0.5) == 0.0);
    assert(rintf(2.5f) == 2.0f && rintf(1.5f) == 2.0f);
    assert(truncf(-1.9f) == -1.0f);

    /* snprintf. The cases that matter are the boundary ones: C99 says the return
       value is what WOULD have been written, and that truncation still
       terminates. Both are what a caller sizing a buffer relies on. */
    char b[32];
    assert(snprintf(b, sizeof b, "%d", 42) == 2 && !strcmp(b, "42"));
    assert(snprintf(b, sizeof b, "%d", -42) == 3 && !strcmp(b, "-42"));
    assert(snprintf(b, sizeof b, "%05d", -42) == 5 && !strcmp(b, "-0042"));
    assert(snprintf(b, sizeof b, "%u", 4000000000u) == 10);
    assert(snprintf(b, sizeof b, "%02x", 5) == 2 && !strcmp(b, "05"));
    assert(snprintf(b, sizeof b, "%04x", 0xabc) == 4 && !strcmp(b, "0abc"));
    assert(snprintf(b, sizeof b, "%X", 0xabc) == 3 && !strcmp(b, "ABC"));
    assert(snprintf(b, sizeof b, "%s|%s", "ab", (char *)0) == 9 && !strcmp(b, "ab|(null)"));
    assert(snprintf(b, sizeof b, "%lld", -1234567890123LL) == 14);
    assert(snprintf(b, sizeof b, "%zu", (size_t)7) == 1 && !strcmp(b, "7"));
    assert(snprintf(b, sizeof b, "100%%") == 4 && !strcmp(b, "100%"));

    /* Truncation: the count is what would have been written, the buffer is still
       terminated, and nothing is written past the end. */
    char t[4] = { 'x', 'x', 'x', 'x' };
    assert(snprintf(t, 4, "abcdefg") == 7);
    assert(!strcmp(t, "abc"));
    /* size 0 must not touch the buffer at all */
    char z = 'Z';
    assert(snprintf(&z, 0, "abc") == 3 && z == 'Z');

    return 0;
}
#endif
