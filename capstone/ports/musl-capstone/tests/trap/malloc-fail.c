/* CDERIVE failure in mallocng: build with -DCASE=1..6 against a libc compiled with
 * -DCAPSTONE_HEAP_DERIVE_INJECTION=1 (build-trap-smoke.sh, TRAP_LIBC_INJECTION_ARCHIVE).
 * fail_next() makes the next derive of malloc, aligned_alloc or realloc give the failure
 * result, an untagged 0, as the hardware does when the lifetime table is full and cannot grow.
 * Each case prints "mf<CASE> ok" and exits 0, or names the broken check and exits nonzero.
 *   1  malloc returns NULL with ENOMEM, and the next malloc succeeds.
 *   2  3000 failed small mallocs leak no slot: later objects stay within one 64 KiB span.
 *   3  64 failed 200000-byte mallocs leak no mapping: the address space does not grow.
 *   4  a failed in-place realloc of a small object returns NULL and keeps it readable.
 *   5  a failed in-place realloc of a mapped object returns NULL and keeps it readable.
 *   6  a failed growing realloc that needs a new mapping returns NULL and keeps the object.
 * A libc that ends the old object first faults in cases 4 to 6 when the object is read. */
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern unsigned long __heap_derive_calls, __heap_derive_fail_at;

static void fail_next(void)
{
    __heap_derive_fail_at = __heap_derive_calls + 1;
}

static void say(const char *text, long n)
{
    char digits[24];
    int at = sizeof digits;

    digits[--at] = '\n';
    do {
        digits[--at] = '0' + n % 10;
        n /= 10;
    } while (n);
    write(1, text, strlen(text));
    write(1, digits + at, sizeof digits - at);
}

static int fail(const char *what, long n)
{
    say(what, n);
    return 1;
}

static long address_space_pages(void)
{
    char buf[64];
    long pages = 0;
    int fd = open("/proc/self/statm", O_RDONLY);

    if (fd < 0) return -1;
    long got = read(fd, buf, sizeof buf - 1);
    close(fd);
    for (long i = 0; i < got && buf[i] >= '0' && buf[i] <= '9'; ++i)
        pages = pages * 10 + (buf[i] - '0');
    return pages;
}

static void fill(unsigned char *p, size_t n)
{
    for (size_t i = 0; i < n; i += 97) p[i] = (unsigned char)(i * 7 + 1);
    p[n - 1] = 0x5a;
}

static int intact(const unsigned char *p, size_t n)
{
    for (size_t i = 0; i < n; i += 97)
        if (p[i] != (unsigned char)(i * 7 + 1)) return 0;
    return p[n - 1] == 0x5a;
}

static int failed_null(void *r)
{
    if (r) return fail("FAIL: answer not NULL, errno ", errno);
    if (errno != ENOMEM) return fail("FAIL: NULL without ENOMEM, errno ", errno);
    return 0;
}

/* A failed realloc: NULL, then the old object still reads back, then ENOMEM. The read comes
 * before the errno check so that a libc which ended the old object faults here. */
static int failed_realloc(void *r, const unsigned char *old, size_t n)
{
    int e = errno;
    if (r) return fail("FAIL: answer not NULL, errno ", e);
    if (!intact(old, n)) return fail("FAIL: contents lost ", 0);
    if (e != ENOMEM) return fail("FAIL: NULL without ENOMEM, errno ", e);
    return 0;
}

#if CASE == 1
static int run(void)
{
    errno = 0;
    fail_next();
    if (failed_null(malloc(32))) return 1;
    char *p = malloc(32);
    if (!p) return fail("FAIL: malloc after the failure, errno ", errno);
    p[31] = 1;
    free(p);
    return 0;
}
#elif CASE == 2
static int run(void)
{
    unsigned long low = ~0UL, high = 0;

    for (int i = 0; i < 3000; ++i) {
        fail_next();
        if (malloc(48)) return fail("FAIL: answer not NULL ", 0);
        char *p = malloc(48);
        if (!p) return fail("FAIL: malloc after the failure, errno ", errno);
        unsigned long a = (unsigned long)__builtin_capstone_cap_get_cursor(p);
        if (a < low) low = a;
        if (a > high) high = a;
        free(p);
    }
    if (high - low > 65536) return fail("FAIL: slots leak, span ", (long)(high - low));
    return 0;
}
#elif CASE == 3
static int run(void)
{
    char *warm = malloc(200000);
    if (!warm) return fail("FAIL: warm-up malloc, errno ", errno);
    free(warm);
    long before = address_space_pages();
    for (int i = 0; i < 64; ++i) {
        fail_next();
        if (malloc(200000)) return fail("FAIL: answer not NULL ", 0);
    }
    long grown = address_space_pages() - before;
    if (before < 0 || grown > 64) return fail("FAIL: mappings leak, pages ", grown);
    return 0;
}
#elif CASE == 4
static int run(void)
{
    unsigned char *p = malloc(100);
    if (!p) return fail("FAIL: malloc, errno ", errno);
    fill(p, 90);
    unsigned char *q = realloc(p, 90);
    if (!q || (unsigned long)q != (unsigned long)p) return fail("FAIL: realloc not in place ", 0);
    errno = 0;
    fail_next();
    if (failed_realloc(realloc(q, 80), q, 90)) return 1;
    free(q);
    return 0;
}
#elif CASE == 5
static int run(void)
{
    /* 50 pages less 512 bytes: the object starts 256 bytes into its mapping wherever mallocng's
     * offset cycling puts the raw slot, so a realloc to the same size keeps the mapping. */
    size_t n = 50 * 4096 - 512;
    unsigned char *p = malloc(n);
    if (!p) return fail("FAIL: malloc, errno ", errno);
    fill(p, n);
    unsigned char *q = realloc(p, n);
    if (!q || (unsigned long)q != (unsigned long)p) return fail("FAIL: realloc not in place ", 0);
    errno = 0;
    fail_next();
    if (failed_realloc(realloc(q, n), q, n)) return 1;
    free(q);
    return 0;
}
#elif CASE == 6
static int run(void)
{
    unsigned char *p = malloc(200000);
    if (!p) return fail("FAIL: malloc, errno ", errno);
    fill(p, 200000);
    errno = 0;
    fail_next();
    if (failed_realloc(realloc(p, 250000), p, 200000)) return 1;
    unsigned char *q = realloc(p, 250000);
    if (!q || !intact(q, 200000)) return fail("FAIL: realloc after the failure, errno ", errno);
    free(q);
    return 0;
}
#else
#error "CASE must be 1 to 6"
#endif

int main(void)
{
    if (run()) return 1;
    say("mf ok, case ", CASE);
    return 0;
}
