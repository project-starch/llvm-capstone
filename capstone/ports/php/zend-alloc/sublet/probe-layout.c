/* Does the nested layer give PHP the stock block layout and stock realloc, and does it still
 * revoke? The unit test of docs/plans/2026-11-15-php-port-layout-fix.md.
 *
 * It links the REAL zend-cap-mode.c, with malloc, realloc and free wrapped at link time
 * (-Wl,--wrap=...), so it sees exactly what the layer asks the heap for. test-probe-layout.sh
 * builds it twice, against the current zend-cap-mode.c and against the file at 5060011844e, and
 * runs both in one boot: the old build is the negative control, and must fail the two layout
 * stages while it still faults on the stale accesses.
 *
 * ONE STAGE PER RUN, selected by argv, because the fault stages end the process. Each prints its
 * READY marker before the access under test.
 *
 *   geometry      every REAL_SIZE from 0 to 16,376 and a list of larger sizes: the request malloc
 *                 receives equals PHP's (48 + REAL_SIZE(n)) whenever the child bound is below
 *                 16 KiB and n > 0; the payload starts 48 bytes into the block; the whole request
 *                 is writable through the child                                 GEOMETRY_PASS
 *   realloc       a 100-byte object grown 8 bytes at a time to 3,000, as strcat grows a string:
 *                 each step calls realloc once with PHP's request, takes no fresh block, and keeps
 *                 the address whenever realloc keeps it; the contents survive    REALLOC_PASS
 *   stale-realloc an in-place realloc, then a read through the old pointer       FAULT cause 25
 *   stale-free    efree, then a read through the old pointer                     FAULT cause 25
 *   bound         a read one past the REAL_SIZE bound                            FAULT cause 28
 *   big           20,000 bytes grown to 40,000, 70,000 and 140,000 and shrunk back to 20,000,
 *                 across every slack change, with the contents checked each time  BIG_PASS
 *   stale-big     the growth to 70,000 (a slack change), then a read through the old pointer
 *                                                                                FAULT cause 25 */
#include "zend-cap-mode.h"
#include <heap_sublet.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HDR 48UL
#define REAL_SIZE(n) (((n) + 7UL) & ~7UL)
#define BLOCK_FOR(n) (HDR + REAL_SIZE(n))
#define MODE (ZEND_CAP_MODE_REVOKE | ZEND_CAP_MODE_RETURN)   /* the protected arm, PHP_CAP_MODE=6 */

/* ---- the heap calls the layer makes ---------------------------------------------------------- */

void *__real_malloc(size_t n);
void *__real_realloc(void *p, size_t n);
void __real_free(void *p);

static int in_heap;   /* mallocng's realloc calls malloc and free itself; those are not the layer's */
static unsigned long n_malloc, n_realloc, n_free;
static size_t last_request;
static unsigned long last_realloc_from, last_return;

void *__wrap_malloc(size_t n)
{
    void *r;
    if (in_heap) { return __real_malloc(n); }
    ++n_malloc;
    last_request = n;
    in_heap = 1;
    r = __real_malloc(n);
    in_heap = 0;
    last_return = __builtin_capstone_cap_get_cursor(r);
    return r;
}

void *__wrap_realloc(void *p, size_t n)
{
    void *r;
    if (in_heap) { return __real_realloc(p, n); }
    ++n_realloc;
    last_request = n;
    last_realloc_from = __builtin_capstone_cap_get_cursor(p);
    in_heap = 1;
    r = __real_realloc(p, n);
    in_heap = 0;
    last_return = __builtin_capstone_cap_get_cursor(r);
    return r;
}

void __wrap_free(void *p)
{
    if (in_heap) { __real_free(p); return; }
    ++n_free;
    in_heap = 1;
    __real_free(p);
    in_heap = 0;
}

/* ---- PHP's calls, in the shape the patched zend_alloc.c makes them -------------------------- */

static unsigned long cur(void *c) { return __builtin_capstone_cap_get_cursor(c); }

static void *emalloc_like(unsigned long n)
{
    void *block = zend_cap_block((size_t)BLOCK_FOR(n));
    if (!block) { printf("PROBE FAIL: block returned NULL for %lu\n", n); exit(2); }
    return zend_cap_issue(block, HDR, (size_t)n);
}

static void efree_like(void *ptr)
{
    zend_cap_release(zend_cap_reclaim(ptr, HDR));
}

static void *erealloc_like(void *ptr, unsigned long oldn, unsigned long n)
{
    void *block = zend_cap_header(ptr, HDR);
    void *grown = zend_cap_reblock(block, (size_t)BLOCK_FOR(n), HDR + oldn);
    if (!grown) { printf("PROBE FAIL: reblock returned NULL for %lu\n", n); exit(2); }
    return zend_cap_issue(grown, HDR, (size_t)n);
}

static void fill(void *p, unsigned long n, unsigned long seed)
{
    unsigned long i;
    for (i = 0; i < n; i++) { ((unsigned char *)p)[i] = (unsigned char)((i * 7 + seed) % 251); }
}

static int same(void *p, unsigned long n, unsigned long seed)
{
    unsigned long i;
    for (i = 0; i < n; i++) {
        if (((unsigned char *)p)[i] != (unsigned char)((i * 7 + seed) % 251)) { return 0; }
    }
    return 1;
}

static void ready(const char *what) { printf("%s\n", what); fflush(stdout); }

/* ---- the stages ------------------------------------------------------------------------------ */

static int geometry(void)
{
    static const unsigned long big[] = { 16376, 16384, 16400, 20000, 32760, 32768, 40000, 65488,
                                         65536, 70000, 131072, 140000, 300000 };
    unsigned long n, small = 0, equal = 0, differ = 0, first = 0, zero_request = 0;
    unsigned int i;
    for (n = 0; n <= 16376; n += 8) {
        void *pay = emalloc_like(n);
        size_t want = BLOCK_FOR(n);
        unsigned long slack = cur(pay) - HDR - last_return;
        fill(pay, n, n);
        if (n == 0) {
            zero_request = last_request;
        } else if (heap_round(REAL_SIZE(n)) < 16384UL) {
            ++small;
            if (last_request == want && slack == 0) { ++equal; }
            else if (!differ++) { first = n; }
        }
        efree_like(pay);
    }
    printf("GEOMETRY small=%lu equal=%lu differ=%lu zero_request=%lu\n", small, equal, differ,
           zero_request);
    for (i = 0; i < sizeof big / sizeof big[0]; i++) {
        void *pay;
        n = big[i];
        pay = emalloc_like(n);
        fill(pay, n, n);
        printf("GEOMETRY big n=%lu php_request=%lu request=%lu slack=%lu payload_aligned=%d\n",
               n, (unsigned long)BLOCK_FOR(n), (unsigned long)last_request,
               cur(pay) - HDR - last_return,
               (int)(cur(pay) % heap_granule(heap_round(REAL_SIZE(n))) == 0));
        efree_like(pay);
    }
    if (differ) {
        printf("GEOMETRY_FAIL first n=%lu: the layer did not pass PHP's request to malloc\n", first);
        return 1;
    }
    ready("GEOMETRY_PASS");
    return 0;
}

static int grow(void)
{
    unsigned long n, steps = 0, stock_request = 0, inplace = 0, moved = 0, kept = 0;
    unsigned long m0, r0, f0;
    void *pay = emalloc_like(100);
    fill(pay, 100, 3);
    m0 = n_malloc; r0 = n_realloc; f0 = n_free;
    for (n = 100; n < 3000; n += 8) {
        unsigned long before = cur(pay), calls = n_realloc;
        pay = erealloc_like(pay, n, n + 8);
        ++steps;
        if (n_realloc == calls + 1 && last_request == BLOCK_FOR(n + 8) &&
            last_realloc_from == before - HDR && last_return == cur(pay) - HDR) {
            ++stock_request;
        }
        if (cur(pay) == before) { ++inplace; } else { ++moved; }
        if (same(pay, 100, 3)) { ++kept; }
    }
    printf("REALLOC steps=%lu realloc_calls=%lu stock_request=%lu inplace=%lu moved=%lu "
           "malloc_calls=%lu free_calls=%lu contents_kept=%lu\n", steps, n_realloc - r0,
           stock_request, inplace, moved, n_malloc - m0, n_free - f0, kept);
    efree_like(pay);
    if (stock_request != steps || n_malloc != m0 || inplace == 0 || kept != steps) {
        ready("REALLOC_FAIL: erealloc did not realloc the block with PHP's request");
        return 1;
    }
    ready("REALLOC_PASS");
    return 0;
}

static int big_blocks(int stale)
{
    static const unsigned long sizes[] = { 40000, 70000, 140000, 20000 };
    unsigned long n = 20000;
    unsigned int i;
    void *pay = emalloc_like(n), *old;
    fill(pay, n, 11);
    printf("BIG n=%lu slack=%lu\n", n, cur(pay) - HDR - last_return);
    for (i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        unsigned long next = sizes[i], keep = n < next ? n : next;
        old = pay;
        pay = erealloc_like(pay, n, next);
        printf("BIG n=%lu slack=%lu inplace=%d\n", next, cur(pay) - HDR - last_return,
               (int)(cur(pay) == cur(old)));
        if (!same(pay, keep, 11)) { printf("BIG_FAIL: contents lost at %lu\n", next); return 1; }
        if (stale && next == 70000) {
            ready("STALE_BIG_READY reading the pointer from before the slack change");
            return ((volatile unsigned char *)old)[0];
        }
        fill(pay, next, 11);
        n = next;
    }
    efree_like(pay);
    ready("BIG_PASS");
    return 0;
}

int main(int argc, char **argv)
{
    const char *stage = argc > 1 ? argv[1] : "";
    void *pay, *old;
    zend_cap_set_mode(MODE);
    printf("PROBE-LAYOUT stage=%s\n", stage);
    fflush(stdout);

    if (!strcmp(stage, "geometry")) { return geometry(); }
    if (!strcmp(stage, "realloc")) { return grow(); }
    if (!strcmp(stage, "big")) { return big_blocks(0); }
    if (!strcmp(stage, "stale-big")) { big_blocks(1); return 1; }
    if (!strcmp(stage, "stale-realloc")) {
        pay = emalloc_like(100);
        fill(pay, 100, 5);
        old = pay;
        pay = erealloc_like(pay, 100, 104);
        printf("STALE_REALLOC_READY inplace=%d\n", (int)(cur(pay) == cur(old)));
        fflush(stdout);
        return ((volatile unsigned char *)old)[0];
    }
    if (!strcmp(stage, "stale-free")) {
        pay = emalloc_like(64);
        fill(pay, 64, 5);
        efree_like(pay);
        ready("STALE_FREE_READY");
        return ((volatile unsigned char *)pay)[0];
    }
    if (!strcmp(stage, "bound")) {
        pay = emalloc_like(100);
        fill(pay, 100, 5);
        ((volatile unsigned char *)pay)[103] = 1;   /* inside REAL_SIZE(100) = 104 */
        ready("BOUND_READY reading index 104 of a 104-byte bound");
        return ((volatile unsigned char *)pay)[104];
    }
    printf("usage: probe-layout geometry|realloc|stale-realloc|stale-free|bound|big|stale-big\n");
    return 2;
}
