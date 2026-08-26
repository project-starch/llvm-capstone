/* The libc gap the census named, and nothing more.
 *
 * Deliberately small. snprintf/vsnprintf come from
 * xlang/lua-cdp/capstone-lua/lua_libc.c and the string routines from
 * beebs_freestanding_string.c; duplicating either would collide at link, which is
 * the mistake lua_libc.c's own header warns about.
 *
 * Self-check: build native and run the asserts --
 *   cc -DCAPSTONE_WAMR_LIBC_SELFTEST -O2 capstone_libc_extra.c -o /tmp/t && /tmp/t
 */
#ifdef CAPSTONE_WAMR_LIBC_SELFTEST
/* No <stdlib.h> here on purpose: it declares labs/abs/bsearch itself, and on a
   current glibc `abs` is a _Generic macro, so including it turns the definitions
   below into syntax errors rather than redefinitions. */
#include <stddef.h>
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

int
abs(int v)
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
    assert(abs(-5) == 5);

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

    return 0;
}
#endif
