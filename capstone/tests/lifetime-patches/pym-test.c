/* pym-test.c: unit test of the standalone pymalloc port's lifetime patch (0003), with inject.h
 * forced into obmalloc.c.
 *   pym-test stale    a block read after free: must fault (cause 25)
 *   pym-test alloc    a failed CDERIVE: the block goes back to its pool (the next request gets it)
 *   pym-test realloc  a failed CDERIVE in an in-place realloc: NULL, old block intact
 *   pym-test arena    a block of an arena that was given back and handed out again: must fault (25)
 *   pym-test newarena a failed CDERIVE for a new arena: the arena goes back, the request is served by
 *                     the raw fallback (as when no arena can be had), and a retry gets a new arena */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "port.h"

long lifec_fail_left, lifec_fail_size, lifec_failed;
unsigned long lifec_failed_at;
void *pym_realloc(void *, size_t);
void pym_free(void *);

_Noreturn void pym_fail(unsigned code) { printf("PYMTEST pym_fail %u\n", code); fflush(stdout); exit(3); }

static int stale(void)
{
    volatile char *p = pym_malloc(64);
    void *keep = pym_malloc(64);   /* keeps the arena */
    p[0] = 7;
    pym_free((void *)p);
    printf("PYMTEST stale: read after free\n"); fflush(stdout);
    printf("PYMTEST stale: read %d, no fault\n", p[0]);
    (void)keep;
    return 1;
}

static int alloc(void)
{
    void *holder = pym_malloc(488);
    lifec_fail_size = 488; lifec_fail_left = 1;
    void *p = pym_malloc(488);
    lifec_fail_left = 0;
    void *q = pym_malloc(488);
    unsigned long qa = __builtin_capstone_cap_get_cursor(q);
    printf("PYMTEST alloc: failed=%ld served=%s next_block_is_the_failed_one=%s\n", lifec_failed,
           p ? "yes" : "no", qa == lifec_failed_at ? "yes" : "no");
    if (p) pym_free(p);
    pym_free(q); pym_free(holder);
    return lifec_failed == 1 && qa == lifec_failed_at ? 0 : 1;
}

static int realloc_inplace(void)
{
    unsigned char *p = pym_malloc(400);
    memset(p, 0x5a, 400);
    lifec_fail_size = 390; lifec_fail_left = 1;
    void *q = pym_realloc(p, 390);
    lifec_fail_left = 0;
    printf("PYMTEST realloc: failed=%ld result=%s\n", lifec_failed, q ? "pointer" : "NULL"); fflush(stdout);
    if (q) return 1;
    for (int i = 0; i < 400; i++) if (p[i] != 0x5a) return 1;
    pym_free(p);
    printf("PYMTEST realloc: old block intact and freed\n");
    return 0;
}

static int arena(void)
{
    volatile char *p = pym_malloc(64);
    p[0] = 7;
    void *alias = (void *)p;
    pym_free((void *)p);                 /* the arena is empty: given back */
    void *again = pym_malloc(64);        /* a new arena over the same memory */
    printf("PYMTEST arena: same address=%s\n",
           __builtin_capstone_cap_get_cursor(again) == __builtin_capstone_cap_get_cursor(alias) ? "yes" : "no");
    fflush(stdout);
    printf("PYMTEST arena: read %d through the old block, no fault\n", ((volatile char *)alias)[0]);
    return 1;
}

static int newarena(void)
{
    lifec_fail_size = 1048576; lifec_fail_left = 1;
    void *p = pym_malloc(64);
    lifec_fail_left = 0;
    void *q = pym_malloc(64);
    printf("PYMTEST newarena: failed=%ld first=%s retry=%s\n", lifec_failed, p ? "pointer" : "NULL",
           q ? "pointer" : "NULL");
    return lifec_failed == 1 && p && q ? 0 : 1;
}

int main(int argc, char **argv)
{
    pym_backing_init(malloc(PYM_META_BYTES), malloc(PYM_ARENA_BYTES));
    pym_allocator_init();
    const char *m = argc > 1 ? argv[1] : "";
    int rc = !strcmp(m, "stale") ? stale() : !strcmp(m, "alloc") ? alloc()
           : !strcmp(m, "realloc") ? realloc_inplace() : !strcmp(m, "arena") ? arena()
           : !strcmp(m, "newarena") ? newarena() : 2;
    printf("PYMTEST %s: rc=%d\n", m, rc); fflush(stdout);
    return rc;
}
