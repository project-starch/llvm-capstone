/* mc-test.c: unit test of memcached's slab and cache lifetime patch (0006), linked with the
 * protected server's objects (main renamed away) and slabs.c and cache.c rebuilt with inject.h.
 *   mc-test stale     a slab chunk read after slabs_free: must fault (cause 25)
 *   mc-test alloc     a failed chunk CDERIVE: NULL, the chunk stays filed (the next request gets it)
 *   mc-test newslab   a failed generation CDERIVE for a new page: NULL, the next request succeeds
 *   mc-test big       chunks of the classes from 4 KiB to 512 KiB: allocated, written, freed
 *   mc-test cache     a failed CRENEW in cache_free: the object goes back to malloc, whose free
 *                     ends its lifetime, so a read through the caller's pointer faults (25) */
#include "memcached.h"
#include "slabs.h"
#include "cache.h"
#include <stdio.h>
#include <string.h>

long lifec_fail_left, lifec_fail_size, lifec_failed;
unsigned long lifec_failed_at;

static void setup(void)
{
    settings.item_size_max = 1024 * 1024;
    settings.slab_page_size = 1024 * 1024;
    settings.slab_chunk_size_max = settings.slab_page_size / 2;
    settings.slab_reassign = true;
    settings.factor = 1.25;
    settings.chunk_size = 48;
    slabs_init(64 * 1024 * 1024, 1.25, false, NULL, NULL, false);
}

static int stale(void)
{
    unsigned int id = slabs_clsid(100);
    volatile char *p = slabs_alloc(id, 0);
    p[sizeof(item)] = 7;
    slabs_free((void *)p, id);
    printf("MCTEST stale: read after free\n"); fflush(stdout);
    printf("MCTEST stale: read %d, no fault\n", p[sizeof(item)]);
    return 1;
}

static int alloc(void)
{
    unsigned int id = slabs_clsid(100);
    void *holder = slabs_alloc(id, 0);
    lifec_fail_size = 0; lifec_fail_left = 1;
    void *r = slabs_alloc(id, 0);
    lifec_fail_left = 0;
    void *q = slabs_alloc(id, 0);
    unsigned long qa = __builtin_capstone_cap_get_cursor(q);
    printf("MCTEST alloc: failed=%ld result=%s next_chunk_is_the_failed_one=%s\n", lifec_failed,
           r ? "pointer" : "NULL", qa == lifec_failed_at ? "yes" : "no");
    (void)holder;
    return lifec_failed == 1 && !r && qa == lifec_failed_at ? 0 : 1;
}

static int newslab(void)
{
    unsigned int id = slabs_clsid(3000);
    lifec_fail_size = 0; lifec_fail_left = 1;
    void *r = slabs_alloc(id, 0);
    lifec_fail_left = 0;
    printf("MCTEST newslab: failed=%ld result=%s\n", lifec_failed, r ? "pointer" : "NULL"); fflush(stdout);
    void *q = slabs_alloc(id, 0);
    printf("MCTEST newslab: retry=%s\n", q ? "pointer" : "NULL"); fflush(stdout);
    if (q) memset((char *)q + sizeof(item), 0x5a, 64);
    return lifec_failed == 1 && !r && q ? 0 : 1;
}

static int big(void)
{
    static const unsigned sizes[] = {5000, 9000, 20000, 70000, 300000, 500000};
    for (unsigned i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        unsigned int id = slabs_clsid(sizes[i]);
        char *p = slabs_alloc(id, 0);
        if (!p) { printf("MCTEST big: %u failed\n", sizes[i]); return 1; }
        memset(p + sizeof(item), 0x5a, sizes[i] - sizeof(item));
        slabs_free(p, id);
        printf("MCTEST big: class %u for %u bytes ok\n", id, sizes[i]);
    }
    return 0;
}

static int cache(void)
{
    cache_t *c = cache_create("t", 64, sizeof(char *));
    void *a = cache_alloc(c);
    cache_free(c, a);                      /* renewed and filed */
    volatile char *b = cache_alloc(c);
    b[0] = 7;
    int total = c->total;
    lifec_fail_size = 0; lifec_fail_left = 1;
    cache_free(c, (void *)b);
    lifec_fail_left = 0;
    printf("MCTEST cache: failed=%ld total %d -> %d freecurr=%d\n", lifec_failed, total, c->total,
           c->freecurr);
    fflush(stdout);
    printf("MCTEST cache: read %d through the freed object, no fault\n", b[0]);
    return 1;
}

int main(int argc, char **argv)
{
    setup();
    const char *m = argc > 1 ? argv[1] : "";
    int rc = !strcmp(m, "stale") ? stale() : !strcmp(m, "alloc") ? alloc()
           : !strcmp(m, "newslab") ? newslab() : !strcmp(m, "big") ? big()
           : !strcmp(m, "cache") ? cache() : 2;
    printf("MCTEST %s: rc=%d\n", m, rc); fflush(stdout);
    return rc;
}
