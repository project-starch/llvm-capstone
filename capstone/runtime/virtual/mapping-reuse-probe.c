/* Qualification probe for CAPSTONE_HEAP_MAPPING_REUSE (malloc-mapping.c). One mode per run; every mode prints one
 * result line whose fields the runner grades, and the runner builds each mode against several SDKs: the switch off,
 * the switch on, and the negative controls (CAPSTONE_HEAP_MAPPING_REUSE_TEST_*), each of which must fail the check
 * it names.
 *
 *   control   allocate, free and reallocate two size classes; prints the counters (UAF-COUNTERS)
 *   stale200, stale16
 *             read a stale pointer into a reused 200-byte or 16-byte class: must fault (cause 25). Build against an
 *             SDK with object lifetimes OFF to isolate the mapping's own revocation: then only the reuse path's
 *             CREVOKE, or the kernel's retirement of the mapping, can make it fault. NO_REVOKE must RETURN.
 *   reuse     a group mapping filled with capabilities is freed and the next group of its size takes it: the cache
 *             must have been hit (hits rise), and the new group reads zero bytes and no tag. NO_ZERO must fail.
 *   alias     a capability to a freed group is kept inside a live object of another size class, and a sibling
 *             group of the same size stays live: after reuse the alias faults and the sibling is intact
 *   protect   a group with a PROT_NONE interior page is freed; the next group of its size is written whole: it
 *             must not be that mapping (no fault). KEEP_PROTECTED must fault (cause 15).
 *   saturate  more live group mappings than the tracking table holds, then free them all and reallocate: must
 *             end, with untracked > 0. Against TEST_LIVE=64 it frees with every slot occupied (review P1).
 *   realloc   grow and shrink a mapped block through realloc, contents kept, a failing realloc keeps the block;
 *             then a block of a released size is zero (realloc's unmap goes through the cache too)
 *   pressure  fill the registered space while the cache holds mappings of other sizes: a new size must still be
 *             allocated (the cache is given back: evicted > 0 with the switch on) */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
extern unsigned long __heap_reuse_hits, __heap_reuse_puts, __heap_reuse_spills, __heap_reuse_untracked,
                     __heap_reuse_protected, __heap_reuse_evicted;
__attribute__((weak)) unsigned long __heap_reuse_hits, __heap_reuse_puts, __heap_reuse_spills,
                     __heap_reuse_untracked, __heap_reuse_protected, __heap_reuse_evicted;
extern unsigned long __heap_region_maps, __heap_region_pieces, __heap_piece_drops;   /* CAPSTONE_HEAP_MAPPING_REGION */
__attribute__((weak)) unsigned long __heap_region_maps, __heap_region_pieces, __heap_piece_drops;
static void *(*volatile p_malloc)(size_t) = malloc;   /* no folding of malloc/free pairs */
static void *(*volatile p_realloc)(void *, size_t) = realloc;
static void (*volatile p_free)(void *) = free;
static unsigned long cursor(void *p) { return __builtin_capstone_cap_get_cursor(p); }
static unsigned long tag(void *p) { return __builtin_capstone_cap_get_tag(p); }
static void counters(const char *mode)
{
    printf("UAF-COUNTERS %s hits=%lu puts=%lu spills=%lu untracked=%lu protected=%lu evicted=%lu regions=%lu pieces=%lu"
           " drops=%lu\n", mode, __heap_reuse_hits, __heap_reuse_puts, __heap_reuse_spills, __heap_reuse_untracked,
           __heap_reuse_protected, __heap_reuse_evicted, __heap_region_maps, __heap_region_pieces, __heap_piece_drops);
    fflush(stdout);
}
static int fail(const char *mode, const char *what)
{
    counters(mode); printf("UAF-FAIL %s %s\n", mode, what); fflush(stdout); return 1;
}
#define G 20000UL   /* above half a page: a group of one slot that owns its mapping (as mapcache-contract.c) */

#define N 4096
static char *a[N], *b[N], *s[N], *t[N];
static int mode_stale(const char *mode)
{
    for (int i = 0; i < N; i++) { a[i] = p_malloc(200); s[i] = p_malloc(16); if (!a[i] || !s[i]) { puts("UAF-ENOMEM1"); return 2; }
                                  memset(a[i], 0x5a, 200); memset(s[i], 0x5b, 16); }
    char *stale200 = a[N / 2], *stale16 = s[N / 2];
    for (int i = 0; i < N; i++) { p_free(a[i]); p_free(s[i]); }
    for (int i = 0; i < N; i++) { b[i] = p_malloc(200); t[i] = p_malloc(16); if (!b[i] || !t[i]) { puts("UAF-ENOMEM2"); return 2; }
                                  memset(b[i], 0x33, 200); memset(t[i], 0x34, 16); }
    int same200 = 0, same16 = 0;
    for (int i = 0; i < N; i++) { same200 += (b[i] == stale200); same16 += (t[i] == stale16); }
    counters(mode);
    printf("UAF-SAME same200=%d same16=%d\n", same200, same16); fflush(stdout);
    if (!strcmp(mode, "stale200") || !strcmp(mode, "stale16")) {
        char *p = !strcmp(mode, "stale200") ? stale200 : stale16;
        printf("UAF-STALE-READ-BEGIN %s\n", mode); fflush(stdout);
        volatile char c = p[0];
        printf("UAF-STALE-READ-RETURNED %s 0x%02x\n", mode, (unsigned char)c); fflush(stdout);
        return 3;
    }
    puts("UAF-END"); return 0;
}

static int mode_reuse(void)
{
    size_t count = G / sizeof(void *);
    unsigned long h0 = __heap_reuse_hits;
    void **p = p_malloc(G);
    if (!p) return fail("reuse", "first block");
    for (size_t i = 8; i < count; ++i) p[i] = &p[i];
    if (!tag(p[100])) return fail("reuse", "the pointers carry no tag");
    unsigned long address = cursor(p);
    p_free(p);
    void **q = p_malloc(G);
    if (!q) return fail("reuse", "second block");
    unsigned char *bytes = (unsigned char *)q;
    for (size_t i = 128; i < G; ++i) if (bytes[i]) return fail("reuse", "data of the previous block is visible");
    for (size_t i = 8; i < count; ++i) if (tag(q[i]) || cursor(q[i])) return fail("reuse", "a capability of the previous block is visible");
    for (size_t i = 8; i < count; ++i) q[i] = &q[i];
    if (q[200] != &q[200] || !tag(q[200])) return fail("reuse", "the second block is not usable");
    counters("reuse");
    printf("UAF-REUSE-OK same_address=%d hit=%lu\n", cursor(q) == address, __heap_reuse_hits - h0); fflush(stdout);
    p_free(q);
    return 0;
}

static char *volatile alias_global;
static int mode_alias(void)
{
    char *x = p_malloc(G), *sib = p_malloc(G);
    char **box = p_malloc(64);                       /* another size class, another mapping */
    if (!x || !sib || !box) return fail("alias", "blocks");
    memset(x, 0x5c, G); memset(sib, 0x6d, G);
    box[0] = x + 4096; alias_global = x + 8192;      /* derived capabilities, kept in a live object and a global */
    p_free(x);
    char *y = p_malloc(G);                           /* takes x's mapping when the cache is on */
    if (!y) return fail("alias", "replacement");
    memset(y, 0x3d, G);
    for (size_t i = 0; i < G; ++i) if ((unsigned char)sib[i] != 0x6d) return fail("alias", "the sibling group changed");
    counters("alias");
    printf("UAF-ALIAS-READ-BEGIN same_address=%d\n", cursor(y) == cursor(alias_global) - 8192); fflush(stdout);
    volatile char c = box[0][0];
    printf("UAF-ALIAS-READ-RETURNED 0x%02x\n", (unsigned char)c); fflush(stdout);
    return 3;
}

static int mode_protect(void)
{
    char *p = p_malloc(G);
    if (!p) return fail("protect", "block");
    memset(p, 0x11, G);
    unsigned long hole = (cursor(p) + 4095) & ~4095UL;
    if (hole + 4096 > cursor(p) + G) return fail("protect", "no whole interior page");
    if (mprotect(p + (hole - cursor(p)), 4096, PROT_NONE)) return fail("protect", "mprotect");
    unsigned long address = cursor(p);
    p_free(p);
    char *q = p_malloc(G);
    if (!q) return fail("protect", "second block");
    printf("UAF-PROTECT-WRITE-BEGIN same_address=%d\n", cursor(q) == address); fflush(stdout);
    memset(q, 0x22, G);
    counters("protect");
    puts("UAF-PROTECT-OK"); fflush(stdout);
    p_free(q);
    return 0;
}

#define SAT 160
#define SATB 140000UL   /* above mallocng's mmap threshold: one mapping per block */
static void *sat[SAT];
static int mode_saturate(void)
{
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < SAT; ++i) { sat[i] = p_malloc(SATB); if (!sat[i]) return fail("saturate", "allocation"); memset(sat[i], i, 64); }
        for (int i = 0; i < SAT; ++i) p_free(sat[i]);
    }
    counters("saturate");
    puts("UAF-SATURATE-OK"); fflush(stdout);
    return 0;
}

static int mode_realloc(void)
{
    size_t n1 = 300000, n2 = 700000;
    unsigned char *p = p_malloc(n1);
    if (!p) return fail("realloc", "block");
    for (size_t i = 0; i < n1; ++i) p[i] = (unsigned char)(i * 7);
    unsigned char *q = p_realloc(p, n2);
    if (!q) return fail("realloc", "grow");
    for (size_t i = 0; i < n1; ++i) if (q[i] != (unsigned char)(i * 7)) return fail("realloc", "grow lost data");
    memset(q + n1, 0x44, n2 - n1);
    unsigned char *r = p_realloc(q, n1);
    if (!r) return fail("realloc", "shrink");
    for (size_t i = 0; i < n1; ++i) if (r[i] != (unsigned char)(i * 7)) return fail("realloc", "shrink lost data");
    void *big = p_realloc(r, 1UL << 40);             /* must fail and keep r */
    if (big) return fail("realloc", "a 1 TiB realloc succeeded");
    for (size_t i = 0; i < n1; ++i) if (r[i] != (unsigned char)(i * 7)) return fail("realloc", "failed realloc lost data");
    p_free(r);
    unsigned char *z = p_malloc(n2);                 /* a block of a size realloc released */
    if (!z) return fail("realloc", "block after realloc");
    for (size_t i = 0; i < n2; ++i) if (z[i]) return fail("realloc", "a released block's data is visible");
    counters("realloc");
    puts("UAF-REALLOC-OK"); fflush(stdout);
    p_free(z);
    return 0;
}

/* Fill the heap's space with blocks of one size while the cache holds 60 MiB of another size. With the switch on,
 * the first failure gives the cache back and the allocation is retried; the runner compares the count with the
 * switch off. Nothing is written, so nothing is pinned. */
#define PA (1UL << 20)
#define PBIG (16UL << 20)
#define PMAX 4096
static void *pbig[PMAX];
static int mode_pressure(void)
{
    enum { CACHED = 60 };
    void *held[CACHED];
    for (unsigned i = 0; i < CACHED; ++i) { held[i] = p_malloc(PA); if (!held[i]) return fail("pressure", "cached block"); }
    for (unsigned i = 0; i < CACHED; ++i) p_free(held[i]);
    unsigned n = 0;
    while (n < PMAX && (pbig[n] = p_malloc(PBIG))) ++n;
    counters("pressure");
    printf("UAF-PRESSURE blocks=%u of %lu MiB, stopped=%s\n", n, PBIG >> 20, n < PMAX ? "enomem" : "cap"); fflush(stdout);
    for (unsigned i = 0; i < n; ++i) p_free(pbig[i]);
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "control";
    if (!strcmp(mode, "reuse")) return mode_reuse();
    if (!strcmp(mode, "alias")) return mode_alias();
    if (!strcmp(mode, "protect")) return mode_protect();
    if (!strcmp(mode, "saturate")) return mode_saturate();
    if (!strcmp(mode, "realloc")) return mode_realloc();
    if (!strcmp(mode, "pressure")) return mode_pressure();
    return mode_stale(mode);
}
