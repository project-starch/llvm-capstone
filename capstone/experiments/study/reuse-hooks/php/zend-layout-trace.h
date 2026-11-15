/* The dynamic layout check of docs/plans/2026-11-15-php-port-layout-fix.md. Test builds only
 * (-DZEND_CAP_LAYOUT_TRACE); build-layout-trace.sh builds both arms with it.
 *
 * After this header, malloc, realloc and free in the including file go through counters. The
 * stock build includes it in zend_alloc.c (through zend-layout-trace-def.h), so it counts
 * ZEND_DO_MALLOC, ZEND_DO_REALLOC and ZEND_DO_FREE; the nested build includes it there too and
 * in zend-cap-mode.c, so it counts the layer's own calls. zend_strndup's persistent malloc is
 * counted in both. What the heap receives is therefore compared call for call.
 *
 * Requests are split at 48 + 16 KiB, where the child bound reaches 16 KiB and the nested layer
 * may add slack. Per class, at exit, on stderr:
 *   ZEND-LAYOUT <class> mallocs=.. reallocs=.. inplace=.. moved=.. failed=.. bytes=.. seq=0x..
 * seq is an FNV-1a hash over the class's calls in order (operation, request, and for realloc
 * whether the block moved), so two equal hashes mean the same requests in the same order. Then
 *   ZEND-LAYOUT frees=..
 *   ZEND-LAYOUT-HIST size:count ...      (requests below the split, by exact size)
 *   ZEND-LAYOUT-BIG bits:count ...       (requests at or above, by bit length) */
#ifndef ZEND_LAYOUT_TRACE_H
#define ZEND_LAYOUT_TRACE_H
#ifdef ZEND_CAP_LAYOUT_TRACE
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void *zend_layout_malloc(size_t n);
void *zend_layout_realloc(void *p, size_t n);
void zend_layout_free(void *p);

#ifdef ZEND_LAYOUT_TRACE_DEFINE
#define ZEND_LAYOUT_SPLIT (48UL + 16384UL)
struct zend_layout_class { uint64_t mallocs, reallocs, inplace, moved, failed, bytes, seq; };
static struct zend_layout_class zend_layout_cls[2];   /* 0: below the split, 1: at or above */
static uint64_t zend_layout_frees;
static uint32_t zend_layout_hist[ZEND_LAYOUT_SPLIT];
static uint64_t zend_layout_big[65];

static void zend_layout_mix(struct zend_layout_class *c, uint64_t word)
{
    uint64_t h = c->seq ? c->seq : 1469598103934665603ULL;
    int i;
    for (i = 0; i < 8; i++) {
        h ^= (word >> (8 * i)) & 0xffU;
        h *= 1099511628211ULL;
    }
    c->seq = h;
}

static struct zend_layout_class *zend_layout_count(size_t n)
{
    if (n < ZEND_LAYOUT_SPLIT) {
        ++zend_layout_hist[n];
        return &zend_layout_cls[0];
    }
    ++zend_layout_big[64 - __builtin_clzl((unsigned long)n)];
    return &zend_layout_cls[1];
}

void *zend_layout_malloc(size_t n)
{
    struct zend_layout_class *c = zend_layout_count(n);
    ++c->mallocs;
    c->bytes += n;
    zend_layout_mix(c, (1ULL << 62) | (uint64_t)n);
    return malloc(n);
}

void *zend_layout_realloc(void *p, size_t n)
{
    struct zend_layout_class *c = zend_layout_count(n);
    unsigned long from = p ? __builtin_capstone_cap_get_cursor(p) : 0;
    void *r = realloc(p, n);
    int moved = !r || __builtin_capstone_cap_get_cursor(r) != from;
    ++c->reallocs;
    c->bytes += n;
    if (!r) { ++c->failed; }
    else if (moved) { ++c->moved; }
    else { ++c->inplace; }
    zend_layout_mix(c, (2ULL << 62) | ((uint64_t)moved << 61) | (uint64_t)n);
    return r;
}

void zend_layout_free(void *p)
{
    if (p) { ++zend_layout_frees; }
    free(p);
}

__attribute__((destructor)) static void zend_layout_report(void)
{
    static const char *const name[2] = { "small", "big" };
    unsigned long i, k;
    for (i = 0; i < 2; i++) {
        struct zend_layout_class *c = &zend_layout_cls[i];
        fprintf(stderr, "ZEND-LAYOUT %s mallocs=%llu reallocs=%llu inplace=%llu moved=%llu failed=%llu "
                "bytes=%llu seq=0x%016llx\n", name[i], (unsigned long long)c->mallocs,
                (unsigned long long)c->reallocs, (unsigned long long)c->inplace,
                (unsigned long long)c->moved, (unsigned long long)c->failed,
                (unsigned long long)c->bytes, (unsigned long long)c->seq);
    }
    fprintf(stderr, "ZEND-LAYOUT frees=%llu\n", (unsigned long long)zend_layout_frees);
    for (i = 0, k = 0; i < ZEND_LAYOUT_SPLIT; i++) {
        if (!zend_layout_hist[i]) { continue; }
        if (k % 64 == 0) { fprintf(stderr, k ? "\nZEND-LAYOUT-HIST" : "ZEND-LAYOUT-HIST"); }
        fprintf(stderr, " %lu:%lu", i, (unsigned long)zend_layout_hist[i]);
        ++k;
    }
    fprintf(stderr, k ? "\n" : "ZEND-LAYOUT-HIST\n");
    fprintf(stderr, "ZEND-LAYOUT-BIG");
    for (i = 0; i < 65; i++) {
        if (zend_layout_big[i]) { fprintf(stderr, " %lu:%llu", i, (unsigned long long)zend_layout_big[i]); }
    }
    fprintf(stderr, "\n");
    fflush(stderr);
}
#endif /* ZEND_LAYOUT_TRACE_DEFINE */

#undef malloc
#undef realloc
#undef free
#define malloc(n) zend_layout_malloc(n)
#define realloc(p, n) zend_layout_realloc((p), (n))
#define free(p) zend_layout_free(p)

#endif /* ZEND_CAP_LAYOUT_TRACE */
#endif
