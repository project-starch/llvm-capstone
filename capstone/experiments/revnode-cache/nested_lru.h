/* One fully associative LRU over 64-bit line ids, every size at once.
 *
 * The nested-LRU of cachesim.c: one list of LMAX lines, most recent first,
 * with a marker at each size boundary; an access in bucket b hits in every
 * size k >= b. Keyed through an open-addressing hash so a line id can be any
 * 64-bit value (a region tag in the top byte, an index below). Shared by
 * clovsim.c and bucketsim.c. */
#ifndef NESTED_LRU_H
#define NESTED_LRU_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef NONE
#define NONE 0xffffffffu
#endif

static const unsigned sizes[] = {16, 64, 256, 1024, 4096, 16384, 65536};
enum { NSZ = sizeof(sizes) / sizeof(sizes[0]) };
#define LMAX 65536
#define HBITS 18
struct lru {
    uint64_t tag[LMAX];
    uint32_t prev[LMAX], next[LMAX], head, tail, count, marker[NSZ];
    uint8_t bucket[LMAX];
    uint64_t hkey[1u << HBITS];     /* open addressing: line -> slot+1 */
    uint32_t hval[1u << HBITS];
    uint64_t hist[NSZ + 1];         /* accesses by bucket; NSZ = miss everywhere */
    uint64_t accesses;
};

static uint32_t hslot(uint64_t k) { return (uint32_t)((k * 0x9E3779B97F4A7C15ull) >> (64 - HBITS)); }
static uint32_t hget(struct lru *c, uint64_t k) {
    for (uint32_t i = hslot(k);; i = (i + 1) & ((1u << HBITS) - 1)) {
        if (!c->hval[i]) return 0;
        if (c->hkey[i] == k) return c->hval[i];
    }
}
static void hput(struct lru *c, uint64_t k, uint32_t v) {
    uint32_t i = hslot(k);
    while (c->hval[i] && c->hkey[i] != k) i = (i + 1) & ((1u << HBITS) - 1);
    c->hkey[i] = k; c->hval[i] = v;
}
static void hdel(struct lru *c, uint64_t k) {
    uint32_t mask = (1u << HBITS) - 1, i = hslot(k);
    while (c->hkey[i] != k || !c->hval[i]) i = (i + 1) & mask;
    c->hval[i] = 0;
    /* backward-shift the cluster after the hole */
    for (uint32_t j = (i + 1) & mask; c->hval[j]; j = (j + 1) & mask) {
        uint32_t home = hslot(c->hkey[j]);
        if (((j - home) & mask) >= ((j - i) & mask)) {
            c->hkey[i] = c->hkey[j]; c->hval[i] = c->hval[j]; c->hval[j] = 0; i = j;
        }
    }
}
static void lru_flush(struct lru *c) {
    memset(c->hval, 0, sizeof c->hval);
    c->count = 0; c->head = c->tail = NONE;
    for (int k = 0; k < NSZ; ++k) c->marker[k] = NONE;
}
static void unlink_(struct lru *c, uint32_t s) {
    if (c->prev[s] != NONE) c->next[c->prev[s]] = c->next[s]; else c->head = c->next[s];
    if (c->next[s] != NONE) c->prev[c->next[s]] = c->prev[s]; else c->tail = c->prev[s];
}
static void front(struct lru *c, uint32_t s) {
    c->prev[s] = NONE; c->next[s] = c->head;
    if (c->head != NONE) c->prev[c->head] = s;
    c->head = s;
    if (c->tail == NONE) c->tail = s;
    c->bucket[s] = 0;
}
static void shift(struct lru *c, int upto) {
    for (int k = 0; k < upto; ++k) {
        uint32_t m = c->marker[k];
        if (m == NONE) break;
        c->bucket[m] = k + 1;
        c->marker[k] = c->prev[m];
    }
}
/* the nested-LRU of cachesim.c, keyed through the hash */
static void touch(struct lru *c, uint64_t line) {
    ++c->accesses;
    uint32_t s1 = hget(c, line);
    if (s1) {
        uint32_t s = s1 - 1;
        unsigned b = c->bucket[s];
        ++c->hist[b];
        if (s == c->head) return;
        shift(c, b);
        if (c->marker[b] == s) c->marker[b] = c->prev[s];
        unlink_(c, s); front(c, s);
        return;
    }
    ++c->hist[NSZ];
    uint32_t s;
    if (c->count == LMAX) {
        s = c->tail;
        c->marker[NSZ - 1] = c->prev[s];
        unlink_(c, s);
        hdel(c, c->tag[s]);
        shift(c, NSZ - 1);
    } else {
        s = c->count++;
        shift(c, NSZ);
        for (int k = 0; k < NSZ; ++k)
            if (c->count == sizes[k]) c->marker[k] = c->tail == NONE ? s : c->tail;
    }
    c->tag[s] = line;
    hput(c, line, s + 1);
    front(c, s);
}


/* misses at size index k: every access in a bucket above k */
static inline uint64_t lru_misses(const struct lru *c, int k) {
    uint64_t m = 0;
    for (int b = k + 1; b <= NSZ; ++b) m += c->hist[b];
    return m;
}

#endif
