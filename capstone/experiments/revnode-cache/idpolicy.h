/* Node-id allocation policies, as a remapping of the trace's node ids.
 *
 * The emulator hands out node ids by a bump counter and recycles them only
 * when its pool runs low, which in every measured run it never did: each id
 * was used once. A design that reuses an id as soon as its node is revoked
 * chooses which free id to hand out next, and the choice decides where the
 * node's record lives. This remaps every allocation in the trace to the id
 * a policy would have chosen, keeping the trace's own ids as keys:
 *
 *   traced   the id as the emulator chose it (no reuse)
 *   lifo     the most recently freed id first; a bump counter when none is free
 *   bitmap   the lowest free id, from a 64-ary bitmap tree (4 levels for 2^24)
 *   hybrid   a free id in the leaf word used last, else the lowest free id
 *   chunk    a free id in the leaf word used last; when that word is full, the
 *            lowest leaf word with all 64 ids free (a fresh run, so the next
 *            allocations are neighbours again), and only failing that the
 *            lowest free id
 *
 * free() returns an id to the policy. An id is handed out again only after
 * it was returned, so reuse never precedes a revoke. The caller keeps the
 * map trace id -> policy id; a trace id freed and still named by later
 * records (the emulator's lazy checks of stale copies) keeps its mapping,
 * which is what hardware would do: the stale check reads the record that now
 * belongs to the new owner of that id.
 *
 * Density: live ids over the footprint (the highest id handed out so far,
 * plus one), taken at every allocation; the mean is over allocations.
 * reused counts allocations that got an id used before; under traced that is
 * the emulator's own reuse (its collector released the id). first_reuse is
 * the allocation at which it first happened (0: never).
 */
#ifndef IDPOLICY_H
#define IDPOLICY_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum idpolicy_kind { IDP_TRACED, IDP_LIFO, IDP_BITMAP, IDP_HYBRID, IDP_CHUNK, IDP_NKINDS };
static const char *idpolicy_name[] = {"traced", "lifo", "bitmap", "hybrid", "chunk"};

#define IDP_LEVELS 4   /* 64^4 = 2^24 ids */

struct idpolicy {
    enum idpolicy_kind kind;
    uint32_t capacity;             /* ids 0 .. capacity-1 */
    uint32_t *map;                 /* trace id -> policy id (NONE32 = unmapped) */
    uint8_t *mapped;               /* trace id has a mapping */
    uint8_t *freed;                /* trace id already returned */
    /* lifo */
    uint32_t *stack, sp, bump;
    /* bitmap: level 0 is one word, level l has 64^l words; leaf = level 3 */
    uint64_t *level[IDP_LEVELS];
    uint64_t *full[IDP_LEVELS - 1]; /* leaf words with all 64 ids free, summarised the same way */
    uint32_t last_leaf;            /* hybrid, chunk: the leaf word used last */
    /* statistics */
    uint64_t allocs, frees, live, peak_live, highwater, samples;
    double density_sum, density_min;
    uint64_t reused;               /* allocations that got an id used before */
    uint64_t first_reuse;          /* the allocation (1-based) that first reused one; 0 = none */
    uint64_t refill_fresh, refill_lowest;  /* hybrid, chunk: word exhausted -> fresh word / lowest id */
};

#define NONE32 0xffffffffu

static inline int idp_ffs64(uint64_t w) { return __builtin_ctzll(w); }

static void idpolicy_init(struct idpolicy *p, enum idpolicy_kind kind, uint32_t capacity) {
    memset(p, 0, sizeof *p);
    p->kind = kind;
    p->capacity = capacity;
    p->map = malloc(sizeof(uint32_t) * capacity);
    p->mapped = calloc(capacity, 1);
    p->freed = calloc(capacity, 1);
    p->density_min = 1.0;
    if (kind == IDP_LIFO) {
        p->stack = malloc(sizeof(uint32_t) * capacity);
    } else if (kind >= IDP_BITMAP) {
        size_t words = 1;
        for (int l = 0; l < IDP_LEVELS; ++l) {
            p->level[l] = malloc(sizeof(uint64_t) * words);
            memset(p->level[l], 0xff, sizeof(uint64_t) * words);   /* every id free */
            words *= 64;
        }
        /* ids at and above capacity do not exist: clear their leaf bits (and
           whole words), and the summary bits of fully absent words */
        size_t leaf_words = words / 64;
        for (size_t w = (capacity + 63) / 64; w < leaf_words; ++w) p->level[IDP_LEVELS - 1][w] = 0;
        if (capacity % 64) p->level[IDP_LEVELS - 1][capacity / 64] &= (~0ull) >> (64 - capacity % 64);
        for (int l = IDP_LEVELS - 2; l >= 0; --l) {
            size_t n = 1; for (int k = 0; k < l; ++k) n *= 64;
            for (size_t w = 0; w < n; ++w) {
                uint64_t s = 0;
                for (int b = 0; b < 64; ++b) if (p->level[l + 1][w * 64 + b]) s |= 1ull << b;
                p->level[l][w] = s;
            }
        }
        /* the fully free leaf words: bit w of full[L-2] is leaf word w, summarised above */
        for (int l = 0; l < IDP_LEVELS - 1; ++l) {
            size_t n = 1; for (int k = 0; k < l; ++k) n *= 64;
            p->full[l] = calloc(n, sizeof(uint64_t));
        }
        for (size_t w = 0; w < leaf_words; ++w)
            if (p->level[IDP_LEVELS - 1][w] == ~0ull) p->full[IDP_LEVELS - 2][w / 64] |= 1ull << (w % 64);
        for (int l = IDP_LEVELS - 3; l >= 0; --l) {
            size_t n = 1; for (int k = 0; k < l; ++k) n *= 64;
            for (size_t w = 0; w < n; ++w) {
                uint64_t s = 0;
                for (int b = 0; b < 64; ++b) if (p->full[l + 1][w * 64 + b]) s |= 1ull << b;
                p->full[l][w] = s;
            }
        }
        p->last_leaf = NONE32;
    }
}

static void idpolicy_reset(struct idpolicy *p) {
    enum idpolicy_kind kind = p->kind;
    uint32_t cap = p->capacity;
    free(p->map); free(p->mapped); free(p->freed); free(p->stack);
    for (int l = 0; l < IDP_LEVELS; ++l) free(p->level[l]);
    for (int l = 0; l < IDP_LEVELS - 1; ++l) free(p->full[l]);
    idpolicy_init(p, kind, cap);
}

/* leaf word w stopped being / became entirely free */
static void idp_full_clear(struct idpolicy *p, uint32_t w) {
    for (int l = IDP_LEVELS - 2; l >= 0; --l) {
        p->full[l][w / 64] &= ~(1ull << (w % 64));
        if (p->full[l][w / 64]) break;
        w /= 64;
    }
}
static void idp_full_set(struct idpolicy *p, uint32_t w) {
    for (int l = IDP_LEVELS - 2; l >= 0; --l) {
        int was_empty = p->full[l][w / 64] == 0;
        p->full[l][w / 64] |= 1ull << (w % 64);
        if (!was_empty) break;
        w /= 64;
    }
}

static uint32_t idp_bitmap_take(struct idpolicy *p, uint32_t leaf_word, int bit) {
    uint32_t id = leaf_word * 64 + bit;
    if (p->level[IDP_LEVELS - 1][leaf_word] == ~0ull) idp_full_clear(p, leaf_word);
    p->level[IDP_LEVELS - 1][leaf_word] &= ~(1ull << bit);
    /* a word that became empty clears its bit one level up, and so on */
    uint32_t w = leaf_word;
    for (int l = IDP_LEVELS - 1; l > 0 && p->level[l][w] == 0; --l) {
        p->level[l - 1][w / 64] &= ~(1ull << (w % 64));
        w /= 64;
    }
    p->last_leaf = leaf_word;
    return id;
}

/* the lowest leaf word with all 64 ids free, or NONE32 */
static uint32_t idp_fresh_word(const struct idpolicy *p) {
    if (!p->full[0][0]) return NONE32;
    uint32_t w = 0;
    for (int l = 0; l < IDP_LEVELS - 1; ++l) w = w * 64 + idp_ffs64(p->full[l][w]);
    return w;
}

static uint32_t idp_bitmap_lowest(struct idpolicy *p) {
    if (!p->level[0][0]) return NONE32;
    uint32_t w = 0;
    for (int l = 0; l < IDP_LEVELS - 1; ++l) w = w * 64 + idp_ffs64(p->level[l][w]);
    return idp_bitmap_take(p, w, idp_ffs64(p->level[IDP_LEVELS - 1][w]));
}

/* a new lifetime for trace id t: the id the policy hands out */
static uint32_t idpolicy_alloc(struct idpolicy *p, uint32_t t) {
    uint32_t id;
    int reuse = 0;
    switch (p->kind) {
    case IDP_TRACED:
        id = t;
        reuse = p->mapped[t];          /* the emulator handed this id out before */
        break;
    case IDP_LIFO:
        if (p->sp) { id = p->stack[--p->sp]; reuse = 1; }
        else id = p->bump++;
        break;
    case IDP_HYBRID:
    case IDP_CHUNK:
        if (p->last_leaf != NONE32 && p->level[IDP_LEVELS - 1][p->last_leaf]) {
            id = idp_bitmap_take(p, p->last_leaf, idp_ffs64(p->level[IDP_LEVELS - 1][p->last_leaf]));
            break;
        }
        if (p->kind == IDP_CHUNK) {
            uint32_t w = idp_fresh_word(p);
            if (w != NONE32) { ++p->refill_fresh; id = idp_bitmap_take(p, w, 0); break; }
        }
        ++p->refill_lowest;
        /* fall through */
    case IDP_BITMAP:
    default:
        id = idp_bitmap_lowest(p);
        break;
    }
    if (id == NONE32 || id >= p->capacity) {
        fprintf(stderr, "idpolicy: out of ids (capacity %u)\n", p->capacity);
        exit(1);
    }
    if (p->kind >= IDP_BITMAP) reuse = id < p->highwater;
    if ((uint64_t)id + 1 > p->highwater) p->highwater = id + 1;
    p->map[t] = id; p->mapped[t] = 1; p->freed[t] = 0;
    ++p->allocs; ++p->live;
    if (reuse) { ++p->reused; if (!p->first_reuse) p->first_reuse = p->allocs; }
    if (p->live > p->peak_live) p->peak_live = p->live;
    double d = (double)p->live / (double)p->highwater;
    p->density_sum += d; ++p->samples;
    if (d < p->density_min) p->density_min = d;
    return id;
}

/* the trace id t's lifetime ended (its revoke completed): its id is free */
static void idpolicy_free(struct idpolicy *p, uint32_t t) {
    if (!p->mapped[t] || p->freed[t]) return;
    p->freed[t] = 1;
    ++p->frees; --p->live;
    uint32_t id = p->map[t];
    switch (p->kind) {
    case IDP_TRACED: default: break;
    case IDP_LIFO: p->stack[p->sp++] = id; break;
    case IDP_BITMAP: case IDP_HYBRID: case IDP_CHUNK: {
        uint32_t w = id / 64;
        int was_empty = p->level[IDP_LEVELS - 1][w] == 0;
        p->level[IDP_LEVELS - 1][w] |= 1ull << (id % 64);
        if (p->level[IDP_LEVELS - 1][w] == ~0ull) idp_full_set(p, w);
        for (int l = IDP_LEVELS - 1; l > 0 && was_empty; --l) {
            was_empty = p->level[l - 1][w / 64] == 0;
            p->level[l - 1][w / 64] |= 1ull << (w % 64);
            w /= 64;
        }
        break;
    }
    }
}

/* the policy id a trace id maps to; a trace id never allocated maps to itself */
static inline uint32_t idpolicy_map(const struct idpolicy *p, uint32_t t) {
    return p->mapped[t] ? p->map[t] : t;
}

static int idpolicy_parse(const char *s, enum idpolicy_kind *out) {
    for (int k = 0; k < IDP_NKINDS; ++k) if (!strcmp(s, idpolicy_name[k])) { *out = (enum idpolicy_kind)k; return 1; }
    return 0;
}

static void idpolicy_print_json(const struct idpolicy *p) {
    printf("\"ids\": {\"policy\": \"%s\", \"allocations\": %llu, \"frees\": %llu, \"reused\": %llu, "
           "\"first_reuse_allocation\": %llu, \"footprint\": %llu, \"peak_live\": %llu, "
           "\"density_mean\": %.4f, \"density_min\": %.4f, \"density_samples\": %llu, "
           "\"refills_fresh_word\": %llu, \"refills_lowest_id\": %llu}",
           idpolicy_name[p->kind], (unsigned long long)p->allocs, (unsigned long long)p->frees,
           (unsigned long long)p->reused, (unsigned long long)p->first_reuse, (unsigned long long)p->highwater,
           (unsigned long long)p->peak_live, p->samples ? p->density_sum / p->samples : 0.0,
           p->samples ? p->density_min : 0.0, (unsigned long long)p->samples,
           (unsigned long long)p->refill_fresh, (unsigned long long)p->refill_lowest);
}

#endif
