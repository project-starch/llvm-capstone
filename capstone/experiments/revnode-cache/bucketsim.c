/* The proposed index: copies listed in the node record, deleted eagerly, with a
 * slot cache in front. Replayed on the node-access trace with alias records
 * (capstone-qemu perf/revnode-trace, 2f16a090 or later), from a file or "-".
 *
 * Design (README, "Ein Index mit einem Zugriff pro Aenderung"):
 *
 *   Node record = one 64-byte line: Capstone's node state and K slot refs of
 *   32 bits (the 16-byte granule) -- the copies of this node. The store unit
 *   sees the old granule it overwrites (the line is in the cache for the
 *   write), so a store knows both the node it removes a copy from (m) and the
 *   node it adds one to (n). No reverse directory.
 *
 *     capability store, same node (m == n)      0 accesses
 *     capability store into an empty slot       RMW node[n]          1
 *     capability store over another node's copy RMW node[m], node[n] 2
 *     data store over a capability              RMW node[m]          1
 *     revoke of a run of nodes                  root RMW, per node: RMW node, tag
 *                                               clear per listed slot, overflow
 *                                               lines; the node after the run
 *                                               (relinked) once
 *
 *   Overflow: a node with more than K copies gets a table of 16-entry lines,
 *   2-choice hashed by granule, insert into the first choice with room, grown
 *   by doubling when both are full. Lookups touch 1 or 2 lines.
 *
 *   Slot cache (W entries, LRU, keyed by granule): the W most recently
 *   capability-written slots are not in the memory index yet. An entry holds
 *   the slot's current node c and the node f the memory index still lists for
 *   it. Rewrites of a cached slot change only c. On eviction the entry is
 *   flushed: delete (f,g) and insert (c,g) if they differ. A revoke clears the
 *   tags of cached slots whose c is revoked, and skips memory entries whose
 *   slot is cached under another c. W = 0 is the design without the cache.
 *
 * Costs: a read-modify-write of one line is one access. Misses at 64 lines
 * per operation, and misses at every size overall, through one fully
 * associative LRU of 64-byte lines per variant. Tag clears are counted apart.
 *
 * The emulator revokes lazily; under Clover a revoke clears the copies' tags
 * at once. The model therefore keeps its own view of which slot holds which
 * node (the shadow): a copy the emulator still reports after Clover cleared
 * it does not exist here, a store of a capability whose node Clover revoked
 * is a data store, and the emulator's collector untags, not Clover
 * operations, update the index for free.
 *
 * Node ids: each variant names an id allocation policy (idpolicy.h: traced,
 * lifo, bitmap, hybrid). The trace's ids stay the keys of the model's own
 * bookkeeping; the policy decides which record a node occupies, and so which
 * line its accesses touch. A node's id is returned to the policy when its
 * revoke has freed its record.
 *
 * Variants: --variants "K,W,B[,P];..." (inline capacity, slot cache entries,
 * node record bytes 64 or 32, policy; default traced). Default
 * "12,64,64,traced;12,64,64,lifo;12,64,64,bitmap;12,64,64,hybrid;12,64,64,chunk;4,64,32,traced;4,64,32,lifo;4,64,32,bitmap;4,64,32,hybrid;4,64,32,chunk".
 *
 *   cc -O2 -o bucketsim bucketsim.c
 *   ./bucketsim [--max-node N] [--no-end] [--variants ...] trace|-
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nested_lru.h"
#include "idpolicy.h"

enum { K_READ, K_WRITE, K_ALLOC, K_FREE, K_RESET, K_REPEAT, K_ALIAS_INC, K_ALIAS_DEC, K_ALIAS_REG,
       K_ALIAS_SLOT, K_END, K_INSN };
enum { S_LDST, S_LDC, S_MREV, S_SPLIT, S_REVOKE, S_DELIN, S_CREATE, S_SUPERVISOR, S_GC,
       S_MEM_CAPSTORE, S_MEM_UNTAG, S_MEM_CLEAR, S_DROP,
       S_RC_REG_INC, S_RC_REG_DEC, S_RC_MEM_INC, S_RC_MEM_DEC, S_RC_SAME, S_RC_FREE,
       S_RC_LD_INC, S_RC_LD_DEC, S_RC_SWEEP_DEC, S_RC_SWEEP_FREE, S_RC_MOVE, S_RC_CALL, N_SITES };

enum { OP_INSERT, OP_DELETE, OP_FLUSH, OP_REUSE, OP_REVOKE, OP_TREE, OP_GROW, N_OPS };
static const char *op_name[N_OPS] = {"insert", "delete", "flush", "reuse_flush", "revoke", "tree", "grow"};

#define R_NODE 1ull
#define R_TAB 2ull
#define LINE(r, v) (((r) << 56) | (uint64_t)(v))
#define KMAX 16
#define TABW 16   /* 32-bit entries per 64-byte table line */
#define MAXV 10

struct bucket {
    uint32_t inl[KMAX];
    uint32_t n_inl, lg, ntab, tab_id, freenext;
    uint32_t *tab;
};

struct variant {
    unsigned K, W, node_bytes;
    struct idpolicy ids;
    struct lru *c;
    uint32_t *bk;               /* node -> bucket index + 1 */
    struct bucket *pool;
    uint32_t pool_n, pool_cap, pool_free, tab_ids;
    /* slot cache: W entries, LRU list, hash granule -> entry */
    uint32_t *sc_g, *sc_c, *sc_f, *sc_prev, *sc_next, sc_head, sc_tail, sc_n, sc_free;
    uint32_t *sc_hk, *sc_hv;
    unsigned sc_hbits;
    uint16_t *ref;              /* node -> cache entries naming it as c or f */
    /* counters */
    uint64_t acc[N_OPS], miss64[N_OPS];
    uint64_t changes, absorbed, absorbed_del, stores_fresh, stores_same, stores_other, stores_stale,
             data_del, gc_silent, overflow_inserts, overflow_nodes, grows, max_lines, fallback,
             tag_clears, cam_skips, cam_purges, evictions, errors, mem_entries, mem_entries_peak,
             revokes, revoke_acc_max;
    int op;
};

static uint32_t max_node = 16777216;
static uint32_t *prv, *nxt, *depth;
static uint8_t *valid;

/* ---- the shadow: slot -> node under Clover, shared by the variants ---- */
#define SHB 24
static uint32_t *sh_k, *sh_v;   /* open addressing, value = node + 1 */
static uint32_t shh(uint32_t g) { return (uint32_t)(((uint64_t)g * 0x9E3779B97F4A7C15ull) >> (64 - SHB)); }
static uint32_t shadow_get(uint32_t g) {
    for (uint32_t i = shh(g);; i = (i + 1) & ((1u << SHB) - 1)) {
        if (!sh_v[i]) return NONE;
        if (sh_k[i] == g) return sh_v[i] - 1;
    }
}
static void shadow_set(uint32_t g, uint32_t n) {
    uint32_t i = shh(g);
    while (sh_v[i] && sh_k[i] != g) i = (i + 1) & ((1u << SHB) - 1);
    sh_k[i] = g; sh_v[i] = n + 1;
}
static void shadow_del(uint32_t g) {
    uint32_t mask = (1u << SHB) - 1, i = shh(g);
    while (sh_k[i] != g || !sh_v[i]) i = (i + 1) & mask;
    sh_v[i] = 0;
    for (uint32_t j = (i + 1) & mask; sh_v[j]; j = (j + 1) & mask) {
        uint32_t home = shh(sh_k[j]);
        if (((j - home) & mask) >= ((j - i) & mask)) { sh_k[i] = sh_k[j]; sh_v[i] = sh_v[j]; sh_v[j] = 0; i = j; }
    }
}

static void acc(struct variant *v, uint64_t line) {
    uint64_t h64 = v->c->hist[0] + v->c->hist[1];
    touch(v->c, line);
    ++v->acc[v->op];
    if (v->c->hist[0] + v->c->hist[1] == h64) ++v->miss64[v->op];
}
static uint64_t node_line(struct variant *v, uint32_t n) {
    return LINE(R_NODE, idpolicy_map(&v->ids, n) / (64 / v->node_bytes));
}

/* ---- buckets: the memory index ---- */
static struct bucket *bucket(struct variant *v, uint32_t n) {
    return v->bk[n] ? &v->pool[v->bk[n] - 1] : NULL;
}
static struct bucket *bucket_make(struct variant *v, uint32_t n) {
    uint32_t i;
    if (v->pool_free != NONE) { i = v->pool_free; v->pool_free = v->pool[i].freenext; }
    else {
        if (v->pool_n == v->pool_cap) {
            v->pool_cap = v->pool_cap ? v->pool_cap * 2 : 4096;
            v->pool = realloc(v->pool, sizeof(struct bucket) * v->pool_cap);
        }
        i = v->pool_n++;
    }
    memset(&v->pool[i], 0, sizeof(struct bucket));
    v->bk[n] = i + 1;
    return &v->pool[i];
}
static void bucket_free(struct variant *v, uint32_t n) {
    struct bucket *b = bucket(v, n);
    free(b->tab);
    b->freenext = v->pool_free;
    v->pool_free = v->bk[n] - 1;
    v->bk[n] = 0;
}
static uint32_t th1(uint32_t g, unsigned lg) { return (uint32_t)(((uint64_t)g * 0x9E3779B97F4A7C15ull) >> (64 - lg)); }
static uint32_t th2(uint32_t g, unsigned lg) { return (uint32_t)(((uint64_t)(g ^ 0x5bd1e995u) * 0xC2B2AE3D27D4EB4Full) >> (64 - lg)); }
static uint64_t tab_line(struct bucket *b, uint32_t l) { return LINE(R_TAB, ((uint64_t)b->tab_id << 24) | l); }

static int line_put(uint32_t *line, uint32_t g) {
    for (int i = 0; i < TABW; ++i) if (!line[i]) { line[i] = g; return 1; }
    return 0;
}
static int line_del(uint32_t *line, uint32_t g) {
    for (int i = 0; i < TABW; ++i) if (line[i] == g) { line[i] = 0; return 1; }
    return 0;
}
static void tab_grow(struct variant *v, struct bucket *b) {
    int saved = v->op; v->op = OP_GROW; ++v->grows;
    uint32_t L = 1u << b->lg;
    uint32_t *old = b->tab;
    for (uint32_t l = 0; l < L; ++l) acc(v, tab_line(b, l));
    b->lg++; b->tab = calloc((size_t)2 * L * TABW, 4); b->tab_id = ++v->tab_ids;
    for (uint32_t l = 0; l < 2 * L; ++l) acc(v, tab_line(b, l));
    for (uint32_t i = 0; i < L * TABW; ++i) {
        uint32_t e = old[i];
        if (!e) continue;
        if (line_put(&b->tab[th1(e, b->lg) * TABW], e) || line_put(&b->tab[th2(e, b->lg) * TABW], e)) continue;
        /* neither choice has room after doubling: first line with a hole;
           a deletion then finds it by a scan. Counted. */
        ++v->fallback;
        int placed = 0;
        for (uint32_t l = 0; l < 2 * L && !placed; ++l) placed = line_put(&b->tab[l * TABW], e);
        if (!placed) { fprintf(stderr, "bucketsim: table full after doubling\n"); exit(1); }
    }
    free(old);
    if ((1u << b->lg) > v->max_lines) v->max_lines = 1u << b->lg;
    v->op = saved;
}
static void tab_insert(struct variant *v, struct bucket *b, uint32_t g) {
    if (!b->tab) {
        b->lg = 1; b->tab = calloc(2 * TABW, 4); b->ntab = 0; b->tab_id = ++v->tab_ids;
        ++v->overflow_nodes;
        if (2 > v->max_lines) v->max_lines = 2;
    }
    for (;;) {
        acc(v, tab_line(b, th1(g, b->lg)));
        if (line_put(&b->tab[th1(g, b->lg) * TABW], g)) break;
        acc(v, tab_line(b, th2(g, b->lg)));
        if (line_put(&b->tab[th2(g, b->lg) * TABW], g)) break;
        tab_grow(v, b);
    }
    ++b->ntab;
}
static int tab_delete(struct variant *v, struct bucket *b, uint32_t g, int silent) {
    if (!b->tab) return 0;
    if (!silent) acc(v, tab_line(b, th1(g, b->lg)));
    if (line_del(&b->tab[th1(g, b->lg) * TABW], g)) { --b->ntab; return 1; }
    if (!silent) acc(v, tab_line(b, th2(g, b->lg)));
    if (line_del(&b->tab[th2(g, b->lg) * TABW], g)) { --b->ntab; return 1; }
    for (uint32_t l = 0; l < (1u << b->lg); ++l)       /* a fallback placement */
        if (line_del(&b->tab[l * TABW], g)) { if (!silent) acc(v, tab_line(b, l)); --b->ntab; return 1; }
    return 0;
}

static void mem_insert(struct variant *v, uint32_t n, uint32_t g) {
    acc(v, node_line(v, n));
    struct bucket *b = bucket(v, n);
    if (!b) b = bucket_make(v, n);
    if (b->n_inl < v->K) b->inl[b->n_inl++] = g;
    else { ++v->overflow_inserts; tab_insert(v, b, g); }
    if (++v->mem_entries > v->mem_entries_peak) v->mem_entries_peak = v->mem_entries;
}
static void mem_delete(struct variant *v, uint32_t m, uint32_t g, int silent) {
    if (!silent) acc(v, node_line(v, m));
    struct bucket *b = bucket(v, m);
    if (!b) { ++v->errors; return; }
    int found = 0;
    for (uint32_t i = 0; i < b->n_inl; ++i)
        if (b->inl[i] == g) { b->inl[i] = b->inl[--b->n_inl]; found = 1; break; }
    if (!found) found = tab_delete(v, b, g, silent);
    if (!found) { ++v->errors; return; }
    --v->mem_entries;
    if (!b->n_inl && !b->ntab) bucket_free(v, m);
}

/* ---- slot cache ---- */
static uint32_t sch(struct variant *v, uint32_t g) { return (uint32_t)(((uint64_t)g * 0x9E3779B97F4A7C15ull) >> (64 - v->sc_hbits)); }
static int sc_find(struct variant *v, uint32_t g) {
    if (!v->W) return -1;
    uint32_t mask = (1u << v->sc_hbits) - 1;
    for (uint32_t i = sch(v, g);; i = (i + 1) & mask) {
        if (!v->sc_hv[i]) return -1;
        if (v->sc_hk[i] == g) return (int)v->sc_hv[i] - 1;
    }
}
static void sc_hput(struct variant *v, uint32_t g, uint32_t idx) {
    uint32_t mask = (1u << v->sc_hbits) - 1, i = sch(v, g);
    while (v->sc_hv[i]) i = (i + 1) & mask;
    v->sc_hk[i] = g; v->sc_hv[i] = idx + 1;
}
static void sc_hdel(struct variant *v, uint32_t g) {
    uint32_t mask = (1u << v->sc_hbits) - 1, i = sch(v, g);
    while (v->sc_hk[i] != g || !v->sc_hv[i]) i = (i + 1) & mask;
    v->sc_hv[i] = 0;
    for (uint32_t j = (i + 1) & mask; v->sc_hv[j]; j = (j + 1) & mask) {
        uint32_t home = sch(v, v->sc_hk[j]);
        if (((j - home) & mask) >= ((j - i) & mask)) {
            v->sc_hk[i] = v->sc_hk[j]; v->sc_hv[i] = v->sc_hv[j]; v->sc_hv[j] = 0; i = j;
        }
    }
}
static void sc_unlink(struct variant *v, uint32_t s) {
    if (v->sc_prev[s] != NONE) v->sc_next[v->sc_prev[s]] = v->sc_next[s]; else v->sc_head = v->sc_next[s];
    if (v->sc_next[s] != NONE) v->sc_prev[v->sc_next[s]] = v->sc_prev[s]; else v->sc_tail = v->sc_prev[s];
}
static void sc_front(struct variant *v, uint32_t s) {
    v->sc_prev[s] = NONE; v->sc_next[s] = v->sc_head;
    if (v->sc_head != NONE) v->sc_prev[v->sc_head] = s;
    v->sc_head = s;
    if (v->sc_tail == NONE) v->sc_tail = s;
}
static void ref_add(struct variant *v, uint32_t n, int d) { if (n != NONE) v->ref[n] += d; }
/* write an entry's state to the memory index and drop the entry */
static void sc_flush(struct variant *v, uint32_t s, int op) {
    uint32_t g = v->sc_g[s], c = v->sc_c[s], f = v->sc_f[s];
    int saved = v->op; v->op = op;
    if (f != c) {
        if (f != NONE) mem_delete(v, f, g, 0);
        if (c != NONE) mem_insert(v, c, g);
    }
    v->op = saved;
    ref_add(v, c, -1); ref_add(v, f, -1);
    sc_unlink(v, s);
    sc_hdel(v, g);
    v->sc_n--;
    v->sc_next[s] = v->sc_free; v->sc_free = s;
}
/* a slot enters the cache; the memory index lists it under f (or nothing) */
static void sc_insert(struct variant *v, uint32_t g, uint32_t c, uint32_t f) {
    if (v->sc_n == v->W) { ++v->evictions; sc_flush(v, v->sc_tail, OP_FLUSH); }
    uint32_t s = v->sc_free; v->sc_free = v->sc_next[s];
    v->sc_g[s] = g; v->sc_c[s] = c; v->sc_f[s] = f;
    ref_add(v, c, +1); ref_add(v, f, +1);
    sc_front(v, s);
    sc_hput(v, g, s);
    v->sc_n++;
}
static void sc_set_c(struct variant *v, int s, uint32_t c) {
    ref_add(v, v->sc_c[s], -1); v->sc_c[s] = c; ref_add(v, c, +1);
    sc_unlink(v, s); sc_front(v, s);
}
static void sc_reset(struct variant *v) {
    if (!v->W) return;
    memset(v->sc_hv, 0, sizeof(uint32_t) << v->sc_hbits);
    v->sc_head = v->sc_tail = NONE; v->sc_n = 0; v->sc_free = 0;
    for (uint32_t i = 0; i < v->W; ++i) v->sc_next[i] = i + 1 < v->W ? i + 1 : NONE;
}

/* ---- the operations ---- */
static void cap_store(struct variant *v, uint32_t g, uint32_t m, uint32_t n) {
    if (m == n) { ++v->stores_same; return; }
    ++v->changes;
    if (m == NONE) ++v->stores_fresh; else ++v->stores_other;
    int s = sc_find(v, g);
    if (s >= 0) { sc_set_c(v, s, n); ++v->absorbed; return; }
    if (v->W) { sc_insert(v, g, n, m); return; }
    if (m != NONE) { v->op = OP_DELETE; mem_delete(v, m, g, 0); }
    v->op = OP_INSERT; mem_insert(v, n, g);
}
static void data_store(struct variant *v, uint32_t g, uint32_t m) {
    ++v->data_del;
    int s = sc_find(v, g);
    if (s >= 0) { sc_set_c(v, s, NONE); ++v->absorbed_del; return; }
    v->op = OP_DELETE; mem_delete(v, m, g, 0);
}
/* the emulator's collector or a map reset untagged g: not a Clover event */
static void silent_untag(struct variant *v, uint32_t g, uint32_t m) {
    ++v->gc_silent;
    int s = sc_find(v, g);
    if (s >= 0) { sc_set_c(v, s, NONE); return; }
    mem_delete(v, m, g, 1);
}
/* one node of a revoked run: its copies lose their tags, its index goes. The
 * primary variant also takes the cleared slots out of the shadow. */
static void revoke_node(struct variant *v, uint32_t c, int primary) {
    acc(v, node_line(v, c));
    uint64_t listed_cached = 0;
    if (v->W && v->ref[c]) {
        for (uint32_t s = v->sc_head; s != NONE; s = v->sc_next[s]) {
            int holds = v->sc_c[s] == c, listed = v->sc_f[s] == c;
            if (holds) {
                ++v->tag_clears; ++v->cam_purges; ref_add(v, c, -1); v->sc_c[s] = NONE;
                if (primary) shadow_del(v->sc_g[s]);
            }
            if (listed) {
                ++listed_cached;
                if (!holds) ++v->cam_skips;   /* listed under c, holds another node now */
                ref_add(v, c, -1); v->sc_f[s] = NONE;
            }
        }
    }
    struct bucket *b = bucket(v, c);
    if (b) {
        uint64_t listed = b->n_inl + b->ntab;
        v->tag_clears += listed - listed_cached;   /* uncached listed slots hold c */
        if (primary) {
            for (uint32_t i = 0; i < b->n_inl; ++i)
                if (sc_find(v, b->inl[i]) < 0) shadow_del(b->inl[i]);
            if (b->tab)
                for (uint32_t i = 0; i < (1u << b->lg) * TABW; ++i)
                    if (b->tab[i] && sc_find(v, b->tab[i]) < 0) shadow_del(b->tab[i]);
        }
        if (b->tab) for (uint32_t l = 0; l < (1u << b->lg); ++l) acc(v, tab_line(b, l));
        v->mem_entries -= listed;
        bucket_free(v, c);
    }
    valid[c] = 0;
    idpolicy_free(&v->ids, c);     /* nothing names it any more: its id is free */
}
/* a node id is handed out again: its index must be empty */
static void node_reuse(struct variant *v, uint32_t id) {
    if (v->W && v->ref[id]) {
        /* entries that still list it in memory: write them out first */
        for (uint32_t s = v->sc_head, nx; s != NONE; s = nx) {
            nx = v->sc_next[s];
            if (v->sc_f[s] == id) sc_flush(v, s, OP_REUSE);
            else if (v->sc_c[s] == id) { ++v->errors; ref_add(v, id, -1); v->sc_c[s] = NONE; }
        }
    }
    if (bucket(v, id)) { ++v->errors; v->mem_entries -= bucket(v, id)->n_inl + bucket(v, id)->ntab; bucket_free(v, id); }
}

static void variant_init(struct variant *v, unsigned K, unsigned W, unsigned B, enum idpolicy_kind P) {
    memset(v, 0, sizeof *v);
    v->K = K; v->W = W; v->node_bytes = B;
    idpolicy_init(&v->ids, P, max_node + 1);
    v->c = calloc(1, sizeof *v->c); lru_flush(v->c);
    v->bk = calloc((size_t)max_node + 1, 4);
    v->pool_free = NONE;
    v->ref = calloc((size_t)max_node + 1, 2);
    if (W) {
        v->sc_g = malloc(W * 4); v->sc_c = malloc(W * 4); v->sc_f = malloc(W * 4);
        v->sc_prev = malloc(W * 4); v->sc_next = malloc(W * 4);
        v->sc_hbits = 2; while ((1u << v->sc_hbits) < 4 * W) ++v->sc_hbits;
        v->sc_hk = malloc(sizeof(uint32_t) << v->sc_hbits); v->sc_hv = malloc(sizeof(uint32_t) << v->sc_hbits);
    }
    sc_reset(v);
}
static void variant_reset(struct variant *v) {
    lru_flush(v->c);
    idpolicy_reset(&v->ids);
    memset(v->bk, 0, ((size_t)max_node + 1) * 4);
    for (uint32_t i = 0; i < v->pool_n; ++i) free(v->pool[i].tab);
    v->pool_n = 0; v->pool_free = NONE;
    memset(v->ref, 0, ((size_t)max_node + 1) * 2);
    v->mem_entries = 0;
    sc_reset(v);
}

int main(int argc, char **argv) {
    const char *path = NULL,
               *spec = "12,64,64,traced;12,64,64,lifo;12,64,64,bitmap;12,64,64,hybrid;12,64,64,chunk;4,64,32,traced;4,64,32,lifo;4,64,32,bitmap;4,64,32,hybrid;4,64,32,chunk";
    int no_end = 0, saw_end = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-end")) no_end = 1;
        else if (!strcmp(argv[i], "--max-node") && i + 1 < argc) max_node = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--variants") && i + 1 < argc) spec = argv[++i];
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: bucketsim [--max-node N] [--no-end] [--variants K,W,B;...] trace|-\n"); return 2; }
        else path = argv[i];
    }
    if (!path) { fprintf(stderr, "usage: bucketsim [--max-node N] [--no-end] [--variants K,W,B;...] trace|-\n"); return 2; }
    static struct variant vs[MAXV];
    int nv = 0;
    {
        char *copy = strdup(spec);
        for (char *tok = strtok(copy, ";"); tok; tok = strtok(NULL, ";")) {
            unsigned K, W, B;
            char pname[16] = "traced";
            enum idpolicy_kind P;
            int n = sscanf(tok, "%u,%u,%u,%15s", &K, &W, &B, pname);
            if (n < 3 || K > KMAX || (B != 64 && B != 32) || W > 65535 || nv == MAXV || !idpolicy_parse(pname, &P)) {
                fprintf(stderr, "bucketsim: bad variant %s\n", tok); return 2;
            }
            variant_init(&vs[nv++], K, W, B, P);
        }
    }
    FILE *in = strcmp(path, "-") ? fopen(path, "rb") : stdin;
    if (!in) { perror(path); return 1; }
    static char inbuf[1 << 22];
    setvbuf(in, inbuf, _IOFBF, sizeof inbuf);
    char magic[8];
    if (fread(magic, 1, 8, in) != 8 || (memcmp(magic, "CRNTRC01", 8) && memcmp(magic, "CRNTRC02", 8) && memcmp(magic, "CRNTRC03", 8) && memcmp(magic, "CRNTRC04", 8) && memcmp(magic, "CRNTRC05", 8))) {
        fprintf(stderr, "bucketsim: %s: no trace header\n", path);
        return 1;
    }
    size_t N = (size_t)max_node + 1;
    prv = malloc(N * 4); nxt = malloc(N * 4); depth = calloc(N, 4); valid = calloc(N, 1);
    sh_k = malloc(sizeof(uint32_t) << SHB); sh_v = calloc((size_t)1 << SHB, sizeof(uint32_t));
    uint32_t last_read[N_SITES];
    uint64_t insns = 0;
    for (int s = 0; s < N_SITES; ++s) last_read[s] = NONE;

    uint64_t checks = 0, n = 0;
    uint32_t slot = NONE;
    int in_revoke = 0;
    uint32_t root = NONE;
    unsigned last_kind = 0, last_site = 0;
    int have_last = 0;
    uint8_t r[8];
    size_t got;
    while ((got = fread(r, 1, 8, in)) == 8) {
        ++n;
        uint32_t id = r[0] | r[1] << 8 | r[2] << 16 | (uint32_t)r[3] << 24;
        unsigned kind = r[4], site = r[5];
        if (saw_end) { fprintf(stderr, "bucketsim: record after END\n"); return 1; }
        if (kind == K_END) { saw_end = 1; continue; }
        if (kind == K_INSN) { insns |= (uint64_t)id << (site ? 32 : 0); continue; }   /* the domain's instructions (format 04) */
        if (kind == K_REPEAT) {
            if (!have_last) { fprintf(stderr, "bucketsim: REPEAT first\n"); return 1; }
            if (last_kind == K_READ && (last_site == S_LDST || last_site == S_LDC)) checks += id;
            else if (last_kind >= K_ALIAS_INC && last_kind != K_ALIAS_REG) {
                fprintf(stderr, "bucketsim: record %llu: repeated alias record\n", (unsigned long long)n);
                return 1;
            }
            continue;
        }
        have_last = kind != K_RESET;
        last_kind = kind; last_site = site;
        if (kind > K_ALIAS_SLOT || site >= N_SITES) {
            fprintf(stderr, "bucketsim: record %llu: bad kind %u site %u\n", (unsigned long long)n, kind, site);
            return 1;
        }
        if (kind == K_ALIAS_SLOT) { slot = id; continue; }
        if (kind == K_ALIAS_REG) continue;
        if (kind == K_RESET) {
            for (int k = 0; k < nv; ++k) variant_reset(&vs[k]);
            memset(valid, 0, N);
            memset(sh_v, 0, sizeof(uint32_t) << SHB);
            in_revoke = 0; slot = NONE;
            continue;
        }
        if (id != NONE && id > max_node && kind != K_ALIAS_SLOT) {
            fprintf(stderr, "bucketsim: node %u above --max-node\n", id);
            return 1;
        }
        if (kind == K_ALIAS_DEC) {
            if (slot == NONE) { fprintf(stderr, "bucketsim: record %llu: DEC without a slot\n", (unsigned long long)n); return 1; }
            if (site == S_MEM_CAPSTORE) continue;      /* the INC that follows does the work */
            uint32_t m = shadow_get(slot);
            if (m == NONE) continue;                   /* Clover cleared this slot already */
            shadow_del(slot);
            for (int k = 0; k < nv; ++k) {
                if (site == S_MEM_UNTAG) data_store(&vs[k], slot, m);
                else silent_untag(&vs[k], slot, m);
            }
            continue;
        }
        if (kind == K_ALIAS_INC) {
            if (slot == NONE) { fprintf(stderr, "bucketsim: record %llu: INC without a slot\n", (unsigned long long)n); return 1; }
            uint32_t m = shadow_get(slot);             /* what the slot holds under Clover */
            if (valid[id]) {
                shadow_set(slot, id);
                for (int k = 0; k < nv; ++k) cap_store(&vs[k], slot, m, id);
            } else {
                /* Clover cleared this capability's tag at its revoke: a data store */
                if (m != NONE) shadow_del(slot);
                for (int k = 0; k < nv; ++k) {
                    ++vs[k].stores_stale;
                    if (m != NONE) data_store(&vs[k], slot, m);
                }
            }
            slot = NONE;
            continue;
        }

        /* node accesses */
        if (id == NONE) {
            if (kind == K_READ && (site == S_LDST || site == S_LDC)) ++checks;
            continue;
        }
        if (kind == K_READ && (site == S_LDST || site == S_LDC)) ++checks;
        if (kind == K_READ) last_read[site] = id;
        if (kind == K_ALLOC) {
            for (int k = 0; k < nv; ++k) {
                struct variant *v = &vs[k];
                node_reuse(v, id);
                idpolicy_alloc(&v->ids, id);       /* the record this node will occupy */
                v->op = OP_TREE;
                if ((site == S_MREV || site == S_SPLIT) && last_read[site] != NONE && valid[last_read[site]]) {
                    uint32_t src = last_read[site];
                    acc(v, node_line(v, id)); acc(v, node_line(v, src));
                    if (prv[src] != NONE) acc(v, node_line(v, prv[src]));
                } else {
                    acc(v, node_line(v, id));
                }
            }
            if ((site == S_MREV || site == S_SPLIT) && last_read[site] != NONE && valid[last_read[site]]) {
                uint32_t src = last_read[site];
                depth[id] = depth[src]; prv[id] = prv[src];
                if (prv[src] != NONE) nxt[prv[src]] = id;
                nxt[id] = src; prv[src] = id; valid[id] = 1;
                if (site == S_MREV) ++depth[src];
            } else {
                prv[id] = nxt[id] = NONE; depth[id] = 0; valid[id] = 1;
            }
        } else if (site == S_REVOKE && kind == K_READ && !in_revoke && valid[id]) {
            in_revoke = 1; root = id;
            for (int k = 0; k < nv; ++k) {
                struct variant *v = &vs[k];
                v->op = OP_REVOKE;
                uint64_t before = v->acc[OP_REVOKE];
                acc(v, node_line(v, root));
                uint32_t c = nxt[root];
                for (; c != NONE && depth[c] > depth[root]; c = nxt[c]) revoke_node(v, c, k == 0);
                if (c != NONE) acc(v, node_line(v, c));   /* the node after the run: its prev is relinked */
                acc(v, node_line(v, root));
                ++v->revokes;
                uint64_t a = v->acc[OP_REVOKE] - before;
                if (a > v->revoke_acc_max) v->revoke_acc_max = a;
            }
            uint32_t c = nxt[root];
            while (c != NONE && depth[c] > depth[root]) { valid[c] = 0; c = nxt[c]; }
            nxt[root] = c;
            if (c != NONE) prv[c] = root;
        } else if (site == S_REVOKE && kind == K_WRITE && in_revoke && id == root) {
            in_revoke = 0;
        }
    }
    if (got != 0 || ferror(in)) {
        fprintf(stderr, "bucketsim: %s: truncated trace (%zu stray bytes)\n", path, got);
        return 1;
    }
    if (!saw_end && !no_end) {
        fprintf(stderr, "bucketsim: %s: no END record; the trace is truncated\n", path);
        return 1;
    }
    if (!checks) { fprintf(stderr, "bucketsim: no lifetime checks in the trace\n"); return 1; }

    printf("{\n  \"trace\": \"%s\",\n  \"records\": %llu,\n  \"domain_instructions\": %llu,\n  \"lifetime_checks\": %llu,\n  \"line_bytes\": 64,\n"
           "  \"sizes_lines\": [", path, (unsigned long long)n, (unsigned long long)insns, (unsigned long long)checks);
    for (int k = 0; k < NSZ; ++k) printf("%s%u", k ? ", " : "", sizes[k]);
    printf("],\n  \"variants\": [");
    int rc = 0;
    for (int k = 0; k < nv; ++k) {
        struct variant *v = &vs[k];
        uint64_t total = 0;
        for (int o = 0; o < N_OPS; ++o) total += v->acc[o];
        printf("%s\n    {\"inline\": %u, \"slot_cache\": %u, \"node_bytes\": %u, \"policy\": \"%s\", \"accesses\": %llu, \"misses\": [",
               k ? "," : "", v->K, v->W, v->node_bytes, idpolicy_name[v->ids.kind], (unsigned long long)total);
        for (int s = 0; s < NSZ; ++s) printf("%s%llu", s ? ", " : "", (unsigned long long)lru_misses(v->c, s));
        printf("],\n     \"by_operation\": {");
        for (int o = 0; o < N_OPS; ++o)
            printf("%s\"%s\": {\"accesses\": %llu, \"misses_64_lines\": %llu}", o ? ", " : "", op_name[o],
                   (unsigned long long)v->acc[o], (unsigned long long)v->miss64[o]);
        printf("},\n     \"capability_stores\": {\"same_node\": %llu, \"fresh\": %llu, \"other_node\": %llu, \"of_revoked\": %llu},"
               " \"data_stores_over_capability\": %llu,\n",
               (unsigned long long)v->stores_same, (unsigned long long)v->stores_fresh,
               (unsigned long long)v->stores_other, (unsigned long long)v->stores_stale,
               (unsigned long long)v->data_del);
        printf("     \"index_changes\": %llu, \"absorbed_by_slot_cache\": %llu, \"deletes_absorbed\": %llu,"
               " \"evictions\": %llu,\n",
               (unsigned long long)v->changes, (unsigned long long)v->absorbed,
               (unsigned long long)v->absorbed_del, (unsigned long long)v->evictions);
        printf("     \"overflow\": {\"inserts\": %llu, \"nodes\": %llu, \"grows\": %llu, \"max_lines\": %llu,"
               " \"fallback_placements\": %llu},\n",
               (unsigned long long)v->overflow_inserts, (unsigned long long)v->overflow_nodes,
               (unsigned long long)v->grows, (unsigned long long)v->max_lines, (unsigned long long)v->fallback);
        printf("     \"revokes\": {\"count\": %llu, \"tag_clears\": %llu, \"accesses_mean\": %.3f, \"accesses_max\": %llu,"
               " \"cam_purges\": %llu, \"cam_skips\": %llu},\n",
               (unsigned long long)v->revokes, (unsigned long long)v->tag_clears,
               v->revokes ? (double)v->acc[OP_REVOKE] / v->revokes : 0.0, (unsigned long long)v->revoke_acc_max,
               (unsigned long long)v->cam_purges, (unsigned long long)v->cam_skips);
        printf("     \"memory_entries_peak\": %llu, \"collector_untags\": %llu, \"errors\": %llu,\n     ",
               (unsigned long long)v->mem_entries_peak, (unsigned long long)v->gc_silent,
               (unsigned long long)v->errors);
        idpolicy_print_json(&v->ids);
        printf("}");
        if (v->errors) rc = 3;
    }
    printf("\n  ]\n}\n");
    return rc;
}
