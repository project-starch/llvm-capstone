/* Capstone's lifetime checks against Clover's capability index, on one trace.
 *
 * Reads the node-access trace with alias and slot records (capstone-qemu
 * perf/revnode-trace; format in target/riscv/cap_rev_tree.h), from a file or
 * "-", and replays two designs' metadata traffic through a fully associative
 * LRU cache of 64-byte lines, 16 .. 65536 lines, one cache per design:
 *
 * Capstone (the trace as it is). Every node access the emulator traced -- the
 * lifetime check of each capability load/store, mrev/split/revoke/delin/drop,
 * allocation -- reads or writes a 16-byte node record: line = node / 4.
 *
 * Clover (nested-allocators-paper ideas/clover, the baseline in
 * detailed/sections/tracking-metadata.tex). No lifetime check. Instead, with
 * its own index kept from the trace's capability stores:
 *   capability store into slot g with node n (records SLOT g, [DEC], INC n):
 *     g registered under n already            -> nothing (same-node replacement)
 *     otherwise: reserve_alias_entry          -> R alias[e]   (pop: read .next)
 *                g registered under m != n    -> unregister(m, g)
 *                register(n, g, e)            -> R node[n], W alias[e],
 *                                                W alias[old head], W node[n],
 *                                                directory walk, W sidecar[g]
 *   data store over a slot the index holds    -> unregister
 *   unregister(m, g): directory walk, R sidecar[g], R alias[e],
 *                     W alias[prev] or W node[m], W alias[next] if any,
 *                     W sidecar[g], W alias[e] (push on the free list)
 *   directory walk: R radix top entry, R radix leaf entry (2-level, 8-byte
 *                   entries, 512 per table)
 *   revoke of root r: R node[r]; per invalidated node c: R node[c], per alias e
 *                   of c: R alias[e], unregister's sidecar/free writes, and the
 *                   tag clear of the slot (counted apart: it is a data-side
 *                   write); W node[c]; then W node[r].
 *   mrev/split: W node[new], W node[src], W node[src's list predecessor]
 *               (stands in for the parent's child link); lone node: W node[new]
 * Record sizes: node 32 B (links, alias head, Capstone state), alias 16 B
 * (slot, node, prev, next as 32-bit fields), sidecar cell 4 B.
 *
 * A capability whose node the index has already revoked is untagged under
 * Clover; the emulator revokes lazily and may still store it, so such a store
 * is treated as a data store and counted. The emulator's own collector sweeps
 * are not Clover operations: their untags update the index without traffic.
 *
 * The same trace's lifetime checks (ldst + ldc) are the common denominator.
 *
 * A complete trace ends with an END record (format note in cap_rev_tree.h); a
 * stream that stops without one is truncated -- the emulator was killed or a
 * reader upstream died -- and is an error. --no-end accepts traces recorded
 * before END existed.
 *
 *   cc -O2 -o clovsim clovsim.c
 *   ./clovsim [--max-node N] trace|-      (JSON on stdout)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K_READ, K_WRITE, K_ALLOC, K_FREE, K_RESET, K_REPEAT, K_ALIAS_INC, K_ALIAS_DEC, K_ALIAS_REG,
       K_ALIAS_SLOT, K_END };
enum { S_LDST, S_LDC, S_MREV, S_SPLIT, S_REVOKE, S_DELIN, S_CREATE, S_SUPERVISOR, S_GC,
       S_MEM_CAPSTORE, S_MEM_UNTAG, S_MEM_CLEAR, S_DROP, N_SITES };
#define NONE 0xffffffffu

#include "nested_lru.h"

/* ---- line ids ---- */
#define R_CAPNODE 0ull
#define R_NODE 1ull
#define R_ALIAS 2ull
#define R_SIDE 3ull
#define R_RADIX1 4ull
#define R_RADIX2 5ull
#define LINE(r, v) (((r) << 56) | (uint64_t)(v))

static struct lru *cap, *clo;
/* Clover traffic by operation */
enum { OP_STORE_REGISTER, OP_STORE_UNREGISTER, OP_DATA_UNREGISTER, OP_REVOKE, OP_TREE, N_OPS };
static const char *op_name[N_OPS] = {"store_register", "store_unregister", "data_store_unregister",
                                     "revoke", "tree"};
static uint64_t op_acc[N_OPS], op_miss64[N_OPS];
static int cur_op;
static void cl(uint64_t line) {
    uint64_t b1 = clo->hist[0] + clo->hist[1];   /* hits within 64 lines */
    touch(clo, line);
    ++op_acc[cur_op];
    if (clo->hist[0] + clo->hist[1] == b1) ++op_miss64[cur_op];
}

/* ---- Clover index ---- */
static uint32_t max_node = 16777216;
static uint32_t *alias_head;                 /* node -> record */
static uint32_t *rec_node, *rec_slot, *rec_prev, *rec_next, rec_cap, rec_hwm, rec_free = NONE, rec_live, rec_peak;
/* slot -> record: hash keyed by granule */
#define SB 26
static uint32_t *skey, *sval;                /* open addressing, val = record+1 */
static uint32_t sh(uint32_t g) { return (uint32_t)(((uint64_t)g * 0x9E3779B97F4A7C15ull) >> (64 - SB)); }
static uint32_t sget(uint32_t g) {
    for (uint32_t i = sh(g);; i = (i + 1) & ((1u << SB) - 1)) {
        if (!sval[i]) return 0;
        if (skey[i] == g) return sval[i];
    }
}
static void sput(uint32_t g, uint32_t v) {
    uint32_t i = sh(g);
    while (sval[i] && skey[i] != g) i = (i + 1) & ((1u << SB) - 1);
    skey[i] = g; sval[i] = v;
}
static void sdel(uint32_t g) {
    uint32_t mask = (1u << SB) - 1, i = sh(g);
    while (skey[i] != g || !sval[i]) i = (i + 1) & mask;
    sval[i] = 0;
    for (uint32_t j = (i + 1) & mask; sval[j]; j = (j + 1) & mask) {
        uint32_t home = sh(skey[j]);
        if (((j - home) & mask) >= ((j - i) & mask)) { skey[i] = skey[j]; sval[i] = sval[j]; sval[j] = 0; i = j; }
    }
}
/* frames with at least one registered slot: they need a sidecar */
static uint32_t *fkey, *fval;
#define FB 22
static uint64_t frames_live, frames_peak;
static uint32_t fh(uint32_t f) { return (uint32_t)(((uint64_t)f * 0x9E3779B97F4A7C15ull) >> (64 - FB)); }
static void frame_add(uint32_t g, int d) {
    uint32_t f = g >> 8, mask = (1u << FB) - 1, i = fh(f);
    while (fval[i] && fkey[i] != f) i = (i + 1) & mask;
    if (!fval[i]) { fkey[i] = f; fval[i] = 0; }
    if (d > 0) {
        if (fval[i]++ == 0) { if (++frames_live > frames_peak) frames_peak = frames_live; }
        /* fval 0 is "empty": a live frame keeps a count >= 1 */
    } else if (--fval[i] == 0) {
        --frames_live;
        /* leave the key with value 0 = empty slot; rehash the cluster */
        for (uint32_t j = (i + 1) & mask; fval[j]; j = (j + 1) & mask) {
            uint32_t home = fh(fkey[j]);
            if (((j - home) & mask) >= ((j - i) & mask)) { fkey[i] = fkey[j]; fval[i] = fval[j]; fval[j] = 0; i = j; }
        }
    }
}

static void walk_dir(uint32_t g) {
    uint32_t f = g >> 8;
    cl(LINE(R_RADIX1, f >> 12));       /* top table: one 8-byte entry per 512 frames */
    cl(LINE(R_RADIX2, f >> 3));        /* leaf table: one 8-byte entry per frame */
}
static uint64_t side_line(uint32_t g) { return LINE(R_SIDE, ((uint64_t)(g >> 8) << 4) | ((g & 255) >> 4)); }

static uint32_t reserve(void) {
    uint32_t e;
    if (rec_free != NONE) { e = rec_free; rec_free = rec_next[e]; }
    else {
        if (rec_hwm == rec_cap) { fprintf(stderr, "clovsim: alias pool model full\n"); exit(1); }
        e = rec_hwm++;
    }
    cl(LINE(R_ALIAS, e / 4));          /* pop: read the record's next field */
    return e;
}
static void unregister(uint32_t g, int with_walk) {
    uint32_t e = sget(g) - 1, n = rec_node[e];
    if (with_walk) { walk_dir(g); cl(side_line(g)); }
    cl(LINE(R_ALIAS, e / 4));
    uint32_t p = rec_prev[e], q = rec_next[e];
    if (p == NONE) { alias_head[n] = q; cl(LINE(R_NODE, n / 2)); }
    else { rec_next[p] = q; cl(LINE(R_ALIAS, p / 4)); }
    if (q != NONE) { rec_prev[q] = p; cl(LINE(R_ALIAS, q / 4)); }
    cl(side_line(g));
    sdel(g);
    rec_next[e] = rec_free; rec_free = e;
    cl(LINE(R_ALIAS, e / 4));
    --rec_live;
    frame_add(g, -1);
}
static void reg(uint32_t n, uint32_t g, uint32_t e) {
    cl(LINE(R_NODE, n / 2));
    uint32_t h = alias_head[n];
    rec_node[e] = n; rec_slot[e] = g; rec_prev[e] = NONE; rec_next[e] = h;
    cl(LINE(R_ALIAS, e / 4));
    if (h != NONE) { rec_prev[h] = e; cl(LINE(R_ALIAS, h / 4)); }
    alias_head[n] = e;
    cl(LINE(R_NODE, n / 2));
    walk_dir(g);
    cl(side_line(g));
    sput(g, e + 1);
    if (++rec_live > rec_peak) rec_peak = rec_live;
    frame_add(g, +1);
}
/* silent removal: the emulator's collector, not a Clover operation */
static void drop_silently(uint32_t g) {
    uint32_t e = sget(g) - 1, n = rec_node[e];
    uint32_t p = rec_prev[e], q = rec_next[e];
    if (p == NONE) alias_head[n] = q; else rec_next[p] = q;
    if (q != NONE) rec_prev[q] = p;
    sdel(g);
    rec_next[e] = rec_free; rec_free = e;
    --rec_live;
    frame_add(g, -1);
}

/* ---- tree mirror (as aliasstat.c) ---- */
static uint32_t *prv, *nxt, *depth;
static uint8_t *valid;

/* histogram of Clover accesses per revoke: 0..8 exact, then powers of two */
#define NB 32
static uint64_t rev_hist[NB], rev_count, rev_sum, rev_max, tag_clears, rev_aliases;
static unsigned bucket(uint64_t v) {
    if (v <= 8) return (unsigned)v;
    unsigned b = 9; uint64_t lim = 16;
    while (v > lim && b < NB - 1) { lim <<= 1; ++b; }
    return b;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    int no_end = 0, saw_end = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-end")) no_end = 1;
        else if (!strcmp(argv[i], "--max-node") && i + 1 < argc) max_node = strtoul(argv[++i], NULL, 0);
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: clovsim [--max-node N] trace|-\n"); return 2; }
        else path = argv[i];
    }
    if (!path) { fprintf(stderr, "usage: clovsim [--max-node N] trace|-\n"); return 2; }
    FILE *in = strcmp(path, "-") ? fopen(path, "rb") : stdin;
    if (!in) { perror(path); return 1; }
    static char inbuf[1 << 22];
    setvbuf(in, inbuf, _IOFBF, sizeof inbuf);
    char magic[8];
    if (fread(magic, 1, 8, in) != 8 || (memcmp(magic, "CRNTRC01", 8) && memcmp(magic, "CRNTRC02", 8))) {
        fprintf(stderr, "clovsim: %s: no trace header\n", path);
        return 1;
    }
    size_t N = (size_t)max_node + 1;
    cap = calloc(1, sizeof *cap); clo = calloc(1, sizeof *clo);
    lru_flush(cap); lru_flush(clo);
    alias_head = malloc(N * 4); memset(alias_head, 0xff, N * 4);
    rec_cap = 1u << 26;
    rec_node = malloc((size_t)rec_cap * 4); rec_slot = malloc((size_t)rec_cap * 4);
    rec_prev = malloc((size_t)rec_cap * 4); rec_next = malloc((size_t)rec_cap * 4);
    skey = malloc((size_t)4 << SB); sval = calloc((size_t)1 << SB, 4);
    fkey = malloc((size_t)4 << FB); fval = calloc((size_t)1 << FB, 4);
    prv = malloc(N * 4); nxt = malloc(N * 4); depth = calloc(N, 4); valid = calloc(N, 1);
    uint32_t last_read[N_SITES];
    for (int s = 0; s < N_SITES; ++s) last_read[s] = NONE;

    uint64_t checks = 0, stores_same = 0, stores_new = 0, stores_stale = 0, data_unreg = 0;
    uint64_t gc_untags = 0, cap_by_site[N_SITES] = {0}, orphaned = 0;
    uint32_t slot = NONE;
    int in_revoke = 0;
    uint32_t root = NONE;
    unsigned last_kind = 0, last_site = 0;
    uint32_t last_id = 0;
    int have_last = 0;
    uint8_t r[8];
    uint64_t n = 0;
    size_t got;
    while ((got = fread(r, 1, 8, in)) == 8) {
        ++n;
        uint32_t id = r[0] | r[1] << 8 | r[2] << 16 | (uint32_t)r[3] << 24;
        unsigned kind = r[4], site = r[5];
        if (saw_end) { fprintf(stderr, "clovsim: record after END\n"); return 1; }
        if (kind == K_END) { saw_end = 1; continue; }
        if (kind == K_REPEAT) {
            if (!have_last) { fprintf(stderr, "clovsim: REPEAT first\n"); return 1; }
            /* a repeated node access: same line again, a hit in Capstone's cache */
            if (last_kind <= K_FREE && last_id != NONE) {
                /* the line is the most recent one: a hit in every size, nothing moves */
                cap->accesses += id; cap->hist[0] += id;
                cap_by_site[last_site] += id;
                if (last_kind == K_READ && (last_site == S_LDST || last_site == S_LDC)) checks += id;
            } else if (last_kind <= K_FREE && last_id == NONE &&
                       (last_site == S_LDST || last_site == S_LDC)) {
                checks += id;
            } else if (last_kind == K_ALIAS_REG) {
                /* several registers holding the same node: a snapshot, no traffic */
            } else if (last_kind >= K_ALIAS_INC) {
                fprintf(stderr, "clovsim: record %llu: repeated alias record\n", (unsigned long long)n);
                return 1;
            }
            continue;
        }
        have_last = kind != K_RESET;
        last_kind = kind; last_site = site; last_id = id;
        if (kind > K_ALIAS_SLOT || site >= N_SITES) {
            fprintf(stderr, "clovsim: record %llu: bad kind %u site %u\n", (unsigned long long)n, kind, site);
            return 1;
        }
        if (kind == K_ALIAS_SLOT) { slot = id; continue; }
        if (kind == K_ALIAS_REG) continue;
        if (kind == K_RESET) {
            lru_flush(cap); lru_flush(clo);
            memset(sval, 0, (size_t)4 << SB); memset(fval, 0, (size_t)4 << FB);
            memset(alias_head, 0xff, N * 4);
            rec_hwm = 0; rec_free = NONE; rec_live = 0; frames_live = 0;
            memset(valid, 0, N);
            in_revoke = 0; slot = NONE;
            continue;
        }
        if (id != NONE && id > max_node && kind != K_ALIAS_SLOT) {
            fprintf(stderr, "clovsim: node %u above --max-node\n", id);
            return 1;
        }
        if (kind == K_ALIAS_DEC) {
            if (slot == NONE) { fprintf(stderr, "clovsim: record %llu: DEC without a slot\n", (unsigned long long)n); return 1; }
            if (site == S_MEM_UNTAG && sget(slot)) {
                cur_op = OP_DATA_UNREGISTER; unregister(slot, 1); ++data_unreg;
            } else if ((site == S_GC || site == S_MEM_CLEAR) && sget(slot)) {
                drop_silently(slot); ++gc_untags;
            }
            /* a DEC at a capability store is followed by the INC; handled there */
            continue;
        }
        if (kind == K_ALIAS_INC) {
            if (slot == NONE) { fprintf(stderr, "clovsim: record %llu: INC without a slot\n", (unsigned long long)n); return 1; }
            uint32_t e1 = sget(slot);
            if (!valid[id]) {
                /* Clover cleared this capability's tag at its revoke: a data store */
                ++stores_stale;
                if (e1) { cur_op = OP_DATA_UNREGISTER; unregister(slot, 1); }
            } else if (e1 && rec_node[e1 - 1] == id) {
                ++stores_same;
            } else {
                ++stores_new;
                cur_op = OP_STORE_REGISTER;
                uint32_t e = reserve();
                if (e1) { cur_op = OP_STORE_UNREGISTER; unregister(slot, 1); cur_op = OP_STORE_REGISTER; }
                reg(id, slot, e);
            }
            slot = NONE;
            continue;
        }

        /* node accesses: Capstone's traffic as traced */
        if (id == NONE) {
            if (kind == K_READ && (site == S_LDST || site == S_LDC)) ++checks;
            continue;
        }
        touch(cap, LINE(R_CAPNODE, id / 4));
        ++cap_by_site[site];
        if (kind == K_READ && (site == S_LDST || site == S_LDC)) ++checks;

        /* the tree, for Clover's revokes and node writes */
        if (kind == K_READ) last_read[site] = id;
        if (kind == K_ALLOC) {
            cur_op = OP_TREE;
            if ((site == S_MREV || site == S_SPLIT) && last_read[site] != NONE && valid[last_read[site]]) {
                uint32_t src = last_read[site];
                uint32_t before = prv[src];
                depth[id] = depth[src]; prv[id] = prv[src];
                if (prv[src] != NONE) nxt[prv[src]] = id;
                nxt[id] = src; prv[src] = id; valid[id] = 1;
                if (site == S_MREV) ++depth[src];
                cl(LINE(R_NODE, id / 2)); cl(LINE(R_NODE, src / 2));
                if (before != NONE) cl(LINE(R_NODE, before / 2));
            } else {
                prv[id] = nxt[id] = NONE; depth[id] = 0; valid[id] = 1;
                cl(LINE(R_NODE, id / 2));
            }
            if (alias_head[id] != NONE) ++orphaned;   /* records left under a reused id */
            alias_head[id] = NONE;
        } else if (site == S_REVOKE && kind == K_READ && !in_revoke && valid[id]) {
            in_revoke = 1; root = id;
            cur_op = OP_REVOKE;
            uint64_t before = clo->accesses;
            cl(LINE(R_NODE, root / 2));
            uint32_t c = nxt[root];
            while (c != NONE && depth[c] > depth[root]) {
                cl(LINE(R_NODE, c / 2));
                for (uint32_t e = alias_head[c]; e != NONE;) {
                    uint32_t nx = rec_next[e], g = rec_slot[e];
                    cl(LINE(R_ALIAS, e / 4));
                    ++tag_clears; ++rev_aliases;
                    cl(side_line(g));
                    sdel(g);
                    rec_next[e] = rec_free; rec_free = e; --rec_live;
                    cl(LINE(R_ALIAS, e / 4));
                    frame_add(g, -1);
                    e = nx;
                }
                alias_head[c] = NONE;
                valid[c] = 0;
                cl(LINE(R_NODE, c / 2));
                c = nxt[c];
            }
            nxt[root] = c;
            if (c != NONE) prv[c] = root;
            cl(LINE(R_NODE, root / 2));
            uint64_t a = clo->accesses - before;
            ++rev_hist[bucket(a)]; ++rev_count; rev_sum += a;
            if (a > rev_max) rev_max = a;
        } else if (site == S_REVOKE && kind == K_WRITE && in_revoke && id == root) {
            in_revoke = 0;
        }
    }
    if (got != 0 || ferror(in)) {
        fprintf(stderr, "clovsim: %s: truncated trace (%zu stray bytes)\n", path, got);
        return 1;
    }
    if (!saw_end && !no_end) {
        fprintf(stderr, "clovsim: %s: no END record; the trace is truncated\n", path);
        return 1;
    }
    if (!checks) { fprintf(stderr, "clovsim: no lifetime checks in the trace\n"); return 1; }
    if (orphaned) { fprintf(stderr, "clovsim: %llu node ids reused with index records left\n",
                            (unsigned long long)orphaned); return 3; }

    printf("{\n  \"trace\": \"%s\",\n  \"records\": %llu,\n  \"lifetime_checks\": %llu,\n", path,
           (unsigned long long)n, (unsigned long long)checks);
    printf("  \"line_bytes\": 64, \"sizes_lines\": [");
    for (int k = 0; k < NSZ; ++k) printf("%s%u", k ? ", " : "", sizes[k]);
    printf("],\n");
    struct lru *L[2] = {cap, clo};
    const char *nm[2] = {"capstone", "clover"};
    for (int d = 0; d < 2; ++d) {
        printf("  \"%s\": {\"accesses\": %llu, \"misses\": [", nm[d], (unsigned long long)L[d]->accesses);
        for (int k = 0; k < NSZ; ++k) {
            uint64_t m = 0;
            for (int b = k + 1; b <= NSZ; ++b) m += L[d]->hist[b];
            printf("%s%llu", k ? ", " : "", (unsigned long long)m);
        }
        printf("]");
        if (d == 0) {
            printf(", \"by_site\": {");
            static const char *sn[N_SITES] = {"ldst", "ldc", "mrev", "split", "revoke", "delin", "create",
                                              "supervisor", "gc", "mem_capstore", "mem_untag", "mem_clear", "drop"};
            for (int s = 0, f = 1; s < N_SITES; ++s) {
                if (!cap_by_site[s]) continue;
                printf("%s\"%s\": %llu", f ? "" : ", ", sn[s], (unsigned long long)cap_by_site[s]); f = 0;
            }
            printf("}},\n");
        } else {
            printf(", \"by_operation\": {");
            for (int o = 0; o < N_OPS; ++o)
                printf("%s\"%s\": {\"accesses\": %llu, \"misses_64_lines\": %llu}", o ? ", " : "", op_name[o],
                       (unsigned long long)op_acc[o], (unsigned long long)op_miss64[o]);
            printf("},\n    \"capability_stores\": {\"index_unchanged_same_node\": %llu, \"index_updated\": %llu,"
                   " \"of_a_revoked_capability\": %llu}, \"data_stores_unregistering\": %llu,"
                   " \"collector_untags_ignored\": %llu,\n",
                   (unsigned long long)stores_same, (unsigned long long)stores_new,
                   (unsigned long long)stores_stale, (unsigned long long)data_unreg,
                   (unsigned long long)gc_untags);
            printf("    \"revokes\": {\"count\": %llu, \"tag_clears\": %llu, \"accesses_mean\": %.3f,"
                   " \"accesses_max\": %llu, \"accesses_hist\": {",
                   (unsigned long long)rev_count, (unsigned long long)tag_clears,
                   rev_count ? (double)rev_sum / rev_count : 0.0, (unsigned long long)rev_max);
            for (unsigned b = 0, f = 1; b < NB; ++b) {
                if (!rev_hist[b]) continue;
                if (b <= 8) printf("%s\"%u\": %llu", f ? "" : ", ", b, (unsigned long long)rev_hist[b]);
                else printf("%s\"%llu-%llu\": %llu", f ? "" : ", ", (1ull << (b - 6)) + 1, 1ull << (b - 5),
                            (unsigned long long)rev_hist[b]);
                f = 0;
            }
            printf("}},\n    \"alias_records_peak\": %u, \"alias_records_end\": %u, \"sidecar_frames_peak\": %llu}\n",
                   rec_peak, rec_live, (unsigned long long)frames_peak);
        }
    }
    printf("}\n");
    (void)rev_aliases;
    return 0;
}
