/* Aliases per revocation node, and the shape of the revocation trees.
 *
 * Reads the same trace as cachesim (capstone-qemu, CAPSTONE_REVNODE_TRACE; format
 * in target/riscv/cap_rev_tree.h), from a file or "-", once, front to back.
 *
 * Aliases. An alias is one copy of a capability. The trace says when a
 * capability naming node n is stored into a 16-byte memory granule (ALIAS_INC)
 * and when it leaves it (ALIAS_DEC: overwritten by another capability, untagged
 * by a data store, swept by the collector, or the map reset), so the number of
 * memory aliases of every node is known at every point. Registers are known at
 * each revoke: the trace snapshots the 32 GPRs and the PCC just before it.
 * Reported:
 *   - per node, the most memory aliases it ever had at once (one sample per node
 *     incarnation: an id handed out again starts a new one);
 *   - per revoke, the aliases the revoked run had at that moment: memory and
 *     registers, for the root and for the nodes the revoke invalidated -- the
 *     copies that become stale;
 *   - at each free, the aliases left (the emulator's collector clears stale tags
 *     before it frees a node, so anything but 0 is an instrument defect).
 *
 * Trees. The tool keeps its own copy of the emulator's tree -- the same list of
 * nodes with depths that cap_rev_tree.c keeps -- by replaying mrev (R src, ALLOC
 * new), split, lone-node creation and revoke from the trace. At each revoke it
 * walks its copy exactly as the emulator does and checks the walk against the
 * nodes the trace says were invalidated; a mismatch is counted and must be 0.
 * Reported: per revoke, the root's direct children and the run it invalidates;
 * and a snapshot of every live node's direct children and depth and of each
 * tree's size, every 2^14 allocations and every 2^24 records, whichever comes
 * first (a snapshot costs time in the live nodes only, which are kept in a
 * list). The histograms sum the snapshots, so they are time-sampled. The
 * end-of-run state is not sampled: by then the programs have torn down.
 *
 * A complete trace ends with an END record (format note in cap_rev_tree.h); a
 * stream that stops without one is truncated -- the emulator was killed or a
 * reader upstream died -- and is an error. --no-end accepts traces recorded
 * before END existed.
 *
 * Errors, which exit non-zero rather than print a plausible table: a memory
 * alias count going negative, a walk mismatch, a truncated trace, no header.
 *
 *   cc -O2 -o aliasstat aliasstat.c
 *   ./aliasstat [--max-node N] trace.bin|-      (JSON on stdout)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K_READ, K_WRITE, K_ALLOC, K_FREE, K_RESET, K_REPEAT, K_ALIAS_INC, K_ALIAS_DEC, K_ALIAS_REG,
       K_ALIAS_SLOT, K_END };
enum { S_LDST, S_LDC, S_MREV, S_SPLIT, S_REVOKE, S_DELIN, S_CREATE, S_SUPERVISOR, S_GC,
       S_MEM_CAPSTORE, S_MEM_UNTAG, S_MEM_CLEAR, S_DROP, N_SITES };
static const char *site_name[N_SITES] = {"ldst", "ldc", "mrev", "split", "revoke", "delin",
                                         "create", "supervisor", "gc", "mem_capstore",
                                         "mem_untag", "mem_clear", "drop"};
#define NONE 0xffffffffu

/* Histogram with exact buckets 0..8 and power-of-two buckets above:
 * 0,1,...,8, 9-16, 17-32, ..., >2^30. */
#define NB 32
struct hist { uint64_t n[NB], count, sum, max; };
static unsigned bucket(uint64_t v) {
    if (v <= 8) return (unsigned)v;
    unsigned b = 9;
    uint64_t lim = 16;
    while (v > lim && b < NB - 1) { lim <<= 1; ++b; }
    return b;
}
static void hadd(struct hist *h, uint64_t v) {
    ++h->n[bucket(v)]; ++h->count; h->sum += v;
    if (v > h->max) h->max = v;
}
static void hprint(const char *name, const struct hist *h, int last) {
    printf("    \"%s\": {\"count\": %llu, \"mean\": %.4f, \"max\": %llu, \"buckets\": {",
           name, (unsigned long long)h->count, h->count ? (double)h->sum / h->count : 0.0,
           (unsigned long long)h->max);
    int first = 1;
    for (unsigned b = 0; b < NB; ++b) {
        if (!h->n[b]) continue;
        char label[32];
        if (b <= 8) snprintf(label, sizeof label, "%u", b);
        else snprintf(label, sizeof label, "%llu-%llu", (1ull << (b - 6)) + 1, 1ull << (b - 5));
        printf("%s\"%s\": %llu", first ? "" : ", ", label, (unsigned long long)h->n[b]);
        first = 0;
    }
    printf("}}%s\n", last ? "" : ",");
}

static uint32_t max_node = 16777216;
static uint32_t *prv, *nxt, *depth, *mem, *maxmem;
static uint8_t *valid, *live;           /* live: allocated and not yet freed */
static uint32_t *live_list, *live_pos, live_n;   /* the live nodes, for snapshots */

static void live_add(uint32_t n) { live_pos[n] = live_n; live_list[live_n++] = n; }
static void live_del(uint32_t n) {
    uint32_t last = live_list[--live_n];
    live_list[live_pos[n]] = last;
    live_pos[last] = live_pos[n];
}
static uint64_t mem_total, mem_peak;

static struct hist h_max_alias;          /* per node incarnation: most memory aliases at once */
static struct hist h_rev_root_mem, h_rev_root_reg, h_rev_stale_mem, h_rev_stale_reg;
static struct hist h_rev_children, h_rev_run, h_rev_depth;
static struct hist h_snap_children, h_snap_depth, h_snap_tree_size;
static uint64_t snapshots;
static struct hist h_free_alias;
static uint64_t dec_by_site[N_SITES], inc_total, negative, walk_mismatch, alloc_live_aliases;
static uint64_t mrevs, splits, creates, allocs;
static uint32_t max_seen;
/* capability stores by what they replaced (Clover leaves its index alone for
 * the same node), and data stores that untagged a capability */
static uint64_t store_fresh, store_same, store_other, data_over_tagged;
#define NONE_EV 0xffffffffffffffffull
static uint64_t prev_alias = NONE_EV;   /* kind<<40 | site<<32 | id of the alias record before */

static void finalize(uint32_t n) {     /* close one incarnation of node n */
    if (live[n]) hadd(&h_max_alias, maxmem[n]);
}

static void node_new(uint32_t n) {
    if (n > max_seen) max_seen = n;
    finalize(n);
    if (mem[n]) ++alloc_live_aliases;    /* an id handed out while copies of its old self exist */
    maxmem[n] = mem[n];
    if (!live[n]) live_add(n);
    live[n] = 1; valid[n] = 1;
    prv[n] = nxt[n] = NONE; depth[n] = 0;
    ++allocs;
}

/* cap_rev_tree.c _cap_rev_tree_dup_node_before */
static void dup_before(uint32_t src, uint32_t n) {
    node_new(n);
    depth[n] = depth[src];
    prv[n] = prv[src];
    if (prv[src] != NONE) nxt[prv[src]] = n;
    nxt[n] = src;
    prv[src] = n;
}

static void snapshot(void) {
    /* every live, valid node with no predecessor heads a list; walk it with a
     * stack of ancestors: a node's parent is the nearest earlier node shallower
     * than it (the emulator's revoke takes exactly the deeper run after a node) */
    static uint32_t *stack, *children;
    if (!stack) { stack = malloc(sizeof(uint32_t) * ((size_t)max_node + 1));
                  children = calloc((size_t)max_node + 1, sizeof(uint32_t)); }
    ++snapshots;
    for (uint32_t k = 0; k < live_n; ++k) {
        uint32_t h = live_list[k];
        if (!valid[h] || prv[h] != NONE) continue;
        uint32_t sp = 0, size = 0;
        for (uint32_t n = h; n != NONE; n = nxt[n]) {
            while (sp && depth[stack[sp - 1]] >= depth[n]) --sp;
            if (sp) ++children[stack[sp - 1]];
            stack[sp++] = n;
            ++size;
        }
        hadd(&h_snap_tree_size, size);
        for (uint32_t n = h; n != NONE; n = nxt[n]) {
            hadd(&h_snap_children, children[n]);
            hadd(&h_snap_depth, depth[n]);
            children[n] = 0;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = NULL;
    int no_end = 0, saw_end = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-end")) no_end = 1;
        else if (!strcmp(argv[i], "--max-node") && i + 1 < argc) max_node = strtoul(argv[++i], NULL, 0);
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: aliasstat [--max-node N] trace|-\n"); return 2; }
        else path = argv[i];
    }
    if (!path) { fprintf(stderr, "usage: aliasstat [--max-node N] trace|-\n"); return 2; }
    FILE *in = strcmp(path, "-") ? fopen(path, "rb") : stdin;
    if (!in) { perror(path); return 1; }
    static char inbuf[1 << 22];
    setvbuf(in, inbuf, _IOFBF, sizeof inbuf);
    char magic[8];
    if (fread(magic, 1, 8, in) != 8 || (memcmp(magic, "CRNTRC01", 8) && memcmp(magic, "CRNTRC02", 8))) {
        fprintf(stderr, "aliasstat: %s: no trace header\n", path);
        return 1;
    }
    size_t N = (size_t)max_node + 1;
    prv = malloc(N * 4); nxt = malloc(N * 4); depth = malloc(N * 4);
    mem = calloc(N, 4); maxmem = calloc(N, 4);
    valid = calloc(N, 1); live = calloc(N, 1);
    live_list = malloc(N * 4); live_pos = malloc(N * 4);
    uint32_t *regs = malloc(64 * sizeof(uint32_t));
    unsigned nregs = 0;
    uint32_t last_read[N_SITES];
    for (int s = 0; s < N_SITES; ++s) last_read[s] = NONE;

    /* state of the revoke being consumed: its invalidated run was computed
     * when its root was read; the trace's own records are checked against it */
    int in_revoke = 0;
    uint32_t root = NONE;
    uint64_t expect_walk = 0, seen_walk = 0;
    uint64_t next_snapshot = 1u << 14, next_snapshot_rec = 1u << 24;

    uint8_t r[8];
    unsigned last_kind = 0, last_site = 0;
    uint32_t last_id = 0;
    int have_last = 0;
    uint64_t n = 0;
    size_t got;
    while ((got = fread(r, 1, 8, in)) == 8) {
        ++n;
        uint32_t id = r[0] | r[1] << 8 | r[2] << 16 | (uint32_t)r[3] << 24;
        unsigned kind = r[4], site = r[5];
        uint64_t times = 1;
        if (saw_end) { fprintf(stderr, "aliasstat: record after END\n"); return 1; }
        if (kind == K_END) { saw_end = 1; continue; }
        if (kind == K_REPEAT) {
            if (!have_last) { fprintf(stderr, "aliasstat: REPEAT first\n"); return 1; }
            times = id; kind = last_kind; site = last_site; id = last_id;
            /* only alias records change state when repeated; a repeated node
             * access is the same access again */
            if (kind != K_ALIAS_INC && kind != K_ALIAS_DEC && kind != K_ALIAS_REG) continue;
        } else {
            have_last = kind != K_RESET;
            last_kind = kind; last_site = site; last_id = id;
        }
        if (kind == K_ALIAS_SLOT) {
            /* the granule the next INC/DEC concern; a new store starts here */
            prev_alias = NONE_EV;
            continue;
        }
        if (site >= N_SITES || kind > K_ALIAS_SLOT) {
            fprintf(stderr, "aliasstat: record %llu: bad kind %u site %u\n", (unsigned long long)n, kind, site);
            return 1;
        }
        if (kind != K_RESET && id != NONE && id > max_node) {
            fprintf(stderr, "aliasstat: node %u above --max-node\n", id);
            return 1;
        }
        switch (kind) {
        case K_ALIAS_INC:
            /* what the store replaced: nothing, the same node, or another node */
            if (prev_alias == (((uint64_t)K_ALIAS_DEC << 40) | ((uint64_t)S_MEM_CAPSTORE << 32) | id))
                store_same += times;
            else if ((prev_alias >> 32) == (((uint64_t)K_ALIAS_DEC << 8) | S_MEM_CAPSTORE))
                store_other += times;
            else
                store_fresh += times;
            prev_alias = NONE_EV;
            mem[id] += times; mem_total += times; inc_total += times;
            if (mem[id] > maxmem[id]) maxmem[id] = mem[id];
            if (mem_total > mem_peak) mem_peak = mem_total;
            continue;
        case K_ALIAS_DEC:
            prev_alias = ((uint64_t)K_ALIAS_DEC << 40) | ((uint64_t)site << 32) | id;
            if (site == S_MEM_UNTAG) data_over_tagged += times;
            if (mem[id] < times) { ++negative; mem_total -= mem[id]; mem[id] = 0; continue; }
            mem[id] -= times; mem_total -= times; dec_by_site[site] += times;
            continue;
        case K_ALIAS_REG:
            for (uint64_t t = 0; t < times; ++t)
                if (nregs < 64) regs[nregs++] = id;
            continue;
        case K_RESET:
            for (uint32_t i = 0; i <= max_seen; ++i) { finalize(i); live[i] = valid[i] = 0; }
            live_n = 0;
            in_revoke = 0; nregs = 0;
            continue;
        default:
            break;
        }
        if (id == NONE) { nregs = 0; continue; }

        if (kind == K_READ) last_read[site] = id;
        if (kind == K_ALLOC) {
            if (site == S_MREV || site == S_SPLIT) {
                uint32_t src = last_read[site];
                if (src == NONE || !live[src]) { ++walk_mismatch; node_new(id); }
                else {
                    dup_before(src, id);
                    if (site == S_MREV) { ++depth[src]; ++mrevs; } else ++splits;
                }
            } else {
                node_new(id);
                ++creates;
            }
            if (allocs >= next_snapshot) {
                snapshot(); next_snapshot = allocs + (1u << 14); next_snapshot_rec = n + (1u << 24);
            }
        } else if (kind == K_FREE) {
            hadd(&h_free_alias, mem[id]);
            finalize(id);
            if (live[id]) live_del(id);
            live[id] = valid[id] = 0;
        } else if (site == S_REVOKE) {
            if (!in_revoke && kind == K_READ) {
                /* the emulator's walk: the run after root while deeper than root */
                in_revoke = 1; root = id; seen_walk = 0;
                uint64_t run = 0, stale_mem = 0, children = 0;
                for (uint32_t c = nxt[id]; c != NONE && depth[c] > depth[id]; c = nxt[c]) {
                    ++run;
                    stale_mem += mem[c];
                    if (depth[c] == depth[id] + 1) ++children;
                }
                uint64_t root_reg = 0, stale_reg = 0;
                for (unsigned i = 0; i < nregs; ++i) {
                    if (regs[i] == id) ++root_reg;
                    else {
                        for (uint32_t c = nxt[id]; c != NONE && depth[c] > depth[id]; c = nxt[c])
                            if (regs[i] == c) { ++stale_reg; break; }
                    }
                }
                hadd(&h_rev_root_mem, mem[id]);
                hadd(&h_rev_root_reg, root_reg);
                hadd(&h_rev_stale_mem, stale_mem);
                hadd(&h_rev_stale_reg, stale_reg);
                hadd(&h_rev_children, children);
                hadd(&h_rev_run, run);
                hadd(&h_rev_depth, depth[id]);
                expect_walk = run;
                /* invalidate and splice, as cap_rev_tree_revoke does */
                uint32_t c = nxt[id];
                while (c != NONE && depth[c] > depth[id]) { valid[c] = 0; c = nxt[c]; }
                nxt[id] = c;
                if (c != NONE) prv[c] = id;
            } else if (in_revoke && kind == K_WRITE && id != root) {
                ++seen_walk;
            } else if (in_revoke && kind == K_WRITE && id == root) {
                if (seen_walk != expect_walk) ++walk_mismatch;
                in_revoke = 0;
            }
        }
        nregs = 0;   /* a snapshot belongs to the revoke that follows it directly */
        if (n >= next_snapshot_rec) {
            snapshot(); next_snapshot_rec = n + (1u << 24); next_snapshot = allocs + (1u << 14);
        }
    }
    if (got != 0 || ferror(in)) {
        fprintf(stderr, "aliasstat: %s: truncated trace (%zu stray bytes)\n", path, got);
        return 1;
    }
    if (!saw_end && !no_end) {
        fprintf(stderr, "aliasstat: %s: no END record; the trace is truncated\n", path);
        return 1;
    }
    for (uint32_t i = 0; i <= max_seen; ++i) finalize(i);

    printf("{\n  \"trace\": \"%s\",\n  \"records\": %llu,\n", path, (unsigned long long)n);
    printf("  \"nodes_allocated\": %llu, \"mrev\": %llu, \"split\": %llu, \"lone\": %llu,\n",
           (unsigned long long)allocs, (unsigned long long)mrevs, (unsigned long long)splits,
           (unsigned long long)creates);
    printf("  \"memory_alias_events\": {\"stored\": %llu, \"left\": {", (unsigned long long)inc_total);
    for (int s = S_MEM_CAPSTORE, f = 1; s <= S_MEM_CLEAR; ++s, f = 0)
        printf("%s\"%s\": %llu", f ? "" : ", ", site_name[s], (unsigned long long)dec_by_site[s]);
    printf(", \"gc\": %llu}, \"live_at_end\": %llu, \"peak_live\": %llu},\n",
           (unsigned long long)dec_by_site[S_GC], (unsigned long long)mem_total,
           (unsigned long long)mem_peak);
    printf("  \"capability_stores\": {\"into_untagged\": %llu, \"over_same_node\": %llu,"
           " \"over_other_node\": %llu}, \"data_stores_over_capability\": %llu,\n",
           (unsigned long long)store_fresh, (unsigned long long)store_same,
           (unsigned long long)store_other, (unsigned long long)data_over_tagged);
    printf("  \"errors\": {\"negative_alias_count\": %llu, \"walk_mismatch\": %llu,"
           " \"alloc_with_live_aliases\": %llu},\n",
           (unsigned long long)negative, (unsigned long long)walk_mismatch,
           (unsigned long long)alloc_live_aliases);
    printf("  \"aliases\": {\n");
    hprint("max_memory_aliases_per_node", &h_max_alias, 0);
    hprint("at_revoke_root_memory", &h_rev_root_mem, 0);
    hprint("at_revoke_root_registers", &h_rev_root_reg, 0);
    hprint("at_revoke_invalidated_memory", &h_rev_stale_mem, 0);
    hprint("at_revoke_invalidated_registers", &h_rev_stale_reg, 0);
    hprint("at_free_memory", &h_free_alias, 1);
    printf("  },\n  \"trees\": {\n    \"snapshots\": %llu,\n", (unsigned long long)snapshots);
    hprint("revoke_root_children", &h_rev_children, 0);
    hprint("revoke_run", &h_rev_run, 0);
    hprint("revoke_root_depth", &h_rev_depth, 0);
    hprint("live_node_children", &h_snap_children, 0);
    hprint("live_node_depth", &h_snap_depth, 0);
    hprint("tree_size", &h_snap_tree_size, 1);
    printf("  }\n}\n");
    return (negative || walk_mismatch) ? 3 : 0;
}
