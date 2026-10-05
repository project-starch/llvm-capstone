/* Revocation-node cache simulator.
 *
 * Reads a node-access trace written by capstone-qemu with
 * CAPSTONE_REVNODE_TRACE=<file> (format: target/riscv/cap_rev_tree.h) and
 * replays it against a set of node caches in one pass, all LRU:
 *
 *   fully associative at 8 .. 65536 entries, and direct-mapped, 4- and 8-way
 *   set-associative at 16 .. 4096 entries; an entry holds one line of
 *   --nodes-per-line consecutive node ids (default 1).
 *
 * The fully associative sizes are simulated together: LRU caches nest (a
 * smaller one holds the most recent lines of a larger one), so one LRU list
 * with a marker at each size boundary gives every size's hit or miss from
 * the position of the accessed line -- O(sizes) per access, not O(sizes) list
 * operations. An access to the same line as the access before it hits in
 * every cache and changes no replacement state, and is counted without
 * touching the caches.
 *
 * Every traced access goes through the cache: lifetime-check reads, the
 * reads and writes of mrev/split/revoke/delin, allocation (the new node is
 * written) and free-list pushes. A write miss allocates the line
 * (write-allocate). A RESET record (tree re-initialised) empties every
 * cache. A check of a capability with no node (id 0xffffffff) reads
 * nothing and is only counted. A REPEAT record (format CRNTRC02) stands for
 * <id> more copies of the record before it: each is a hit in every cache
 * and leaves LRU order as it is.
 *
 * A complete trace ends with an END record (format note in cap_rev_tree.h); a
 * stream that stops without one is truncated -- the emulator was killed or a
 * reader upstream died -- and is an error. --no-end accepts traces recorded
 * before END existed.
 *
 * --exclude SITE[,SITE...] drops sites before the caches see them, e.g.
 * "gc,supervisor" to leave out the emulated supervisor's own node reads.
 *
 * Output: one JSON object on stdout.
 *
 * The trace is read once, front to back, so it can be a pipe: "-" reads
 * stdin, and a FIFO given as CAPSTONE_REVNODE_TRACE lets a run that would
 * write hundreds of GB be simulated without storing it. --max-node bounds
 * the node ids (default 16777216, the largest pool the runs configure).
 *
 *   cc -O2 -o cachesim cachesim.c
 *   ./cachesim [--max-node N] [--nodes-per-line N] [--exclude gc,supervisor] trace.bin|-
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K_READ, K_WRITE, K_ALLOC, K_FREE, K_RESET, N_KINDS, K_REPEAT = N_KINDS,
       K_ALIAS_INC, K_ALIAS_DEC, K_ALIAS_REG, K_ALIAS_SLOT,  /* alias records: skipped here */
       K_END };
static const char *kind_name[N_KINDS] = {"read", "write", "alloc", "free", "reset"};

#define N_SITES 13
static const char *site_name[N_SITES] = {"ldst", "ldc", "mrev", "split", "revoke",
                                         "delin", "create", "supervisor", "gc",
                                         "mem_capstore", "mem_untag", "mem_clear", "drop"};

#define NODE_NONE 0xffffffffu
#define REC_SIZE 8

/* Set-associative LRU cache. */
struct cache {
    unsigned entries, ways, sets;
    uint32_t *tag;                /* sets*ways line ids, NODE_NONE = empty */
    uint64_t *stamp;              /* last use, for LRU */
    uint64_t hit[N_SITES], miss[N_SITES];
    uint64_t read_hit, read_miss;
};

static uint64_t now;

static void cache_flush(struct cache *c) {
    for (unsigned i = 0; i < c->entries; ++i) c->tag[i] = NODE_NONE;
}

/* Returns 1 on a hit. */
static int cache_access(struct cache *c, uint32_t line) {
    unsigned set = line % c->sets;
    uint32_t *t = &c->tag[set * c->ways];
    uint64_t *st = &c->stamp[set * c->ways];
    for (unsigned w = 0; w < c->ways; ++w) {
        if (t[w] == line) {
            st[w] = now;
            return 1;
        }
    }
    /* Miss: an empty way if there is one, else the least recently used. */
    unsigned victim = 0;
    for (unsigned w = 0; w < c->ways; ++w) {
        if (t[w] == NODE_NONE) { victim = w; break; }
        if (st[w] < st[victim]) victim = w;
    }
    t[victim] = line;
    st[victim] = now;
    return 0;
}

/* All fully associative LRU sizes at once. One LRU list of FULL_MAX slots,
 * most recent first; bucket_of[slot] is the smallest size index k with the
 * slot's position < full_sizes[k], and marker[k] is the slot at position
 * full_sizes[k] - 1 (NODE_NONE while the list is shorter). An access in
 * bucket b hits in every size k >= b; a line not in the list (bucket
 * N_FULL) misses in all. hist[b][site] counts accesses by bucket. */
static const unsigned full_sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                                      8192, 16384, 32768, 65536};
enum { N_FULL = sizeof(full_sizes) / sizeof(full_sizes[0]) };
#define FULL_MAX 65536

static struct {
    uint32_t *slot_of;             /* line -> slot+1, 0 = absent */
    uint32_t tag[FULL_MAX], prev[FULL_MAX], next[FULL_MAX];
    uint8_t bucket_of[FULL_MAX];
    uint32_t marker[N_FULL];
    uint32_t head, tail, count;
    uint64_t hist[N_FULL + 1][N_SITES], read_hist[N_FULL + 1];
} full;

static void full_flush(void) {
    for (uint32_t i = 0; i < full.count; ++i) full.slot_of[full.tag[i]] = 0;
    full.count = 0;
    full.head = full.tail = NODE_NONE;
    for (int k = 0; k < N_FULL; ++k) full.marker[k] = NODE_NONE;
}

static void full_unlink(uint32_t s) {
    if (full.prev[s] != NODE_NONE) full.next[full.prev[s]] = full.next[s]; else full.head = full.next[s];
    if (full.next[s] != NODE_NONE) full.prev[full.next[s]] = full.prev[s]; else full.tail = full.prev[s];
}

static void full_push_front(uint32_t s) {
    full.prev[s] = NODE_NONE;
    full.next[s] = full.head;
    if (full.head != NODE_NONE) full.prev[full.head] = s;
    full.head = s;
    if (full.tail == NODE_NONE) full.tail = s;
    full.bucket_of[s] = 0;
}

/* Every line above each boundary k < upto moves one position down: the one
 * at full_sizes[k] - 1 crosses into bucket k + 1. */
static void full_shift_markers(int upto) {
    for (int k = 0; k < upto; ++k) {
        uint32_t m = full.marker[k];
        if (m == NODE_NONE) break;   /* list shorter than this size, and every larger one */
        full.bucket_of[m] = k + 1;
        full.marker[k] = full.prev[m];
    }
}

/* Returns the bucket of the access (N_FULL = miss everywhere). */
static unsigned full_access(uint32_t line) {
    uint32_t s1 = full.slot_of[line];
    if (s1) {
        uint32_t s = s1 - 1;
        unsigned b = full.bucket_of[s];
        if (s == full.head) return b;
        full_shift_markers(b);
        if (full.marker[b] == s) full.marker[b] = full.prev[s];
        full_unlink(s);
        full_push_front(s);
        return b;
    }
    uint32_t s;
    if (full.count == FULL_MAX) {
        s = full.tail;               /* at FULL_MAX - 1: marker of the largest size */
        full.marker[N_FULL - 1] = full.prev[s];
        full_unlink(s);
        full.slot_of[full.tag[s]] = 0;
        full_shift_markers(N_FULL - 1);
    } else {
        s = full.count++;
        full_shift_markers(N_FULL);
        for (int k = 0; k < N_FULL; ++k)
            if (full.count == full_sizes[k]) full.marker[k] = full.tail == NODE_NONE ? s : full.tail;
    }
    full.tag[s] = line;
    full.slot_of[line] = s + 1;
    full_push_front(s);
    return N_FULL;
}

static void usage(void) {
    fprintf(stderr, "usage: cachesim [--max-node N] [--nodes-per-line N] [--exclude site,...] trace.bin|-\n");
    exit(2);
}

int main(int argc, char **argv) {
    unsigned npl = 1;
    uint32_t max_node = 16777216; /* largest CAPSTONE_REV_NODES the runs use */
    int no_end = 0, saw_end = 0;
    int excluded[N_SITES] = {0};
    const char *path = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-end")) {
            no_end = 1;
        } else if (!strcmp(argv[i], "--max-node") && i + 1 < argc) {
            max_node = strtoul(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--nodes-per-line") && i + 1 < argc) {
            npl = strtoul(argv[++i], NULL, 0);
            if (!npl) usage();
        } else if (!strcmp(argv[i], "--exclude") && i + 1 < argc) {
            char *list = strdup(argv[++i]);
            for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
                int found = 0;
                for (int s = 0; s < N_SITES; ++s)
                    if (!strcmp(tok, site_name[s])) excluded[s] = found = 1;
                if (!found) { fprintf(stderr, "cachesim: unknown site %s\n", tok); return 2; }
            }
        } else if (argv[i][0] == '-' && argv[i][1]) {
            usage();
        } else {
            path = argv[i];
        }
    }
    if (!path) usage();

    FILE *in = strcmp(path, "-") ? fopen(path, "rb") : stdin;
    if (!in) { perror(path); return 1; }
    static char inbuf[1 << 22];
    setvbuf(in, inbuf, _IOFBF, sizeof(inbuf));
    char magic[8];
    if (fread(magic, 1, 8, in) != 8 ||
        (memcmp(magic, "CRNTRC01", 8) && memcmp(magic, "CRNTRC02", 8))) {
        fprintf(stderr, "cachesim: %s: no trace header (empty or not a node trace)\n", path);
        return 1;
    }
    uint32_t max_id = 0;
    uint32_t max_line = max_node / npl;

    static const unsigned sizes[] = {16, 64, 256, 1024, 4096};
    static const unsigned waysv[] = {1, 4, 8};
    enum { NS = sizeof(sizes) / sizeof(sizes[0]), NW = sizeof(waysv) / sizeof(waysv[0]) };
    struct cache caches[NS * NW];
    int nc = 0;
    for (int si = 0; si < NS; ++si) {
        for (int wi = 0; wi < NW; ++wi) {
            struct cache *c = &caches[nc++];
            memset(c, 0, sizeof(*c));
            c->entries = sizes[si];
            c->ways = waysv[wi];
            c->sets = c->entries / c->ways;
            c->tag = calloc(c->entries, sizeof(uint32_t));
            c->stamp = calloc(c->entries, sizeof(uint64_t));
            cache_flush(c);
        }
    }
    full.slot_of = calloc((size_t)max_line + 1, sizeof(uint32_t));
    full.count = 0;
    full_flush();
    uint32_t prev_line = NODE_NONE;   /* the line of the access before, for the fast path */

    uint64_t count[N_KINDS][N_SITES] = {{0}};
    uint64_t no_node_checks[N_SITES] = {0};
    uint64_t resets = 0, excluded_recs = 0;
    uint8_t *seen = calloc((size_t)max_node + 1, 1);
    /* Compulsory misses: the first access to a line since the last reset
     * misses in every cache, whatever its size. */
    uint8_t *line_seen = calloc((size_t)max_line + 1, 1);
    uint64_t compulsory[N_SITES] = {0};
    uint64_t distinct = 0;

    /* Revoke walk lengths. capstone-qemu traces a revoke as
     *   R root, (R n, W n)*, [R end], W root, [W end]
     * so a walked node is a WRITE of another node between root's read and
     * root's write; the optional R end is the node that stopped the walk
     * and the trailing W end, outside the walk, is its relink. A READ at
     * the revoke site outside a walk starts the next one.
     * Buckets: 0,1,2,3-4,5-8,...,>256. */
    uint64_t walk_hist[11] = {0}, walks = 0, walked_total = 0, walk_max = 0;
    int in_revoke = 0;
    uint32_t revoke_root = 0;
    uint64_t walked = 0;
    uint64_t logical = 0, alias_recs = 0;
    /* The record a REPEAT repeats. */
    int have_last = 0;
    uint32_t last_id = 0;
    unsigned last_kind = 0, last_site = 0;

    uint8_t r[REC_SIZE];
    uint64_t n = 0;
    size_t got;
    for (uint64_t i = 0; (got = fread(r, 1, REC_SIZE, in)) == REC_SIZE; ++i, ++n) {
        uint32_t id = r[0] | r[1] << 8 | r[2] << 16 | (uint32_t)r[3] << 24;
        unsigned kind = r[4], site = r[5];
        if (saw_end) {
            fprintf(stderr, "cachesim: record %llu after END\n", (unsigned long long)i);
            return 1;
        }
        if (kind == K_END) { saw_end = 1; continue; }
        if (kind == K_REPEAT) {
            if (!have_last) {
                fprintf(stderr, "cachesim: record %llu: REPEAT with nothing before it\n",
                        (unsigned long long)i);
                return 1;
            }
            logical += id;
            if (last_kind >= K_ALIAS_INC) { alias_recs += id; continue; }
            if (excluded[last_site]) { excluded_recs += id; continue; }
            if (last_id == NODE_NONE) { no_node_checks[last_site] += id; continue; }
            count[last_kind][last_site] += id;
            for (int k = 0; k < nc; ++k) {
                caches[k].hit[last_site] += id;
                if (last_kind == K_READ) caches[k].read_hit += id;
            }
            full.hist[0][last_site] += id;
            if (last_kind == K_READ) full.read_hist[0] += id;
            continue;
        }
        ++logical;
        if (kind >= K_ALIAS_INC && kind <= K_ALIAS_SLOT && site < N_SITES) {
            ++alias_recs;
            have_last = 1;
            last_id = id; last_kind = kind; last_site = site;
            continue;
        }
        if (kind >= N_KINDS || site >= N_SITES) {
            fprintf(stderr, "cachesim: record %llu: bad kind %u / site %u\n",
                    (unsigned long long)i, kind, site);
            return 1;
        }
        have_last = kind != K_RESET;
        last_id = id; last_kind = kind; last_site = site;
        if (kind == K_RESET) {
            ++resets;
            for (int k = 0; k < nc; ++k) cache_flush(&caches[k]);
            full_flush();
            prev_line = NODE_NONE;
            memset(line_seen, 0, (size_t)max_line + 1);
            in_revoke = 0;
            continue;
        }
        if (excluded[site]) { ++excluded_recs; continue; }
        if (id == NODE_NONE) { ++no_node_checks[site]; continue; }
        if (id > max_node) {
            fprintf(stderr, "cachesim: record %llu: node %u above --max-node %u\n",
                    (unsigned long long)i, id, max_node);
            return 1;
        }
        if (id > max_id) max_id = id;
        ++count[kind][site];
        if (!seen[id]) { seen[id] = 1; ++distinct; }

        if (site == 4 /* revoke */) {
            if (!in_revoke && kind == K_READ) {
                in_revoke = 1; revoke_root = id; walked = 0;
            } else if (in_revoke && kind == K_WRITE && id != revoke_root) {
                ++walked;
            } else if (in_revoke && kind == K_WRITE && id == revoke_root) {
                in_revoke = 0;
                ++walks; walked_total += walked;
                if (walked > walk_max) walk_max = walked;
                unsigned b = walked == 0 ? 0 : walked == 1 ? 1 : walked == 2 ? 2 : 0;
                if (walked > 2) { b = 3; uint64_t lim = 4; while (walked > lim && b < 10) { lim *= 2; ++b; } }
                ++walk_hist[b];
            }
        }

        ++now;
        uint32_t line = id / npl;
        if (!line_seen[line]) { line_seen[line] = 1; ++compulsory[site]; }
        if (line == prev_line) {
            /* most recent in every cache already: a hit that changes nothing */
            for (int k = 0; k < nc; ++k) {
                ++caches[k].hit[site];
                if (kind == K_READ) ++caches[k].read_hit;
            }
            ++full.hist[0][site];
            if (kind == K_READ) ++full.read_hist[0];
            continue;
        }
        prev_line = line;
        unsigned b = full_access(line);
        ++full.hist[b][site];
        if (kind == K_READ) ++full.read_hist[b];
        for (int k = 0; k < nc; ++k) {
            struct cache *c = &caches[k];
            if (cache_access(c, line)) {
                ++c->hit[site];
                if (kind == K_READ) ++c->read_hit;
            } else {
                ++c->miss[site];
                if (kind == K_READ) ++c->read_miss;
            }
        }
    }

    if (got != 0 || ferror(in)) {
        fprintf(stderr, "cachesim: %s: trace ends inside a record (%zu stray bytes); truncated\n",
                path, got);
        return 1;
    }
    if (!saw_end && !no_end) {
        fprintf(stderr, "cachesim: %s: no END record; the trace is truncated\n", path);
        return 1;
    }

    printf("{\n  \"trace\": \"%s\",\n  \"records\": %llu,\n  \"logical_records\": %llu,\n"
           "  \"nodes_per_line\": %u,\n",
           path, (unsigned long long)n, (unsigned long long)logical, npl);
    printf("  \"excluded_sites\": [");
    for (int s = 0, first = 1; s < N_SITES; ++s)
        if (excluded[s]) { printf("%s\"%s\"", first ? "" : ", ", site_name[s]); first = 0; }
    printf("],\n  \"excluded_records\": %llu,\n", (unsigned long long)excluded_recs);
    printf("  \"alias_records\": %llu,\n", (unsigned long long)alias_recs);
    printf("  \"resets\": %llu,\n  \"max_node_id\": %u,\n  \"distinct_nodes\": %llu,\n",
           (unsigned long long)resets, max_id, (unsigned long long)distinct);
    printf("  \"accesses\": {");
    for (int k = 0; k < N_KINDS; ++k) {
        if (k == K_RESET) continue;
        printf("%s\n    \"%s\": {", k ? "," : "", kind_name[k]);
        for (int s = 0; s < N_SITES; ++s)
            printf("%s\"%s\": %llu", s ? ", " : "", site_name[s], (unsigned long long)count[k][s]);
        printf("}");
    }
    printf("\n  },\n  \"checks_without_node\": {");
    for (int s = 0; s < N_SITES; ++s)
        printf("%s\"%s\": %llu", s ? ", " : "", site_name[s], (unsigned long long)no_node_checks[s]);
    printf("},\n  \"compulsory_misses\": {");
    for (int s = 0; s < N_SITES; ++s)
        printf("%s\"%s\": %llu", s ? ", " : "", site_name[s], (unsigned long long)compulsory[s]);
    printf("},\n  \"revoke_walks\": {\"count\": %llu, \"nodes_walked\": %llu, \"max\": %llu, "
           "\"hist_buckets\": [\"0\",\"1\",\"2\",\"3-4\",\"5-8\",\"9-16\",\"17-32\",\"33-64\",\"65-128\",\"129-256\",\">256\"], "
           "\"hist\": [",
           (unsigned long long)walks, (unsigned long long)walked_total, (unsigned long long)walk_max);
    for (int b = 0; b < 11; ++b) printf("%s%llu", b ? ", " : "", (unsigned long long)walk_hist[b]);
    printf("]},\n  \"caches\": [");
    /* fully associative: size k hits every access in buckets 0..k */
    for (int k = 0; k < N_FULL; ++k) {
        uint64_t hs[N_SITES] = {0}, ms[N_SITES] = {0}, h = 0, m = 0, rh = 0, rm = 0;
        for (int b = 0; b <= N_FULL; ++b) {
            for (int st = 0; st < N_SITES; ++st) {
                if (b <= k) hs[st] += full.hist[b][st]; else ms[st] += full.hist[b][st];
            }
            if (b <= k) rh += full.read_hist[b]; else rm += full.read_hist[b];
        }
        for (int st = 0; st < N_SITES; ++st) { h += hs[st]; m += ms[st]; }
        printf("%s\n    {\"entries\": %u, \"ways\": \"full\", \"ways_n\": %u, \"hits\": %llu, \"misses\": %llu, "
               "\"read_hits\": %llu, \"read_misses\": %llu, \"by_site\": {",
               k ? "," : "", full_sizes[k], full_sizes[k], (unsigned long long)h, (unsigned long long)m,
               (unsigned long long)rh, (unsigned long long)rm);
        for (int st = 0; st < N_SITES; ++st)
            printf("%s\"%s\": [%llu, %llu]", st ? ", " : "", site_name[st],
                   (unsigned long long)hs[st], (unsigned long long)ms[st]);
        printf("}}");
    }
    for (int k = 0; k < nc; ++k) {
        struct cache *c = &caches[k];
        uint64_t h = 0, m = 0;
        for (int s = 0; s < N_SITES; ++s) { h += c->hit[s]; m += c->miss[s]; }
        printf(",\n    {\"entries\": %u, \"ways\": %u, \"hits\": %llu, \"misses\": %llu, "
               "\"read_hits\": %llu, \"read_misses\": %llu, \"by_site\": {",
               c->entries, c->ways, (unsigned long long)h, (unsigned long long)m,
               (unsigned long long)c->read_hit, (unsigned long long)c->read_miss);
        for (int s = 0; s < N_SITES; ++s)
            printf("%s\"%s\": [%llu, %llu]", s ? ", " : "", site_name[s],
                   (unsigned long long)c->hit[s], (unsigned long long)c->miss[s]);
        printf("}}");
    }
    printf("\n  ]\n}\n");
    return 0;
}
