/* Revocation-node cache simulator.
 *
 * Reads a node-access trace written by capstone-qemu with
 * CAPSTONE_REVNODE_TRACE=<file> (format: target/riscv/cap_rev_tree.h) and
 * replays it against a set of node caches in one pass, all LRU:
 *
 *   direct-mapped, 2-, 4- and 8-way set-associative, and fully associative,
 *   each at 8 .. 65536 entries; an entry holds one line of --nodes-per-line
 *   consecutive node ids (default 1).
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

enum { K_READ, K_WRITE, K_ALLOC, K_FREE, K_RESET, N_KINDS, K_REPEAT = N_KINDS };
static const char *kind_name[N_KINDS] = {"read", "write", "alloc", "free", "reset"};

#define N_SITES 9
static const char *site_name[N_SITES] = {"ldst", "ldc", "mrev", "split", "revoke",
                                         "delin", "create", "supervisor", "gc"};

#define NODE_NONE 0xffffffffu
#define REC_SIZE 8

struct cache {
    unsigned entries, ways, sets; /* ways == entries: fully associative */
    uint32_t *tag;                /* sets*ways line ids, NODE_NONE = empty */
    uint64_t *stamp;              /* last use, for LRU (set-associative) */
    /* fully associative: index by line id into a doubly linked LRU list */
    uint32_t *slot_of;            /* line -> slot+1, 0 = absent */
    uint32_t *prev, *next;        /* per slot */
    uint32_t head, tail, used;
    uint64_t hit[N_SITES], miss[N_SITES];
    uint64_t read_hit, read_miss;
};

static uint64_t now;

static void cache_flush(struct cache *c, uint32_t max_line) {
    (void)max_line;
    if (c->ways == c->entries) {
        for (uint32_t s = 0; s < c->used; ++s) c->slot_of[c->tag[s]] = 0;
        c->used = 0;
        c->head = c->tail = NODE_NONE;
    } else {
        for (unsigned i = 0; i < c->entries; ++i) c->tag[i] = NODE_NONE;
    }
}

static void lru_unlink(struct cache *c, uint32_t s) {
    if (c->prev[s] != NODE_NONE) c->next[c->prev[s]] = c->next[s]; else c->head = c->next[s];
    if (c->next[s] != NODE_NONE) c->prev[c->next[s]] = c->prev[s]; else c->tail = c->prev[s];
}

static void lru_push_front(struct cache *c, uint32_t s) {
    c->prev[s] = NODE_NONE;
    c->next[s] = c->head;
    if (c->head != NODE_NONE) c->prev[c->head] = s;
    c->head = s;
    if (c->tail == NODE_NONE) c->tail = s;
}

/* Returns 1 on a hit. */
static int cache_access(struct cache *c, uint32_t line) {
    if (c->ways == c->entries) {
        uint32_t s1 = c->slot_of[line];
        if (s1) {
            lru_unlink(c, s1 - 1);
            lru_push_front(c, s1 - 1);
            return 1;
        }
        uint32_t s;
        if (c->used < c->entries) {
            s = c->used++;
        } else {
            s = c->tail;
            lru_unlink(c, s);
            c->slot_of[c->tag[s]] = 0;
        }
        c->tag[s] = line;
        c->slot_of[line] = s + 1;
        lru_push_front(c, s);
        return 0;
    }
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

static void usage(void) {
    fprintf(stderr, "usage: cachesim [--max-node N] [--nodes-per-line N] [--exclude site,...] trace.bin|-\n");
    exit(2);
}

int main(int argc, char **argv) {
    unsigned npl = 1;
    uint32_t max_node = 16777216; /* largest CAPSTONE_REV_NODES the runs use */
    int excluded[N_SITES] = {0};
    const char *path = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--max-node") && i + 1 < argc) {
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

    static const unsigned sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    static const unsigned waysv[] = {1, 2, 4, 8, 0 /* full */};
    enum { NS = sizeof(sizes) / sizeof(sizes[0]), NW = sizeof(waysv) / sizeof(waysv[0]) };
    struct cache caches[NS * NW];
    int nc = 0;
    for (int si = 0; si < NS; ++si) {
        for (int wi = 0; wi < NW; ++wi) {
            if (waysv[wi] >= sizes[si]) continue; /* that is the fully associative one */
            struct cache *c = &caches[nc++];
            memset(c, 0, sizeof(*c));
            c->entries = sizes[si];
            c->ways = waysv[wi] ? waysv[wi] : sizes[si];
            c->sets = c->entries / c->ways;
            c->tag = calloc(c->entries, sizeof(uint32_t));
            if (c->ways == c->entries) {
                c->slot_of = calloc((size_t)max_line + 1, sizeof(uint32_t));
                c->prev = calloc(c->entries, sizeof(uint32_t));
                c->next = calloc(c->entries, sizeof(uint32_t));
            } else {
                c->stamp = calloc(c->entries, sizeof(uint64_t));
            }
            cache_flush(c, max_line);
        }
    }

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
    uint64_t logical = 0;
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
        if (kind == K_REPEAT) {
            if (!have_last) {
                fprintf(stderr, "cachesim: record %llu: REPEAT with nothing before it\n",
                        (unsigned long long)i);
                return 1;
            }
            logical += id;
            if (excluded[last_site]) { excluded_recs += id; continue; }
            if (last_id == NODE_NONE) { no_node_checks[last_site] += id; continue; }
            count[last_kind][last_site] += id;
            for (int k = 0; k < nc; ++k) {
                caches[k].hit[last_site] += id;
                if (last_kind == K_READ) caches[k].read_hit += id;
            }
            continue;
        }
        ++logical;
        if (kind >= N_KINDS || site >= N_SITES) {
            fprintf(stderr, "cachesim: record %llu: bad kind %u / site %u\n",
                    (unsigned long long)i, kind, site);
            return 1;
        }
        have_last = kind != K_RESET;
        last_id = id; last_kind = kind; last_site = site;
        if (kind == K_RESET) {
            ++resets;
            for (int k = 0; k < nc; ++k) cache_flush(&caches[k], max_line);
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

    printf("{\n  \"trace\": \"%s\",\n  \"records\": %llu,\n  \"logical_records\": %llu,\n"
           "  \"nodes_per_line\": %u,\n",
           path, (unsigned long long)n, (unsigned long long)logical, npl);
    printf("  \"excluded_sites\": [");
    for (int s = 0, first = 1; s < N_SITES; ++s)
        if (excluded[s]) { printf("%s\"%s\"", first ? "" : ", ", site_name[s]); first = 0; }
    printf("],\n  \"excluded_records\": %llu,\n", (unsigned long long)excluded_recs);
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
    for (int k = 0; k < nc; ++k) {
        struct cache *c = &caches[k];
        uint64_t h = 0, m = 0;
        for (int s = 0; s < N_SITES; ++s) { h += c->hit[s]; m += c->miss[s]; }
        printf("%s\n    {\"entries\": %u, \"ways\": %s%u%s, \"hits\": %llu, \"misses\": %llu, "
               "\"read_hits\": %llu, \"read_misses\": %llu, \"by_site\": {",
               k ? "," : "", c->entries, c->ways == c->entries ? "\"full\", \"ways_n\": " : "",
               c->ways, "", (unsigned long long)h, (unsigned long long)m,
               (unsigned long long)c->read_hit, (unsigned long long)c->read_miss);
        for (int s = 0; s < N_SITES; ++s)
            printf("%s\"%s\": [%llu, %llu]", s ? ", " : "", site_name[s],
                   (unsigned long long)c->hit[s], (unsigned long long)c->miss[s]);
        printf("}}");
    }
    printf("\n  ]\n}\n");
    return 0;
}
