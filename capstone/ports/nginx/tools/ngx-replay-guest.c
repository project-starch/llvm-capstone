/* Create the domain, hand it three regions, run the replay, print what it wrote back.
 *
 *   ngx-replay-guest <domain image> <trace file> [--arena-linear] [--arena BYTES]
 *
 * Three regions, in the order the domain expects:
 *   1  the trace, sized from the file, so one build serves traces of different lengths
 *   2  the arena the level below carves, shared as REV_TRANSFERRED under --arena-linear
 *   3  a scratch for the replay's identity tables and its result block
 *
 * The result comes back through the scratch and not through the return value: two hundred
 * thousand records do not fit in the 24 bits a marker has. The marker still carries the reason
 * when the replay refuses to start.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include "capstone.h"
#include "lib/libcapstone.h"

#define NGX_PERM_INOUT       0x1UL
#define NGX_REV_SHARED       0x2UL
#define NGX_REV_TRANSFERRED  0x3UL
#define NGX_SCRATCH_BYTES    (1UL << 20)

struct ngx_replay_out {
    uint64_t magic, records, executed, skipped, failures;
    uint64_t ops[16];
    uint64_t pools_at_end, objs_at_end, level0_live, carved, reused, tables_full;
    uint64_t n_split, n_mrev, n_delin, n_revoke, n_init;
    uint64_t reset_unsupported;
};
#define NGX_REPLAY_MAGIC 0x4E47585245504CAull

static const char *OPNAME[16] = { "?", "create", "destroy", "reset", "palloc", "pnalloc",
                                  "pcalloc", "pmemalign", "pfree", "cleanup", "blocks", "end" };

/* The kernel declines to read(2) straight into a shared region's mapping, so the file comes in
   through a bounce buffer. In pieces rather than in one, because the trace is megabytes. */
static long load_trace(const char *path, char *dst, unsigned long room) {
    static char buf[1 << 20];
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    unsigned long got = 0;
    for (;;) {
        ssize_t n = read(fd, buf, sizeof buf);
        if (n < 0) { close(fd); return -1; }
        if (n == 0) break;
        if (got + (unsigned long) n > room) { close(fd); return -2; }
        memcpy(dst + got, buf, (size_t) n);
        got += (unsigned long) n;
    }
    close(fd);
    return (long) got;
}

int main(int argc, char **argv) {
    int linear = 0;
    unsigned long arena_bytes = 1UL << 20;
    if (argc < 3) {
        fprintf(stderr, "usage: %s DOMAIN TRACE [--arena-linear] [--arena BYTES]\n", argv[0]);
        return 2;
    }
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--arena-linear") == 0) { linear = 1; continue; }
        if (strcmp(argv[i], "--arena") == 0 && i + 1 < argc) { arena_bytes = strtoul(argv[++i], NULL, 0); continue; }
        fprintf(stderr, "unknown option: %s\n", argv[i]);
        return 2;
    }

    int tfd = open(argv[2], O_RDONLY);
    if (tfd < 0) { fprintf(stderr, "no trace at %s\n", argv[2]); return 2; }
    long tsize = (long) lseek(tfd, 0, SEEK_END);
    close(tfd);
    /* rounded up to a megabyte, so one build serves traces of different lengths */
    unsigned long trace_bytes = ((unsigned long) tsize + 0xFFFFFUL) & ~0xFFFFFUL;
    if (trace_bytes == 0) trace_bytes = 1UL << 20;

    int rc = capstone_init();
    if (rc != 0) { fprintf(stderr, "Failed to initialise Capstone: %d\n", rc); return rc; }

    dom_id_t dom = create_dom(argv[1], NULL);
    printf("Created domain ID = %lu\n", dom);

    region_id_t r_trace = create_region(trace_bytes);
    void *m_trace = map_region(r_trace, trace_bytes);
    region_id_t r_arena = create_region(arena_bytes);
    void *m_arena = map_region(r_arena, arena_bytes);
    region_id_t r_scratch = create_region(NGX_SCRATCH_BYTES);
    void *m_scratch = map_region(r_scratch, NGX_SCRATCH_BYTES);
    if (!m_trace || !m_arena || !m_scratch) {
        fprintf(stderr, "Failed to map a region\n");
        capstone_cleanup();
        return 3;
    }

    long got = load_trace(argv[2], (char *) m_trace, trace_bytes);
    if (got < 0) { fprintf(stderr, "Failed to load the trace: %ld\n", got); capstone_cleanup(); return 3; }
    /* The tail of the region, zeroed. A region is rounded up to a megabyte so that one build
       serves traces of different lengths, and the domain cannot see the file's own length. The
       format has no op 0, so a zeroed record is an unambiguous end and nothing else can be. */
    memset((char *) m_trace + got, 0, trace_bytes - (unsigned long) got);
    printf("trace %ld bytes into a %lu region\n", got, trace_bytes);
    memset(m_arena, 0, arena_bytes);
    memset(m_scratch, 0, NGX_SCRATCH_BYTES);

    shared_region_annotated(dom, r_trace, NGX_PERM_INOUT, NGX_REV_SHARED);
    shared_region_annotated(dom, r_arena, NGX_PERM_INOUT,
                            linear ? NGX_REV_TRANSFERRED : NGX_REV_SHARED);
    shared_region_annotated(dom, r_scratch, NGX_PERM_INOUT, NGX_REV_SHARED);

    printf("Called dom (1-th time) retval = %lu\n", call_dom(dom));
    unsigned long v = call_dom(dom);
    printf("ngx retval = %lu\n", v);

    const struct ngx_replay_out *o = (const struct ngx_replay_out *) m_scratch;
    if (o->magic != NGX_REPLAY_MAGIC) {
        printf("no result block: the replay did not finish\n");
        capstone_cleanup();
        return 1;
    }
    printf("replay records   %lu\n", o->records);
    printf("replay executed  %lu\n", o->executed);
    printf("replay skipped   %lu\n", o->skipped);
    printf("replay failures  %lu\n", o->failures);
    for (int i = 1; i < 12; i++) {
        if (o->ops[i]) printf("replay op %-9s %lu\n", OPNAME[i], o->ops[i]);
    }
    printf("replay at end    pools %lu objects %lu\n", o->pools_at_end, o->objs_at_end);
    printf("replay level0    %lu\n", o->level0_live);         /* the number last, so a gate can read it */
    printf("replay blocks    carved %lu reused %lu\n", o->carved, o->reused);
    printf("replay sublet    split %lu mrev %lu delin %lu revoke %lu init %lu\n",
           o->n_split, o->n_mrev, o->n_delin, o->n_revoke, o->n_init);
    printf("replay noreset   %lu\n", o->reset_unsupported);
    printf("replay tables    %lu\n", o->tables_full);

    capstone_cleanup();
    return 0;
}
