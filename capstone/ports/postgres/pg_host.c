/* The host side of the PostgreSQL memory-manager domain.
 *
 *   pg_host.user <domain image> <trace> [--tail]
 *
 * Four regions, in the order the domain expects them, because the domain
 * counts the shares and assigns by order:
 *
 *   0  the metadata, whose `length` says how much of the payload is written
 *   1  the payload, which the domain writes and this prints
 *   2  the arena: the driver's identity tables at the front, the level below the rest
 *   3  the trace, read from the file named on the command line
 *
 * This is the region half of what sqlite_host.c does and none of the rest: the
 * memory manager makes no hostcalls, so there is no protocol to service. It
 * enters the domain once and prints what came back.
 *
 * A region is one physically contiguous block. Since the module took to
 * dma_alloc_pages the block comes from the CMA area the kernel reserved at
 * boot, so a region far above the buddy allocator's 4 MiB is possible and the
 * whole trace fits in one. The guest's command line needs cma= large enough
 * for the arena and the trace together.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>

/* By name and not by path: the build puts the module's lib directory on the
   include path, resolved from CAPSTONE_BUILDROOT_DIR. A relative path into the
   submodule works only where the submodule is checked out, which a worktree is
   not, and the workaround there is a symlink that makes the commit scanner
   block every commit in the tree. */
#include "libcapstone.h"

/* Both halves must agree on these, and they are separate compilations, so the
   host publishes what it used in the metadata and the domain checks it. */
#ifndef PG_META_SIZE
#define PG_META_SIZE 4096UL
#endif
#ifndef PG_REPLAY_PAYLOAD_SIZE
#define PG_REPLAY_PAYLOAD_SIZE 65536UL
#endif
#ifndef PG_REPLAY_ARENA_SIZE
#define PG_REPLAY_ARENA_SIZE (32UL * 1024UL * 1024UL)
#endif
#ifndef PG_REPLAY_TRACE_SIZE
#define PG_REPLAY_TRACE_SIZE (64UL * 1024UL * 1024UL)
#endif

#define PERM_INOUT 0x1UL
/* What the monitor hands the domain for a shared region, and the difference
 * decides whether the Sublet port can work at all.
 *
 *   REV_DEFAULT    non-linear to the domain, a handle kept by the monitor
 *   REV_BORROWED   linear to the domain, a handle kept by the monitor
 *
 * The unprotected arm wants the first: pg_level0.c walks its arena with
 * ordinary pointer arithmetic, and a linear capability copied by ordinary C
 * code is what the hardware refuses. The Sublet arm needs the second, because
 * a region that is not linear cannot be split and a handle senior to it has
 * nothing to revoke. The first run of the sub-pool test said so, type 1 where
 * it wanted 0, and that is why this is an argument and not a constant. */
#define REV_DEFAULT 0x0UL
#define REV_BORROWED 0x1UL

struct pg_hostcall_v0 {
    unsigned long long phase, opcode, offset, length;
    long long result, error;
};

static void
mark(const char *s)
{
    (void) write(STDOUT_FILENO, s, strlen(s));
}

static void
mark_u(const char *s, unsigned long v)
{
    char line[96];
    int n = snprintf(line, sizeof line, "%s%lu\n", s, v);

    if (n > 0)
        (void) write(STDOUT_FILENO, line, (size_t) n);
}

static int
fail(const char *what, unsigned long value)
{
    mark_u(what, value);
    capstone_cleanup();
    return 1;
}

/* --tail: print the payload while the domain runs, from where it left off. A
   domain that never returns then still shows what it wrote up to that point,
   which is the difference between a diagnosable wedge and a silent one. */
struct tail_state {
    volatile struct pg_hostcall_v0 *meta;
    const char *payload;
    unsigned long printed;
    volatile int stop;
};

static void *
tail_main(void *arg)
{
    struct tail_state *t = arg;

    for (;;) {
        unsigned long len = t->meta->length;

        if (len > t->printed && len <= PG_REPLAY_PAYLOAD_SIZE) {
            (void) write(STDOUT_FILENO, t->payload + t->printed,
                         (size_t) (len - t->printed));
            t->printed = len;
        }
        if (__atomic_load_n(&t->stop, __ATOMIC_ACQUIRE))
            return NULL;
        usleep(500000);
    }
}

/* The kernel declines to read(2) straight into a shared region's mapping: the
   first read returns zero and nothing is written. A userspace copy into it
   works, so the file goes through a staging buffer, one chunk at a time rather
   than in one piece, because the trace is tens of megabytes and the guest is
   not generous. */
static long
load_trace(const char *path, char *dst, unsigned long room)
{
    enum { CHUNK = 1UL << 20 };
    char *staging = malloc(CHUNK);
    if (!staging)
        return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        free(staging);
        return -2;
    }
    unsigned long got = 0;
    for (;;) {
        unsigned long want = room - got < CHUNK ? room - got : CHUNK;
        if (want == 0) {                 /* full: is there more? */
            char probe;
            ssize_t more = read(fd, &probe, 1);
            close(fd);
            free(staging);
            return more > 0 ? -3 : (long) got;
        }
        ssize_t n = read(fd, staging, (size_t) want);
        if (n < 0) {
            close(fd);
            free(staging);
            return -4;
        }
        if (n == 0)
            break;
        memcpy(dst + got, staging, (size_t) n);
        got += (unsigned long) n;
    }
    close(fd);
    free(staging);
    return (long) got;
}

int
main(int argc, char **argv)
{
    int want_tail = 0;
    int linear_arena = 0;

    if (argc < 3) {
        mark("usage: pg_host.user <domain image> <trace> [--tail]\n");
        return 2;
    }
    for (int i = 3; i < argc; i++)
        if (!strcmp(argv[i], "--tail"))
            want_tail = 1;
        else if (!strcmp(argv[i], "--linear-arena"))
            linear_arena = 1;

    if (capstone_init())
        return fail("PG: capstone_init failed=", 0);

    dom_id_t domain = create_dom(argv[1], NULL);
    mark_u("PG: dom=", (unsigned long) domain);
    if ((long) domain < 0)
        return fail("PG: create_dom failed=", (unsigned long) domain);

    region_id_t r_meta = create_region(PG_META_SIZE);
    region_id_t r_pay = create_region(PG_REPLAY_PAYLOAD_SIZE);
    region_id_t r_arena = create_region(PG_REPLAY_ARENA_SIZE);
    region_id_t r_trace = create_region(PG_REPLAY_TRACE_SIZE);
    mark_u("PG: r0=", (unsigned long) r_meta);
    mark_u("PG: r1=", (unsigned long) r_pay);
    mark_u("PG: r2=", (unsigned long) r_arena);
    mark_u("PG: r3=", (unsigned long) r_trace);
    /* A region above 4 MiB needs the CMA area, so a refusal here is most often
       cma= missing or too small on the guest's command line, not a bug. */
    if ((long) r_meta < 0 || (long) r_pay < 0)
        return fail("PG: create_region for the small regions failed=", 0);
    if ((long) r_arena < 0)
        return fail("PG: create_region(arena) failed, wanted=", PG_REPLAY_ARENA_SIZE);
    if ((long) r_trace < 0)
        return fail("PG: create_region(trace) failed, wanted=", PG_REPLAY_TRACE_SIZE);

    volatile struct pg_hostcall_v0 *meta =
        (struct pg_hostcall_v0 *) map_region(r_meta, PG_META_SIZE);
    char *payload = (char *) map_region(r_pay, PG_REPLAY_PAYLOAD_SIZE);
    char *trace = (char *) map_region(r_trace, PG_REPLAY_TRACE_SIZE);
    /* map_region returns mmap's value raw, so a refusal is MAP_FAILED and not
       NULL; a bare null test accepts it and the fault surfaces on first use. */
    if (!meta || meta == (void *) -1 || !payload || payload == (void *) -1
        || !trace || trace == (void *) -1)
        return fail("PG: map_region failed=", 0);
    /* The arena is never touched here: the domain owns what is in it. */

    memset((void *) meta, 0, PG_META_SIZE);
    memset(payload, 0, PG_REPLAY_PAYLOAD_SIZE);
    meta->result = (long long) PG_REPLAY_PAYLOAD_SIZE;   /* what the host used */

    long got = load_trace(argv[2], trace, PG_REPLAY_TRACE_SIZE);
    if (got < 0)
        return fail("PG: the trace did not load, code=", (unsigned long) -got);
    mark_u("PG: trace bytes=", (unsigned long) got);

    shared_region_annotated(domain, r_meta, PERM_INOUT, REV_DEFAULT);
    shared_region_annotated(domain, r_pay, PERM_INOUT, REV_DEFAULT);
    shared_region_annotated(domain, r_arena, PERM_INOUT,
                            linear_arena ? REV_BORROWED : REV_DEFAULT);
    shared_region_annotated(domain, r_trace, PERM_INOUT, REV_DEFAULT);
    mark(linear_arena ? "PG: shared, the arena linear\n" : "PG: shared\n");

    struct tail_state tail = {meta, payload, 0, 0};
    pthread_t tail_thread;
    int tail_started = want_tail
        && pthread_create(&tail_thread, NULL, tail_main, &tail) == 0;

    unsigned long result = call_dom(domain);

    if (tail_started) {
        __atomic_store_n(&tail.stop, 1, __ATOMIC_RELEASE);
        pthread_join(tail_thread, NULL);
    }
    if (meta->length > tail.printed && meta->length <= PG_REPLAY_PAYLOAD_SIZE)
        (void) write(STDOUT_FILENO, payload + tail.printed,
                     (size_t) (meta->length - tail.printed));
    mark_u("PG: result=", result);
    capstone_cleanup();
    return 0;
}
