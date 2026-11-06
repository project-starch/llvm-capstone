/* Qualification program for the launcher's mapping pool (CAPSTONE_VM_MAPCACHE, exec.c).
 *
 * The pool keeps the mapping of a freed heap group and hands it to the next group of the same
 * size. Every mode below states what a program must not be able to observe because of that, and
 * run-mapcache.py runs each of them with the pool off and on. The groups are mallocng's: a block
 * of NBYTES is a few pages, which is above half a page and so owns its mapping.
 *
 *   reuse    the second block of the same size is the first one's backing, and holds zero data and no
 *            capability tag (what the first one stored has been overwritten, not merely hidden)
 *   stale    a pointer to the freed block still faults with a lifetime cause (25) after the backing
 *            has been given out again, and after the pool took it but before anyone did
 *   churn    random sizes, live set, patterns checked before every free: the output is a function of
 *            the operations only, so it is byte-identical with the pool off and on
 *   bounded  many distinct blocks freed at once: the pool stays under its byte cap
 *   protect  a block with a PROT_NONE page is never pooled: the next block of its size is whole
 *   threads  four threads churn; join; the totals match
 *   nodes    many reissues of one mapping: lifetime records are recycled, the run does not run out
 *   procmem  a page populated by a native writer while the mapping is parked is zero in the next grant
 *   pressure the pool is given back at the first ENOMEM instead of failing an allocation
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <capstone/virtual.h>
#include <capstone/capability.h>
#include "vm.h"

static void *(*volatile mc_malloc)(size_t) = malloc;
static void *(*volatile mc_calloc)(size_t, size_t) = calloc;
static void (*volatile mc_free)(void *) = free;
#define malloc(n) mc_malloc(n)
#define calloc(n, m) mc_calloc(n, m)
#define free(p) mc_free(p)

#define NBYTES 20000UL
static void *volatile stale;
static void fail(const char *what) { fprintf(stderr, "MAPCACHE_FAIL:%s\n", what); fflush(stderr); exit(1); }
static void require(int ok, const char *what) { if (!ok) fail(what); }
static unsigned long cursor(void *p) { return __builtin_capstone_cap_get_cursor(p); }
static unsigned long tag(void *p) { return __builtin_capstone_cap_get_tag(p); }

static __thread unsigned long long rng = 0x9e3779b97f4a7c15ULL;
static unsigned long long next_random(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

/* Fill a block with capabilities to itself, so every granule carries a tag. */
static void fill_with_pointers(void **block, size_t count)
{
    for (size_t i = 8; i < count; ++i) block[i] = &block[i];
}

static int mode_reuse(void)
{
    size_t count = NBYTES / sizeof(void *);
    void **p = malloc(NBYTES);
    require(p != NULL, "first block");
    fill_with_pointers(p, count);
    require(tag(p[100]) == 1, "the pointers carry tags");
    unsigned long address = cursor(p);
    free(p);
    void **q = malloc(NBYTES);
    require(q != NULL, "second block");
    int same = cursor(q) == address;
    unsigned char *bytes = (unsigned char *)q;
    for (size_t i = 128; i < NBYTES; ++i)
        if (bytes[i]) fail("data of the previous block is visible");
    for (size_t i = 8; i < count; ++i) {
        if (tag(q[i])) fail("a capability tag of the previous block is visible");
        if (cursor(q[i])) fail("a capability cursor of the previous block is visible");
    }
    /* The new block works: store, load back, as a fresh mapping does. */
    fill_with_pointers(q, count);
    require(q[200] == &q[200] && tag(q[200]), "the second block is usable");
    free(q);
    printf("MAPCACHE_REUSE_OK same_address=%d\n", same);
    return 0;
}

static int mode_stale(int reuse)
{
    char *p = malloc(NBYTES);
    require(p != NULL, "block");
    memset(p, 0x5c, NBYTES);
    stale = p;
    free(p);
    if (reuse) {
        char *q = malloc(NBYTES);
        require(q != NULL, "replacement");
        memset(q, 0x3d, NBYTES);
        printf("MAPCACHE_STALE_REUSED same_address=%d\n", cursor(q) == cursor(stale));
    } else {
        printf("MAPCACHE_STALE_FREED\n");
    }
    fflush(stdout);
    (void)*(volatile char *)((char *)stale + 4096);
    fail("stale access survived");
    return 1;
}

/* The mapping-level grant, without the allocator: a copy of it is not revoked by any free(), so only the
 * retirement of the mapping's lifetime can make it fault. (A freed malloc block faults on its own object
 * lifetime whether or not the pool retired the mapping, so mode stale cannot tell the two apart.) */
#define GRANT_BYTES 20480UL
static int mode_stale_grant(int reuse, int hole)
{
    sublet_cap slot, again;
    require(!cap_vm_acquire(&slot, GRANT_BYTES, 4096, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP), "grant");
    char *p = cap_vm_copyable(&slot);
    require(p != NULL, "copy of the grant");
    memset(p, 0x5c, hole ? 4096 : GRANT_BYTES);   /* hole: only the first page is ever resident */
    stale = p;
    unsigned long address = cursor(p);
    require(!munmap(p, GRANT_BYTES), "unmap of the grant");
    if (reuse) {
        require(!cap_vm_acquire(&again, GRANT_BYTES, 4096, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP), "second grant");
        char *q = cap_vm_copyable(&again);
        require(q != NULL, "copy of the second grant");
        for (size_t i = 0; i < (hole ? 4096 : GRANT_BYTES); ++i) if (q[i]) fail("data of the previous grant is visible");
        memset(q, 0x3d, hole ? 4096 : GRANT_BYTES);
        printf("MAPCACHE_STALE_GRANT_REUSED same_address=%d\n", cursor(q) == address);
    } else {
        printf("MAPCACHE_STALE_GRANT_FREED\n");
    }
    fflush(stdout);
    (void)*(volatile char *)((char *)stale + (hole ? 3 * 4096 : 4096));
    fail("stale grant access survived");
    return 1;
}

/* A page populated while its mapping is parked, by a native writer that no capability and no module pin
 * stands in front of: the domain's own delegated pwrite to /proc/self/mem, at the address of a page of the
 * parked mapping that was never resident. The next grant of the mapping must read zero there. Where the
 * guest does not allow the write the mode says so, and the control is then not exercised. */
static int mode_procmem(void)
{
    sublet_cap slot, again;
    require(!cap_vm_acquire(&slot, GRANT_BYTES, 4096, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP), "grant");
    char *p = cap_vm_copyable(&slot);
    require(p != NULL, "copy of the grant");
    p[0] = 1;                                  /* page 0 only is resident */
    unsigned long address = cursor(p);
    require(!munmap(p, GRANT_BYTES), "unmap of the grant");
    int fd = open("/proc/self/mem", O_RDWR);
    if (fd < 0) { printf("MAPCACHE_PROCMEM_UNAVAILABLE open errno=%d\n", errno); return 0; }
    char marker[16];
    memset(marker, 0xa7, sizeof marker);
    ssize_t n = pwrite(fd, marker, sizeof marker, (off_t)(address + 2 * 4096));
    if (n != (ssize_t)sizeof marker) { printf("MAPCACHE_PROCMEM_UNAVAILABLE write=%zd errno=%d\n", n, errno); return 0; }
    close(fd);
    require(!cap_vm_acquire(&again, GRANT_BYTES, 4096, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP), "second grant");
    char *q = cap_vm_copyable(&again);
    require(q != NULL, "copy of the second grant");
    int same = cursor(q) == address;
    for (size_t i = 0; i < GRANT_BYTES; ++i) if (q[i]) fail("a byte written while the mapping was parked is visible");
    printf("MAPCACHE_PROCMEM_OK reused=%d\n", same);
    return 0;
}

#define LIVE 48
static unsigned long churn_run(unsigned long rounds, unsigned long long seed, unsigned long *check)
{
    struct { unsigned char *p; size_t n; unsigned char v; } live[LIVE] = {0};
    unsigned long sum = 0, ops = 0;
    rng = seed;
    for (unsigned long r = 0; r < rounds; ++r) {
        unsigned i = next_random() % LIVE;
        if (live[i].p) {
            for (size_t k = 0; k < live[i].n; k += 509)
                if (live[i].p[k] != live[i].v) fail("block corrupted");
            if (live[i].p[live[i].n - 1] != live[i].v) fail("block tail corrupted");
            sum = sum * 31 + live[i].n + live[i].v;
            free(live[i].p); live[i].p = NULL;
        } else {
            size_t n = 3000 + (size_t)(next_random() % 9) * 4096 + next_random() % 700;
            int zeroed = next_random() % 3 == 0;
            unsigned char *p = zeroed ? calloc(1, n) : malloc(n);
            require(p != NULL, "allocation");
            if (zeroed) for (size_t k = 0; k < n; k += 61) if (p[k]) fail("calloc block not zero");
            unsigned char v = (unsigned char)(1 + next_random() % 250);
            for (size_t k = 0; k < n; k += 509) p[k] = v;
            p[n - 1] = v;
            live[i].p = p; live[i].n = n; live[i].v = v;
            sum = sum * 131 + n;
        }
        ++ops;
    }
    for (unsigned i = 0; i < LIVE; ++i) if (live[i].p) free(live[i].p);
    *check = sum;
    return ops;
}

static int mode_churn(void)
{
    unsigned long check;
    unsigned long ops = churn_run(6000, 0x1234567ULL, &check);
    printf("MAPCACHE_CHURN_OK ops=%lu check=%lu\n", ops, check);
    return 0;
}

static int mode_bounded(void)
{
    enum { BLOCKS = 200 };
    static void *held[BLOCKS];
    for (unsigned i = 0; i < BLOCKS; ++i) {
        held[i] = malloc(NBYTES + (size_t)i * 4096);
        require(held[i] != NULL, "block");
        memset(held[i], 0x11, 64);
    }
    for (unsigned i = 0; i < BLOCKS; ++i) free(held[i]);
    printf("MAPCACHE_BOUNDED_DONE blocks=%u\n", (unsigned)BLOCKS);
    return 0;
}

static int mode_protect(void)
{
    char *p = malloc(NBYTES);
    require(p != NULL, "block");
    unsigned long page = (cursor(p) + 8192) & ~4095UL;
    char *hole = (char *)p + (page - cursor(p));
    require(!mprotect(hole, 4096, PROT_NONE), "mprotect");
    free(p);
    char *q = malloc(NBYTES);
    require(q != NULL, "second block");
    for (size_t i = 128; i < NBYTES; i += 64) q[i] = 1;
    free(q);
    printf("MAPCACHE_PROTECT_OK\n");
    return 0;
}

#define THREADS 4
static unsigned long thread_check[THREADS];
static void *churn_thread(void *arg)
{
    unsigned long id = *(unsigned long *)arg;
    churn_run(1500, 0xabcdef0ULL + id, &thread_check[id]);
    return NULL;
}
static int mode_threads(void)
{
    pthread_t t[THREADS];
    static unsigned long ids[THREADS];
    for (unsigned long i = 0; i < THREADS; ++i) {
        ids[i] = i;
        require(!pthread_create(&t[i], NULL, churn_thread, &ids[i]), "thread create");
    }
    for (unsigned long i = 0; i < THREADS; ++i) require(!pthread_join(t[i], NULL), "thread join");
    unsigned long total = 0;
    for (unsigned i = 0; i < THREADS; ++i) total += thread_check[i];
    printf("MAPCACHE_THREADS_OK total=%lu\n", total);
    return 0;
}

static int mode_nodes(void)
{
    /* Above mallocng's mmap threshold every block is a group of its own, mapped and unmapped each time. */
    enum { BIG = 140000 };
    unsigned long cycles = getenv("MAPCACHE_CYCLES") ? strtoul(getenv("MAPCACHE_CYCLES"), 0, 0) : 20000;
    unsigned long sum = 0;
    for (unsigned long i = 0; i < cycles; ++i) {
        unsigned char *p = malloc(BIG);
        require(p != NULL, "block");
        p[300] = (unsigned char)i; p[BIG - 1] = 1;
        sum += p[300];
        free(p);
    }
    printf("MAPCACHE_NODES_OK cycles=%lu sum=%lu\n", cycles, sum);
    return 0;
}

/* The pool is given back at the first ENOMEM: fill the registered space with sparse mappings, leave less
 * room than the pooled blocks hold, and ask for a block that is not pooled. */
static int mode_pressure(void)
{
    enum { POOLED = 8 };
    void *held[POOLED];
    for (unsigned i = 0; i < POOLED; ++i) { held[i] = malloc(NBYTES + (size_t)i * 8192); require(held[i] != NULL, "pooled block"); }
    for (unsigned i = 0; i < POOLED; ++i) free(held[i]);
    static void *maps[40];
    static size_t sizes[40];
    unsigned n = 0, failed = 0;
    for (size_t sz = 1UL << 31; sz >= 4096; sz >>= 1) {
        void *m = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { ++failed; continue; }
        maps[n] = m; sizes[n++] = sz;
    }
    require(failed >= 1, "the registered space was not filled");
    /* Give back the small ones: less than 64 KiB is left, and the pool holds more. */
    unsigned kept = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (sizes[i] <= 32768) require(!munmap(maps[i], sizes[i]), "munmap");
        else kept++;
    }
    char *big = malloc(100000);
    require(big != NULL, "a block of 100000 bytes after the registered space was full");
    big[99999] = 1;
    free(big);
    printf("MAPCACHE_PRESSURE_OK mappings=%u failed=%u\n", kept, failed);
    return 0;
}

int main(int argc, char **argv)
{
    require(argc == 2, "mode");
    const char *m = argv[1];
    if (!strcmp(m, "reuse")) return mode_reuse();
    if (!strcmp(m, "stale")) return mode_stale(1);
    if (!strcmp(m, "stale-pooled")) return mode_stale(0);
    if (!strcmp(m, "stale-grant")) return mode_stale_grant(1, 0);
    if (!strcmp(m, "stale-grant-pooled")) return mode_stale_grant(0, 0);
    if (!strcmp(m, "procmem")) return mode_procmem();
    if (!strcmp(m, "stale-grant-hole")) return mode_stale_grant(1, 1);
    if (!strcmp(m, "stale-grant-hole-pooled")) return mode_stale_grant(0, 1);
    if (!strcmp(m, "churn")) return mode_churn();
    if (!strcmp(m, "bounded")) return mode_bounded();
    if (!strcmp(m, "protect")) return mode_protect();
    if (!strcmp(m, "threads")) return mode_threads();
    if (!strcmp(m, "nodes")) return mode_nodes();
    if (!strcmp(m, "pressure")) return mode_pressure();
    fail("unknown mode");
    return 2;
}
