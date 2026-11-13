#define _GNU_SOURCE
#include "wire.h"
#include "capstone/linux-domain-fault.h"
#include "capstone/spawn.h"
#include <linux/binfmts.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <sys/auxv.h>
#include "../linux/application-image.h"
#include "../linux/delegate-service.h"
#include <elf.h>
#include <errno.h>
#include <stdbool.h>
#include <limits.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <capstone/representable.h>
extern char **environ;
struct mapping {
    void *address;
    size_t bytes, visible;
    uint64_t id;
    int heap, kind;
    /* The worker whose transport block (its meta block or its exchange region) lies in this
     * mapping, or 0. This launcher writes into a worker's transport for the worker's whole
     * attachment, through Linux and with service_lock released while a service blocks, so the
     * mapping must outlive the attachment: while a holder is set, retirement (unmap), protection
     * changes and madvise are refused to EVERY caller, the holder included, and hence the address
     * is never reused under it and its pages stay writable. The attachment requires the blocks'
     * pages to be writable first (mapping_writable), so the hold keeps them so. The hold is released
     * only by the holder's own detach (the SDK detaches before it gives the blocks back) or by its
     * exit. One holder per mapping and one block per mapping: the SDK maps each block on its own
     * (pthread.c), and the attachment refuses a mapping that already has a holder. Without the
     * hold, a second worker could retire the mapping under a blocked read, map a fresh one at the
     * same address, and the read's copy-back would create backing the module never admitted. On
     * dev since linear arenas became lazy: the transport blocks were heap blocks. */
    uint64_t holder;
    /* mprotect was applied inside this mapping: its page protections are no longer the ones a fresh
     * grant has, so the mapping is never pooled (see the mapping pool below). */
    int protected_range;
    /* The protection Linux maps on each page of the visible length (PROT_READ, PROT_WRITE,
     * PROT_EXEC), exact per page: prot for every page while pages is NULL, else one byte per page
     * (see "Page state" below). This launcher writes into the program's memory on its behalf (an
     * exit word, a worker's transport) only into pages that Linux maps writable, so neither a
     * protection the program set nor the image's read-execute text can make that write fault. */
    unsigned char prot, *pages;
    /* The program's grant carries the write right (CV_GRANT_WRITE). */
    int grant_writes;
};
/* The write right of a grant's permissions (cv_map.permissions, CAPSTONE_PERM_WRITE). */
#define CV_GRANT_WRITE 2
/* Registered mappings, one node each, in an AVL tree keyed by address. Registrations never
 * overlap, so the highest mapping starting at or below an address is the only one that can
 * contain it, and map_floor, map_find, map_insert and map_remove each visit one root-to-leaf
 * path: O(log n) in the number of live mappings. The registry was an array sorted by address
 * that moved its tail on every insertion and removal; the virtual heap gives every mallocng group
 * its own mapping and frees the oldest groups first, so with a few thousand live mappings each
 * unmap moved most of the array (PHP's ary2(150000): 2.29e9 instruction attempts in memmove,
 * 53 % of that test's Capstone overhead). A pointer to a registered mapping stays valid until
 * that mapping is removed. Removed nodes go to a free list that map_insert takes from, and
 * map_reserve refills it (64 nodes per malloc) before the kernel mints a grant, so map_insert
 * cannot fail after CV_ADD succeeded and a steady map/unmap pattern calls no allocator. */
struct map_node {
    struct mapping m;
    struct map_node *child[2];   /* [0] lower addresses, [1] higher */
    int height;                  /* of the subtree rooted here; a leaf is 1 */
};
static struct map_node *map_root, *map_free;   /* map_free chains through child[0] */
static size_t map_count;
static void *reuse_address;
static size_t reuse_bytes;
static int device = -1, image_fd = -1;
static const char *image_path;
static struct capstone_spawner spawner = {.socket = -1};
static struct capstone_delegate_host host;
static __thread struct capstone_delegate_host *worker_host;
/* The process transport's host state (host: its bounce buffer, its signal ring) is shared by every
 * context that names the process transport. The domain serializes those contexts with its own
 * transport lock, but that lock is the domain's: a context that skips it (a forged gp identity, or
 * code that calls the service directly) must not reach this launcher's state from two native
 * threads at once. host_lock is this launcher's own serialization of every serve on &host. The
 * transport service takes it with service_lock released (a blocked read on the process transport
 * stops only the contexts that share that transport, as their domain lock already does); the
 * legacy service, which serves under service_lock, takes it inside. Nothing takes service_lock
 * while holding host_lock. */
static pthread_mutex_t host_lock = PTHREAD_MUTEX_INITIALIZER;
static __thread int host_signals_attached;   /* this worker's thread records into host.signals */
/* The thread table: one entry per live thread, the first one (the main thread) included. An exit
 * frees its entry at once unless joiners wait on it; then the last joiner to leave frees it
 * (thread_finish, thread_join). So an entry outlives its thread only while a joiner waits, and
 * threads that are never joined cannot fill the table. */
#define CV_MAX_THREADS 32
static uint64_t thread_ids[CV_MAX_THREADS] = {0};
/* Set by the exit of an entry's thread while joiners wait on the entry. */
static unsigned thread_done[CV_MAX_THREADS];
/* The joiners that wait in thread_join on each entry. */
static unsigned thread_waiters[CV_MAX_THREADS];
/* The highest thread id of a create that succeeded (thread_publish). The module issues ids in order,
 * never reuses the id of a thread that ran, and gives back the id of a create that the launcher
 * rolls back (capstone_vm.c, thread_exit), so every id at or below this one names a thread that ran,
 * and one without an entry names a thread that has exited. */
static uint64_t thread_last_id;
/* Each thread's frame mapping, by address and identity (the module's arena id, never reused): the
 * exit names it to the module for release (thread_frame_mapping). */
static struct { uintptr_t address; uint64_t id; } thread_frames[CV_MAX_THREADS];
static unsigned thread_count = 1;
static pthread_mutex_t service_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t thread_state_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t thread_state_cond = PTHREAD_COND_INITIALIZER;
static unsigned long long launch_ns, start_ns;
static int trace;
/* CAPSTONE_VM_GRANTS (or the trace): one CAPSTONE_VM_ADD_OK line per admitted grant. Off by
 * default: a gcc test run admits ~400 grants, and each line is an unbuffered write(2) to wherever
 * stderr goes -- a 9p file in a harness that does not redirect it. */
static int grants;
/* CV_STEP returns by kind (1 quantum, 3 service, 4 fault, else other) and services by their
 * vm-abi.h number (CV_SERVICE_DELEGATE 0 .. CV_SERVICE_ADVISE 16; not Linux syscall numbers),
 * for the CAPSTONE_VM_STEPS line: which exits the step protocol actually makes, and how many. A
 * number above the last service counts as other. A retried EINTR/EAGAIN is not a return.
 * Relaxed atomics on both sides: every virtual thread's loop counts, and the main thread reads
 * at exit while a worker may still be returning. */
#define STEP_SERVICE_COUNT (CV_SERVICE_ADVISE + 1)
static unsigned long step_kinds[8], step_services[STEP_SERVICE_COUNT + 1];
static unsigned long step_count(const unsigned long *counter)
{ return __atomic_load_n(counter, __ATOMIC_RELAXED); }
static void *meta;
static void *virtual_thread_worker(void *opaque);
static struct mapping *map_find(uintptr_t base);

/* Record thread id, which CV_THREAD_CREATE has just issued and whose frame lies in the mapping
 * frame. -EAGAIN if the table is full. A join sees the id only after thread_publish; until then the
 * create can still fail (thread_withdraw). Joiners cannot name it in between: the creator holds
 * service_lock until the create service ends, and a join starts under service_lock. */
static int thread_add(uint64_t id, const struct mapping *frame)
{
    int result = -EAGAIN;
    if (!id) return -EINVAL;
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; thread_count < CV_MAX_THREADS && i < CV_MAX_THREADS; ++i) {
        if (thread_ids[i]) continue;
        thread_ids[i] = id;
        thread_done[i] = 0;
        thread_waiters[i] = 0;
        thread_frames[i].address = (uintptr_t)frame->address;
        thread_frames[i].id = frame->id;
        ++thread_count;
        result = 0;
        break;
    }
    pthread_mutex_unlock(&thread_state_lock);
    return result;
}
/* The create of thread id has succeeded: a join of the id waits for its exit from here on. */
static void thread_publish(uint64_t id)
{
    pthread_mutex_lock(&thread_state_lock);
    if (id > thread_last_id) thread_last_id = id;
    pthread_mutex_unlock(&thread_state_lock);
}
static void thread_slot_free(unsigned i);
/* The create of thread id failed after thread_add, and its thread never ran: free its entry. The
 * id was never published, so no joiner waits on it, and the module issues it again. */
static void thread_withdraw(uint64_t id)
{
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i)
        if (thread_ids[i] == id) { thread_slot_free(i); break; }
    pthread_mutex_unlock(&thread_state_lock);
}
/* The registered mapping that holds thread id's frame, found by identity, or NULL. */
static struct mapping *thread_frame_mapping(uint64_t id)
{
    uintptr_t address = 0;
    uint64_t frame_id = 0;
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i)
        if (thread_ids[i] == id) {
            address = thread_frames[i].address;
            frame_id = thread_frames[i].id;
            break;
        }
    pthread_mutex_unlock(&thread_state_lock);
    struct mapping *m = address ? map_find(address) : NULL;
    return m && m->id == frame_id ? m : NULL;
}
/* Free entry i of the thread table. Called with thread_state_lock held. */
static void thread_slot_free(unsigned i)
{
    thread_ids[i] = 0;
    thread_done[i] = 0;
    thread_waiters[i] = 0;
    thread_frames[i].address = 0;
    thread_frames[i].id = 0;
    --thread_count;
}
/* Publish thread id's exit, after all of its cleanup: a joiner through the launcher (thread_join)
 * returns from here on. The last one to
 * leave frees the entry. With no joiner waiting, that is this exit, with or without an exit word.
 * Otherwise the exit marks the entry done and wakes the joiners, and the last joiner to leave
 * frees it. */
static void thread_finish(uint64_t id)
{
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i) {
        if (thread_ids[i] != id) continue;
        if (thread_waiters[i]) {
            thread_done[i] = 1;
            pthread_cond_broadcast(&thread_state_cond);
        } else {
            thread_slot_free(i);
        }
        break;
    }
    pthread_mutex_unlock(&thread_state_lock);
}
/* Thread tid waits until thread id has exited (CV_SERVICE_THREAD_JOIN). 0 once it has exited,
 * whether the exit comes before or after the call: an id at or below thread_last_id without an
 * entry belongs to a thread that has exited. -EDEADLK for the caller's own id, the main thread's 0
 * included. -ESRCH for an id that names no thread that ran: 0 from another thread, an id never
 * issued, and the id of a failed create, which the module issues again to the next create. Any
 * number of joiners may wait on one thread: each counts itself in thread_waiters while it waits,
 * the last one to leave frees the entry, and the others leave it alone. The entry cannot be freed,
 * nor reused, while a joiner waits on it, so every joiner sees the exit of the thread it named. */
static int thread_join(uint64_t tid, uint64_t id)
{
    int result;
    if (id == tid) return -EDEADLK;
    if (!id) return -ESRCH;
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i) {
        if (thread_ids[i] != id) continue;
        ++thread_waiters[i];
        while (!thread_done[i])
            pthread_cond_wait(&thread_state_cond, &thread_state_lock);
        if (!--thread_waiters[i]) thread_slot_free(i);
        pthread_mutex_unlock(&thread_state_lock);
        return 0;
    }
    result = id <= thread_last_id ? 0 : -ESRCH;
    pthread_mutex_unlock(&thread_state_lock);
    return result;
}
static unsigned long long now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000 + t.tv_nsec;
}

static size_t rounded(size_t n)
{
    size_t size = 4096;
    if (!n || n > CV_MAX_REGION_BYTES) return 0;
    while (size < n) size <<= 1;
    return size;
}
static void *private_pages(void *p, size_t bytes)
{
    /* Demand resolution accounts for one base page at a time. The module
     * requires this VMA policy on kernels that can instantiate huge pages.
     * A kernel built without them answers EINVAL; it does every time, so it
     * is asked only once. */
    static int no_huge_pages;
    if (no_huge_pages) return p;
    if (madvise(p, bytes, MADV_NOHUGEPAGE)) {
        if (errno == EINVAL) { no_huge_pages = 1; return p; }
        int error = errno; munmap(p, bytes); errno = error; return NULL;
    }
    return p;
}
static void *reserve(size_t bytes)
{
    /* Let Linux reuse the most recently retired range if it is still free.
     * A fresh grant always creates new identities, even at this same VA. */
    if (reuse_address && bytes == reuse_bytes) {
        void *p = mmap(reuse_address, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        reuse_address = NULL;
        if (p != MAP_FAILED) return private_pages(p, bytes);
    }
    char *raw = mmap(NULL, bytes * 2, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t base = ((uintptr_t)raw + bytes - 1) & ~(bytes - 1);
    if (base != (uintptr_t)raw) munmap(raw, base - (uintptr_t)raw);
    size_t tail = (uintptr_t)raw + bytes * 2 - base - bytes;
    if (tail) munmap((void *)(base + bytes), tail);
    return private_pages((void *)base, bytes);
}
/* A per-process virtual-address cursor for aligned arenas, growing DOWN. It is seeded by the
 * first over-map-and-trim reservation and holds the base of the most recently placed arena;
 * thereafter each arena is tried at the next `alignment`-aligned slot below the previous one. */
static uintptr_t aligned_cursor;
static void *reserve_aligned(size_t bytes, size_t alignment)
{
    /* Fast path: place the arena at the cursor with MAP_FIXED_NOREPLACE, so the kernel confirms
     * the slot is free with one maple-tree probe instead of running its top-down free-area search
     * (mas_empty_area_rev). Nothing about the mapping or its registration changes -- same bytes,
     * same PROT_READ|PROT_WRITE, same madvise via private_pages(), and the base is aligned to
     * `alignment` (>= the bounds granule) exactly as the trim path below produces, so grant()'s
     * representability check and the CV_ADD are identical; only HOW the free virtual range is found
     * differs. A taken hint -- EEXIST, or an older kernel without MAP_FIXED_NOREPLACE handing back
     * a different address -- falls back to the trim path and reseeds the cursor, so two arenas can
     * never land at one address and correctness never depends on the hint being honoured. */
    if (aligned_cursor > bytes) {
        uintptr_t hint = (aligned_cursor - bytes) & ~(uintptr_t)(alignment - 1);
        if (hint) {
            void *p = mmap((void *)hint, bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p == (void *)hint) { aligned_cursor = hint; return private_pages(p, bytes); }
            if (p != MAP_FAILED) munmap(p, bytes);
        }
    }
    /* Fallback / first call. mmap() promises page alignment only. Over-map by the worst-case
     * distance to the next alignment boundary -- the raw base is already page-aligned, so that is
     * alignment - 4096 -- then trim both ends, leaving EXACTLY `bytes` mapped. Mapping exactly
     * `bytes` is the point: a heap mapping has no padding (m->bytes == m->visible), which the
     * mapping pool requires (mc_eligible). At alignment == 4096 the slack is zero and this
     * reduces to a plain mmap() of `bytes`, which is what the heap path used to do. Seed the
     * cursor at the base so the next arena is placed just below this one. */
    size_t slack = alignment - 4096;
    char *raw = mmap(NULL, bytes + slack, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t base = ((uintptr_t)raw + alignment - 1) & ~(uintptr_t)(alignment - 1);
    if (base != (uintptr_t)raw) munmap(raw, base - (uintptr_t)raw);
    size_t tail = (uintptr_t)raw + bytes + slack - (base + bytes);
    if (tail) munmap((void *)(base + bytes), tail);
    aligned_cursor = base;
    return private_pages((void *)base, bytes);
}
/* The highest mapping starting at or below p, or NULL. */
static struct mapping *map_floor(uintptr_t p)
{
    struct map_node *n = map_root, *best = NULL;
    while (n) {
        int higher = (uintptr_t)n->m.address <= p;
        if (higher) best = n;
        n = n->child[higher];
    }
    return best ? &best->m : NULL;
}
static struct mapping *map_find(uintptr_t base)
{
    struct mapping *m = map_floor(base);
    return m && (uintptr_t)m->address == base ? m : NULL;
}
/* Room for one more mapping, made before the kernel mints its grant. */
static int map_reserve(void)
{
    if (map_free) return 0;
    struct map_node *chunk = malloc(64 * sizeof *chunk);
    if (!chunk) { errno = ENOMEM; return -1; }
    for (unsigned i = 0; i < 64; ++i) { chunk[i].child[0] = map_free; map_free = &chunk[i]; }
    return 0;
}
static int map_height(const struct map_node *n) { return n ? n->height : 0; }
static void map_update(struct map_node *n)
{
    int low = map_height(n->child[0]), high = map_height(n->child[1]);
    n->height = 1 + (low > high ? low : high);
}
/* Rotate n's child[!side] up into n's place; returns the new subtree root. */
static struct map_node *map_rotate(struct map_node *n, int side)
{
    struct map_node *up = n->child[!side];
    n->child[!side] = up->child[side];
    up->child[side] = n;
    map_update(n); map_update(up);
    return up;
}
/* Restore the AVL property at n after one of its subtrees changed height by one. */
static struct map_node *map_balance(struct map_node *n)
{
    map_update(n);
    int skew = map_height(n->child[1]) - map_height(n->child[0]);
    if (skew > 1 || skew < -1) {
        int heavy = skew > 0;   /* the taller side */
        struct map_node *c = n->child[heavy];
        if (map_height(c->child[!heavy]) > map_height(c->child[heavy]))
            n->child[heavy] = map_rotate(c, heavy);
        return map_rotate(n, !heavy);
    }
    return n;
}
static struct map_node *map_link(struct map_node *n, struct map_node *add)
{
    if (!n) return add;
    int higher = (uintptr_t)add->m.address > (uintptr_t)n->m.address;
    n->child[higher] = map_link(n->child[higher], add);
    return map_balance(n);
}
static void map_insert(struct mapping m)
{
    /* map_reserve ran first; a missing node is a registry defect. */
    struct map_node *add = map_free;
    if (!add) abort();
    map_free = add->child[0];
    *add = (struct map_node){.m = m, .height = 1};
    map_root = map_link(map_root, add);
    ++map_count;
}
/* Detach the lowest node of the subtree n into *lowest; returns the rest, rebalanced. */
static struct map_node *map_unlink_lowest(struct map_node *n, struct map_node **lowest)
{
    if (!n->child[0]) { *lowest = n; return n->child[1]; }
    n->child[0] = map_unlink_lowest(n->child[0], lowest);
    return map_balance(n);
}
/* Detach the node holding *m (found by its address, then by identity) from the subtree n. */
static struct map_node *map_unlink(struct map_node *n, const struct mapping *m)
{
    if (!n) abort();   /* a mapping that is not registered: a registry defect */
    if (&n->m != m) {
        int higher = (uintptr_t)m->address > (uintptr_t)n->m.address;
        n->child[higher] = map_unlink(n->child[higher], m);
        return map_balance(n);
    }
    if (!n->child[0] || !n->child[1]) return n->child[!n->child[0]];
    struct map_node *next;
    struct map_node *rest = map_unlink_lowest(n->child[1], &next);
    next->child[0] = n->child[0]; next->child[1] = rest;
    return map_balance(next);
}
static void page_free(struct mapping *m);
static void map_remove(struct mapping *m)
{
    struct map_node *gone = (struct map_node *)((char *)m - offsetof(struct map_node, m));
    page_free(m);
    map_root = map_unlink(map_root, m);
    --map_count;
    /* Kept for the next registration. */
    gone->child[0] = map_free; map_free = gone;
}
/* ---- Page state ----
 *
 * Each mapping records the protection Linux maps on every page of its visible length, exactly:
 * PROT_READ, PROT_WRITE and PROT_EXEC as mprotect takes them. The padding behind the visible length
 * is PROT_NONE for good and never named by the program, so it has no state. A mapping whose pages
 * all share one protection keeps it in m->prot alone (m->pages NULL), which is every mapping that
 * no mprotect ever changed in part: a registration and its removal call no allocator. The first
 * change of part of a mapping gives it one byte per page (page_split), and a change of the whole
 * mapping returns it to the one value. A byte per page and not a bit: three protection bits, and
 * room for the state that stage 4d adds when a partial munmap removes a page (a removed page stays
 * a PROT_NONE placeholder that the program can map again only under the MAP_FIXED rule). The cost is
 * bounded: a mapping is at most CV_MAX_REGION_BYTES (512 Ki pages, 512 KiB of state) and all
 * mappings of a context together at most CV_MAX_BYTES (1 MiB), paid only where a protection changed
 * in part. Only protect_range changes a protection, and it keeps the state conservative where
 * Linux can fail halfway: before mprotect each page of the range keeps only the rights that its old
 * and its new protection share (page_and), and after a successful mprotect it holds the new one. */
static size_t page_count(const struct mapping *m) { return m->visible / 4096; }
/* Page i's protection. */
static unsigned page_get(const struct mapping *m, size_t i)
{
    return m->pages ? m->pages[i] : m->prot;
}
/* Give m one byte per page, each holding the protection of every page now. 0, or -ENOMEM with
 * nothing changed. A change of part of a mapping asks for it before anything else changes. */
static int page_split(struct mapping *m)
{
    if (m->pages) return 0;
    m->pages = malloc(page_count(m));
    if (!m->pages) return -ENOMEM;
    memset(m->pages, m->prot, page_count(m));
    return 0;
}
/* Pages [first, end) now have protection prot. A whole-mapping change returns to the one value; a
 * change of part needs page_split first. */
static void page_set(struct mapping *m, size_t first, size_t end, unsigned prot)
{
    if (!first && end == page_count(m)) {
        page_free(m);
        m->prot = prot;
        return;
    }
    if (!m->pages) abort();   /* the caller's page_split was skipped: a launcher defect */
    memset(m->pages + first, prot, end - first);
}
/* Pages [first, end) keep only the rights in mask. A change of part needs page_split first. */
static void page_and(struct mapping *m, size_t first, size_t end, unsigned mask)
{
    if (!m->pages && !first && end == page_count(m)) { m->prot &= mask; return; }
    if (!m->pages) abort();
    for (size_t i = first; i < end; ++i) m->pages[i] &= mask;
}
/* Do pages [first, end) all have every right in want? With exact, and no other right? */
static int page_all(const struct mapping *m, size_t first, size_t end, unsigned want, int exact)
{
    for (size_t i = first; i < end; ++i) {
        unsigned prot = page_get(m, i);
        if ((prot & want) != want || (exact && prot != want)) return 0;
        if (!m->pages) return 1;   /* every page holds the same protection */
    }
    return 1;
}
static void page_free(struct mapping *m)
{
    free(m->pages);
    m->pages = NULL;
}
/* Register [base, + bytes) with the module and grant it in register reg of thread. visible: the
 * length the program's mmap exposes (the rest is PROT_NONE padding); the module checks a request
 * to end the mapping against it (wire.h, cv_map). */
static int grant_mapping(void *base, size_t bytes, size_t visible, unsigned reg, unsigned perms,
                         uintptr_t cursor, int linear, uint64_t thread, size_t rx_bytes)
{
    if (map_reserve()) return -1;
    /* Option (iii): refuse to mint an arena whose bounds the encoder cannot represent, rather
     * than hand the domain a capability silently widened below its base or above its end.
     *
     * CORRECTION (2026-10-17): an earlier version of this comment claimed the check "cannot
     * fire today", on the grounds that every size reaching a grant is a power of two. That was
     * wrong in the case that matters most. The local-malloc heap grants `visible` bytes, which
     * is page-rounded and granule-rounded but NOT a power of two -- a 6 MiB heap grants
     * 6291456 -- so this check is load-bearing for exactly the path option (iii) changed, and
     * it is COUPLED to the alignment heap.c requests: it passes only because acquire_mapping()
     * now aligns that mapping to the granule. The two must land together. Every arena passes
     * through this function or revives a pooled one of the same range. Negative-tested in
     * tests/exact-bounds-iii/grant-repr-test.c. */
    if (!capstone_repr_ok((uintptr_t)base, bytes)) { errno = ERANGE; return -1; }
    /* Every range this launcher grants is mapped read-write past rx_bytes (the image's text,
     * read-execute), so only the image needs a byte per page, made before the kernel mints the
     * grant; acquire_mapping sets another protection afterwards. */
    unsigned char *pages = NULL;
    if (rx_bytes) {
        pages = malloc(visible / 4096);
        if (!pages) { errno = ENOMEM; return -1; }
        memset(pages, PROT_READ | PROT_EXEC, rx_bytes / 4096);
        memset(pages + rx_bytes / 4096, PROT_READ | PROT_WRITE, (visible - rx_bytes) / 4096);
    }
    struct cv_map r = {thread, (uintptr_t)base, bytes, perms, reg, cursor, linear, 0, rx_bytes, visible};
    if (ioctl(device, CV_ADD, &r)) {
        /* Evaluate add()'s two disjuncts on the REAL values the module saw. The domain cannot do
         * this: cap_vm_acquire clears its slot on failure, so the base never reaches it, and a
         * power-of-two size refused with EINVAL is then indistinguishable from a misaligned one. */
        /* NOT gated on `trace`. A refusal is rare, so this cannot flood anything, and gating
         * it is what made its absence unprovable: four of this suite's own invocations
         * (population, churn, remap-align, the denial loop) run without CAPSTONE_VM_TRACE, so an
         * "ADD_REFUSED absent" check over them was an unproven zero. Turning the trace on
         * globally is not the fix -- churn alone performs 70,000 allocations and would bury the
         * serial in boundary lines. A diagnostic on an error path does not need a verbosity flag. */
        int err = errno;
        fprintf(stderr, "CAPSTONE_VM_ADD_REFUSED base=0x%lx bytes=%zu errno=%d"
                        " pow2=%d naturally_aligned=%d\n",
                (unsigned long)r.address, bytes, err,
                (bytes & (bytes - 1)) == 0, (r.address & (bytes - 1)) == 0);
        free(pages);
        errno = err;
        return -1;
    }
    /* Every ADMITTED grant, in order, so two modules' grant sequences can be diffed. The
     * refusal side is always printed; without the success side a diff cannot say WHICH grant
     * first differs, only that something did. Asked for with CAPSTONE_VM_GRANTS=1 (or the
     * trace), unlike the refusal: at one write(2) per grant it is the launcher's largest
     * stderr traffic. The board lane's parsers, the benchmark console audit and the three
     * gates that find their arenas by address (run-pin-ahead.py, run-step-hooks.py,
     * run-pthreads.py) set it. */
    /* slot= is the number of mappings registered before this one: the old array slot until the
     * first unmap, so the image's grant still reads slot=0 for the board lane's parsers. */
    if (grants)
        fprintf(stderr, "CAPSTONE_VM_ADD_OK slot=%zu base=0x%lx bytes=%zu reg=%u perms=%u linear=%d\n",
                map_count, (unsigned long)(uintptr_t)base, bytes, reg, perms, linear);
    map_insert((struct mapping){.address = base, .bytes = bytes, .visible = visible,
                                .id = r.id, .heap = linear,
                                .kind = linear ? CV_MAP_HEAP : -1,
                                .prot = PROT_READ | PROT_WRITE, .pages = pages,
                                .grant_writes = (perms & CV_GRANT_WRITE) != 0});
    return 0;
}
/* The launcher's own grants (image, stack, transport, startup data, regions): no padding. */
static int grant(void *base, size_t bytes, unsigned reg, unsigned perms,
                 uintptr_t cursor, int linear, uint64_t thread, size_t rx_bytes)
{
    return grant_mapping(base, bytes, bytes, reg, perms, cursor, linear, thread, rx_bytes);
}
/* The calling worker's attachment: its meta block and exchange region, the two spans this
 * launcher writes for it, each with the identity of the mapping it holds. The identity is the
 * module's arena id (CV_ADD's r.id): a per-context counter, never reused, so it names one
 * registration, not an address or a slot that a later mapping can take over. Zero while
 * detached: the worker's next delegate service attaches again, CV_TOUCH and all. */
static __thread struct { uintptr_t at; uint64_t id; } transport_at[2];
/* The mapping a worker's attachment holds, found by exact identity: the address must lie inside
 * the mapping, which must carry the recorded id and this worker as holder. Anything else is a
 * launcher defect (the hold refuses retirement, so it cannot happen through the services); it is
 * logged and nothing is released. */
static struct mapping *transport_mapping(uint64_t tid, unsigned i)
{
    uintptr_t at = transport_at[i].at;
    struct mapping *m = map_floor(at);
    if (m && at - (uintptr_t)m->address < m->visible && m->id == transport_at[i].id &&
        m->holder == tid) return m;
    fprintf(stderr, "CAPSTONE_VM_TRANSPORT_MISMATCH tid=%llu block=%u at=0x%lx id=%llu found=%llu\n",
            (unsigned long long)tid, i, (unsigned long)at, (unsigned long long)transport_at[i].id,
            m ? (unsigned long long)m->id : 0ULL);
    return NULL;
}
/* Release the calling worker's holds and its host state: the worker's own detach service and
 * its exit, nothing else. After it this launcher writes nothing into the blocks. */
static void transport_detach(uint64_t tid)
{
    for (unsigned i = 0; i < 2; ++i) {
        if (!transport_at[i].at) continue;
        struct mapping *m = transport_mapping(tid, i);
        if (m) m->holder = 0;
        transport_at[i].at = 0; transport_at[i].id = 0;
    }
    if (worker_host) {
        capstone_signals_detach();
        capstone_delegate_host_free(worker_host);
        free(worker_host);
        worker_host = NULL;
    } else if (host_signals_attached) {
        /* Only the calling thread's signal mask and TLS attachment change.
         * A blocking serve holds host_lock with service_lock released; waiting
         * for it here would stop the service that can unblock that serve. */
        capstone_signals_detach();
    }
    host_signals_attached = 0;
}
/* ---- Mapping pool (EXPERIMENTAL, default off: CAPSTONE_VM_MAPCACHE=<bytes>, K and M suffixes) ----
 *
 * Mallocng gives every group above half a page its own mapping and gives it back as soon as the
 * group is empty, so a program that grows and shrinks a few groups maps and unmaps over and over:
 * each pair is an mmap, a CV_ADD, a CV_RETIRE and a munmap, and every page touched in between is
 * faulted, pinned, zeroed, unpinned and freed again. The pool keeps such a mapping instead and
 * gives it to the next grant of the same size.
 *
 * What is pooled: a whole heap mapping (kind CV_MAP_HEAP) whose domain unmap asked for exactly its
 * range, that nobody holds, whose protections were never changed. Anything else (application
 * mmaps with their PROT_NONE padding, metadata, thread frames and stacks, the unmaps this launcher
 * does itself) takes the ordinary path.
 *
 * The safe sequence is the module's (CV_UNMAP with park, CV_REVIVE; capstone_vm.c): the domain's
 * access ends at the park (the arena's lifetime is retired, so a pointer from before it faults
 * with a lifetime cause for ever), a pooled mapping is not in the registry (no service sees it:
 * not unmap, not protect, not futex, not a delegated copy), and the revive overwrites the pinned
 * pages with zeros (data and tags) before it mints a new lifetime and a new arena id. Bounded by
 * bytes and by entries, oldest first; emptied at the first ENOMEM of a new mapping. Nothing is done
 * at exit: the module retires every arena it still holds and Linux unmaps the rest. Pinned mode
 * only: when the module refuses a park with EOPNOTSUPP (unpinned) the pool turns itself off. */
#define MC_MAX_ENTRIES 256
#define MC_BUCKETS 64
struct mc_entry {
    struct mapping m;
    struct mc_entry *older, *newer;   /* parked order: oldest is evicted first */
    struct mc_entry *chain;           /* same size bucket; also the free list */
};
static struct {
    size_t cap, bytes;
    unsigned entries;
    struct mc_entry *oldest, *newest, *bucket[MC_BUCKETS], *spare;
    struct mc_entry pool[MC_MAX_ENTRIES];
    unsigned long parks, hits, misses, evictions, flushes, refused, lost;
    int broken;   /* a retirement failed for good: no new mapping enters the pool */
} mc;
static void mc_init(const char *text)
{
    char *end;
#ifdef CAPSTONE_MAPCACHE_DEFAULT_BYTES
    /* Test builds only (the gates run every runner unchanged, so the pool is switched on in the binary):
     * the variable still wins, and CAPSTONE_VM_MAPCACHE=0 turns it off. */
    if (!text) text = CAPSTONE_MAPCACHE_DEFAULT_BYTES;
    else if (!strcmp(text, "0")) return;
#endif
    if (!text || !*text) return;
    unsigned long long n = strtoull(text, &end, 10);
    if (*end == 'k' || *end == 'K') { n <<= 10; ++end; }
    else if (*end == 'm' || *end == 'M') { n <<= 20; ++end; }
    if (*end || n < 8192 || n > CV_MAX_BYTES) {
        fprintf(stderr, "capstone-vexec: CAPSTONE_VM_MAPCACHE=%s ignored (bytes, 8192 .. %lu, K or M suffix)\n",
                text, (unsigned long)CV_MAX_BYTES);
        return;
    }
    mc.cap = n;
    for (unsigned i = 0; i < MC_MAX_ENTRIES; ++i) { mc.pool[i].chain = mc.spare; mc.spare = &mc.pool[i]; }
}
static unsigned mc_slot(size_t bytes) { return (bytes >> 12) % MC_BUCKETS; }
static void mc_unlink(struct mc_entry *e)
{
    if (e->older) e->older->newer = e->newer; else mc.oldest = e->newer;
    if (e->newer) e->newer->older = e->older; else mc.newest = e->older;
    for (struct mc_entry **at = &mc.bucket[mc_slot(e->m.bytes)]; *at; at = &(*at)->chain)
        if (*at == e) { *at = e->chain; break; }
    mc.bytes -= e->m.bytes; --mc.entries;
}
static void mc_release(struct mc_entry *e) { e->chain = mc.spare; mc.spare = e; }
/* Give a pooled mapping back to Linux, as release_range does for an ordinary one. A retirement that
 * fails for good (not an interrupted ioctl, which is repeated) leaves the arena registered in the
 * module and mapped here with nobody remembering it: counted as lost, and the pool stops taking new
 * mappings, so a defect shows in the statistics and cannot grow without bound. */
static void mc_evict(struct mc_entry *e)
{
    int rc;
    mc_unlink(e);
    do { rc = ioctl(device, CV_RETIRE, &e->m.id); } while (rc && errno == EINTR);
    if (!rc) {
        if (!munmap(e->m.address, e->m.bytes)) { reuse_address = e->m.address; reuse_bytes = e->m.bytes; }
        else { ++mc.lost; mc.broken = 1; }
    } else { ++mc.lost; mc.broken = 1; }
    ++mc.evictions;
    mc_release(e);
}
static unsigned mc_flush(void)
{
    unsigned n = 0;
#ifdef CAPSTONE_MAPCACHE_MUTANT_NOFLUSH
    return 0;   /* Negative control: the pool is never given back under pressure. */
#endif
    while (mc.oldest) { mc_evict(mc.oldest); ++n; }
    if (n) ++mc.flushes;
    return n;
}
static void mc_trim(void)
{
#ifndef CAPSTONE_MAPCACHE_MUTANT_UNBOUNDED
    while (mc.oldest && (mc.bytes > mc.cap || mc.entries > MC_MAX_ENTRIES)) mc_evict(mc.oldest);
#else
    /* Negative control: no bound. */
#endif
}
/* May m (a registered, unheld mapping the domain asked to unmap) enter the pool? */
static int mc_eligible(const struct mapping *m)
{
#ifdef CAPSTONE_MAPCACHE_MUTANT_PROT
    /* Negative control: pool a mapping whose protections were changed. */
#define MC_PROTECTED 0
#else
#define MC_PROTECTED m->protected_range
#endif
    return mc.cap && !mc.broken && m->kind == CV_MAP_HEAP && !MC_PROTECTED &&
           m->bytes == m->visible && m->bytes <= mc.cap / 2 && mc.spare;
}
/* Pool m, which the module has parked (CV_UNMAP answered CV_UNMAP_PARKED), and remove it from the
 * registry. The park is the domain's unmap and goes through the same CV_UNMAP as the ordinary
 * path: one call decides park or retire, so with root_grants a refused reference is refused once
 * (unmap_arena_for). */
static void mc_pool(struct mapping *m)
{
    struct mc_entry *e = mc.spare;
    mc.spare = e->chain;
    e->m = *m;
    e->m.pages = NULL;   /* a pooled mapping's pages share one protection (mc_eligible) */
    map_remove(m);
    e->older = mc.newest; e->newer = NULL;
    if (mc.newest) mc.newest->newer = e; else mc.oldest = e;
    mc.newest = e;
    unsigned slot = mc_slot(e->m.bytes);
    e->chain = mc.bucket[slot]; mc.bucket[slot] = e;
    mc.bytes += e->m.bytes; ++mc.entries; ++mc.parks;
    if (trace) fprintf(stderr, "CAPSTONE_VM_MAPCACHE_PARK base=0x%lx bytes=%zu id=%llu\n",
                       (unsigned long)(uintptr_t)e->m.address, e->m.bytes, (unsigned long long)e->m.id);
    mc_trim();
}
/* A pooled mapping of exactly this size and an address that satisfies the alignment, newest first. */
static struct mc_entry *mc_find(size_t bytes, size_t alignment)
{
    for (struct mc_entry *e = mc.bucket[mc_slot(bytes)]; e; e = e->chain)
        if (e->m.bytes == bytes && !((uintptr_t)e->m.address & (alignment - 1))) return e;
    return NULL;
}
static long mapping_gate(const struct mapping *m, size_t bytes);
static long release_range(struct mapping *m);
/* Unmap the whole mapping at base for the domain thread tid (CV_SERVICE_UNMAP, base the cursor
 * of the reference it passed in a0): one CV_UNMAP pools the mapping where the pool takes it and
 * the module can, and retires it otherwise; with root_grants the module checks that reference,
 * the address and the length first. A partial unmap is refused here, with or without root_grants,
 * and again by the module with root_grants. */
static long unmap_arena_for(uintptr_t base, size_t bytes, uint64_t tid)
{
    struct mapping *m = map_find(base);
    long gate = mapping_gate(m, bytes);
    if (gate) return gate;
    int pool = mc_eligible(m);
    struct cv_unmap request = {.thread = tid, .id = m->id, .park = pool};
    if (ioctl(device, CV_UNMAP, &request)) return -errno;
    if (request.park == CV_UNMAP_PARKED) { mc_pool(m); return 0; }
    if (pool) {
        /* The module retired the mapping instead of parking it. */
        ++mc.refused;
        if (request.park == CV_UNMAP_NO_POOL) { mc_flush(); mc.cap = 0; }
    }
    return release_range(m);
}
/* May a request of the program end m, a mapping of exactly bytes: munmap, the frame's hand-over
 * at thread creation, the stack at thread exit? Only a registered heap mapping of that visible
 * length that is not the allocator's metadata (else EINVAL) and that no worker holds (else
 * EBUSY). The launcher's own mappings (image, stack, transport, startup data) never are. */
static long mapping_gate(const struct mapping *m, size_t bytes)
{
    if (!m || m->visible != bytes || !m->heap || m->kind == CV_MAP_METADATA)
        return -EINVAL;
    if (m->holder) return -EBUSY;
    return 0;
}
/* Give the range of m, whose arena the module has released, back to Linux and drop its
 * registration. */
static long release_range(struct mapping *m)
{
    if (munmap(m->address, m->bytes)) return -errno;
    reuse_address = m->address; reuse_bytes = m->bytes;
    map_remove(m);
    return 0;
}
/* The registered mapping that holds the page range [base, + bytes) that a request of the program
 * names (mprotect, madvise), or NULL with *error set: a nonempty range of whole pages inside the
 * visible length of one registered heap mapping that is not the allocator's metadata (EINVAL), and
 * that no worker holds (EBUSY, as mapping_gate answers): this launcher writes a held mapping
 * through Linux for the worker's whole attachment, and a protection change or a madvise would take
 * the pages from under that write. */
static struct mapping *range_mapping(uintptr_t base, size_t bytes, long *error)
{
    struct mapping *m = map_floor(base);
    uintptr_t at = m ? (uintptr_t)m->address : 0;
    *error = -EINVAL;
    if (!bytes || (base & 4095) || (bytes & 4095) || !m || !m->heap ||
        m->kind == CV_MAP_METADATA || base - at > m->visible || bytes > m->visible - (base - at))
        return NULL;
    *error = -EBUSY;
    if (m->holder) return NULL;
    *error = 0;
    return m;
}
/* The domain thread tid's mprotect of [base, + bytes) (CV_SERVICE_PROTECT, base the cursor of the
 * reference it passed in a0): a range that range_mapping accepts. With root_grants the module
 * checks the reference first (CV_PROTECT): a live reference to that mapping whose bounds cover the
 * range. The page state follows Linux exactly where mprotect succeeds and conservatively where it
 * fails (page_and before, page_set after); a change of part of the mapping gets its byte per page
 * before anything changes, so a failed allocation changes nothing. */
static long protect_range(uint64_t tid, uintptr_t base, size_t bytes, int prot)
{
    long error;
    if (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) return -EINVAL;
    struct mapping *m = range_mapping(base, bytes, &error);
    if (!m) return error;
    size_t first = (base - (uintptr_t)m->address) / 4096, end = first + bytes / 4096;
    if ((first || end != page_count(m)) && page_split(m)) return -ENOMEM;
    struct cv_protect request = {.thread = tid, .id = m->id};
    if (ioctl(device, CV_PROTECT, &request)) return -errno;
    m->protected_range = 1;
    page_and(m, first, end, prot);
    if (mprotect((void *)base, bytes, prot)) return -errno;
    page_set(m, first, end, prot);
    return 0;
}
/* The domain thread tid's madvise(MADV_DONTNEED or MADV_FREE) of [base, + bytes)
 * (CV_SERVICE_ADVISE, base the cursor of the reference in a0): a range that range_mapping accepts,
 * and with root_grants a reference that the module accepts as for mprotect (CV_ADVISE). Where the
 * module holds the pages (pinned) it overwrites them with zeros itself; otherwise (unpinned) Linux
 * gives the range back. Either way the range reads zero afterwards, which MADV_DONTNEED promises
 * for private anonymous memory and MADV_FREE allows. */
static long advise_range(uint64_t tid, uintptr_t base, size_t bytes, int advice)
{
    long error;
    if (advice != MADV_DONTNEED && advice != MADV_FREE) return -EINVAL;
    struct mapping *m = range_mapping(base, bytes, &error);
    if (!m) return error;
    struct cv_advise request = {.thread = tid, .id = m->id};
    if (ioctl(device, CV_ADVISE, &request)) return -errno;
    if (!request.zeroed && madvise((void *)base, bytes, MADV_DONTNEED)) return -errno;
    return 0;
}
/* The domain thread tid asks, before it moves a mapping (CV_SERVICE_MAPPING_CHECK, a0 the
 * reference, a1 the length; mapping.c, mremap), whether a request to end the mapping at base, of
 * exactly bytes, would be accepted, and whether every page of the mapping is read-write. Changes
 * nothing. First the gate of every request that ends a mapping (mapping_gate: EINVAL, or EBUSY
 * for a held mapping), then with root_grants the module's check of the reference (CV_PROTECT,
 * which reads the service from the caller's a7 and then checks the whole mapping, as for munmap),
 * then the protection: ENOTSUP unless every page of the visible length is exactly read-write and
 * the grant may write (the libc's mremap moves only such a mapping). */
static long check_mapping(uint64_t tid, uintptr_t base, size_t bytes)
{
    struct mapping *m = map_find(base);
    long gate = mapping_gate(m, bytes);
    if (gate) return gate;
    struct cv_protect request = {.thread = tid, .id = m->id};
    if (ioctl(device, CV_PROTECT, &request)) return -errno;
    if (!m->grant_writes || !page_all(m, 0, page_count(m), PROT_READ | PROT_WRITE, 1))
        return -ENOTSUP;
    return 0;
}

/* Before this launcher writes a span of domain memory through Linux on the domain's behalf, the
 * module admits every page of it that a lazily admitted arena does not hold yet (CV_TOUCH: the
 * same pin and checks as the domain's own fault). Without it the write would create backing the
 * domain can store a capability into with no pin and no sweep: at exit the frame is freed with
 * its tags before the module runs. The destinations: a thread's exit word and a worker's meta
 * block and exchange region, each after its checks (exit_word_refusal, transport_attach), and the
 * process transport's exchange region (exchange_prepare_write). */
static int touch(uintptr_t address, size_t bytes)
{
    struct cv_touch r = {address, bytes};
    int rc;
    /* Admission is idempotent, including pages pinned before an interruption. */
    do { rc = ioctl(device, CV_TOUCH, &r); } while (rc < 0 && errno == EINTR);
    return rc;
}
/* The initial exchange cannot be retired by domain services. Cache
 * only successful admissions for this mapping; worker transports instead repeat
 * their whole-span touch on every attachment, including address reuse. */
static unsigned char *exchange_admitted;
static int exchange_prepare_write(struct capstone_delegate_host *h, size_t offset, size_t bytes)
{
    size_t first = offset / 4096, end = (offset + bytes + 4095) / 4096;
    for (size_t i = first; i < end;) {
        if (__atomic_load_n(&exchange_admitted[i], __ATOMIC_RELAXED)) { ++i; continue; }
        size_t start = i++;
        while (i < end && !__atomic_load_n(&exchange_admitted[i], __ATOMIC_RELAXED)) ++i;
        if (touch((uintptr_t)h->exchange + start * 4096, (i - start) * 4096)) return errno;
        for (size_t j = start; j < i; ++j)
            __atomic_store_n(&exchange_admitted[j], 1, __ATOMIC_RELAXED);
    }
    return 0;
}
/* The registered mapping that holds the int at p inside its visible length, or NULL. */
static struct mapping *word_mapping(uintptr_t p)
{
    struct mapping *m = map_floor(p);
    size_t offset = m ? p - (uintptr_t)m->address : 0;
    return m && offset <= m->visible && m->visible - offset >= sizeof(int) ? m : NULL;
}
/* May this launcher write the bytes bytes at p through Linux for the program? They must lie inside
 * the visible length of one registered mapping whose grant may write, in pages that Linux maps
 * writable (the page state). */
static int writable_span(uintptr_t p, size_t bytes)
{
    struct mapping *m = map_floor(p);
    if (!m || !bytes || !m->grant_writes) return 0;
    size_t offset = p - (uintptr_t)m->address;
    if (offset > m->visible || bytes > m->visible - offset) return 0;
    return page_all(m, offset / 4096, (offset + bytes + 4095) / 4096, PROT_WRITE, 0);
}
/* Is every page of m writable for this launcher? */
static int mapping_writable(const struct mapping *m)
{
    return writable_span((uintptr_t)m->address, m->visible);
}
/* Does m hold a transport that this launcher writes through Linux by itself: the process
 * transport's meta block or exchange region, or a block that a worker holds? */
static int transport_holds(const struct mapping *m)
{
    return m->holder || m->address == meta || m->address == host.exchange;
}
/* Does [p, + bytes) overlap the whole extent of m, padding included? Overflow-safe: both lengths
 * are far below 2^63. */
static int span_overlaps(uintptr_t p, size_t bytes, const struct mapping *m)
{
    uintptr_t base = (uintptr_t)m->address;
    return p - base < m->bytes || base - p < bytes;
}
/* The kernel's reasons to refuse a word (CV_WORD_*, wire.h). */
static const char *const word_reasons[CV_WORD_REASONS] = {
    "accepted", "not a reference", "not live", "does not cover the word", "unaligned",
    "not inside one mapping", "in a thread's frame",
};
/* Why the kernel refuses the word at a0's cursor of thread tid, paused at its exit or its futex
 * wait (CV_WORD), or NULL: with root_grants a0 must be a live reference that covers the word, and
 * in both modes the word must be aligned, inside one mapping and in no thread's frame. */
static const char *word_refusal(uint64_t tid)
{
    struct cv_word request = {.thread = tid};
    if (ioctl(device, CV_WORD, &request) || request.reason >= CV_WORD_REASONS)
        return "refused by the kernel";
    return request.reason ? word_reasons[request.reason] : NULL;
}
/* Why the exit word at p, a0's cursor at thread tid's exit, may not be written, or NULL. First
 * this launcher's own conditions, on the address: an int inside one registered mapping, in pages
 * it may write (writable_span), outside every transport that it writes by itself. Then the
 * kernel's, on the reference in a0 (word_refusal), which also keeps the word out of every thread's
 * frame, the exiting thread's own included (the exit releases it before the store). Then the
 * module admits the word's page (touch). The caller checks the stack the exit names, which stays
 * mapped when the word lies in it. */
static const char *exit_word_refusal(uint64_t tid, uintptr_t p)
{
    struct mapping *m = word_mapping(p);
    const char *refused;
    if (!m) return "not inside one mapping";
    if (!writable_span(p, sizeof(int))) return "not writable";
    if (transport_holds(m)) return "in a transport";
    if ((refused = word_refusal(tid))) return refused;
    if (touch(p, sizeof(int))) return "not admitted";
    return NULL;
}
static long acquire_mapping(uint64_t tid, struct cv_step *step)
{
    size_t visible = step->args[0], alignment = step->args[1];
    unsigned prot = step->args[2], rights = step->args[3], kind = step->args[4];
    size_t bytes = rounded(visible);
    if (!bytes || (visible & 4095) || alignment < 4096 ||
        (alignment & (alignment - 1)) || alignment > CV_MAX_REGION_BYTES ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) || rights > 7 ||
        kind > CV_MAP_METADATA ||
        (kind != CV_MAP_APPLICATION && (prot != (PROT_READ | PROT_WRITE) || rights != 6)) ||
        (kind == CV_MAP_APPLICATION && rights != 7)) return -EINVAL;
    if (bytes < alignment) bytes = alignment;
    void *p;
    if (kind == CV_MAP_HEAP && mc.cap) {
        struct mc_entry *e = mc_find(visible, alignment);
        if (e && !map_reserve()) {
            struct cv_map r = {tid, (uintptr_t)e->m.address, visible, 6, 10, (uintptr_t)e->m.address, 1, e->m.id, 0, visible};
            if (!ioctl(device, CV_REVIVE, &r)) {
                mc_unlink(e);
                if (grants)
                    fprintf(stderr, "CAPSTONE_VM_ADD_OK slot=%zu base=0x%lx bytes=%zu reg=%u perms=%u linear=%d\n",
                            map_count, (unsigned long)(uintptr_t)e->m.address, visible, 10u, 6u, 1);
                /* A pooled mapping never had its protection changed (mc_eligible): read-write. */
                map_insert((struct mapping){.address = e->m.address, .bytes = visible, .visible = visible,
                                            .id = r.id, .heap = 1, .kind = CV_MAP_HEAP,
                                            .prot = PROT_READ | PROT_WRITE, .grant_writes = 1});
                if (trace) fprintf(stderr, "CAPSTONE_VM_MAPCACHE_HIT base=0x%lx bytes=%zu id=%llu\n",
                                   (unsigned long)(uintptr_t)e->m.address, visible, (unsigned long long)r.id);
                mc_release(e);
                ++mc.hits;
                return 0;
            }
            /* The module refused: nothing was handed out. Give the mapping back and map afresh. */
            ++mc.refused;
            mc_evict(e);
        }
        ++mc.misses;
    }
    for (int retried = 0;; retried = 1) {
    if (kind == CV_MAP_HEAP) {
        /* Keyed on the HEAP, not on the alignment value. A heap mapping has no padding
         * (bytes == visible, which the mapping pool requires), and the old `alignment == 4096`
         * spelling conflated that with the alignment mmap() happens to provide for free. Option
         * (iii) asks for the bounds granule, which exceeds a page above 4 MiB, so every heap of
         * 4 MiB or more silently left this path and got padded to a power of two by reserve(). */
        bytes = visible;
        p = reserve_aligned(bytes, alignment);
    } else p = reserve(bytes);
    if (!p && errno == ENOMEM && !retried && mc_flush()) continue;
    if (!p) return -errno;
    /* Padding remains inaccessible, including after later mprotect calls.
     * No capability reply exists until registration has succeeded. The range
     * is mapped readable and writable, so that protection needs no call. */
    if ((visible < bytes && mprotect((char *)p + visible, bytes - visible, PROT_NONE)) ||
        (prot != (PROT_READ | PROT_WRITE) && mprotect(p, visible, prot)) ||
        grant_mapping(p, bytes, visible, 10, rights, (uintptr_t)p, 1, tid, 0)) {
        int error = errno;
        munmap(p, bytes);
        if (error == ENOMEM && !retried && mc_flush()) continue;
        return -error;
    }
    break;
    }
    /* grant() registered p under this address; a miss is a registry defect,
     * and the launcher has nothing to answer with. */
    struct mapping *m = map_find((uintptr_t)p);
    if (!m) abort();
    m->visible = visible; m->kind = kind;
    m->prot = prot;
    return 0;
}
/* The error exit's cleanup. It does not unmap the registered mappings: die() runs on any worker
 * while the others may still be inside a delegated service that writes into their transport,
 * and an unmap here would bypass the transport hold (mapping_gate) and reopen the address to
 * reuse under that write. Linux releases the mappings at the process's exit, after exit_group
 * has stopped every worker, as on the normal exit path. */
static void cleanup(void)
{
    if (device >= 0) { close(device); device = -1; }
    if (image_fd >= 0) { close(image_fd); image_fd = -1; }
    capstone_spawner_stop(&spawner);
    capstone_delegate_host_free(&host);
}
/* End the process with 125. Its caller ends the process: a worker with _exit, the main thread by
 * returning from main. One thread does it. Another thread that fails afterwards (its CV_STEP finds
 * the device closed by the first one's cleanup) waits for that exit instead of running cleanup a
 * second time beside the first, which freed the host state twice (found in stage 4c: an attachment
 * refused by one worker while the main thread stepped ended with SIGSEGV instead of 125). */
static int die(const char *what)
{
    static int ending;
    int error = errno;
    if (__atomic_exchange_n(&ending, 1, __ATOMIC_ACQ_REL))
        for (;;) pause();
    fprintf(stderr, "capstone-vexec: %s: %s\n", what, strerror(error));
    cleanup(); return 125;
}
/* Whether the NUL-terminated section name at `at` in a string table of `len` bytes is `want`,
 * without reading past the table. */
static int section_named(const char *names, uint64_t len, uint64_t at, const char *want)
{
    size_t n = strlen(want) + 1;
    return at <= len && n <= len - at && !memcmp(names + at, want, n);
}
/* The image as virtual/link.ld lays it out, checked against the sealed file:
 * exactly two load segments, read+execute text from the image's first page
 * and read+write data starting on the page right after it. The text segment
 * is whole pages of the file (p_filesz == p_memsz), so mapping it maps nothing
 * but image bytes; the data segment's file part may end inside a page. */
struct image_layout {
    uint64_t low, text_offset, text_bytes, data_offset, data_file, data_bytes, entry;
    size_t stack;
};
static int image_layout(const unsigned char *raw, size_t size, struct image_layout *l)
{
    /* image_fd_layout validated this mapping before asking. A change below the VFS in between
     * (the residual named there) may still make these bytes garbage; it must not make them an
     * out-of-bounds read in this process. So each header field is read ONCE into a local,
     * checked against the file size before it is used as an offset, count or length, and the
     * checked local is what every later step uses. */
    const Elf64_Ehdr *h = (const void *)raw;
    const uint64_t phoff = h->e_phoff, shoff = h->e_shoff, entry_va = h->e_entry;
    const unsigned phnum = h->e_phnum, shnum = h->e_shnum, shstr = h->e_shstrndx;
    if (phoff > size || phnum > (size - phoff) / sizeof(Elf64_Phdr) ||
        shoff > size || shnum > (size - shoff) / sizeof(Elf64_Shdr) || shstr >= shnum)
        return -1;
    const Elf64_Phdr *ph = (const void *)(raw + phoff);
    const Elf64_Shdr *sh = (const void *)(raw + shoff);
    const uint64_t names_at = sh[shstr].sh_offset, names_len = sh[shstr].sh_size;
    if (names_at > size || names_len > size - names_at) return -1;
    const char *names = (const void *)(raw + names_at);
    Elf64_Phdr seg[2];
    unsigned loads = 0;
    int virtual = 0;
    for (unsigned i = 0; i < phnum; ++i) {
        const Elf64_Phdr p = ph[i];
        if (p.p_type != PT_LOAD) continue;
        if (loads == 2) return -1;
        seg[loads++] = p;
    }
    if (loads != 2) return -1;
    const Elf64_Phdr *t = &seg[0], *d = &seg[1];
    if (t->p_flags != (PF_R | PF_X) || d->p_flags != (PF_R | PF_W) ||
        (t->p_vaddr | t->p_offset | t->p_filesz | d->p_vaddr | d->p_offset) & 4095 ||
        !t->p_filesz || t->p_filesz != t->p_memsz ||
        d->p_vaddr != t->p_vaddr + t->p_memsz || d->p_filesz > d->p_memsz || !d->p_memsz ||
        entry_va < t->p_vaddr || entry_va - t->p_vaddr >= t->p_filesz)
        return -1;
    /* Each segment's sizes are bounded, so no sum below can wrap, and its file part lies in
     * the file. */
    if (t->p_memsz > CV_MAX_REGION_BYTES || d->p_memsz > CV_MAX_REGION_BYTES ||
        t->p_offset > size || t->p_filesz > size - t->p_offset ||
        d->p_offset > size || d->p_filesz > size - d->p_offset)
        return -1;
    l->stack = 1u << 20;
    for (unsigned i = 0; i < shnum; ++i) {
        uint64_t v[3];
        const Elf64_Shdr s = sh[i];
        if (s.sh_type != SHT_PROGBITS) continue;
        if (section_named(names, names_len, s.sh_name, ".capstone_virtual") && s.sh_size == 8 &&
            s.sh_offset <= size && 8 <= size - s.sh_offset) {
            memcpy(v, raw + s.sh_offset, 8);
            virtual = v[0] == CV_IMAGE_MAGIC;
        }
        if (section_named(names, names_len, s.sh_name, ".capstone_domreq") && s.sh_size == 24 &&
            s.sh_offset <= size && 24 <= size - s.sh_offset) {
            memcpy(v, raw + s.sh_offset, 24);
            l->stack = rounded(v[2]);
        }
    }
    if (!virtual || !l->stack) return -1;
    l->low = t->p_vaddr;
    l->text_offset = t->p_offset; l->text_bytes = t->p_memsz;
    l->data_offset = d->p_offset; l->data_file = d->p_filesz;
    l->data_bytes = (d->p_memsz + 4095) & ~(uint64_t)4095;
    l->entry = entry_va;
    return 0;
}
/* Map the image, validate the mapping and read the layout from it. One mapping, so the
 * headers and spans checked are the ones indexed; a second mapping would trust a check
 * made on other pages. The denial (CV_IMAGE) keeps every writer Linux counts out from
 * before this until the process ends; a writer below the VFS (a raw block device, a loop
 * device's backing file) is not stopped. Both the loader and exec in place (before it
 * replaces this process) ask. */
static int image_fd_layout(int fd, struct capstone_application_descriptor_v2 *desc,
                           struct image_layout *l)
{
    struct stat st;
    if (fstat(fd, &st)) return -1;
    if (!S_ISREG(st.st_mode) || st.st_size < (off_t)sizeof(Elf64_Ehdr) ||
        st.st_size > 512 * 1024 * 1024) { errno = ENOEXEC; return -1; }
    unsigned char *raw = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (raw == MAP_FAILED) return -1;
    if (capstone_application_validate(raw, st.st_size, desc)) {
        int error = errno;
        munmap(raw, st.st_size); errno = error; return -1;
    }
    int rc = image_layout(raw, st.st_size, l);
    munmap(raw, st.st_size);
    if (rc) errno = ENOEXEC;
    return rc;
}
/* Map the image instead of copying it. Text is a private, read+execute
 * mapping of the sealed file: no store reaches it, so it holds no tag and
 * the module neither pins nor sweeps it, and Linux faults it in from the page
 * cache as it is executed. Data is a private writable mapping of the file
 * (the module's pins break copy-on-write, so the domain owns its pages), the
 * page where the file part ends is built in anonymous memory, and the rest of
 * .bss stays anonymous. Linux synchronises the instruction cache when it maps
 * an executable page; the launcher writes none. */
/* As private_pages: kernels without transparent huge pages refuse the advice. */
static int no_huge_pages(void *p, size_t bytes)
{
    return madvise(p, bytes, MADV_NOHUGEPAGE) && errno != EINVAL ? -1 : 0;
}
static int load(int fd, struct capstone_application_descriptor_v2 *desc, void **image,
                size_t *bytes, size_t *rx_bytes, uintptr_t *entry, size_t *stack)
{
    struct image_layout l;
    if (image_fd_layout(fd, desc, &l)) return -1;
    /* The grant's size, rounded as every grant is. The tail past the data's own pages stays
     * anonymous and zero, like bss. */
    *bytes = rounded(l.text_bytes + l.data_bytes);
    *rx_bytes = l.text_bytes;
    *stack = l.stack;
    if (!*bytes || *bytes > CV_MAX_EAGER_BYTES) { errno = ENOEXEC; return -1; }
    char *base = reserve(*bytes);
    if (!base) return -1;
    char *data = base + l.text_bytes;
    size_t whole = l.data_file & ~(uint64_t)4095, part = l.data_file - whole;
    if (mmap(base, l.text_bytes, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_FIXED, fd,
             l.text_offset) == MAP_FAILED ||
        no_huge_pages(base, l.text_bytes) ||
        (whole && (mmap(data, whole, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd,
                        l.data_offset) == MAP_FAILED ||
                   no_huge_pages(data, whole))) ||
        (part && pread(fd, data + whole, part, l.data_offset + whole) != (ssize_t)part)) {
        int error = errno ? errno : EIO;
        munmap(base, *bytes); errno = error; return -1;
    }
    *image = base;
    *entry = (uintptr_t)base + l.entry - l.low;
    return 0;
}
struct exec_state {
  uint64_t magic;
  struct capstone_spawner spawner;
  unsigned child_count;
  int image;
  pid_t children[CAPSTONE_DELEGATE_CHILDREN];
};
#define EXEC_STATE_MAGIC UINT64_C(0x4350455845433031)

/* Deny writers to the image held by *fd (CV_IMAGE). Where the module cannot
 * (EOPNOTSUPP: a filesystem that does not keep one inode per file, such as
 * 9p), *fd becomes a sealed copy read from that same open file, reopened
 * read-only, and writers to the copy are denied instead. */
static int image_denied(int *fd, struct capstone_application_descriptor_v2 *desc)
{
    char path[64];
    if (capstone_application_fixed(*fd))
        return -1;
    if (!ioctl(device, CV_IMAGE, fd))
        return 0;
    if (errno == ENOTTY || errno == EINVAL) {
        /* An ioctl the module does not know: a module from before CV_IMAGE.
         * Module and launcher are one revision; there is no older path. */
        fprintf(stderr, "capstone-vexec: the capstone_vm module does not implement CV_IMAGE: "
                        "it is from another revision than this launcher\n");
        errno = ENOSYS;
        return -1;
    }
    if (errno != EOPNOTSUPP)
        return -1;
    int copy = capstone_application_copy(*fd, desc);
    if (copy < 0)
        return -1;
    snprintf(path, sizeof path, "/proc/self/fd/%d", copy);
    int sealed = capstone_application_open(path);
    int error = errno;
    close(copy);
    if (sealed < 0) {
        errno = error;
        return -1;
    }
    close(*fd);
    *fd = image_fd = sealed;
    return capstone_application_fixed(sealed) || ioctl(device, CV_IMAGE, fd) ? -1 : 0;
}

static int above_stdio(int fd) {
  if (fd < 0 || fd >= 3) return fd;
  int parked = fcntl(fd, F_DUPFD_CLOEXEC, 3);
  int error = errno;
  close(fd);
  errno = error;
  return parked;
}

static long exec_in_place(void) {
  static char *argv[CAPSTONE_SPAWN_STRINGS + 7], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  struct capstone_application_descriptor_v2 descriptor;
  host.exec_requested = 0;
  if (capstone_spawn_unpack(host.exec_block, host.exec_bytes, argv + 6,
                            CAPSTONE_SPAWN_STRINGS + 1, envp, CAPSTONE_SPAWN_STRINGS + 1, paths,
                            CAPSTONE_SPAWN_ACTIONS, &view))
    return -EINVAL;
  /* The new image is copied into a sealed memfd and checked here, while the
   * caller still exists to receive any error. The copy, not the file, is what
   * crosses the exec: nothing can change it between this check and the new
   * launcher's, and the module's denial of writers ends with this process. */
  int checked = above_stdio(capstone_application_image(view.path, &descriptor));
  if (checked < 0) return -errno;
  if (!view.argc) { close(checked); return -EINVAL; }
  /* An image this launcher cannot load fails here, while the caller still
   * exists to receive the error, not after the exec. */
  struct image_layout layout;
  if (image_fd_layout(checked, &descriptor, &layout)) { close(checked); return -ENOEXEC; }
  struct exec_state state = {.magic = EXEC_STATE_MAGIC, .spawner = spawner,
                            .child_count = host.child_count, .image = checked};
  memcpy(state.children, host.children, sizeof state.children);
  int fd = above_stdio(memfd_create("capstone-exec-state", MFD_CLOEXEC));
  if (fd < 0) { int error = errno; close(checked); return -error; }
  int error = 0;
  if (write(fd, &state, sizeof state) != sizeof state || lseek(fd, 0, SEEK_SET) < 0 ||
      fcntl(fd, F_SETFD, 0) < 0 || fcntl(checked, F_SETFD, 0) < 0 ||
      (spawner.socket >= 0 && fcntl(spawner.socket, F_SETFD, 0) < 0)) {
    error = errno ? errno : EIO;
  } else {
    char fd_string[32];
    snprintf(fd_string, sizeof fd_string, "%d", fd);
    argv[0] = spawner.self;
    argv[1] = "--resume";
    argv[2] = fd_string;
    argv[3] = "--application-argv";
    argv[4] = (char *)view.path;
    argv[5] = "--";
    execve(spawner.self, argv, envp);
    error = errno;
  }
  close(fd);
  close(checked);
  if (spawner.socket >= 0) fcntl(spawner.socket, F_SETFD, FD_CLOEXEC);
  return -error;
}

/* The domain's main stack, as reserved and granted in main(): the bounds the frame walk below
 * checks every read against. */
static void *domain_stack;
static size_t domain_stack_bytes;

/* THE FRAME CHAIN: the return addresses up the domain's stack from the faulting frame, read by
 * the launcher (the stack is this process's own mapping, granted to the domain, and the domain
 * is stopped on a terminal step), each frame validated before it is read, bounded. The
 * compiler's prologue stores ra as a scalar at fp-16 and the caller's fp as a capability at
 * fp-32 (`sd ra, 0x40(sp); stc s0, 0x30(sp); s0 = sp + 0x50`), and a capability slot's low word
 * is its cursor, which is how the module reads the register slots (capstone_vm.c `r.fp =
 * t->frame[24]`). So a frame is {fp-32: the caller's fp, fp-16: the return address}. The chain
 * stops at a frame outside the stack or misaligned, a zero return address, a caller's fp not
 * above this one, or the 16th frame. A thread's stack is not this mapping, so a worker's fault
 * gets no chain rather than a wrong one. Diagnostic: nothing here decides on it. */
static unsigned frame_chain(uint64_t fp, uint64_t *out, unsigned max)
{
    uint64_t lo = (uint64_t)(uintptr_t)domain_stack;
    if (!domain_stack) return 0;
    /* the walk itself lives in the delegate service, where a native test can hand it a
     * buffer (runtime/tests/application/delegate-service-test.c) */
    return capstone_delegate_frame_chain(fp, lo, lo + domain_stack_bytes, out, max);
}

static void fault(const struct cv_step *step)
{
    struct capstone_delegate_host *h = worker_host ? worker_host : &host;
    uint64_t chain[16];
    unsigned chain_len = step->fp ? frame_chain(step->fp, chain, 16) : 0;
    if (!h->image_sha256[0] && image_fd >= 0)
        capstone_application_hash(image_fd, h->image_sha256);
    const char *record = getenv("CAPSTONE_FAULT_RECORD");
    int fd = record && *record ? open(record, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600) : -1;
    if (fd >= 0) {
        capstone_delegate_fault_record(fd, h, image_path, step->cause, step->pc, step->address,
                                       step->ra, step->fp, chain, chain_len);
        close(fd);
    }
    if (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))
        capstone_delegate_fault_record(2, h, image_path, step->cause, step->pc, step->address,
                                       step->ra, step->fp, chain, chain_len);
    /* Signal death stops the process before Linux releases shared mappings. */
    capstone_domain_exit_on_fault(CAPSTONE_DOMAIN_FAULT_RETVAL, NULL, NULL);
}
static int reserve_stdio(unsigned *mask) {
  *mask = 0;
  for (int i = 0; i < 3; ++i) {
    if (fcntl(i, F_GETFD) >= 0) {
      *mask |= 1u << i;
      continue;
    }
    if (errno != EBADF)
      return -1;
    int fd = open("/dev/null", O_RDWR);
    if (fd < 0)
      return -1;
    if (fd != i) {
      int rc = dup2(fd, i);
      close(fd);
      if (rc < 0)
        return -1;
    }
  }
  return 0;
}

/* Undo a CV_THREAD_CREATE that succeeded: stop the new thread (it never ran) and release its
 * frame's mapping, which the module names in its answer. The create ended every reference of the
 * program to the frame (with root_grants the module ended the reservation; without, it retired
 * the frame's ancestor), so the program cannot unmap the frame itself any more; without this
 * release the frame would stay pinned and registered until exit. */
static void thread_rollback(uint64_t thread, struct mapping *frame)
{
    struct cv_thread_exit rollback = {.thread = thread, .frame = frame->id};
    long rc = ioctl(device, CV_THREAD_EXIT, &rollback) ? -errno : 0;
    if (!rc && !(rollback.released & CV_EXIT_FRAME)) rc = -EBUSY;
    if (!rc) rc = release_range(frame);
    if (rc)
        fprintf(stderr, "CAPSTONE_VM_FRAME_RELEASE_FAILED frame=0x%lx error=%ld\n",
                (unsigned long)(uintptr_t)frame->address, -rc);
}

/* Attach the calling worker's own transport at its first delegate service: a0 the meta block's
 * reference, a1 the exchange region's, a2 its length (request is the meta block). This launcher
 * writes both blocks from here on, so each must be a whole mapping of the program that the program
 * may end and that no worker holds (mapping_gate, which also refuses this launcher's own transport:
 * it is no heap mapping), the two distinct, every page of both writable (mapping_writable: a
 * protection the program set before the attachment would make those writes fault, and the hold
 * keeps it from changing afterwards), and with root_grants both references must pass the module's
 * whole-mapping check, as for munmap and ending nothing (CV_ATTACH, which also refuses a mapping
 * that holds a thread's frame). Then every page of both is admitted (CV_TOUCH, at every
 * attachment: a re-attachment after a detach may name a fresh mapping at the same address), and the
 * worker holds both mappings until it detaches or exits. NULL, or the reason of a refusal, which
 * ends the program, as every malformed transport does. */
static const char *transport_attach(uint64_t tid, const struct cv_step *step,
                                    struct capstone_delegate_entry *request)
{
    struct mapping *held_meta = map_find(step->args[0]);
    struct mapping *held_exchange = map_find(step->args[1]);
    long gate = mapping_gate(held_meta, CAPSTONE_DELEGATE_META_BYTES);
    if (!gate) gate = mapping_gate(held_exchange, step->args[2]);
    if (gate == -EBUSY) return "held by a worker";
    if (gate || held_meta == held_exchange) return "not a whole mapping of the program";
    if (!mapping_writable(held_meta) || !mapping_writable(held_exchange)) return "not writable";
    struct cv_attach check = {.thread = tid, .meta = held_meta->id, .exchange = held_exchange->id,
                              .meta_bytes = CAPSTONE_DELEGATE_META_BYTES};
    if (ioctl(device, CV_ATTACH, &check))
        return errno == EBUSY ? "holds a thread's frame" : "reference refused";
    if (touch(step->args[0], CAPSTONE_DELEGATE_META_BYTES) || touch(step->args[1], step->args[2]))
        return "not admitted";
    worker_host = calloc(1, sizeof *worker_host);
    if (!worker_host) return "no memory";
    held_meta->holder = held_exchange->holder = tid;
    transport_at[0].at = step->args[0]; transport_at[0].id = held_meta->id;
    transport_at[1].at = step->args[1]; transport_at[1].id = held_exchange->id;
    worker_host->exchange = (void *)(uintptr_t)step->args[1];
    worker_host->exchange_bytes = step->args[2];
    worker_host->private_count = host.private_count;
    memcpy(worker_host->private_fds, host.private_fds, sizeof host.private_fds);
    worker_host->entry_address = host.entry_address;
    worker_host->code_base = host.code_base;
    worker_host->code_end = host.code_end;
    memcpy(worker_host->image_sha256, host.image_sha256, sizeof host.image_sha256);
    struct capstone_signal_block *block = (void *)((char *)request + CAPSTONE_SIGNAL_OFFSET);
    uint64_t inherited_mask = block->initial_mask;
    capstone_signals_init_context(&worker_host->signals, block, &host.signals);
    worker_host->signals.logical = inherited_mask;
    block->initial_mask = inherited_mask;
    capstone_signals_attach(&worker_host->signals);
    return NULL;
}

static int virtual_service(uint64_t tid, struct cv_step *step)
{
    if (trace) fprintf(stderr, "CAPSTONE_VM_BOUNDARY kind=%llu\n",
                       (unsigned long long)step->args[7]);
    step->reply = 1;
    step->result = 0;
    if (step->args[7] == CV_SERVICE_THREAD_SELF) {
        step->result = tid + 1; /* zero remains the wire lock's vacant value */
    } else if (step->args[7] == CV_SERVICE_THREAD_CREATE) {
        /* a0 is the frame's reference (thread.c), its cursor the frame's address. The frame must
         * be a whole one-page mapping that the program may end (mapping_gate): the hand-over ends
         * every reference of the program to it. With root_grants the module checks the reference
         * in the caller's a0 and ends the frame's mapping only if the thread is created. After a
         * successful CV_THREAD_CREATE the frame belongs to the module: if a later step here fails,
         * the rollback stops the thread and releases the frame's mapping, since every reference of
         * the program to it has ended. EAGAIN tells the program so (vm-abi.h); every other error
         * leaves the frame the program's. */
        struct mapping *frame = map_find(step->args[0]);
        struct cv_thread_create request = {.frame = step->args[0], .creator = tid};
        long gate = mapping_gate(frame, 4096);
        if (gate) {
            step->result = gate;
        } else if (ioctl(device, CV_THREAD_CREATE, &request)) {
            step->result = -errno;
        } else if (thread_add(request.thread, frame)) {
            /* The rollback gives the id back to the module (the thread never ran). */
            thread_rollback(request.thread, frame);
            step->result = -EAGAIN;
        } else {
            uint64_t *child = malloc(sizeof(*child));
            pthread_t worker;
            if (!child || pthread_create(&worker, NULL, virtual_thread_worker,
                                         (child ? (*child = request.thread), child : NULL))) {
                free(child);
                /* The thread never ran and its id was never published: the entry goes, and the
                 * module issues the id again. */
                thread_rollback(request.thread, frame);
                thread_withdraw(request.thread);
                step->result = -EAGAIN;
            } else {
                pthread_detach(worker);
                thread_publish(request.thread);
                step->result = request.thread;
            }
        }
    } else if (step->args[7] == CV_SERVICE_THREAD_EXIT || step->args[7] == CV_SERVICE_PTHREAD_EXIT) {
        int posix = step->args[7] == CV_SERVICE_PTHREAD_EXIT;
        uintptr_t clear = posix ? step->args[0] : 0;
        if (tid == 0) return -1;
        /* The exit word (a0, its reference; clear is its cursor) is the program's: this launcher
         * stores 0 into it through Linux and wakes its futex waiters after the thread has stopped.
         * With root_grants the kernel accepts it only through a live reference that covers it, so
         * the launcher never writes memory that the program holds no reference to; in both modes
         * the kernel keeps it out of every thread's frame (exit_word_refusal). The module releases
         * the thread's frame and the stack it names before the store, so the word must lie in
         * neither: a store into a range just given back to Linux would land in whatever Linux maps
         * there next. musl's word is a global (the thread-list lock), never in the thread's stack.
         * The stack check uses the mapping this launcher found, with its whole extent, never the
         * base and length the program names. A word that fails (exit_word_refusal) is not written
         * and the refusal is logged: the exit goes on, joiners through the launcher return at
         * thread_finish as after every exit, and futex waiters on the word stay blocked, which is
         * the program's bug. A word in the stack makes the stack request a wrong one: the stack
         * stays mapped and registered, and the word is written. */
        struct mapping *frame = thread_frame_mapping(tid);
        struct mapping *stack = posix && step->args[1] ? map_find(step->args[1]) : NULL;
        const char *refused = clear ? exit_word_refusal(tid, clear) : NULL;
        if (refused) {
            fprintf(stderr, "CAPSTONE_VM_EXIT_WORD_REFUSED tid=%llu word=0x%lx reason=%s\n",
                    (unsigned long long)tid, (unsigned long)clear, refused);
            clear = 0;
        }
        int word_in_stack = clear && stack && span_overlaps(clear, sizeof(int), stack);
        /* A worker that exits still attached gives its holds back here, so they keep none of its
         * mappings. */
        transport_detach(tid);
        /* The frame, and the stack that the program asks to unmap (a1 the stack's reference,
         * pthread.c __unmapself, a2 its length), each only where the program may end it. With
         * root_grants the stack is the program's request: the module checks the reference while
         * the frame still holds it and ends the stack's reservation after the thread has stopped.
         * A refused stack stays mapped and registered, and the exit goes on. A stack that is the
         * thread's own frame is the exception: the module refuses it as a stack, and the frame's
         * release still ends that mapping. */
        struct cv_thread_exit request = {.thread = tid};
        if (frame && !mapping_gate(frame, 4096)) request.frame = frame->id;
        if (stack && !word_in_stack && !mapping_gate(stack, step->args[2]))
            request.stack = stack->id;
        if (ioctl(device, CV_THREAD_EXIT, &request))
            return -1;
        if (!(request.released & CV_EXIT_FRAME) || release_range(frame))
            fprintf(stderr, "CAPSTONE_VM_FRAME_RELEASE_FAILED frame=0x%lx\n",
                    frame ? (unsigned long)(uintptr_t)frame->address : 0UL);
        if (posix && step->args[1] &&
            (!(request.released & CV_EXIT_STACK) || release_range(stack)))
            fprintf(stderr, "CAPSTONE_VM_STACK_UNMAP_REFUSED tid=%llu base=0x%lx bytes=%lu\n",
                    (unsigned long long)tid, (unsigned long)step->args[1],
                    (unsigned long)step->args[2]);
        /* musl join may retire the child's stack/TLS after this store.
         * The virtual CPU has stopped and its tagged frame is gone first. */
        if (clear) {
            __atomic_store_n((int *)clear, 0, __ATOMIC_RELEASE);
            syscall(SYS_futex, (int *)clear, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
        }
        /* A joiner through the launcher (CV_SERVICE_THREAD_JOIN) returns only after everything
         * above: the frame and the stack are released or kept, and the exit word is stored or
         * refused. */
        thread_finish(tid);
        step->reply = 0;
        return 1;
    } else if (step->args[7] == CV_SERVICE_FUTEX) {
        /* a0 the word's reference. A wait makes Linux read the word, so the kernel checks the
         * reference first, as for an exit word (word_refusal). A wake neither reads nor writes the
         * word: Linux only hashes the address of a private futex. Its integer address must still
         * be an aligned int inside one registered mapping, which keeps it off this launcher's own
         * futexes. The wait itself runs with service_lock released, after the check: another thread
         * of the program can end the mapping in between, and the wait then compares against
         * whatever Linux maps there, which affects only the program itself. */
        int op = (int)step->args[1];
        int command = op & FUTEX_CMD_MASK;
        int wait = command == FUTEX_WAIT || command == FUTEX_WAIT_BITSET;
        const char *refused = NULL;
        if ((step->args[0] & 3) || !word_mapping(step->args[0])) step->result = -EFAULT;
        else if (!wait && command != FUTEX_WAKE && command != FUTEX_WAKE_BITSET)
            step->result = -ENOSYS;
        else if (wait && (refused = word_refusal(tid))) {
            fprintf(stderr, "CAPSTONE_VM_FUTEX_REFUSED tid=%llu word=0x%lx reason=%s\n",
                    (unsigned long long)tid, (unsigned long)step->args[0], refused);
            step->result = -EFAULT;
        } else {
            struct timespec timeout = {(long)step->args[3], (long)step->args[4]};
            pthread_mutex_unlock(&service_lock);
            long result = syscall(SYS_futex, (int *)(uintptr_t)step->args[0],
                                  op | FUTEX_PRIVATE_FLAG, (int)step->args[2],
                                  timeout.tv_sec < 0 ? NULL : &timeout, NULL, (int)step->args[5]);
            int error = errno;
            pthread_mutex_lock(&service_lock);
            step->result = result < 0 ? -error : result;
        }
    } else if (step->args[7] == CV_SERVICE_THREAD_JOIN) {
        /* The child must acquire the same transport lock to publish its
         * exit.  Do not wait while holding it, or join would deadlock. */
        pthread_mutex_unlock(&service_lock);
        step->result = thread_join(tid, step->args[0]);
        pthread_mutex_lock(&service_lock);
    } else if (step->args[7] == CV_SERVICE_DELEGATE ||
               step->args[7] == CV_SERVICE_THREAD_DELEGATE) {
        int separate = step->args[7] == CV_SERVICE_THREAD_DELEGATE;
        struct capstone_delegate_entry *request = separate ? (void *)(uintptr_t)step->args[0] : meta;
        struct capstone_delegate_host *h = &host;
        if (separate) {
            if (step->args[0] == (uintptr_t)meta) {
                /* The process transport: this launcher's own meta block and exchange region, the
                 * system-call region it shares with the program. They are named by address and
                 * compared with the launcher's own, never resolved. */
                if (h->exchange != (void *)(uintptr_t)step->args[1] ||
                    h->exchange_bytes != step->args[2]) return -1;
            } else if (tid) {
                if (!worker_host) {
                    const char *refused = transport_attach(tid, step, request);
                    if (refused) {
                        fprintf(stderr, "CAPSTONE_VM_TRANSPORT_REFUSED tid=%llu meta=0x%lx"
                                        " exchange=0x%lx reason=%s\n",
                                (unsigned long long)tid, (unsigned long)step->args[0],
                                (unsigned long)step->args[1], refused);
                        return -1;
                    }
                }
                h = worker_host;
                if (h->exchange != (void *)(uintptr_t)step->args[1] ||
                    h->exchange_bytes != step->args[2] ||
                    h->signals.block != (void *)((char *)request + CAPSTONE_SIGNAL_OFFSET)) return -1;
            } else return -1;
        }
        if (request->version != CAPSTONE_DELEGATE_VERSION) return -1;
        /* Each native worker owns its bounce buffer and signal ring, or serves the shared
         * process host under host_lock. A blocked epoll/read must not stop another virtual
         * context that does not share its transport. */
        if (separate) pthread_mutex_unlock(&service_lock);
        if (h == &host) {
            pthread_mutex_lock(&host_lock);
            /* A worker on the process transport records its signals into the process ring
             * (it shares the domain's signal state with every context on that transport). */
            if (tid && separate) {
                capstone_signals_attach(&host.signals);
                host_signals_attached = 1;
            }
        }
        capstone_delegate_serve(h, request);
        if (h == &host) pthread_mutex_unlock(&host_lock);
        if (separate) pthread_mutex_lock(&service_lock);
        if (trace) fprintf(stderr, "CAPSTONE_VM_SERVICE tid=%llu nr=%llu result=%lld\n",
                           (unsigned long long)tid, (unsigned long long)request->nr,
                           (long long)request->result);
        if (h->exec_requested) request->result = tid ? -ENOSYS : exec_in_place();
        if (h->exiting) {
            struct cv_stats stats;
            if (tid == 0 && getenv("CAPSTONE_VM_STATS_MARK")) {
                /* A mark in the kernel log right before the counters are read, so a gate can
                 * set the module's per-arena census (dmesg) against pages=: the arenas retired
                 * after the mark are the ones pages= still counts. Its own variable, not
                 * CAPSTONE_VM_STATS: a guest whose console is up prints the line to its UART,
                 * ~0.5 M instruction attempts in a profile that sets CAPSTONE_VM_STATS. */
                int kmsg = open("/dev/kmsg", O_WRONLY | O_CLOEXEC);
                if (kmsg >= 0) {
                    if (write(kmsg, "CAPSTONE_VM_STATS_MARK\n", 23) < 0) { /* diagnostic only */ }
                    close(kmsg);
                }
            }
            if (tid == 0 && (getenv("CAPSTONE_VM_STATS") || getenv("CAPSTONE_DELEGATE_STATS")) &&
                !ioctl(device, CV_STATS, &stats)) {
                fprintf(stderr, "CAPSTONE_VM_STATS arenas=%llu pages=%llu peak=%llu nodes=%llu steps=%llu faults=%llu collections=%llu reclaimed=%llu rounds=%llu bytes_in=%llu bytes_out=%llu node_bytes=%llu node_capacity=%llu node_growths=%llu launch_ns=%llu elapsed_ns=%llu\n",
                    stats.arenas, stats.pinned_pages, stats.peak_pages, stats.nodes, stats.steps,
                    stats.faults, stats.collections, stats.reclaimed, (unsigned long long)host.rounds,
                    (unsigned long long)host.bytes_in, (unsigned long long)host.bytes_out,
                    stats.node_bytes, stats.node_capacity, stats.node_growths,
                    launch_ns, now_ns() - start_ns);
                /* Linux's own residency of this process, an independent witness for gates that
                 * must not rely on the module's counters (unpinned mode keeps none). Off unless
                 * CAPSTONE_VM_RSS is set: reading /proc costs instructions a profile run counts. */
                if (getenv("CAPSTONE_VM_RSS")) {
                    FILE *status = fopen("/proc/self/status", "re");
                    unsigned long rss = 0, hwm = 0, anon = 0;
                    char text[256];
                    while (status && fgets(text, sizeof text, status)) {
                        sscanf(text, "VmRSS: %lu kB", &rss);
                        sscanf(text, "VmHWM: %lu kB", &hwm);
                        sscanf(text, "RssAnon: %lu kB", &anon);
                    }
                    if (status) fclose(status);
                    fprintf(stderr, "CAPSTONE_VM_RSS rss_pages=%lu hwm_pages=%lu anon_pages=%lu\n",
                            rss / 4, hwm / 4, anon / 4);
                }
                /* Its own line: the STATS line's fields are all integers, and its collectors
                 * int() every value. The service keys are vm-abi.h CV_SERVICE_* ids. */
                fprintf(stderr, "CAPSTONE_VM_STEPS quantum=%lu service=%lu fault=%lu other=%lu"
                                " service0_delegate=%lu service1_map=%lu service2_unmap=%lu"
                                " service3_thread_create=%lu service4_thread_exit=%lu"
                                " service5_thread_join=%lu service6_protect=%lu service7_wait=%lu"
                                " service8_futex=%lu service9_pthread_exit=%lu"
                                " service10_thread_delegate=%lu service11_thread_self=%lu"
                                " service12_nodes=%lu service13_unassigned=%lu"
                                " service14_transport_detach=%lu service15_mapping_check=%lu"
                                " service16_advise=%lu service_other=%lu\n",
                    step_count(&step_kinds[1]), step_count(&step_kinds[3]), step_count(&step_kinds[4]),
                    step_count(&step_kinds[0]) + step_count(&step_kinds[2]) + step_count(&step_kinds[5]) +
                        step_count(&step_kinds[6]) + step_count(&step_kinds[7]),
                    step_count(&step_services[0]), step_count(&step_services[1]),
                    step_count(&step_services[2]), step_count(&step_services[3]),
                    step_count(&step_services[4]), step_count(&step_services[5]),
                    step_count(&step_services[6]), step_count(&step_services[7]),
                    step_count(&step_services[8]), step_count(&step_services[9]),
                    step_count(&step_services[10]), step_count(&step_services[11]),
                    step_count(&step_services[12]), step_count(&step_services[13]),
                    step_count(&step_services[14]), step_count(&step_services[15]),
                    step_count(&step_services[16]), step_count(&step_services[STEP_SERVICE_COUNT]));
                /* The thread table's census at the program's exit: count is the number of entries
                 * in use, the main thread's included, so 1 once every other thread has exited and
                 * its last joiner (if any) has left. An entry freed twice shows as 0 or less. */
                pthread_mutex_lock(&thread_state_lock);
                fprintf(stderr, "CAPSTONE_VM_THREADS count=%u issued=%llu\n", thread_count,
                        (unsigned long long)thread_last_id);
                pthread_mutex_unlock(&thread_state_lock);
            }
            if (tid == 0 && mc.cap && (getenv("CAPSTONE_VM_STATS") || getenv("CAPSTONE_DELEGATE_STATS")))
                fprintf(stderr, "CAPSTONE_VM_MAPCACHE cap=%zu parks=%lu hits=%lu misses=%lu evictions=%lu"
                                " flushes=%lu refused=%lu lost=%lu pooled_entries=%u pooled_bytes=%zu\n",
                        mc.cap, mc.parks, mc.hits, mc.misses, mc.evictions, mc.flushes, mc.refused,
                        mc.lost, mc.entries, mc.bytes);
            /* The image's identity on a run that ENDS, not only on one that faults: a
             * transcript that carries no digest cannot be tied to the image it claims to be
             * from, and a judge that reads a probe's successful run (the putenv probe) has
             * nothing else to check it against. Same gate as the fault record. */
            if (tid == 0 && (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))) {
                if (!host.image_sha256[0] && image_fd >= 0)
                    capstone_application_hash(image_fd, host.image_sha256);
                fprintf(stderr, "capstone-exec: exit image=%s sha256=%s\n",
                        image_path ? image_path : "?", host.image_sha256);
            }
            const char *profile = tid == 0 ? getenv("CAPSTONE_PROFILE_RECORD") : NULL;
            if (profile && *profile) {
                /* No digest here: hashing a large image is most of a short run's
                 * instructions. The profiler hashes the image itself. */
                /* O_NONBLOCK: a FIFO without a reader fails to open instead of blocking the exit,
                 * and a full pipe fails the write instead of waiting; a regular file is unaffected. */
                int fd = open(profile, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | O_NONBLOCK, 0600);
                if (fd >= 0) {
                    capstone_delegate_image_record(fd, h, image_path);
                    close(fd);
                }
            }
            /* exit_group stops all native workers before Linux releases
             * mappings and the adapter fd. Unmapping here would race a
             * worker completing its blocked delegated read. */
            _exit(h->exit_status);
        }
    } else if (step->args[7] == CV_SERVICE_MAP) {
        step->result = acquire_mapping(tid, step);
    } else if (step->args[7] == CV_SERVICE_MAPPING_CHECK) {
        step->result = check_mapping(tid, step->args[0], step->args[1]);
    } else if (step->args[7] == CV_SERVICE_UNMAP) {
        step->result = unmap_arena_for(step->args[0], step->args[1], tid);
    } else if (step->args[7] == CV_SERVICE_TRANSPORT_DETACH) {
        /* The worker gives its transport back (pthread.c, before it unmaps the blocks). */
        transport_detach(tid);
    } else if (step->args[7] == CV_SERVICE_PROTECT) {
        step->result = protect_range(tid, step->args[0], step->args[1], step->args[2]);
    } else if (step->args[7] == CV_SERVICE_ADVISE) {
        step->result = advise_range(tid, step->args[0], step->args[1], (int)step->args[2]);
    } else if (step->args[7] == CV_SERVICE_WAIT) {
        /* The heap's contended atomic mutex must let its owner run, including
         * the owner's mapping service. Never yield with the wire lock held. */
        pthread_mutex_unlock(&service_lock);
        sched_yield();
        pthread_mutex_lock(&service_lock);
    } else if (step->args[7] == CV_SERVICE_NODES) {
        struct cv_nodes nodes = {.thread = tid, .available = step->args[0]};
        step->result = ioctl(device, CV_NODES, &nodes) ? -errno : 0;
    } else {
        step->result = -ENOSYS;
    }
    return 0;
}

static int virtual_service_loop(uint64_t tid)
{
    struct cv_step step = {.thread = tid};
    for (;;) {
        while (ioctl(device, CV_STEP, &step))
            if (errno != EINTR && errno != EAGAIN) return die("step");
        __atomic_fetch_add(&step_kinds[step.kind < 8 ? step.kind : 7], 1, __ATOMIC_RELAXED);
        if (step.kind == 3)
            __atomic_fetch_add(&step_services[step.args[7] < STEP_SERVICE_COUNT
                                                  ? step.args[7] : STEP_SERVICE_COUNT],
                               1, __ATOMIC_RELAXED);
        step.reply = 0;
        if (step.kind == 1) {
            /* The supervisor quantum is the preemption point. Yielding here
             * lets Linux schedule another launcher thread on this one hart. */
            sched_yield();
            continue;
        }
        if (step.kind == 4) {
            struct cv_thread_control control = {.thread = step.thread};
            if (!ioctl(device, CV_RESOLVE, &control)) continue;
        }
        if (step.kind != 3) {
            pthread_mutex_lock(&service_lock);
            if (worker_host)
                worker_host->preparing_nr = ((struct capstone_delegate_entry *)
                    ((char *)worker_host->signals.block - CAPSTONE_SIGNAL_OFFSET))->nr;
            else
                host.preparing_nr = ((struct capstone_delegate_entry *)meta)->nr;
            fault(&step);
            pthread_mutex_unlock(&service_lock);
            return 125;
        }
        pthread_mutex_lock(&service_lock);
        int result = virtual_service(tid, &step);
        pthread_mutex_unlock(&service_lock);
        if (result == 1) return 0;
        if (result < 0) return die("service");
        if (result & 2) return result >> 8;
    }
}

static void *virtual_thread_worker(void *opaque)
{
    uint64_t tid = *(uint64_t *)opaque;
    free(opaque);
    int result = virtual_service_loop(tid);
    /* A virtual thread is hosted by a Linux worker, so normal virtual-thread
     * exit must return from this worker rather than terminate the process. */
    if (result) _exit(result);
    /* The exit service detached the worker's transport (transport_detach). */
    return NULL;
}

int main(int argc, char **argv)
{
    int binfmt = (getauxval(AT_FLAGS) & AT_FLAGS_PRESERVE_ARGV0) != 0;
    if (binfmt && argc < 3) return 125;
    struct exec_state resumed = {0};
    if (!binfmt && argc > 3 && !strcmp(argv[1], "--resume")) {
        char *end; long fd = strtol(argv[2], &end, 10);
        if (*end || fd < 3 || fd > 65535 || read(fd, &resumed, sizeof resumed) != sizeof resumed ||
            resumed.magic != EXEC_STATE_MAGIC || resumed.child_count > CAPSTONE_DELEGATE_CHILDREN)
            return 125;
        close(fd);
        if (resumed.spawner.socket >= 0 && fcntl(resumed.spawner.socket, F_SETFD, FD_CLOEXEC) < 0)
            return 125;
        spawner = resumed.spawner; host.child_count = resumed.child_count;
        memcpy(host.children, resumed.children, sizeof resumed.children);
        argc -= 2; argv += 2;
    }
    int app_args = !binfmt && argc >= 5 && !strcmp(argv[1], "--application-argv") && !strcmp(argv[3], "--");
    int literal = binfmt || (argc > 1 && !strcmp(argv[1], "--"));
    if (literal && !binfmt) { --argc; ++argv; }
    if (argc < 2) { fprintf(stderr, "usage: capstone-vexec [--] PROGRAM.dom [ARG...]\n"); return 2; }
    if (!literal && argc == 2 && !strcmp(argv[1], "--stats")) {
        int fd = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
        struct cv_global stats;
        if (fd < 0 || ioctl(fd, CV_GLOBAL, &stats)) return 125;
        close(fd);
        printf("{\"version\":1,\"live_domains\":%llu,\"live_regions\":%llu,\"live_bytes\":%llu,\"cached_bytes\":0,\"poisoned_blocks\":0,\"nodes_high_water\":%llu,\"nodes_live\":%llu,\"nodes_retired\":%llu,\"nodes_allocated_total\":%llu,\"tag_pages\":%llu,\"node_capacity\":%llu,\"node_bytes\":%llu,\"collections\":%llu,\"nodes_reclaimed\":%llu}\n",
            stats.contexts, stats.arenas, stats.pinned_pages * 4096,
            stats.nodes_high_water, stats.nodes_live, stats.nodes_retired, stats.nodes_allocated,
            stats.pinned_pages, stats.node_capacity, stats.node_bytes, stats.collections, stats.reclaimed);
        return 0;
    }
    image_path = app_args ? argv[2] : argv[1];
    unsigned stdio_mask;
    if (reserve_stdio(&stdio_mask)) return 125;
    start_ns = now_ns();
    trace = getenv("CAPSTONE_VM_TRACE") != NULL;
    grants = trace || getenv("CAPSTONE_VM_GRANTS") != NULL;
    mc_init(getenv("CAPSTONE_VM_MAPCACHE"));
    struct capstone_application_descriptor_v2 desc;
    char resumed_path[64];
    if (resumed.magic) snprintf(resumed_path, sizeof resumed_path, "/proc/self/fd/%d", resumed.image);
    /* No copy of the image on a local filesystem: the module denies writers
     * to the file (CV_IMAGE) before it is validated, and holds that until
     * this process ends, as Linux does for a running executable. load()
     * validates the one mapping it loads from, so the bytes validated are
     * the bytes loaded, and the domain cannot write its own image while it
     * runs. After exec in place, and on 9p, NFS or FUSE, the image is a
     * sealed copy, which the denial accepts as fixed. */
    int fd = capstone_application_open(resumed.magic ? resumed_path : image_path);
    int image_error = errno;
    if (resumed.magic) close(resumed.image);
    errno = image_error;
    if (fd < 0) { perror("capstone-vexec: image"); return errno == ENOENT ? 127 : 126; }
    image_fd = fd;
    if (spawner.socket < 0) { int error = capstone_spawner_start(&spawner);
        if (error) { errno = error; return die("spawner"); } }
    host.spawner = &spawner;
    device = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    if (device < 0) return die("open");
    if (image_denied(&fd, &desc)) { perror("capstone-vexec: image"); cleanup(); return 126; }
    void *image; size_t image_bytes, image_rx, stack_bytes; uintptr_t entry;
    if (load(fd, &desc, &image, &image_bytes, &image_rx, &entry, &stack_bytes)) {
        perror("capstone-vexec: image"); cleanup(); return 126;
    }
    host.private_fds[host.private_count++] = fd;
    /* Option (iii): no exact-bounds profile bit is required. Bounds are representable by
     * construction -- the allocator's geometry keeps every rounded bound inside its own slot --
     * so the launcher runs on the default profile. The bit may still be set; it is not read. */
    if (grant(image, image_bytes, 3, 7, entry, 0, 0, image_rx)) { munmap(image, image_bytes); return die("image grant"); }
    void *stack = reserve(stack_bytes);
    domain_stack = stack; domain_stack_bytes = stack_bytes;
    meta = reserve(CAPSTONE_DELEGATE_META_BYTES);
    size_t exchange_bytes = rounded(desc.exchange_bytes);
    void *exchange = exchange_bytes ? reserve(exchange_bytes) : NULL;
    void *startup = reserve(CAPSTONE_LAUNCH_BYTES);
    if (!stack || !meta || !exchange || !startup) return die("allocate startup");
    char *cwd = getcwd(NULL, 0);
    struct timespec rt, mt;
    clock_gettime(CLOCK_REALTIME, &rt); clock_gettime(CLOCK_MONOTONIC, &mt);
    struct capstone_launch_task task = { .pid = getpid(), .ppid = getppid(),
        .uid = getuid(), .euid = geteuid(), .gid = getgid(), .egid = getegid(),
        .realtime_ns = (uint64_t)rt.tv_sec * 1000000000 + rt.tv_nsec,
        .monotonic_ns = (uint64_t)mt.tv_sec * 1000000000 + mt.tv_nsec };
    int error = cwd ? capstone_launch_pack(startup, CAPSTONE_LAUNCH_BYTES,
        argc - (app_args ? 4 : binfmt ? 2 : 1), argv + (app_args ? 4 : binfmt ? 2 : 1), environ, cwd, stdio_mask, &task) : ENOMEM;
    free(cwd);
    if (error) { errno = error; return die("startup"); }
    if (grant(stack, stack_bytes, 2, 6, (uintptr_t)stack + stack_bytes, 0, 0, 0) ||
        grant(meta, CAPSTONE_DELEGATE_META_BYTES, 10, 6, (uintptr_t)meta, 0, 0, 0) ||
        grant(exchange, exchange_bytes, 11, 6, (uintptr_t)exchange, 0, 0, 0) ||
        grant(startup, CAPSTONE_LAUNCH_BYTES, 12, 4, (uintptr_t)startup, 0, 0, 0)) return die("grant");
    if (desc.v1.heap_bytes) {
        size_t bytes = rounded(desc.v1.heap_bytes);
        void *region = bytes ? reserve(bytes) : NULL;
        if (!region || grant(region, bytes, 13, 6, (uintptr_t)region, 1, 0, 0))
            return die("nested allocator grant");
    }
    host.exchange = exchange; host.exchange_bytes = exchange_bytes;
    exchange_admitted = calloc(exchange_bytes / 4096, 1);
    if (!exchange_admitted) return die("exchange admission cache");
    host.prepare_write = exchange_prepare_write;
    host.private_fds[host.private_count++] = device;
    capstone_signals_settle_libc();
    capstone_signals_init(&host.signals, (void *)((char *)meta + CAPSTONE_SIGNAL_OFFSET));
    if ((error = capstone_delegate_seccomp())) { errno = error; return die("seccomp"); }
    for (int fd = 0; fd < 3; ++fd) if (!(stdio_mask & (1u << fd))) close(fd);
    launch_ns = now_ns() - start_ns;
    return virtual_service_loop(0);
}
