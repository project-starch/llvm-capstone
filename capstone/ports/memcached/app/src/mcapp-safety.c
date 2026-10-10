/* Safety fixtures for the memcached domain: what does a heap overflow, a use after free, a stale
 * free, or the same thing on one of memcached's own slab items actually DO here? The oracle (M2-M5)
 * shows the server is CORRECT under capability enforcement; it says nothing about safety.
 *
 * The tshark port's src/tsapp-safety.c, carried over with two changes:
 * - ONE IMAGE PER ARM, NOT PER FIXTURE. The fixtures are compiled into memcached itself (patch 0005,
 *   only with -DMC_CAPSTONE_SAFETY_FIXTURES, which #includes this file into proto_text.c), and a
 *   hidden text command `mc_capstone_fixture <n>` runs fixture n on the WORKER thread that took the
 *   connection: a fault lands in a further context, as a real one in a request would. A fixture
 *   that returns prints its mark and ends the process with exit(mark), so the exit status carries
 *   the mark's low byte, as an image's return does in the tshark port.
 * - Fixtures 9 and 10 are memcached's own: two items of one slab class, through item_alloc and
 *   item_remove, the allocator every stored value goes through.
 *
 * EACH FIXTURE HAS TO CREATE ITS CONDITION, or its result means nothing. So every fixture prints,
 * BEFORE the access it tests, the addresses and capability bounds it is about to use, then
 *     MCAPP-FIX <n> target=<hex>
 *     MCAPP-FIX <n> touch
 * A fault counts as the fixture's only after its touch line and with nothing printed after it, and a
 * bounds fault must also name the printed target (ports/common/application/check-safety.py). Marks
 * carry the byte read, so an unprotected arm's answer is a VALUE, not merely "it came back".
 *
 * Every access under test goes through mcapp_fix_touch/mcapp_fix_poke, so a fault pc names one of
 * them. Addresses are read with the cursor builtin and only ever compared or printed.
 *
 *   1 heap_len        two 64-byte mallocs, written and read in bounds; prints each pointer's
 *                     capability length. The setup control: always returns.
 *   2 heap_neighbour  writes through the FIRST at the SECOND's first byte, reads it back through the
 *                     second's own pointer.
 *   3 heap_one_past   reads one byte past the end of a 64-byte object.
 *   4 uaf_noreuse     reads a freed object, nothing allocated since.
 *   5 uaf_reuse       frees, allocates the same size again (reports whether it got the same
 *                     address), writes a new pattern, reads through the OLD pointer.
 *   6 stale_free      frees p, lets q take p's memory, frees p AGAIN, allocates r: does r alias q?
 *   7 global_oob      one byte past a 64-byte global.      (controls: expected to fault on every
 *   8 stack_oob       one byte past a 64-byte stack array.  heap arm alike)
 *   9 slab_neighbour  two items of one slab class (item_alloc), adjacent in one slab page: writes
 *                     through the first item's data pointer at the second item's first data byte.
 *  10 slab_reuse      an item removed (item_remove: back on its class's free list) and a new item
 *                     of the same class allocated; reads the old item's data through the old pointer.
 *  12-16 the five cases of bug-corpora/memcached/allocator-repros, each performing its own
 *                     premature free through memcached's REAL allocator inside the running server.
 *                     The corpus drives the same sequences in a freestanding harness; these are the
 *                     same allocator-level shapes, reduced the same way (its PROVENANCE files say
 *                     what each reduction leaves out: the hash table, the LRU queues, the real
 *                     consumer path and, for 13 and 16, the thread interleaving, which is written
 *                     out in program order here exactly as the corpus writes it).
 *                     12 case 00: a read buffer handed back to its cache and THEN copied out of.
 *                     13 case 01: a pending-IO list walked while the body frees and reissues the
 *                        current entry, so the step reads the next link out of a dead object.
 *                     14 case 02: tail repair assigns refcount = 1 over a live holder's reference
 *                        and unlinks, freeing a chunk the holder still has.
 *                     15 case 03 (CVE-2018-1000127): the unsigned short refcount wraps past the
 *                        holders, so the next release frees a held item. 1.6.45 still has the
 *                        narrow counter and the bare ++; its fix is a ceiling in the multiget
 *                        consumer, which this fixture does not go through.
 *                     16 case 04: an unlocked decrement loses a concurrent get, leaving the count
 *                        one low, so the next release frees a held item.
 *                     12 and 13 are cache.c objects; 14-16 are slabs.c chunks.
 *  11 chunked_reuse   fixture 10 for a CHUNKED item (a 700 KB value: its header in a small class,
 *                     one 512 KiB data chunk attached with do_item_alloc_chunk). item_remove frees
 *                     it through do_slabs_free_chunked, which files the header and then the chunk;
 *                     a new item of the same shape takes both back; reads the old chunk's data
 *                     through the old pointer. Added with the slabsublet arm, whose oracle never
 *                     frees a chunked item, so this is the only run of that release path.
 *  22 mover_floating  upstream a836eab (2015, shipped 1.4.23-1.4.24) REVERSED: the page mover takes
 *                     an item that is allocated but not yet linked (its upload in progress, at its
 *                     page's first chunk) for
 *                     cleared, so the page is wiped and handed to another class under its holder.
 *                     The holder then writes its next byte, and a live item of the other class reads
 *                     it back. The reversal is patch 0005's run-time flag in slabs_mover.c.
 *  23 mover_waits     the same sequence as shipped at the pin: the mover waits for the upload, so
 *                     the page never moves while it is held.
 *  24 mover_chunk     upstream c0e5a99 (2020, shipped 1.5.20-1.5.21) REVERSED: an expired CHUNKED
 *                     item's header is freed header-only when its page moves, orphaning its data
 *                     chunks with a dangling head. A later move of a data chunk's page reads that
 *                     head. The reversal is patch 0005's mcapp_reverse_c0e5a99 in slabs_mover.c.
 *                     The stale read is in the mover thread, at the header's it_flags, the printed
 *                     target. The plain arm has no per-chunk revocation, so the head still names
 *                     live (reused) storage and the read does not fault; its mark is not fixed.
 *  25 mover_chunk_ok  the same as shipped: the full chunked free runs, no orphan, the page moves.
 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MCAPP_MARK(n, v) (0x100000 * (n) + ((v) & 0xFFFFF))

unsigned char mcapp_fix_global[64];

/* Read by slabs_mover.c (patch 0005): set only by fixture 22, to reverse a836eab. */
volatile int mcapp_reverse_a836eab;

/* Read by slabs_mover.c (patch 0005): set only by fixture 24, to reverse c0e5a99. */
volatile int mcapp_reverse_c0e5a99;

static unsigned long mcapp_cur(const volatile void *p) { return __builtin_capstone_cap_get_cursor((void *)p); }
static unsigned long mcapp_end(const volatile void *p) { return __builtin_capstone_cap_get_end((void *)p); }
static unsigned long mcapp_base(const volatile void *p) { return __builtin_capstone_cap_get_base((void *)p); }

static void mcapp_show(int n, const char *name, const volatile void *p)
{
    printf("MCAPP-FIX %d %s cursor=%lx bounds=[%lx,%lx) len-from-cursor=%lu\n", n, name, mcapp_cur(p),
           mcapp_base(p), mcapp_end(p), mcapp_end(p) - mcapp_cur(p));
}

unsigned mcapp_fix_touch(const volatile unsigned char *b, long i);
void mcapp_fix_poke(volatile unsigned char *b, long i, unsigned char v);

__attribute__((noinline)) unsigned mcapp_fix_touch(const volatile unsigned char *b, long i)
{
    return b[i];
}

__attribute__((noinline)) void mcapp_fix_poke(volatile unsigned char *b, long i, unsigned char v)
{
    b[i] = v;
}

/* The target is an ADDRESS computed before any free: a revoked capability reloads untagged, and even
   reading its cursor must not be what faults. */
static void mcapp_touching(int n, unsigned long target)
{
    printf("MCAPP-FIX %d target=%lx\n", n, target);
    printf("MCAPP-FIX %d touch\n", n);
    fflush(stdout);
}

static void mcapp_fill(volatile unsigned char *p, unsigned char v0, int len)
{
    for (int i = 0; i < len; i++)
        mcapp_fix_poke(p, i, (unsigned char)(v0 + i));
}

/* an item of `len` value bytes under key "mcapp-fix-<tag>"; its data pointer, or NULL */
static item *mcapp_item(int n, const char *tag, int len, unsigned char **data)
{
    char key[32];
    int nkey = snprintf(key, sizeof key, "mcapp-fix-%s", tag);
    item *it = item_alloc(key, (size_t)nkey, 0, 0, len + 2);
    if (!it) {
        printf("MCAPP-FIX %d item_alloc %s FAILED\n", n, tag);
        return NULL;
    }
    *data = (unsigned char *)ITEM_data(it);
    printf("MCAPP-FIX %d item %s slab-class=%u\n", n, tag, (unsigned)ITEM_clsid(it));
    mcapp_show(n, tag, *data);
    return it;
}

/* ---- the corpus's reduced item layer (02/03/04), copied rather than called ----------------
 * Upstream's item_free / do_item_remove / do_item_unlink_nolock, with the refcount arithmetic and
 * the one call to the real allocator kept and nothing else. The app's own do_item_unlink_nolock is
 * deliberately NOT used: it also does assoc_delete, do_item_unlink_q and STORAGE_delete on an item
 * that was never linked or put on an LRU. */
static void mcapp_item_free_reduced(item *it)
{
    unsigned int clsid = ITEM_clsid(it);
    slabs_free(it, clsid);
}
/* 1 when this release took the count to zero and freed the chunk. */
static int mcapp_item_remove_reduced(item *it)
{
    if (it->refcount == 0)
        return 0;
    if (--it->refcount == 0) {
        mcapp_item_free_reduced(it);
        return 1;
    }
    return 0;
}
static int mcapp_item_unlink_nolock_reduced(item *it)
{
    if ((it->it_flags & ITEM_LINKED) != 0) {
        it->it_flags &= ~ITEM_LINKED;
        return mcapp_item_remove_reduced(it);
    }
    return 0;
}
/* An item in class `id`, linked, with its payload filled: the LRU tail a client stored. Returns the
 * payload pointer, whose alias is the chunk's under slabsublet. */
static unsigned char *mcapp_store_item(int n, const char *tag, unsigned id, unsigned char fill)
{
    item *it = slabs_alloc(id, 0);
    if (!it) {
        printf("MCAPP-FIX %d slabs_alloc %s FAILED\n", n, tag);
        return NULL;
    }
    it->slabs_clsid = (uint8_t)id;
    it->nkey = 0;
    it->nbytes = 16;
    it->it_flags = ITEM_LINKED;
    it->refcount++;                      /* 2: linked, plus the storing client */
    mcapp_item_remove_reduced(it);       /* the storing client lets go -> 1 */
    unsigned char *payload = (unsigned char *)it + sizeof(item);
    mcapp_fill(payload, fill, 16);
    printf("MCAPP-FIX %d item %s class=%u refcount=%u\n", n, tag, id, (unsigned)it->refcount);
    mcapp_show(n, tag, payload);
    return payload;
}
/* The item an mcapp_store_item payload belongs to. */
static item *mcapp_item_of(unsigned char *payload) { return (item *)(payload - sizeof(item)); }

/* ---- fixture 13's object, the corpus's reduction of io_pending_proxy_t -------------------- */
struct mcapp_io_pending {
    void *thread, *conn, *client_resp;
    void (*return_cb)(void *);
    void (*finalize_cb)(void *);
    int status;
    STAILQ_ENTRY(mcapp_io_pending) io_next;
    char data[120];
};
STAILQ_HEAD(mcapp_io_head_s, mcapp_io_pending);

#define MCAPP_STORED 0xA7   /* what the holder wrote and expects to read back */

static int mcapp_fixture(int n)
{
    volatile long idx;
    unsigned v;
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("MCAPP-FIX %d begin\n", n);
    switch (n) {
    case 1: {
        unsigned char *p = malloc(64), *q = malloc(64);
        mcapp_fill(p, 0xA0, 64);
        mcapp_fill(q, 0x5B, 64);
        mcapp_show(n, "p", p);
        mcapp_show(n, "q", q);
        v = mcapp_fix_touch(p, 63) == 0xA0 + 63 && mcapp_fix_touch(q, 63) == 0x5B + 63;
        printf("MCAPP-FIX 1 in-bounds %s\n", v ? "ok" : "WRONG");
        return MCAPP_MARK(n, v ? 0x1 : 0xE0002);
    }
    case 2: {
        unsigned char *p = malloc(64), *q = malloc(64);
        mcapp_fill(p, 0xA0, 64);
        mcapp_fill(q, 0x5B, 64);
        mcapp_show(n, "p", p);
        mcapp_show(n, "q", q);
        idx = (long)(mcapp_cur(q) - mcapp_cur(p));
        printf("MCAPP-FIX 2 q-p=%ld\n", idx);
        mcapp_touching(n, mcapp_cur(q));
        mcapp_fix_poke(p, idx, 0xEE);
        v = mcapp_fix_touch(q, 0);
        printf("MCAPP-FIX 2 returned q[0]=%02x (0x5b untouched, 0xee overwritten through p)\n", v);
        return MCAPP_MARK(n, v);
    }
    case 3: {
        unsigned char *p = malloc(64);
        mcapp_fill(p, 0xA0, 64);
        mcapp_show(n, "p", p);
        idx = 64;
        mcapp_touching(n, mcapp_cur(p) + 64);
        v = mcapp_fix_touch(p, idx);
        printf("MCAPP-FIX 3 returned p[64]=%02x\n", v);
        return MCAPP_MARK(n, v);
    }
    case 4: {
        unsigned char *p = malloc(64);
        mcapp_fill(p, 0xA0, 64);
        mcapp_show(n, "p", p);
        unsigned long p_addr = mcapp_cur(p);
        free(p);
        idx = 0;
        mcapp_touching(n, p_addr);
        v = mcapp_fix_touch(p, idx);
        printf("MCAPP-FIX 4 returned p[0]=%02x (0xa0 = the freed object's own byte)\n", v);
        return MCAPP_MARK(n, v);
    }
    case 5: {
        unsigned char *p = malloc(64);
        mcapp_fill(p, 0xA0, 64);
        mcapp_show(n, "p", p);
        unsigned long p_addr = mcapp_cur(p);
        free(p);
        unsigned char *q = malloc(64);
        mcapp_fill(q, 0x5B, 64);
        mcapp_show(n, "q", q);
        unsigned same = mcapp_cur(q) == p_addr;
        printf("MCAPP-FIX 5 same-address=%u\n", same);
        idx = 0;
        mcapp_touching(n, p_addr);
        v = mcapp_fix_touch(p, idx);
        printf("MCAPP-FIX 5 returned p[0]=%02x (0x5b = the NEW occupant's byte)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 6: {
        unsigned char *p = malloc(64);
        mcapp_show(n, "p", p);
        unsigned long p_addr = mcapp_cur(p);
        free(p);
        unsigned char *q = malloc(64);
        mcapp_fill(q, 0x5B, 64);
        mcapp_show(n, "q", q);
        printf("MCAPP-FIX 6 q-took-p's-address=%u\n", mcapp_cur(q) == p_addr);
        mcapp_touching(n, p_addr);      /* the "touch" here is the stale free */
        free(p);
        unsigned char *r = malloc(64);
        mcapp_show(n, "r", r);
        unsigned alias = mcapp_cur(r) == mcapp_cur(q);
        if (alias)
            mcapp_fill(r, 0x77, 64);    /* the new owner writes; the live q sees it */
        v = mcapp_fix_touch(q, 0);
        printf("MCAPP-FIX 6 returned r-aliases-q=%u q[0]=%02x\n", alias, v);
        return MCAPP_MARK(n, (alias << 8) | v);
    }
    case 7: {
        mcapp_fill(mcapp_fix_global, 0xA0, 64);
        mcapp_show(n, "global", mcapp_fix_global);
        printf("MCAPP-FIX 7 in-bounds g[63]=%02x\n", mcapp_fix_touch(mcapp_fix_global, 63));
        idx = 64;
        mcapp_touching(n, mcapp_cur(mcapp_fix_global) + 64);
        v = mcapp_fix_touch(mcapp_fix_global, idx);
        printf("MCAPP-FIX 7 returned g[64]=%02x\n", v);
        return MCAPP_MARK(n, v);
    }
    case 8: {
        unsigned char buf[64];
        mcapp_fill(buf, 0xA0, 64);
        mcapp_show(n, "stack", buf);
        printf("MCAPP-FIX 8 in-bounds buf[63]=%02x\n", mcapp_fix_touch(buf, 63));
        idx = 64;
        mcapp_touching(n, mcapp_cur(buf) + 64);
        v = mcapp_fix_touch(buf, idx);
        printf("MCAPP-FIX 8 returned buf[64]=%02x\n", v);
        return MCAPP_MARK(n, v);
    }
    case 9: {
        unsigned char *a, *b;
        if (!mcapp_item(n, "a", 64, &a) || !mcapp_item(n, "b", 64, &b))
            return MCAPP_MARK(n, 0xE0009);
        mcapp_fill(a, 0xA0, 64);
        mcapp_fill(b, 0x5B, 64);
        idx = (long)(mcapp_cur(b) - mcapp_cur(a));
        printf("MCAPP-FIX 9 b-a=%ld\n", idx);
        mcapp_touching(n, mcapp_cur(b));
        mcapp_fix_poke(a, idx, 0xEE);
        v = mcapp_fix_touch(b, 0);
        printf("MCAPP-FIX 9 returned b[0]=%02x (0x5b untouched, 0xee overwritten through a)\n", v);
        return MCAPP_MARK(n, v);
    }
    case 10: {
        unsigned char *a, *b;
        item *ia = mcapp_item(n, "a", 64, &a);
        if (!ia)
            return MCAPP_MARK(n, 0xE000A);
        mcapp_fill(a, 0xA0, 64);
        unsigned long a_addr = mcapp_cur(a);
        item_remove(ia);                /* unlinked, refcount 1 -> 0: back on the class's free list */
        if (!mcapp_item(n, "b", 64, &b))
            return MCAPP_MARK(n, 0xE000A);
        mcapp_fill(b, 0x5B, 64);
        unsigned same = mcapp_cur(b) == a_addr;
        printf("MCAPP-FIX 10 same-address=%u\n", same);
        idx = 0;
        mcapp_touching(n, a_addr);
        v = mcapp_fix_touch(a, idx);
        printf("MCAPP-FIX 10 returned a[0]=%02x (0x5b = the NEW item's byte)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 11: {
        /* 700000 bytes is over slab_chunk_size_max (512 KiB), so the item is chunked: a header, then
           chunks attached as the value arrives. One chunk is attached here, as the first read would. */
        const int big = 700000;
        char key[32];
        int nkey = snprintf(key, sizeof key, "mcapp-fix-a");
        item *ia = item_alloc(key, (size_t)nkey, 0, 0, big + 2);
        if (!ia || !(ia->it_flags & ITEM_CHUNKED)) {
            printf("MCAPP-FIX 11 item_alloc a FAILED or not chunked\n");
            return MCAPP_MARK(n, 0xE000B);
        }
        item_chunk *ca = do_item_alloc_chunk((item_chunk *)ITEM_schunk(ia), (size_t)big);
        if (!ca) {
            printf("MCAPP-FIX 11 do_item_alloc_chunk a FAILED\n");
            return MCAPP_MARK(n, 0xE000B);
        }
        unsigned char *a = (unsigned char *)ca->data;
        printf("MCAPP-FIX 11 item a header-class=%u chunk-class=%u chunk-size=%d\n",
               (unsigned)((item_chunk *)ITEM_schunk(ia))->orig_clsid, (unsigned)ca->slabs_clsid, ca->size);
        mcapp_show(n, "a-header", ia);
        mcapp_show(n, "a", a);
        mcapp_fill(a, 0xA0, 64);
        unsigned long a_addr = mcapp_cur(a);
        item_remove(ia);                /* refcount 1 -> 0: do_slabs_free_chunked files the header, then the chunk */
        item *ib = item_alloc(key, (size_t)nkey, 0, 0, big + 2);
        item_chunk *cb = ib ? do_item_alloc_chunk((item_chunk *)ITEM_schunk(ib), (size_t)big) : NULL;
        if (!ib || !cb) {
            printf("MCAPP-FIX 11 item b FAILED\n");
            return MCAPP_MARK(n, 0xE000B);
        }
        unsigned char *b = (unsigned char *)cb->data;
        mcapp_show(n, "b", b);
        mcapp_fill(b, 0x5B, 64);
        unsigned same = mcapp_cur(b) == a_addr;
        printf("MCAPP-FIX 11 same-address=%u\n", same);
        idx = 0;
        mcapp_touching(n, a_addr);
        v = mcapp_fix_touch(a, idx);
        printf("MCAPP-FIX 11 returned a[0]=%02x (0x5b = the NEW item's chunk byte)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 12: {
        /* case 00: rbuf_switch_to_malloc before 7af02b0c87 -- give the read buffer back to the
           thread's cache, THEN copy the unparsed command out of it. A previous buffer is parked on
           the free list first, so the link written into the freed buffer is a real pointer. */
        static char switched[2 * READ_BUFFER_SIZE];
        static const char command[] = "get key0001 key0002 key0003 key0004 key0005 key0006 key0007\r\n";
        const size_t rbytes = sizeof command - 1;
        cache_t *rbuf_cache = cache_create("mcapp-rbuf", READ_BUFFER_SIZE, sizeof(char *));
        if (!rbuf_cache) { printf("MCAPP-FIX 12 cache_create FAILED\n"); return MCAPP_MARK(n, 0xE000C); }
        char *previous = cache_alloc(rbuf_cache);
        char *rbuf = cache_alloc(rbuf_cache);
        if (!previous || !rbuf || previous == rbuf) {
            printf("MCAPP-FIX 12 cache_alloc FAILED or aliased\n"); return MCAPP_MARK(n, 0xE000C);
        }
        cache_free(rbuf_cache, previous);          /* the free list is now non-empty */
        memcpy(rbuf, command, rbytes);
        mcapp_show(n, "rbuf", (unsigned char *)rbuf);
        unsigned long rbuf_addr = mcapp_cur(rbuf);
        cache_free(rbuf_cache, rbuf);              /* the defect: back to the cache first */
        mcapp_touching(n, rbuf_addr);
        v = mcapp_fix_touch((unsigned char *)rbuf, 0);
        memcpy(switched, rbuf, rbytes);            /* ... and only now the copy */
        unsigned damaged = memcmp(switched, command, rbytes) != 0;
        char *next = cache_alloc(rbuf_cache);
        unsigned same = next && mcapp_cur(next) == rbuf_addr;
        printf("MCAPP-FIX 12 returned byte0=%02x damaged=%u reissued-to-next-connection=%u\n",
               v, damaged, same);
        if (next) cache_free(rbuf_cache, next);
        cache_destroy(rbuf_cache);
        return MCAPP_MARK(n, (same << 8) | damaged);
    }
    case 13: {
        /* case 01: _reset_bad_backend before 0ad4de66ae -- STAILQ_FOREACH over a backend's pending
           IOs whose body returns (frees) the current IO, so the loop's step reads io_next out of an
           object the worker has already taken again for another request. */
        cache_t *io_cache = cache_create("mcapp-io", sizeof(struct mcapp_io_pending), sizeof(char *));
        if (!io_cache) { printf("MCAPP-FIX 13 cache_create FAILED\n"); return MCAPP_MARK(n, 0xE000D); }
        struct mcapp_io_head_s io_head = STAILQ_HEAD_INITIALIZER(io_head);
        STAILQ_INIT(&io_head);
        for (int i = 0; i < 3; i++) {
            struct mcapp_io_pending *io = cache_alloc(io_cache);
            if (!io) { printf("MCAPP-FIX 13 cache_alloc FAILED\n"); return MCAPP_MARK(n, 0xE000D); }
            memset(io, 0, sizeof *io);
            STAILQ_INSERT_TAIL(&io_head, io, io_next);
        }
        struct mcapp_io_pending *io = STAILQ_FIRST(&io_head);
        unsigned long first = mcapp_cur(io);
        mcapp_show(n, "io0", (unsigned char *)io);
        unsigned returned = 0, same = 0;
        v = 0;
        while (io) {
            unsigned long here = mcapp_cur(io);
            io->status = -1;
            ++returned;
            cache_free(io_cache, io);                       /* returned to the worker's cache */
            struct mcapp_io_pending *fresh = cache_alloc(io_cache);  /* its next request */
            if (fresh) memset(fresh, 0, sizeof *fresh);     /* zeroed by its new owner */
            if (here == first) {
                same = fresh && mcapp_cur(fresh) == first;
                printf("MCAPP-FIX 13 reissued-to-next-request=%u\n", same);
                mcapp_touching(n, first);
                v = mcapp_fix_touch((unsigned char *)io, 0);
            }
            io = STAILQ_NEXT(io, io_next);                  /* the step, through the dead object */
        }
        unsigned damaged = returned != 3;
        printf("MCAPP-FIX 13 returned byte0=%02x walked=%u/3 damaged=%u\n", v, returned, damaged);
        cache_destroy(io_cache);
        return MCAPP_MARK(n, (same << 8) | damaged);
    }
    case 14: {
        /* case 02: items.c tail repair with -o tail_repair_time set and the item older than it --
             search->refcount = 1;  do_item_unlink_nolock(search, hv);
           The assignment discards every outstanding reference, the unlink takes the count to zero,
           and the chunk goes back to its class while a holder still has it. The branch's own
           guards (tail_repair_time, the clock, the LRU tail, an exhausted class) are not set up;
           the two statements are performed directly, as the corpus does. */
        unsigned id = slabs_clsid(sizeof(item) + 64);
        unsigned char *held = mcapp_store_item(n, "a", id, MCAPP_STORED);
        if (!held) return MCAPP_MARK(n, 0xE000E);
        item *search = mcapp_item_of(held);
        search->refcount++;                 /* 2: linked, plus the client that fetched it */
        unsigned long held_addr = mcapp_cur(held);
        if (++search->refcount != 2) {      /* the eviction probe: not 2, somebody holds it */
            search->refcount = 1;
            mcapp_item_unlink_nolock_reduced(search);
        }
        unsigned char *fresh = mcapp_store_item(n, "b", id, 0x5C);
        unsigned same = fresh && mcapp_cur(fresh) == held_addr;
        printf("MCAPP-FIX 14 same-address=%u\n", same);
        mcapp_touching(n, held_addr);
        v = mcapp_fix_touch(held, 0);
        printf("MCAPP-FIX 14 returned held[0]=%02x (0x5c = the new item's byte)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 15: {
        /* case 03, CVE-2018-1000127: refcount is an unsigned short and refcount_incr is a bare ++
           (memcached.h), so 65536 references wrap it back past the holders and the next release
           frees an item they still hold. 1.6.45's fix is a ceiling in the multiget consumer, which
           this fixture does not go through -- so this shows the primitive is still reachable, not
           that the shipped server is exploitable. */
        unsigned id = slabs_clsid(sizeof(item) + 64);
        unsigned char *held = mcapp_store_item(n, "a", id, MCAPP_STORED);
        if (!held) return MCAPP_MARK(n, 0xE000F);
        item *it = mcapp_item_of(held);
        it->refcount++;                     /* 2: linked, plus this holder */
        unsigned long held_addr = mcapp_cur(held);
        unsigned long taken = 0;
        for (unsigned long i = 0; i < 65536UL; i++) { it->refcount++; taken++; }
        unsigned wrapped = it->refcount == 2 && taken == 65536UL;
        printf("MCAPP-FIX 15 took=%lu refcount-now=%u wrapped=%u\n", taken, (unsigned)it->refcount, wrapped);
        if (!wrapped) { printf("MCAPP-FIX 15 the counter did not wrap\n"); return MCAPP_MARK(n, 0xE000F); }
        int freed = 0;
        for (int i = 0; i < 2 && !freed; i++) freed = mcapp_item_remove_reduced(it);
        printf("MCAPP-FIX 15 freed-while-held=%d\n", freed);
        unsigned char *fresh = mcapp_store_item(n, "b", id, 0x2E);
        unsigned same = fresh && mcapp_cur(fresh) == held_addr;
        printf("MCAPP-FIX 15 same-address=%u\n", same);
        mcapp_touching(n, held_addr);
        v = mcapp_fix_touch(held, 0);
        printf("MCAPP-FIX 15 returned held[0]=%02x (0x2e = the new item's byte)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 16: {
        /* case 04: the mget error path before 152ddb68f7 calls the lock-assuming do_item_remove
           without the item lock, so a concurrent do_item_get's increment lands inside the
           read-subtract-store and is lost. The interleaving is written out in program order, as the
           corpus does: the race itself is not reproduced, the corruption step is. */
        unsigned id = slabs_clsid(sizeof(item) + 64);
        unsigned char *held = mcapp_store_item(n, "a", id, MCAPP_STORED);
        if (!held) return MCAPP_MARK(n, 0xE0010);
        item *it = mcapp_item_of(held);
        it->refcount++;                     /* 2: linked, plus client A */
        unsigned long held_addr = mcapp_cur(held);
        unsigned short seen = it->refcount;             /* B reads 2 */
        it->refcount++;                                 /* C's do_item_get lands here: 3 */
        it->refcount = (unsigned short)(seen - 1);      /* B stores 1; C's get is lost */
        printf("MCAPP-FIX 16 refcount-after-unlocked-decrement=%u\n", (unsigned)it->refcount);
        int freed = mcapp_item_remove_reduced(it);      /* A releases -> 0 -> slabs_free */
        printf("MCAPP-FIX 16 freed-while-C-holds=%d\n", freed);
        unsigned char *fresh = mcapp_store_item(n, "b", id, 0x91);
        unsigned same = fresh && mcapp_cur(fresh) == held_addr;
        printf("MCAPP-FIX 16 same-address=%u\n", same);
        mcapp_touching(n, held_addr);
        v = mcapp_fix_touch(held, 0);
        printf("MCAPP-FIX 16 returned held[0]=%02x (0x91 = the new item's byte)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 17: {
        /* commit 204019d (a 2006 contributed patch), the connection read buffer grown in place.
           try_read_network reallocs a malloc'd read buffer, which may MOVE the block and release
           the old one. The fix survives verbatim at the pin and updates BOTH pointers:
             1.6.45:memcached.c:2467  c->rcurr = c->rbuf = new_rbuf;
             1.6.45:memcached.c:2468  c->rsize *= 2;
           Reversed here to `c->rbuf = new_rbuf;` alone, which leaves c->rcurr pointing into the
           released block; the text parser then reads straight through it:
             1.6.45:proto_text.c:243  st = c->rcurr;
             1.6.45:proto_text.c:244  el = memchr(c->rcurr, '\n', c->rbytes);
           Plain malloc/realloc -- NOT the slab allocator and NOT cache.c's rbuf cache. This is the
           malloc'd path a conn reaches through rbuf_switch_to_malloc (1.6.45:memcached.c:424-434).
           HISTORICAL: the upstream fix is reversed here, as cases 0, 1, 3 and 4 are.
           Same OBJECT as fixture 12 (the connection read buffer) but a DIFFERENT allocator seam:
           12's lifetime-ender is cache_free pushing onto cache.c's STAILQ; this one's is realloc
           releasing the old block. That is why both are worth having.
           Both "condition not created" exits below are ERRORS, not zero marks: realloc is free to
           grow in place, and if it does there is no stale pointer and the fixture has measured
           nothing. A directed test that does not create its triggering condition must say so. */
        size_t sz = 512;
        unsigned char *rbuf = malloc(sz);
        unsigned char *blocker = malloc(sz);     /* occupy the space after rbuf so realloc moves */
        if (!rbuf || !blocker) return MCAPP_MARK(n, 0xE0011);
        mcapp_fill(rbuf, 0xA0, 64);
        unsigned char *rcurr = rbuf + 8;         /* the parser's cursor, interior to the buffer */
        mcapp_show(n, "rbuf", rbuf);
        mcapp_show(n, "rcurr", rcurr);
        unsigned long rbuf_addr = mcapp_cur(rbuf);
        unsigned long rcurr_addr = mcapp_cur(rcurr);
        unsigned char *grown = realloc(rbuf, sz * 2);
        if (!grown) return MCAPP_MARK(n, 0xE0011);
        unsigned moved = mcapp_cur(grown) != rbuf_addr;
        if (!moved) {
            printf("MCAPP-FIX 17 realloc grew IN PLACE: triggering condition not created\n");
            return MCAPP_MARK(n, 0xE0017);
        }
        unsigned char *taker = NULL;
        unsigned same = 0;
        /* mallocng deliberately cycles offsets. Immediate reuse is not an
           allocator guarantee: search for actual reissue without changing
           its policy. A bounded failure still means no triggering condition. */
        for (unsigned attempt = 0; attempt < 4096; ++attempt) {
            taker = malloc(sz);
            if (!taker) return MCAPP_MARK(n, 0xE0011);
            same = mcapp_cur(taker) == rbuf_addr;
            if (same) break;
            free(taker);
            taker = NULL;
        }
        if (!same) {
            printf("MCAPP-FIX 17 released block NOT reissued: triggering condition not created\n");
            return MCAPP_MARK(n, 0xE0017);
        }
        mcapp_fill(taker, 0x5B, 64);
        printf("MCAPP-FIX 17 realloc-moved=%u reissued-to-next-owner=%u\n", moved, same);
        mcapp_touching(n, rcurr_addr);
        v = mcapp_fix_touch(rcurr, 0);            /* proto_text.c reads through the stale cursor */
        printf("MCAPP-FIX 17 returned rcurr[0]=%02x (0x63 = the new owner's byte at offset 8)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 18: {
        /* e779381 (2026-07-07) "logger: fix use-after-free of closed watcher".
           logger_thread_close_watcher both clears the global slot and frees the watcher:
             1.6.45:logger.c:734  watchers[w->id] = NULL;
             1.6.45:logger.c:737  bipbuf_free(w->buf);
             1.6.45:logger.c:738  free(w);
           and the fix present at the pin is the CALLER's recheck:
             1.6.45:logger.c:691  // Oddity; poll_watchers can free *w, recheck it.
             1.6.45:logger.c:692  if (watchers[x] == NULL)
             1.6.45:logger.c:693      break;
           Reversed here: the caller keeps its own `w` and writes w->failed_flush through it.
           The watcher is plain calloc (1.6.45:logger.c:1120), NOT an item and NOT a slab chunk,
           so the slab arms cannot see this one -- which is the point of recording it as plain heap.
           HISTORICAL, like 17.
           This is the corpus's only WRITE-after-free: every other case here reads. The damage is
           therefore visible from the NEW owner's side, which is how it is measured -- poke through
           the dead pointer, then read back through the live pointer. */
        static unsigned char *watcher_slot[4];
        unsigned char *w = calloc(1, 64);
        if (!w) return MCAPP_MARK(n, 0xE0012);
        mcapp_fill(w, 0xA0, 64);
        watcher_slot[1] = w;
        mcapp_show(n, "watcher", w);
        unsigned long w_addr = mcapp_cur(w);
        watcher_slot[1] = NULL;                   /* logger.c:734 clears the slot ... */
        free(w);                                  /* ... logger.c:738 frees it */
        unsigned char *nu = calloc(1, 64);        /* the next allocation takes the storage */
        if (!nu) return MCAPP_MARK(n, 0xE0012);
        unsigned same = mcapp_cur(nu) == w_addr;
        if (!same) {
            printf("MCAPP-FIX 18 storage NOT reissued: triggering condition not created\n");
            return MCAPP_MARK(n, 0xE0018);
        }
        mcapp_fill(nu, 0x5B, 64);
        printf("MCAPP-FIX 18 same-address=%u slot-cleared=%u\n", same, watcher_slot[1] == NULL);
        mcapp_touching(n, w_addr);
        mcapp_fix_poke(w, 32, 0xEE);              /* w->failed_flush = true, into freed storage */
        v = mcapp_fix_touch(nu, 32);              /* read back through the NEW owner's pointer */
        printf("MCAPP-FIX 18 returned new[32]=%02x (0xee = the write landed in the new owner)\n", v);
        return MCAPP_MARK(n, (same << 8) | v);
    }
    case 19: {
        /* CLASS 3 -- REUSE-NOT-FREE -- on memcached's REAL bipbuffer, which is the logger's
           per-watcher output buffer (logger.h:197,215) and also carries items.c's lru_bump_entry
           records. This is NOT a use-after-free: no free() of any kind occurs anywhere in it.

           bipbuf_new is ONE allocation -- malloc(sizeof(bipbuf_t) + size) with a flexible data[] --
           so every record lives inside a single malloc. Then:
             bipbuf_request  returns (unsigned char *)me->data + me->a_end   -- a pointer INSIDE it
             bipbuf_poll     void *end = me->data + me->a_start; me->a_start += size;
                             ... me->a_start = me->a_end = 0;  return end;  -- CURSORS ONLY
           So a consumer that polled a record, and then lets the producer request again, is holding a
           pointer to bytes that now belong to a DIFFERENT record -- while the pointer was never
           freed, is still tagged, and is still in bounds of the one malloc. Only the data's identity
           changed. That is class 3 in docs/design/sharing-bug-taxonomy-and-novelty.md, the row where
           ASan, GC, Rust, CHERI spatial, CHERI async AND CHERI eager are all listed blind.

           PREDICTION: EVERY arm RETURNS -- level0, shrink, sublet, slabsublet0 and slabsublet1
           alike. The revoking arms are blind here too, because the runtime heap never sees a free
           and there is nothing to revoke. That blindness is the POINT of this fixture: it is the
           measurement that motivates hooking the bipbuffer, not a failure of the arms.

           THE POSITIVE CONTROL IS FIXTURE 18, in this same file and on the same arms: same
           stale-pointer-then-reuse shape, but its release DOES reach the allocator, and it faults on
           every revoking arm. So a RETURN here cannot be dismissed as a harness that never fires.
           Each "condition not created" exit below is an ERROR mark, never a quiet pass. */
        bipbuf_t *bb = bipbuf_new(4096);
        if (!bb) return MCAPP_MARK(n, 0xE0013);
        unsigned char *first = bipbuf_request(bb, 64);
        if (!first) return MCAPP_MARK(n, 0xE0013);
        mcapp_fill(first, 0xA0, 64);
        bipbuf_push(bb, 64);
        mcapp_show(n, "record1", first);
        unsigned long rec_addr = mcapp_cur(first);
        unsigned char *polled = bipbuf_poll(bb, 64);      /* the consumer takes record 1 ... */
        unsigned same_ptr = polled && mcapp_cur(polled) == rec_addr;
        unsigned char *second = bipbuf_request(bb, 64);   /* ... and the producer reuses the bytes */
        unsigned reissued = second && mcapp_cur(second) == rec_addr;
        if (!same_ptr || !reissued) {
            printf("MCAPP-FIX 19 bipbuf did not recycle in place: condition NOT created "
                   "(polled-same=%u reissued=%u)\n", same_ptr, reissued);
            return MCAPP_MARK(n, 0xE0019);
        }
        mcapp_fill(second, 0x5B, 64);
        bipbuf_push(bb, 64);
        mcapp_show(n, "record2", second);
        printf("MCAPP-FIX 19 polled-same=%u reissued-in-place=%u frees-performed=0\n",
               same_ptr, reissued);
        mcapp_touching(n, rec_addr);
        v = mcapp_fix_touch(polled, 0);   /* the consumer reads what it believes is its own record */
        printf("MCAPP-FIX 19 returned polled[0]=%02x (0x5b = the SECOND record's byte)\n", v);
        return MCAPP_MARK(n, (reissued << 8) | v);
    }
    case 20: {
        /* ddee3e2 "Fix minor severity heap buffer overflow reading `--auth-file`".
           SPATIAL, class A: the access leaves a PLAIN malloc'd buffer, so per-object bounds see it.
           The authfile parser sized its buffer to the file exactly and then scanned past the end
           looking for a terminator:
             ddee3e2^:authfile.c  auth_data = calloc(1, sb.st_size);
             ddee3e2^:authfile.c  if (!found && auth_cur[x] == ':') { ... }
           The fix widens the allocation; the pin carries an even wider form, verified:
             1.6.45:authfile.c:50  auth_data = calloc(1, sb.st_size + 2);
             1.6.45:authfile.c:56  char *auth_end = auth_data + sb.st_size + 1;
           Reduced to the first byte the scan reads past the allocation. NOT the slab allocator and
           NOT cache.c -- plain calloc, which is the point of this row: it is the baseline where
           malloc-granular bounds DO fire, and the contrast that makes the nested rows mean
           something. HISTORICAL: the upstream fix is reversed here. */
        size_t sz = 64;
        unsigned char *auth_data = calloc(1, sz);
        unsigned char *after = malloc(sz);     /* whatever the scan runs into */
        if (!auth_data || !after) return MCAPP_MARK(n, 0xE0011);
        mcapp_fill(auth_data, 0x41, sz);       /* no terminator anywhere in the buffer */
        mcapp_fill(after, 0x5B, sz);
        mcapp_show(n, "auth_data", auth_data);
        mcapp_show(n, "after", after);
        printf("MCAPP-FIX 20 size=%lu first-read-past=%lu\n", (unsigned long)sz, (unsigned long)sz);
        mcapp_touching(n, mcapp_cur(auth_data) + sz);
        v = mcapp_fix_touch(auth_data, (long)sz);  /* auth_cur[sz]: one past the allocation */
        printf("MCAPP-FIX 20 returned auth_data[%lu]=%02x (past the calloc)\n",
               (unsigned long)sz, v);
        return MCAPP_MARK(n, v);
    }
    case 21: {
        /* d5d9ff0 (cited by hash: its subject names an outside contributor).
           SPATIAL, class A, and a WRITE. do_item_cachedump reserved room for "END\r\n" but not for
           its terminating NUL, so the last strcpy could put that NUL one byte past the buffer:
             d5d9ff0^:items.c  buffer = malloc(memlimit);
             d5d9ff0^:items.c  if (bufcurr + len +5 > memlimit)   // 5 is END\r\n
           The fix makes the reserve 6, "END\r\n\0". Reduced to the single byte at index memlimit --
           the one a reserve of 5 permits and a reserve of 6 does not. Plain malloc, so this is a
           baseline row like fixture 20. HISTORICAL: the upstream fix is reversed here. */
        size_t memlimit = 64;
        unsigned char *buffer = malloc(memlimit);
        unsigned char *after = malloc(memlimit);
        if (!buffer || !after) return MCAPP_MARK(n, 0xE0011);
        mcapp_fill(buffer, 0xA0, memlimit);
        mcapp_fill(after, 0x5B, memlimit);
        mcapp_show(n, "buffer", buffer);
        mcapp_show(n, "after", after);
        /* bufcurr advanced so that bufcurr + len + 5 == memlimit passes the old check, and the
           terminator then lands at index memlimit. */
        size_t len = 5, bufcurr = memlimit - len - 5;
        printf("MCAPP-FIX 21 memlimit=%lu bufcurr=%lu len=%lu terminator-at=%lu\n",
               (unsigned long)memlimit, (unsigned long)bufcurr, (unsigned long)len,
               (unsigned long)memlimit);
        mcapp_touching(n, mcapp_cur(buffer) + memlimit);
        mcapp_fix_poke(buffer, (long)memlimit, 0x00);  /* the NUL, one byte past */
        v = mcapp_fix_touch(after, 0);
        printf("MCAPP-FIX 21 returned after[0]=%02x (0x5b untouched, 0x00 overwritten)\n", v);
        return MCAPP_MARK(n, v);
    }
    case 22:
    case 23: {
        /* a836eab "fix memory corruption in slab page mover": an item with neither ITEM_SLABBED nor
           ITEM_LINKED -- allocated by item_alloc, its value still being read from the client -- fell
           through the mover's checks as MOVE_PASS, so the mover counted its chunk as clear, wiped the
           page and gave it to another class while the uploading connection still held the item.
           The fix is the MOVE_BUSY_FLOATING branch: wait until the upload links or frees the item.
           22 runs with that branch reversed (mcapp_reverse_a836eab), 23 as shipped. The page move
           is the real one: slabs_reassign signals the server's page-mover thread. */
        const int len_a = 1500, len_b = 150000;
        /* The class's first page hands out its chunks from the top down, so the LAST of its
           perslab allocations is the page's first chunk. a must be that one: the dst class's chunks
           need not reach the page's tail (1 MiB is not a multiple of their size), so only an a near
           the page's start is guaranteed to lie under a dst item. (First run, 2026-10-11: a was the
           page's top chunk, in that tail, and both arms returned 0xE0016.) */
        unsigned char *a = NULL, *d0;
        item *probe = mcapp_item(n, "probe", len_a, &d0);
        if (!probe)
            return MCAPP_MARK(n, 0xE0016);
        unsigned src = ITEM_clsid(probe), dst = slabs_clsid((size_t)len_b + 200), perslab = 0;
        slabs_available_chunks(src, NULL, &perslab);
        printf("MCAPP-FIX %d src=%u perslab=%u pages=%d dst=%u\n", n, src, perslab, slabs_page_count(src), dst);
        if (slabs_page_count(src) != 1 || perslab < 2 || dst == src) {
            printf("MCAPP-FIX %d triggering condition not created (class not fresh)\n", n);
            return MCAPP_MARK(n, 0xE0016);
        }
        /* the probe was the page's top chunk; perslab - 1 more fill the page down to its first chunk
           (a), and one more opens a second page, so the class has a page to spare */
        item **fill = malloc(sizeof(item *) * (perslab + 1));
        if (!fill)
            return MCAPP_MARK(n, 0xE0016);
        fill[0] = probe;
        unsigned made = 1;
        for (; made <= perslab; made++) {
            char key[32];
            int nkey = snprintf(key, sizeof key, "mcapp-fill-%u", made);
            if (!(fill[made] = item_alloc(key, (size_t)nkey, 0, 0, len_a + 2)))
                break;
        }
        int pages = slabs_page_count(src);
        item *ia = made > perslab ? fill[perslab - 1] : NULL;  /* the page's first chunk */
        if (ia) {
            a = (unsigned char *)ITEM_data(ia);
            mcapp_show(n, "a", a);
        }
        /* free every filler but a: a stays allocated and never linked (upload in progress) */
        for (unsigned i = 0; i < made; i++)
            if (fill[i] != ia)
                item_remove(fill[i]);
        free(fill);
        printf("MCAPP-FIX %d fillers=%u pages=%d\n", n, made, pages);
        if (!ia || made != perslab + 1 || pages != 2) {
            printf("MCAPP-FIX %d triggering condition not created (no second page)\n", n);
            return MCAPP_MARK(n, 0xE0016);
        }
        mcapp_fill(a, 0xA0, 64);                    /* the upload has begun */
        unsigned long a_addr = mcapp_cur(a);
        mcapp_reverse_a836eab = (n == 22);
        int r = (int)slabs_reassign(settings.slab_rebal, (int)src, (int)dst, 0);
        printf("MCAPP-FIX %d reassign result=%d reversed=%d\n", n, r, mcapp_reverse_a836eab);
        if (r != REASSIGN_OK)
            return MCAPP_MARK(n, 0xE0016);
        int moved = 0;
        for (int w = 0; w < 300 && !moved; w++) {
            usleep(10000);
            moved = slabs_page_count(src) == 1;
        }
        printf("MCAPP-FIX %d page-moved=%d\n", n, moved);
        if (!moved)
            return MCAPP_MARK(n, 0x7);              /* the mover waited for the upload */
        /* a's page now belongs to dst: allocate dst items until one covers a's address */
        unsigned long chunk = slabs_size((int)dst);
        item *ib = NULL;
        for (int k = 0; k < 64 && !ib; k++) {
            unsigned char *d;
            char tag[16];
            snprintf(tag, sizeof tag, "b%d", k);
            item *it = mcapp_item(n, tag, len_b, &d);
            if (!it)
                break;
            if (ITEM_clsid(it) != dst) {
                printf("MCAPP-FIX %d item b%d is class %u, not %u\n", n, k, (unsigned)ITEM_clsid(it), dst);
                break;
            }
            if (mcapp_cur(it) <= a_addr && a_addr < mcapp_cur(it) + chunk)
                ib = it;
        }
        if (!ib) {
            printf("MCAPP-FIX %d no item of class %u covers a's address\n", n, dst);
            return MCAPP_MARK(n, 0xE0016);
        }
        unsigned long off = a_addr - mcapp_cur(ib);
        printf("MCAPP-FIX %d new-owner class=%u covers a at offset %lu\n", n, dst, off);
        mcapp_touching(n, a_addr);
        mcapp_fix_poke(a, 0, 0xEE);                 /* the uploader writes its next byte */
        v = mcapp_fix_touch((unsigned char *)ib, (long)off);
        printf("MCAPP-FIX %d returned new-owner[%lu]=%02x (0xee = the write landed in the new owner)\n",
               n, off, v);
        return MCAPP_MARK(n, (1 << 8) | v);
    }
    case 24:
    case 25: {
        /* Two real page moves in the running server. A chunked item (header in a small class, one
           data chunk in the largest class) is linked and expired. The header's page is moved first:
           with c0e5a99 reversed (fixture 24) the expired header is freed header-only, so its data
           chunk is orphaned with head still pointing at the (now revoked, under patch 0006) header.
           The data chunk's page is moved next; the mover reads the orphan's head. ia and three more
           big items fill both classes to two pages; all four are linked+expired so none blocks the
           header-page move, and plain fillers bring the header class to two pages. */
        const int big = 700000;   /* > slab_chunk_size_max (512 KiB): a chunked item */
        item *big_it[4]; item_chunk *big_ch[4];
        for (int k = 0; k < 4; k++) {
            char key[32]; int nkey = snprintf(key, sizeof key, "mcapp-fix-c%d", k);
            big_it[k] = item_alloc(key, (size_t)nkey, 0, 0, big + 2);
            if (!big_it[k] || !(big_it[k]->it_flags & ITEM_CHUNKED)) {
                printf("MCAPP-FIX %d item_alloc c%d FAILED or not chunked\n", n, k);
                return MCAPP_MARK(n, 0xE0018);
            }
            big_ch[k] = do_item_alloc_chunk((item_chunk *)ITEM_schunk(big_it[k]), (size_t)big);
            if (!big_ch[k]) { printf("MCAPP-FIX %d alloc_chunk c%d FAILED\n", n, k); return MCAPP_MARK(n, 0xE0018); }
        }
        item *ia = big_it[0];
        unsigned hdr_cls = ITEM_clsid(ia), chk_cls = big_ch[0]->slabs_clsid;
        unsigned hdr_perslab = 0, chk_perslab = 0;
        slabs_available_chunks(hdr_cls, NULL, &hdr_perslab);
        slabs_available_chunks(chk_cls, NULL, &chk_perslab);
        printf("MCAPP-FIX %d hdr_cls=%u (perslab %u, pages %d) chk_cls=%u (perslab %u, pages %d)\n",
               n, hdr_cls, hdr_perslab, slabs_page_count(hdr_cls), chk_cls, chk_perslab, slabs_page_count(chk_cls));
        /* the chunk class is the largest: two chunks per 1 MiB page, so four items = two pages,
           ia's chunk the first chunk of page 0 */
        if (chk_cls == hdr_cls || slabs_page_count(chk_cls) != 2) {
            printf("MCAPP-FIX %d triggering condition not created (chunk class not two pages)\n", n);
            return MCAPP_MARK(n, 0xE0018);
        }
        /* bring the header class to two pages with plain items of its own class, then free them:
           their slots are on the free list (ITEM_SLABBED) and do not block the header-page move */
        item **fill = malloc(sizeof(item *) * (hdr_perslab + 1));
        if (!fill) return MCAPP_MARK(n, 0xE0018);
        unsigned made = 0;
        for (; made <= hdr_perslab; made++) {
            char key[32]; int nkey = snprintf(key, sizeof key, "mcapp-hf-%u", made);
            item *it = item_alloc(key, (size_t)nkey, 0, 0, 8);
            if (!it || ITEM_clsid(it) != hdr_cls) { if (it) item_remove(it); break; }
            fill[made] = it;
        }
        for (unsigned i = 0; i < made; i++) item_remove(fill[i]);
        free(fill);
        printf("MCAPP-FIX %d hdr fillers=%u hdr pages=%d\n", n, made, slabs_page_count(hdr_cls));
        if (slabs_page_count(hdr_cls) < 2) {
            printf("MCAPP-FIX %d triggering condition not created (header class not two pages)\n", n);
            return MCAPP_MARK(n, 0xE0018);
        }
        /* link and expire all four big items, so each takes the header-only free path and none is a
           floating (upload-in-progress) item that would stall the move */
        for (int k = 0; k < 4; k++) {
            item_link(big_it[k]);                 /* refcount 1 -> 2, ITEM_LINKED, assoc + LRU */
            item_remove(big_it[k]);               /* the storing client lets go -> 1 */
            big_it[k]->exptime = 1;               /* 1 < current_time: expired */
        }
        if ((ia->it_flags & ITEM_LINKED) == 0) {
            printf("MCAPP-FIX %d ia not linked\n", n);
            return MCAPP_MARK(n, 0xE0018);
        }
        unsigned long hdr_flags_addr = (unsigned long)(void *)&ia->it_flags;
        unsigned hdr_dst = 0;
        for (unsigned c = POWER_SMALLEST; c <= 63; c++)
            if (c != hdr_cls && c != chk_cls && slabs_page_count(c) == 0) { hdr_dst = c; break; }
        unsigned chk_dst = 0;
        for (unsigned c = POWER_SMALLEST; c <= 63; c++)
            if (c != hdr_cls && c != chk_cls && c != hdr_dst && slabs_page_count(c) == 0) { chk_dst = c; break; }
        printf("MCAPP-FIX %d hdr_dst=%u chk_dst=%u hdr_flags_addr=%lx\n", n, hdr_dst, chk_dst, hdr_flags_addr);
        if (!hdr_dst || !chk_dst) return MCAPP_MARK(n, 0xE0018);
        mcapp_reverse_c0e5a99 = (n == 24);
        /* first move: the header's page. After it, under 24 the chunks are orphaned. */
        if (slabs_reassign(settings.slab_rebal, (int)hdr_cls, (int)hdr_dst, 0) != REASSIGN_OK)
            return MCAPP_MARK(n, 0xE0018);
        int hdr_moved = 0;
        for (int w = 0; w < 300 && !hdr_moved; w++) { usleep(10000); hdr_moved = slabs_page_count(hdr_cls) == 1; }
        printf("MCAPP-FIX %d header-page-moved=%d reversed=%d\n", n, hdr_moved, mcapp_reverse_c0e5a99);
        if (!hdr_moved) return MCAPP_MARK(n, 0x7);     /* the mover waited: header page not moved */
        /* second move: a data chunk's page. The mover reads the orphan chunk's head here. */
        mcapp_touching(n, hdr_flags_addr);
        enum reassign_result_type r2 = slabs_reassign(settings.slab_rebal, (int)chk_cls, (int)chk_dst, 0);
        int chk_moved = 0;
        for (int w = 0; w < 300 && !chk_moved; w++) { usleep(10000); chk_moved = slabs_page_count(chk_cls) < 2; }
        printf("MCAPP-FIX %d chunk-reassign=%d chunk-page-moved=%d\n", n, (int)r2, chk_moved);
        /* On the lifetimes arm the mover faulted at hdr_flags_addr and never got here. Reaching this
           line means no fault: the plain arm, or the shipped arm (25) with no orphan. */
        return MCAPP_MARK(n, (chk_moved ? 0x20 : 0x10) | (mcapp_reverse_c0e5a99 ? 1 : 0));
    }
    default:
        printf("MCAPP-FIX %d unknown\n", n);
        return MCAPP_MARK(0xF, n);
    }
}

/* The hidden command's body: run fixture n here, on this worker, and end the process with its mark. */
static void mcapp_run_fixture(int n)
{
    int mark = mcapp_fixture(n);
    printf("MCAPP-FIX %d mark=%x\n", n, mark);
    fflush(stdout);
    exit(mark);
}
