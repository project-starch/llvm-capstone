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
 *  11 chunked_reuse   fixture 10 for a CHUNKED item (a 700 KB value: its header in a small class,
 *                     one 512 KiB data chunk attached with do_item_alloc_chunk). item_remove frees
 *                     it through do_slabs_free_chunked, which files the header and then the chunk;
 *                     a new item of the same shape takes both back; reads the old chunk's data
 *                     through the old pointer. Added with the slabsublet arm, whose oracle never
 *                     frees a chunked item, so this is the only run of that release path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "items.h"

#define MCAPP_MARK(n, v) (0x100000 * (n) + ((v) & 0xFFFFF))

unsigned char mcapp_fix_global[64];

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
