/* Safety fixtures for the FFmpeg domain: what does a heap overflow, a use-after-free or a
 * double free actually DO in this domain? Written after the question "do temporal violations
 * still fault?" was asked of the M1-M5 result and could not be answered from it: that result
 * shows the decode is CORRECT under capability enforcement, and says nothing about safety.
 *
 * ONE FIXTURE PER IMAGE (FFAPP_FIXTURE), because a capability fault inside a domain ends the
 * emulator (capstone-qemu cpu_helper.c, "domain halted by capability fault" then exit), so a
 * fixture that faults can report nothing beside the fault, and must be the LAST image of its
 * boot. A fixture that returns reports a mark instead.
 *
 * EACH FIXTURE HAS TO CREATE ITS CONDITION, or its result means nothing (CLAUDE.md, "the
 * synthetic test must CREATE the triggering condition"). So every fixture prints, BEFORE the
 * touch, the addresses and capability bounds it is about to use, and then the line
 *     FFAPP-FIX <n> touch
 * A fault is attributable to the touch only if that line was printed and nothing after it
 * was; an OOB fault must also name the address the fixture printed as its target. The marks
 * carry the byte read, so an unprotected arm's answer is a VALUE (the neighbour's pattern, the
 * new occupant's pattern), not merely "it came back".
 *
 * Every touch goes through ffapp_fix_touch/ffapp_fix_poke, so a fault pc names one of them.
 * Addresses are read with the cursor builtin and only ever compared or printed, never turned
 * back into pointers.
 *
 *   1 heap_len        two 64-byte av_malloc objects, written and read in bounds; prints each
 *                     heap pointer's capability length. The setup control: always returns.
 *   2 heap_neighbour  writes through the FIRST object at the address of the SECOND's first
 *                     byte, then reads that byte back through the second's own pointer.
 *   3 heap_one_past   reads one byte past the end of a 64-byte object.
 *   4 uaf_noreuse     reads a freed object, nothing allocated since.
 *   5 uaf_reuse       frees, allocates the same size again (and reports whether it got the
 *                     same address), writes a new pattern, reads through the OLD pointer.
 *   6 avbuffer_uaf    reads the data of an AVBufferRef after av_buffer_unref, i.e. through
 *                     FFmpeg's own reference-counted API rather than av_free.
 *   7 stale_free      frees p, lets q take p's memory, frees p AGAIN (the stale pointer),
 *                     then allocates r: does r alias the live q?
 *   8 global_oob      reads one byte past a 64-byte global.     (control: expected to fault
 *   9 stack_oob       reads one byte past a 64-byte stack array. on every heap arm alike)
 *  10 global_merged   two static 64-byte arrays in this file; reads the SECOND's first byte
 *                     through the FIRST. Added after fixture 8 showed the global's bounds
 *                     covering a merged group (.L_MergedGlobals), not the object alone: if the
 *                     compiler merges the pair, per-object global bounds do not hold for it.
 *
 * FFmpeg's OWN pools (built only on the pool arms, build-domain.sh FFAPP_POOL). These go
 * through the real API -- av_buffer_pool_*, av_refstruct_* -- which a pool returns to and
 * reissues from WITHOUT calling free, so no heap arm can see them:
 *  11 pool_len        an AVBufferPool buffer's capability length (setup control: returns)
 *  12 pool_after_ret  a pool buffer read after av_buffer_unref returned it to the pool
 *  13 pool_reuse      ... after the pool reissued it to a new get that wrote a new pattern
 *  14 pool_one_past   one byte past a 64-byte pool buffer
 *  15 rs_after_ret    a refstruct pool object read after av_refstruct_unref returned it
 *  16 rs_underflow    one byte BELOW a refstruct object, where stock keeps its RefCount
 *  17 rs_stale_unref  a stale refstruct pointer unref'd after its entry went to a new owner:
 *                     does the live owner lose its reference (a third get then aliases it)?
 */
#include <stdio.h>
#include <string.h>

#include "libavutil/buffer.h"
#include "libavutil/mem.h"
#include "libavutil/refstruct.h"
#ifdef FFAPP_POOL_MODE
#include "ffapp_pool.h"
#endif

#ifndef FFAPP_FIXTURE
#error "FFAPP_FIXTURE selects the fixture (1..9)"
#endif

#define FX_MARK(v) (0x100000 * FFAPP_FIXTURE + ((v) & 0xFFFFF))

extern char **__environ;
static char *ffapp_empty_environ[1] = { 0 };

unsigned char ffapp_fix_global[64];

static unsigned long cur(const volatile void *p) { return __builtin_capstone_cap_get_cursor((void *)p); }
static unsigned long end(const volatile void *p) { return __builtin_capstone_cap_get_end((void *)p); }
static unsigned long base(const volatile void *p) { return __builtin_capstone_cap_get_base((void *)p); }

static void show(const char *name, const volatile void *p)
{
    printf("FFAPP-FIX %d %s cursor=%lx bounds=[%lx,%lx) len-from-cursor=%lu\n", FFAPP_FIXTURE,
           name, cur(p), base(p), end(p), end(p) - cur(p));
}

__attribute__((noinline)) unsigned ffapp_fix_touch(const volatile unsigned char *b, long i)
{
    return b[i];
}

__attribute__((noinline)) void ffapp_fix_poke(volatile unsigned char *b, long i, unsigned char v)
{
    b[i] = v;
}

/* The target is an ADDRESS computed before any free: a revoked capability reloads untagged,
   and even reading its cursor must not be what faults. */
static void touching(unsigned long target)
{
    printf("FFAPP-FIX %d target=%lx\n", FFAPP_FIXTURE, target);
    printf("FFAPP-FIX %d touch\n", FFAPP_FIXTURE);
    fflush(stdout);
}

static void fill(volatile unsigned char *p, unsigned char v0, int n)
{
    for (int i = 0; i < n; i++)
        ffapp_fix_poke(p, i, (unsigned char)(v0 + i));
}

static int fixture(void)
{
    volatile long idx;
    unsigned v;

#if FFAPP_FIXTURE == 1
    unsigned char *p = av_malloc(64), *q = av_malloc(64);
    if (!p || !q)
        return FX_MARK(0xE0001);
    fill(p, 0xA0, 64);
    fill(q, 0x5B, 64);
    show("p", p);
    show("q", q);
    v = ffapp_fix_touch(p, 63) == 0xA0 + 63 && ffapp_fix_touch(q, 63) == 0x5B + 63;
    printf("FFAPP-FIX 1 in-bounds %s\n", v ? "ok" : "WRONG");
    return FX_MARK(v ? 0x1 : 0xE0002);

#elif FFAPP_FIXTURE == 2
    unsigned char *p = av_malloc(64), *q = av_malloc(64);
    if (!p || !q)
        return FX_MARK(0xE0001);
    fill(p, 0xA0, 64);
    fill(q, 0x5B, 64);
    show("p", p);
    show("q", q);
    idx = (long)(cur(q) - cur(p));   /* the second object's first byte, as an offset from p */
    printf("FFAPP-FIX 2 q-p=%ld\n", idx);
    touching(cur(q));
    ffapp_fix_poke(p, idx, 0xEE);
    v = ffapp_fix_touch(q, 0);        /* read back through the neighbour's OWN pointer */
    printf("FFAPP-FIX 2 returned q[0]=%02x (0x5b untouched, 0xee overwritten through p)\n", v);
    return FX_MARK(v);

#elif FFAPP_FIXTURE == 3
    unsigned char *p = av_malloc(64);
    if (!p)
        return FX_MARK(0xE0001);
    fill(p, 0xA0, 64);
    show("p", p);
    idx = 64;
    touching(cur(p) + 64);
    v = ffapp_fix_touch(p, idx);
    printf("FFAPP-FIX 3 returned p[64]=%02x\n", v);
    return FX_MARK(v);

#elif FFAPP_FIXTURE == 4
    unsigned char *p = av_malloc(64);
    if (!p)
        return FX_MARK(0xE0001);
    fill(p, 0xA0, 64);
    show("p", p);
    unsigned long p_addr = cur(p);
    av_free(p);
    idx = 0;
    touching(p_addr);
    v = ffapp_fix_touch(p, idx);
    printf("FFAPP-FIX 4 returned p[0]=%02x (0xa0 = the freed object's own byte)\n", v);
    return FX_MARK(v);

#elif FFAPP_FIXTURE == 5
    unsigned char *p = av_malloc(64);
    if (!p)
        return FX_MARK(0xE0001);
    fill(p, 0xA0, 64);
    show("p", p);
    unsigned long p_addr = cur(p);
    av_free(p);
    unsigned char *q = av_malloc(64);
    if (!q)
        return FX_MARK(0xE0002);
    fill(q, 0x5B, 64);
    show("q", q);
    unsigned same = cur(q) == p_addr;
    printf("FFAPP-FIX 5 same-address=%u\n", same);
    idx = 0;
    touching(p_addr);
    v = ffapp_fix_touch(p, idx);
    printf("FFAPP-FIX 5 returned p[0]=%02x (0x5b = the NEW occupant's byte)\n", v);
    return FX_MARK((same << 8) | v);

#elif FFAPP_FIXTURE == 6
    AVBufferRef *r = av_buffer_alloc(64);
    if (!r)
        return FX_MARK(0xE0001);
    unsigned char *d = r->data;
    fill(d, 0xA0, 64);
    show("data", d);
    unsigned long d_addr = cur(d);
    av_buffer_unref(&r);
    printf("FFAPP-FIX 6 unref done, ref=%s\n", r ? "SET" : "null");
    idx = 0;
    touching(d_addr);
    v = ffapp_fix_touch(d, idx);
    printf("FFAPP-FIX 6 returned data[0]=%02x\n", v);
    return FX_MARK(v);

#elif FFAPP_FIXTURE == 7
    unsigned char *p = av_malloc(64);
    if (!p)
        return FX_MARK(0xE0001);
    show("p", p);
    unsigned long p_addr = cur(p);
    av_free(p);
    unsigned char *q = av_malloc(64);
    if (!q)
        return FX_MARK(0xE0002);
    fill(q, 0x5B, 64);
    show("q", q);
    printf("FFAPP-FIX 7 q-took-p's-address=%u\n", cur(q) == p_addr);
    touching(p_addr);                   /* the "touch" here is the stale free */
    av_free(p);
    unsigned char *r = av_malloc(64);
    if (!r)
        return FX_MARK(0xE0003);
    show("r", r);
    unsigned alias = cur(r) == cur(q);
    if (alias)
        fill(r, 0x77, 64);              /* the new owner writes; the live q sees it */
    v = ffapp_fix_touch(q, 0);
    printf("FFAPP-FIX 7 returned r-aliases-q=%u q[0]=%02x\n", alias, v);
    return FX_MARK((alias << 8) | v);

#elif FFAPP_FIXTURE == 8
    fill(ffapp_fix_global, 0xA0, 64);
    show("global", ffapp_fix_global);
    printf("FFAPP-FIX 8 in-bounds g[63]=%02x\n", ffapp_fix_touch(ffapp_fix_global, 63));
    idx = 64;
    touching(cur(ffapp_fix_global) + 64);
    v = ffapp_fix_touch(ffapp_fix_global, idx);
    printf("FFAPP-FIX 8 returned g[64]=%02x\n", v);
    return FX_MARK(v);

#elif FFAPP_FIXTURE == 9
    unsigned char buf[64];
    fill(buf, 0xA0, 64);
    show("stack", buf);
    printf("FFAPP-FIX 9 in-bounds buf[63]=%02x\n", ffapp_fix_touch(buf, 63));
    idx = 64;
    touching(cur(buf) + 64);
    v = ffapp_fix_touch(buf, idx);
    printf("FFAPP-FIX 9 returned buf[64]=%02x\n", v);
    return FX_MARK(v);

#elif FFAPP_FIXTURE == 10
    static unsigned char ga[64], gb[64];
    fill(ga, 0xA0, 64);
    fill(gb, 0x5B, 64);
    show("ga", ga);
    show("gb", gb);
    idx = (long)(cur(gb) - cur(ga));
    printf("FFAPP-FIX 10 gb-ga=%ld\n", idx);
    touching(cur(gb));
    v = ffapp_fix_touch(ga, idx);
    printf("FFAPP-FIX 10 returned gb[0]-through-ga=%02x (0x5b = the second array's byte)\n", v);
    return FX_MARK(v);

#elif FFAPP_FIXTURE >= 11 && FFAPP_FIXTURE <= 14
    AVBufferPool *pool = av_buffer_pool_init(64, av_buffer_alloc);
    AVBufferRef *r = pool ? av_buffer_pool_get(pool) : NULL;
    if (!r)
        return FX_MARK(0xE0001);
    unsigned char *d = r->data;
    fill(d, 0xA0, 64);
    show("p", d);
    unsigned long d_addr = cur(d);
#if FFAPP_FIXTURE == 11
    v = ffapp_fix_touch(d, 63) == 0xA0 + 63;
    printf("FFAPP-FIX 11 in-bounds %s\n", v ? "ok" : "WRONG");
    return FX_MARK(v ? 0x1 : 0xE0002);
#elif FFAPP_FIXTURE == 14
    idx = 64;
    touching(d_addr + 64);
    v = ffapp_fix_touch(d, idx);
    printf("FFAPP-FIX 14 returned p[64]=%02x\n", v);
    return FX_MARK(v);
#else
    av_buffer_unref(&r);                /* back to its pool: not freed */
#if FFAPP_FIXTURE == 12
    idx = 0;
    touching(d_addr);
    v = ffapp_fix_touch(d, idx);
    printf("FFAPP-FIX 12 returned p[0]=%02x (0xa0 = its own byte, still readable)\n", v);
    return FX_MARK(v);
#else
    AVBufferRef *r2 = av_buffer_pool_get(pool);
    if (!r2)
        return FX_MARK(0xE0002);
    fill(r2->data, 0x5B, 64);
    show("q", r2->data);
    unsigned same = cur(r2->data) == d_addr;
    printf("FFAPP-FIX 13 same-address=%u\n", same);
    idx = 0;
    touching(d_addr);
    v = ffapp_fix_touch(d, idx);
    printf("FFAPP-FIX 13 returned p[0]=%02x (0x5b = the new owner's byte)\n", v);
    return FX_MARK((same << 8) | v);
#endif
#endif

#elif FFAPP_FIXTURE == 15 || FFAPP_FIXTURE == 17
    AVRefStructPool *rp = av_refstruct_pool_alloc(64, 0);
    unsigned char *o = rp ? av_refstruct_pool_get(rp) : NULL;
    if (!o)
        return FX_MARK(0xE0001);
    fill(o, 0xA0, 64);
    show("p", o);
    unsigned long o_addr = cur(o);
    unsigned char *stale = o;
    av_refstruct_unref(&o);             /* back to its pool */
#if FFAPP_FIXTURE == 15
    idx = 0;
    touching(o_addr);
    v = ffapp_fix_touch(stale, idx);
    printf("FFAPP-FIX 15 returned p[0]=%02x\n", v);
    return FX_MARK(v);
#else
    unsigned char *o2 = av_refstruct_pool_get(rp);
    if (!o2)
        return FX_MARK(0xE0002);
    fill(o2, 0x5B, 64);
    show("q", o2);
    printf("FFAPP-FIX 17 q-took-p's-address=%u\n", cur(o2) == o_addr);
    touching(o_addr);                   /* the "touch" is the stale unref */
    av_refstruct_unref(&stale);
    unsigned char *o3 = av_refstruct_pool_get(rp);
    if (!o3)
        return FX_MARK(0xE0003);
    show("r", o3);
    unsigned alias = cur(o3) == cur(o2);
    if (alias)
        fill(o3, 0x77, 64);             /* the new owner writes; the live o2 sees it */
    v = ffapp_fix_touch(o2, 0);
    printf("FFAPP-FIX 17 returned r-aliases-q=%u q[0]=%02x\n", alias, v);
    return FX_MARK((alias << 8) | v);
#endif

#elif FFAPP_FIXTURE == 16
    unsigned char *o = av_refstruct_allocz(64);
    if (!o)
        return FX_MARK(0xE0001);
    show("p", o);
    idx = -1;
    touching(cur(o) - 1);
    v = ffapp_fix_touch(o, idx);
    printf("FFAPP-FIX 16 returned p[-1]=%02x (a byte of the RefCount header, in stock)\n", v);
    return FX_MARK(v);

#else
#error "unknown FFAPP_FIXTURE"
#endif
}

int capstone_main(void)
{
    __environ = ffapp_empty_environ;
    setvbuf(stdout, NULL, _IOLBF, 0);
#ifdef FFAPP_POOL_MODE
    ffapp_pool_init();
#endif
    printf("FFAPP-FIX %d begin\n", FFAPP_FIXTURE);
    int status = fixture();
    printf("FFAPP-FIX %d mark=%x\n", FFAPP_FIXTURE, status);
    fflush(stdout);
    return status;
}
