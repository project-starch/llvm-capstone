/* Safety fixtures for the tshark domain: what does a heap overflow, a use after free or a stale
 * pointer into a reset wmem pool actually DO here? The M1-M5 result and the oracle show that the
 * dissection is CORRECT under capability enforcement. They say nothing about safety
 * (ports/ffmpeg/app/results/2026-09-23-qemu-safety, where the same question was first asked).
 *
 * This is the FFmpeg port's src/capstone-domain/ffapp_safety.c, carried over to what tshark
 * allocates with: GLib's g_malloc/g_free instead of av_malloc/av_free, and wmem's block
 * allocators (the ones tshark's packet and file scopes use) instead of FFmpeg's pools. Built by
 * host/build-domain.sh once per fixture (-DTSAPP_FIXTURE=<n>) with tshark.c's own compile command,
 * and linked in tshark.c.o's place, as the staged images M1-M4 are: the same runtime, heap, GLib,
 * wmem and link as M5, and only main() differs.
 *
 * ONE FIXTURE PER IMAGE. A capability fault inside a domain ends the emulator (capstone-qemu
 * prints "domain halted by capability fault", then exits), so a fixture that faults can report
 * nothing beside the fault, and must be the LAST image of its boot. A fixture that returns
 * reports a mark as its exit status instead: 0x100000 * fixture + a value.
 *
 * EACH FIXTURE HAS TO CREATE ITS CONDITION, or its result means nothing. So every fixture prints,
 * BEFORE the access it tests, the addresses and capability bounds it is about to use, then
 *     TSAPP-FIX <n> target=<hex>
 *     TSAPP-FIX <n> touch
 * A fault counts as the fixture's only after its touch line and with nothing printed after it;
 * a bounds fault must also name the printed target (host/safety-verdict.py). The marks carry the
 * byte read, so an unprotected arm's answer is a VALUE (the neighbour's pattern, the new
 * occupant's pattern), not merely "it came back".
 *
 * Every access under test goes through tsapp_fix_touch/tsapp_fix_poke, so a fault pc names one of
 * them. Addresses are read with the cursor builtin and only ever compared or printed, never turned
 * back into pointers.
 *
 *   1 heap_len        two 64-byte g_malloc objects, written and read in bounds; prints each
 *                     pointer's capability length. The setup control: always returns.
 *   2 heap_neighbour  writes through the FIRST object at the address of the SECOND's first byte,
 *                     then reads that byte back through the second's own pointer.
 *   3 heap_one_past   reads one byte past the end of a 64-byte object.
 *   4 uaf_noreuse     reads a g_free'd object, nothing allocated since.
 *   5 uaf_reuse       frees, allocates the same size again (and reports whether it got the same
 *                     address), writes a new pattern, reads through the OLD pointer.
 *   6 stale_free      frees p, lets q take p's memory, frees p AGAIN (the stale pointer), then
 *                     allocates r: does r alias the live q?
 *   7 global_oob      reads one byte past a 64-byte global.     (controls: expected to fault
 *   8 stack_oob       reads one byte past a 64-byte stack array. on every heap arm alike)
 *   9 global_merged   two static 64-byte arrays in this file; reads the SECOND's first byte
 *                     through the FIRST (FFmpeg's fixture 10: the compiler merged such a pair
 *                     into one .L_MergedGlobals, so per-object global bounds did not hold).
 *  10 wmem_fast_reset a stale pointer after a reset of a BLOCK_FAST allocator, the kind tshark's
 *                     packet scope is: wmem_free_all rewinds the block, the next allocation
 *                     takes the same bytes, and the old pointer reads the new occupant's pattern.
 *  11 wmem_block_reset the same after wmem_free_all on a BLOCK allocator, the kind a file scope is.
 *  12 wmem_neighbour  fixture 2 inside one BLOCK allocator: writes through the first wmem
 *                     allocation at the second's first byte.
 * On every heap arm of this port, 10-12 are predicted to return: a wmem block is one g_malloc,
 * and every allocation carved from it carries the whole block's bounds. That is the gap the
 * wmem hooks (ports/wireshark/wmem) are for; it is measured here, not assumed.
 *
 * NO FIXTURE ROUTES INTO A DISSECTOR, and none may: the handles that whitelisted code looks up
 * but this build never registers (ipx and ccsds, packet-ieee8023.c:122,124; http2,
 * packet-http.c:5113; tls-echconfig, packet-dns.c:6298; tpkt, prefs.c:5947; file, packet.c:258;
 * sport, packet-tcp.c) throw "Dissector bug" through the exception path, which would read as a
 * result (docs/plans/2026-09-23-tshark-full-app-port.md, "Absent handles").
 */
#include <stdio.h>
#include <string.h>

#include <glib.h>
#include <wsutil/wmem/wmem.h>

#ifndef TSAPP_FIXTURE
#error "TSAPP_FIXTURE selects the fixture (1..12)"
#endif

#define FX_MARK(v) (0x100000 * TSAPP_FIXTURE + ((v) & 0xFFFFF))

unsigned char tsapp_fix_global[64];

static unsigned long cur(const volatile void *p) { return __builtin_capstone_cap_get_cursor((void *)p); }
static unsigned long end(const volatile void *p) { return __builtin_capstone_cap_get_end((void *)p); }
static unsigned long base(const volatile void *p) { return __builtin_capstone_cap_get_base((void *)p); }

static void show(const char *name, const volatile void *p)
{
    printf("TSAPP-FIX %d %s cursor=%lx bounds=[%lx,%lx) len-from-cursor=%lu\n", TSAPP_FIXTURE,
           name, cur(p), base(p), end(p), end(p) - cur(p));
}

__attribute__((noinline)) unsigned tsapp_fix_touch(const volatile unsigned char *b, long i)
{
    return b[i];
}

__attribute__((noinline)) void tsapp_fix_poke(volatile unsigned char *b, long i, unsigned char v)
{
    b[i] = v;
}

/* The target is an ADDRESS computed before any free: a revoked capability reloads untagged, and
   even reading its cursor must not be what faults. */
static void touching(unsigned long target)
{
    printf("TSAPP-FIX %d target=%lx\n", TSAPP_FIXTURE, target);
    printf("TSAPP-FIX %d touch\n", TSAPP_FIXTURE);
    fflush(stdout);
}

static void fill(volatile unsigned char *p, unsigned char v0, int n)
{
    for (int i = 0; i < n; i++)
        tsapp_fix_poke(p, i, (unsigned char)(v0 + i));
}

static int fixture(void)
{
    volatile long idx;
    unsigned v;

#if TSAPP_FIXTURE == 1
    unsigned char *p = g_malloc(64), *q = g_malloc(64);
    fill(p, 0xA0, 64);
    fill(q, 0x5B, 64);
    show("p", p);
    show("q", q);
    v = tsapp_fix_touch(p, 63) == 0xA0 + 63 && tsapp_fix_touch(q, 63) == 0x5B + 63;
    printf("TSAPP-FIX 1 in-bounds %s\n", v ? "ok" : "WRONG");
    return FX_MARK(v ? 0x1 : 0xE0002);

#elif TSAPP_FIXTURE == 2
    unsigned char *p = g_malloc(64), *q = g_malloc(64);
    fill(p, 0xA0, 64);
    fill(q, 0x5B, 64);
    show("p", p);
    show("q", q);
    idx = (long)(cur(q) - cur(p));   /* the second object's first byte, as an offset from p */
    printf("TSAPP-FIX 2 q-p=%ld\n", idx);
    touching(cur(q));
    tsapp_fix_poke(p, idx, 0xEE);
    v = tsapp_fix_touch(q, 0);        /* read back through the neighbour's OWN pointer */
    printf("TSAPP-FIX 2 returned q[0]=%02x (0x5b untouched, 0xee overwritten through p)\n", v);
    return FX_MARK(v);

#elif TSAPP_FIXTURE == 3
    unsigned char *p = g_malloc(64);
    fill(p, 0xA0, 64);
    show("p", p);
    idx = 64;
    touching(cur(p) + 64);
    v = tsapp_fix_touch(p, idx);
    printf("TSAPP-FIX 3 returned p[64]=%02x\n", v);
    return FX_MARK(v);

#elif TSAPP_FIXTURE == 4
    unsigned char *p = g_malloc(64);
    fill(p, 0xA0, 64);
    show("p", p);
    unsigned long p_addr = cur(p);
    g_free(p);
    idx = 0;
    touching(p_addr);
    v = tsapp_fix_touch(p, idx);
    printf("TSAPP-FIX 4 returned p[0]=%02x (0xa0 = the freed object's own byte)\n", v);
    return FX_MARK(v);

#elif TSAPP_FIXTURE == 5
    unsigned char *p = g_malloc(64);
    fill(p, 0xA0, 64);
    show("p", p);
    unsigned long p_addr = cur(p);
    g_free(p);
    unsigned char *q = g_malloc(64);
    fill(q, 0x5B, 64);
    show("q", q);
    unsigned same = cur(q) == p_addr;
    printf("TSAPP-FIX 5 same-address=%u\n", same);
    idx = 0;
    touching(p_addr);
    v = tsapp_fix_touch(p, idx);
    printf("TSAPP-FIX 5 returned p[0]=%02x (0x5b = the NEW occupant's byte)\n", v);
    return FX_MARK((same << 8) | v);

#elif TSAPP_FIXTURE == 6
    unsigned char *p = g_malloc(64);
    show("p", p);
    unsigned long p_addr = cur(p);
    g_free(p);
    unsigned char *q = g_malloc(64);
    fill(q, 0x5B, 64);
    show("q", q);
    printf("TSAPP-FIX 6 q-took-p's-address=%u\n", cur(q) == p_addr);
    touching(p_addr);                   /* the "touch" here is the stale free */
    g_free(p);
    unsigned char *r = g_malloc(64);
    show("r", r);
    unsigned alias = cur(r) == cur(q);
    if (alias)
        fill(r, 0x77, 64);              /* the new owner writes; the live q sees it */
    v = tsapp_fix_touch(q, 0);
    printf("TSAPP-FIX 6 returned r-aliases-q=%u q[0]=%02x\n", alias, v);
    return FX_MARK((alias << 8) | v);

#elif TSAPP_FIXTURE == 7
    fill(tsapp_fix_global, 0xA0, 64);
    show("global", tsapp_fix_global);
    printf("TSAPP-FIX 7 in-bounds g[63]=%02x\n", tsapp_fix_touch(tsapp_fix_global, 63));
    idx = 64;
    touching(cur(tsapp_fix_global) + 64);
    v = tsapp_fix_touch(tsapp_fix_global, idx);
    printf("TSAPP-FIX 7 returned g[64]=%02x\n", v);
    return FX_MARK(v);

#elif TSAPP_FIXTURE == 8
    unsigned char buf[64];
    fill(buf, 0xA0, 64);
    show("stack", buf);
    printf("TSAPP-FIX 8 in-bounds buf[63]=%02x\n", tsapp_fix_touch(buf, 63));
    idx = 64;
    touching(cur(buf) + 64);
    v = tsapp_fix_touch(buf, idx);
    printf("TSAPP-FIX 8 returned buf[64]=%02x\n", v);
    return FX_MARK(v);

#elif TSAPP_FIXTURE == 9
    static unsigned char ga[64], gb[64];
    fill(ga, 0xA0, 64);
    fill(gb, 0x5B, 64);
    show("ga", ga);
    show("gb", gb);
    idx = (long)(cur(gb) - cur(ga));
    printf("TSAPP-FIX 9 gb-ga=%ld\n", idx);
    touching(cur(gb));
    v = tsapp_fix_touch(ga, idx);
    printf("TSAPP-FIX 9 returned gb[0]-through-ga=%02x (0x5b = the second array's byte)\n", v);
    return FX_MARK(v);

#elif TSAPP_FIXTURE == 10 || TSAPP_FIXTURE == 11
    wmem_allocator_t *a = wmem_allocator_new(TSAPP_FIXTURE == 10 ? WMEM_ALLOCATOR_BLOCK_FAST
                                                                 : WMEM_ALLOCATOR_BLOCK);
    unsigned char *p = wmem_alloc(a, 64);
    fill(p, 0xA0, 64);
    show("p", p);
    unsigned long p_addr = cur(p);
    wmem_free_all(a);                   /* the scope ends: every allocation in it is dead */
    unsigned char *q = wmem_alloc(a, 64);
    fill(q, 0x5B, 64);
    show("q", q);
    unsigned same = cur(q) == p_addr;
    printf("TSAPP-FIX %d same-address=%u\n", TSAPP_FIXTURE, same);
    idx = 0;
    touching(p_addr);
    v = tsapp_fix_touch(p, idx);
    printf("TSAPP-FIX %d returned p[0]=%02x (0x5b = the new scope's byte)\n", TSAPP_FIXTURE, v);
    return FX_MARK((same << 8) | v);

#elif TSAPP_FIXTURE == 12
    wmem_allocator_t *a = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
    unsigned char *p = wmem_alloc(a, 64), *q = wmem_alloc(a, 64);
    fill(p, 0xA0, 64);
    fill(q, 0x5B, 64);
    show("p", p);
    show("q", q);
    idx = (long)(cur(q) - cur(p));
    printf("TSAPP-FIX 12 q-p=%ld\n", idx);
    touching(cur(q));
    tsapp_fix_poke(p, idx, 0xEE);
    v = tsapp_fix_touch(q, 0);
    printf("TSAPP-FIX 12 returned q[0]=%02x (0x5b untouched, 0xee overwritten through p)\n", v);
    return FX_MARK(v);

#else
#error "unknown TSAPP_FIXTURE"
#endif
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("TSAPP-FIX %d begin\n", TSAPP_FIXTURE);
    int status = fixture();
    printf("TSAPP-FIX %d mark=%x\n", TSAPP_FIXTURE, status);
    fflush(stdout);
    return status;
}
