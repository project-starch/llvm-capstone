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
 * a bounds fault must also name the printed target (ports/common/application/check-safety.py). The marks carry the
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
 *  13 wmem_chunk_free one allocation of a BLOCK allocator freed with wmem_free while its block
 *                     lives on, then read. Added with the chunks arm, where that free is a revoke;
 *                     elsewhere the free changes nothing the old pointer can see, except that
 *                     upstream writes its free-list links into the chunk's first bytes.
 *  14 zigbee_touchlink CVE-2026-95391, live at our v4.6.8 pin: a global container keeps a record
 *                     across the release that frees it, AND holds a key pointing INSIDE that same
 *                     record; the stale interior key is dereferenced. Fixture 5's shape plus the
 *                     global and the interior pointer. Heap class (wmem_gc returns the block to
 *                     the OS), not wmem-nested -- see 06a775b2e1b6.
 *  15 http2_regex_unref live at our v4.6.8 pin: a refcount reaching zero frees the object while the
 *                     owner's file-static pointer stays set, so its own `== NULL` validity test
 *                     still passes and the next use reads freed storage. Differs from 14 in the
 *                     LIFETIME-ENDER: a refcount, not an explicit free.
 * 14 and 15 are the two upstream defects this port carries that are confirmed live at the pin by
 * the cherry-pick probe; both are HEAP class, so `chunks` is predicted to behave as `sublet`.
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
#error "TSAPP_FIXTURE selects the fixture (1..13)"
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

#elif TSAPP_FIXTURE == 13
    wmem_allocator_t *a = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
    unsigned char *p = wmem_alloc(a, 64), *keep = wmem_alloc(a, 64);
    fill(p, 0xA0, 64);
    fill(keep, 0x5B, 64);               /* the block stays live, with p's neighbour in use */
    show("p", p);
    unsigned long p_addr = cur(p);
    wmem_free(a, p);                    /* one chunk back, inside a block that lives on */
    idx = 0;
    touching(p_addr);
    v = tsapp_fix_touch(p, idx);
    printf("TSAPP-FIX 13 returned p[0]=%02x (0xa0 = its own byte, unless a free-list link covers it)\n", v);
    return FX_MARK(v);

#elif TSAPP_FIXTURE == 14
    /* ZigBee ZCL Touchlink, CVE-2026-95391 / wnpa-sec-2026-92, upstream 030bf6ad011c (fix
     * 609134fa7c55, first released in v4.6.9 and so ABSENT from our v4.6.8 pin). A file-scope
     * global GHashTable keeps commissioning records across a redissect that frees them, AND its
     * keys point INSIDE those same records (&commissioning_data->transaction_id), so the key and
     * the value go stale together -- which is what makes it worse than fixture 5.
     *
     * Live at the pin, read from the pinned tree: the global at packet-zbee-zcl-general.c:15899,
     * created exactly once at :16917 with NO register_init_routine to empty it, the record
     * allocated at :16591 and inserted at :16593. Verified not backported: the cherry-pick probe
     * over v4.6.8 is empty for this sha while the identical probe returns a sha for a fix that WAS
     * backported, so the negative is a tested one.
     *
     * HEAP class. Precisely: the OBJECT is a wmem file-scope allocation (wmem_new0(wmem_file_scope(),
     * ...) at :16591), but the LIFETIME-ENDING FREE bottoms out in g_free of the containing block --
     * wmem_leave_file_scope() ends in wmem_gc, and wmem_block_gc returns a wholly-unused block to the
     * OS via wmem_free(NULL, cur). So it is not a wmem RESET that ends the lifetime, which is what
     * the nested class means here. Classified as nested once and retracted (06a775b2e1b6); do not
     * re-file it, and do not state it as "not at a wmem scope" either -- the allocation is.
     *
     * Reduced to the allocator seam, no dissector and no GHashTable traversal: a global holds both
     * the record and an INTERIOR key, the storage is released, a new owner takes it, and the stale
     * INTERIOR key is dereferenced -- which is what upstream's g_hash_table_lookup does first. */
    static unsigned char *tl_map_value;   /* the global container's value */
    static unsigned char *tl_map_key;     /* its key, interior to the SAME object */
    unsigned char *rec = g_malloc(64);
    fill(rec, 0xA0, 64);
    show("rec", rec);
    tl_map_value = rec;
    tl_map_key   = rec + 8;               /* &commissioning_data->transaction_id */
    show("key-interior", tl_map_key);
    unsigned long rec_addr = cur(rec);
    unsigned long key_addr = cur(tl_map_key);   /* read BEFORE the free: cur() on a revoked
                                                 * capability would fault outside the touch
                                                 * helper, breaking the oracle's pc rule */
    g_free(rec);                          /* the redissect's release */
    unsigned char *nu = g_malloc(64);
    fill(nu, 0x5B, 64);
    show("new-owner", nu);
    unsigned same14 = cur(nu) == rec_addr;
    printf("TSAPP-FIX 14 same-address=%u map-still-holds=%u\n", same14, tl_map_value != NULL);
    idx = 0;
    touching(key_addr);
    v = tsapp_fix_touch(tl_map_key, idx); /* the lookup hashes the stale interior key */
    printf("TSAPP-FIX 14 returned key[0]=%02x (0x63 = the NEW occupant's byte at offset 8)\n", v);
    return FX_MARK((same14 << 8) | v);

#elif TSAPP_FIXTURE == 15
    /* http2 3GPP header decoding, upstream 6e61bca421 (on master, ABSENT from our v4.6.8 pin and
     * not backported -- probed the same way as fixture 14). Two file-static GRegex pointers are
     * created lazily behind an `if (regex == NULL)` guard and released with g_regex_unref, which
     * frees the object once its refcount reaches zero WITHOUT nulling the static. The guard then
     * passes on freed storage and g_regex_match reads through it.
     *
     * Live at the pin: packet-http2.c:2148-2149 declare the statics, :2160 is the NULL guard,
     * :2171 matches, :2185-2186 unref; the pattern repeats at :2213 and :2262-2263. HEAP class --
     * GRegex is GLib-allocated, and no wmem call appears in the fix.
     *
     * The distinguishing feature against 14 is the LIFETIME-ENDER: not an explicit free of a
     * container's entry but a REFCOUNT reaching zero, with the owner's own validity test -- the
     * NULL check -- left satisfied. The fixture creates that condition itself rather than relying
     * on the http2.3gpp_session preference being set at run time: a directed test that does not
     * create its triggering condition comes back clean and void. */
    static unsigned char *h2_regex;       /* the file-static the guard tests */
    static int h2_refcount;
    if (h2_regex == NULL) {               /* the lazy-create guard */
        h2_regex = g_malloc(64);
        h2_refcount = 1;
        fill(h2_regex, 0xA0, 64);
    }
    show("regex", h2_regex);
    unsigned long rx_addr = cur(h2_regex);
    if (--h2_refcount == 0)
        g_free(h2_regex);                 /* g_regex_unref: frees, leaves the static set */
    unsigned char *nu15 = g_malloc(64);
    fill(nu15, 0x5B, 64);
    show("new-owner", nu15);
    unsigned same15 = cur(nu15) == rx_addr;
    printf("TSAPP-FIX 15 same-address=%u guard-passes=%u\n", same15, h2_regex != NULL);
    idx = 0;
    touching(rx_addr);
    v = tsapp_fix_touch(h2_regex, idx);   /* the guard passed; g_regex_match reads here */
    printf("TSAPP-FIX 15 returned regex[0]=%02x (0x5b = the NEW occupant's byte)\n", v);
    return FX_MARK((same15 << 8) | v);

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
