/* PHP 5.0.0 Zend allocator, ported to a Capstone domain.
 *
 * WHAT IS AND IS NOT CHANGED
 *
 * The allocator's own logic is the PHP 5.0.0 original, UNPATCHED. In particular
 * REAL_SIZE still rounds to 8 and the size-class cache is still on:
 *
 *     #define REAL_SIZE(size) ((size+7) & ~0x7)        zend_alloc.c:132
 *
 * The corpus has to patch both of those away for ASan to see anything
 * (scripts/build-php-5.0.0-variant.sh:61-65). That is the point of this port:
 * neither patch is applied here, and the overflow is caught anyway, because the
 * capability handed to the caller is bounded by the TRUE request rather than by
 * the rounded allocation. `_emalloc` already knows the true size -- it records
 * it in `p->size` (zend_alloc.c:201) purely so the cache can report it.
 *
 * Two substitutions were unavoidable:
 *
 *   1. ZEND_DO_MALLOC is a bump arena, not libc malloc. A domain has no libc and
 *      no OS; the arena is a static array. Consequence: no reclamation, so this
 *      port measures overflow, not fragmentation.
 *
 *   2. The block is carved, then the returned capability is NARROWED with
 *      cap_shrink. See ZEND_CAP_BOUND_END below -- that one expression is the
 *      whole experiment.
 *
 * BOUNDS MODEL (option A)
 *
 *      base   = the header
 *      end    = header + sizeof(hdr) + <size or REAL_SIZE(size)>
 *      cursor = header + sizeof(hdr)   (what the caller is handed)
 *
 * Bounding from the HEADER rather than from the payload is deliberate. _efree
 * (zend_alloc.c:250) and _erealloc (:320) recover the header by subtracting
 * from the caller's own pointer:
 *
 *      p = (zend_mem_header *)((char *)ptr - sizeof(zend_mem_header) - MEM_HEADER_PADDING);
 *
 * With payload-only bounds that subtraction is out of bounds and EVERY free
 * faults, long before any bug is reached -- the port would measure its own
 * porting decision instead of PHP's defect. Covering the header keeps
 * zend_alloc.c's structure intact. The cost is that a header underflow is not
 * caught; that is not the defect under test.
 *
 * NUMBERS DO NOT MATCH THE CORPUS, AND CANNOT. zend_mem_header holds pNext and
 * pLast, which are 128-bit capabilities on this target, so sizeof(zend_mem_header)
 * is not the 24 bytes measured on x86-64 and the region is not 35 bytes. The
 * SHAPE reproduces exactly; the arithmetic does not. Do not quote ASan's figures
 * against this build.
 */
#ifndef ZEND_CAPSTONE_ALLOC_H
#define ZEND_CAPSTONE_ALLOC_H

typedef unsigned long size_t;

/* ---- zend_alloc.h:63-64, verbatim ---- */
#define MAX_CACHED_MEMORY   11
#define MAX_CACHED_ENTRIES  256

/* ---- zend_alloc.h, the !ZEND_DEBUG && !ZEND_MM arm of zend_mem_header ----
 * ZEND_MM is NOT defined in the corpus build (zend_mm.h:34 is commented out),
 * so pNext/pLast are present and the free list is live. */
typedef struct _zend_mem_header {
    struct _zend_mem_header *pNext;
    struct _zend_mem_header *pLast;
    unsigned int size:31;
    unsigned int cached:1;
} zend_mem_header;

/* PLATFORM_ALIGNMENT is 16 here, not 8: anything holding a capability must be
 * 16-aligned or the tag-preserving ldc/stc path does not apply. */
#define PLATFORM_ALIGNMENT  16
#define MEM_HEADER_PADDING \
    (((PLATFORM_ALIGNMENT - sizeof(zend_mem_header)) % PLATFORM_ALIGNMENT + PLATFORM_ALIGNMENT) % PLATFORM_ALIGNMENT)

/* ---- zend_alloc.c:132, UNPATCHED ---- */
#define REAL_SIZE(size) ((size + 7) & ~0x7)

/* THE EXPERIMENT, and the only difference between the two arms.
 *
 *   fault arm   (default)                 end = header + sizeof(hdr) + size
 *   control arm (-DZEND_CAP_BOUNDS_REAL_SIZE) end = ... + REAL_SIZE(size)
 *
 * The control reproduces stock PHP: the 7 bytes of slack are inside the
 * capability, the overflow lands in them, and the program completes -- which is
 * exactly the `asan-stock`/`asan-nocache` OK row the corpus measured. It is what
 * distinguishes "the bounds check fired" from "something else faulted". */
#ifdef ZEND_CAP_BOUNDS_REAL_SIZE
#  define ZEND_CAP_BOUND_BYTES(size) ((unsigned long)REAL_SIZE(size))
#else
#  define ZEND_CAP_BOUND_BYTES(size) ((unsigned long)(size))
#endif

#ifndef ZEND_ARENA_BYTES
#  define ZEND_ARENA_BYTES 16384
#endif

/* AG(...) globals, flattened: one non-threaded heap. Building non-ZTS is not a
 * simplification but a requirement -- __thread cannot be lowered on this target
 * (ISSUES.md C-47, "Cannot select: c128 = GlobalTLSAddress"). */
static zend_mem_header *AG_head;
/* Block capabilities, UNSHRUNK. PHP parks a zend_mem_header* here; that cannot
 * work when the block is bounded per-request, because the bound belongs to the
 * SIZE THE BLOCK WAS LAST HANDED OUT AT, not to the next request. The cache is a
 * size-CLASS cache: bucket i holds anything that rounded to i*8, so a block freed
 * as 9 bytes can be re-served for 11. Parking the unshrunk block and re-bounding
 * on handout keeps the size-class policy exactly as PHP has it while giving each
 * new lifetime its own correct bound. */
static void *AG_cache[MAX_CACHED_MEMORY][MAX_CACHED_ENTRIES] __attribute__((aligned(16)));
static unsigned int     AG_cache_count[MAX_CACHED_MEMORY];

/* zend_alloc.c:117-126 */
#define ADD_POINTER_TO_LIST(p)      \
    (p)->pNext = AG_head;           \
    if (AG_head) { AG_head->pLast = (p); } \
    AG_head = (p);                  \
    (p)->pLast = (zend_mem_header *) 0;

/* zend_alloc.c:103-111 */
#define REMOVE_POINTER_FROM_LIST(p) \
    if ((p) == AG_head) { AG_head = (p)->pNext; } \
    else { (p)->pLast->pNext = (p)->pNext; }      \
    if ((p)->pNext) { (p)->pNext->pLast = (p)->pLast; }

/* ---------------------------------------------------------------------------
 * ARENA
 *
 * Every allocation is carved with SPLIT, which is the ONLY derivation that
 * produces a FRESH revocation-tree node (revoke_on_free_alloc.h:31-36). A
 * cincoffset into a shared pool inherits the pool's node, so revoking one
 * allocation would sweep the entire heap -- that is precisely why memsys5 cannot
 * do revoke-on-free. SHRINK is not a substitute: it copies rev_node_id unchanged.
 * So the order is SPLIT (fresh node) -> MREV (handle, senior, while still LIN)
 * -> DELIN (the copyable alias the caller gets) -> SHRINK (the option-A bound).
 *
 * The arena is made LINEAR with csdebuggencap, a QEMU DEBUG OP. That is how the
 * in-tree revoke probes mint a linear capability without firmware
 * (capstone-qemu/tests/capstone-revoke-probes/README.md), and it means this port
 * runs under the emulator only, not on silicon.
 * --------------------------------------------------------------------------- */

static unsigned char zend_arena[ZEND_ARENA_BYTES] __attribute__((aligned(16)));
static void *zend_arena_lin;    /* LINEAR; never handed out */
static int   zend_arena_ready;

/* csdebuggencap rd, rs1, rs2 -> LINEAR capability over [rs1, rs2), tag = 1. */
static inline void *zend_gencap(unsigned long b, unsigned long e)
{
    void *c;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x40, %0, %1, %2" : "=r"(c) : "r"(b), "r"(e));
    return c;
}

/* cssplit rd, rs1, rs2: rs1 keeps [base,mid), rd gets [mid,end) with a fresh
 * node. No builtin exists. The emulator no-ops it unless rd != rs1, hence the
 * early-clobber. Copied from revoke_on_free_alloc.h:37-45. */
static inline void *zend_split(void **lo, unsigned long mid)
{
    void *hi; void *l = *lo;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x06, %0, %1, %2" : "=&r"(hi), "+r"(l) : "r"(mid));
    *lo = l;
    return hi;
}

#ifndef ZEND_MAX_SLOTS
#define ZEND_MAX_SLOTS 64u
#endif
/* `rev` is a capability, so this struct is 16-aligned with rev at offset 0. */
typedef struct {
    void          *rev;     /* revocation handle, senior to the alias */
    void          *blk;     /* the UNSHRUNK block capability; what the cache parks */
    unsigned long  base;    /* the SPLIT mid; 0 marks a free slot */
} zend_slot;
static zend_slot   zend_slots[ZEND_MAX_SLOTS] __attribute__((aligned(16)));
static unsigned    zend_nslots;

static void zend_arena_init(void)
{
    unsigned long b = __builtin_capstone_cap_get_cursor((void *)&zend_arena[0]);
    zend_arena_lin  = zend_gencap(b, b + (unsigned long)ZEND_ARENA_BYTES);
    zend_arena_ready = 1;
}

/* Carve `bytes`, return an alias bounded to [block, block + bound_bytes). */
static void *zend_arena_carve(unsigned long bytes, unsigned long bound_bytes)
{
    if (!zend_arena_ready) { zend_arena_init(); }

    bytes = (bytes + 15UL) & ~15UL;             /* SPLIT wants 16-byte grain */
    void *cur = zend_arena_lin;
    unsigned long base = __builtin_capstone_cap_get_base(cur);
    unsigned long end  = __builtin_capstone_cap_get_end(cur);
    /* cssplit asserts base < mid < end, so the arena must keep a non-empty head. */
    if (end <= base || end - base <= bytes) { return (void *)0; }

    void *hi   = zend_split(&cur, end - bytes); /* [end-bytes, end), fresh node, LIN */
    zend_arena_lin = cur;

    void *rev   = __builtin_capstone_cap_mrev(hi);    /* senior, while still LIN */
    void *alias = __builtin_capstone_cap_delin(hi);   /* NONLIN, what callers hold */

    unsigned i;
    for (i = 0; i < zend_nslots; ++i) { if (zend_slots[i].base == 0) { break; } }
    if (i == zend_nslots) {
        if (zend_nslots >= ZEND_MAX_SLOTS) { return (void *)0; }
        i = zend_nslots++;
    }
    zend_slots[i].rev  = rev;
    zend_slots[i].blk  = alias;          /* unshrunk: the cache parks THIS */
    zend_slots[i].base = end - bytes;

    unsigned long ab = __builtin_capstone_cap_get_cursor(alias);
    return __builtin_capstone_cap_shrink(alias, ab, ab + bound_bytes);
}

static zend_slot *zend_find(void *p)
{
    unsigned long b = __builtin_capstone_cap_get_base(p);
    for (unsigned i = 0; i < zend_nslots; ++i) {
        if (zend_slots[i].base == b) { return &zend_slots[i]; }
    }
    return (zend_slot *)0;
}

/* ---- zend_alloc.c:142-217, _emalloc, structure preserved ---- */
static void *_emalloc(size_t size)
{
    zend_mem_header *p;
    unsigned int real_size;
    unsigned int cache_index;

    real_size   = REAL_SIZE(size);      /* :135 */
    cache_index = real_size >> 3;       /* :136 */

    /* zend_alloc.c:151-168. The cache is ON, exactly as PHP ships it, in BOTH
     * arms. Recycling is a capability operation here, not a pointer assignment:
     * the parked block is re-bounded for the new request, and under
     * ZEND_TEMPORAL it is also given a FRESH revocation node, so the new
     * lifetime is revocable independently of the old one -- and the previous
     * tenant's alias, revoked at its own _efree, stays dead. Verified by
     * probes/revoke-reuse-safety.c. */
    if ((cache_index < MAX_CACHED_MEMORY) && (AG_cache_count[cache_index] > 0)) {
        void *blk = AG_cache[cache_index][--AG_cache_count[cache_index]];

        unsigned i;
        for (i = 0; i < zend_nslots; ++i) { if (zend_slots[i].base == 0) { break; } }
        if (i == zend_nslots) {
            if (zend_nslots >= ZEND_MAX_SLOTS) { return (void *)0; }
            i = zend_nslots++;
        }
#if defined(ZEND_TEMPORAL) && !defined(ZEND_NO_REVOKE)
        /* `blk` is the authority REVOKE handed back at _efree: tagged, LINEAR,
         * bounds intact over the block. MREV while it is still LIN, exactly as a
         * first allocation does, so the new lifetime gets its own node.
         *
         * THE GUARD IS NOT OPTIONAL. Without the revoke there is no reclaimed
         * LIN authority and `blk` is still the DELINEARIZED alias -- MREV on a
         * NONLIN capability trips `assert(rs1_v->val.cap.type == CAP_TYPE_LIN)`
         * in QEMU's helper_csmrev and ABORTS THE EMULATOR, which reads as an
         * infrastructure failure rather than a result. Not re-minting is a
         * direct consequence of removing the revoke, not a second difference:
         * the control then recycles the block by handing the same alias straight
         * back, which is exactly what stock PHP does. */
        zend_slots[i].rev = __builtin_capstone_cap_mrev(blk);
        blk = __builtin_capstone_cap_delin(blk);
#else
        zend_slots[i].rev = (void *)0;
#endif
        zend_slots[i].blk  = blk;
        zend_slots[i].base = __builtin_capstone_cap_get_base(blk);

        unsigned long bb = __builtin_capstone_cap_get_cursor(blk);
        p = (zend_mem_header *) __builtin_capstone_cap_shrink(blk, bb,
                bb + sizeof(zend_mem_header) + MEM_HEADER_PADDING + ZEND_CAP_BOUND_BYTES(size));
        p->cached = 0;
        p->size   = size;
        return (void *)((char *)p + sizeof(zend_mem_header) + MEM_HEADER_PADDING);
    }

    /* zend_alloc.c:182 -- one allocation of header + REAL_SIZE(size).
     * The full rounded amount is carved, as PHP does; only the BOUND differs. */
    p = (zend_mem_header *) zend_arena_carve(
            sizeof(zend_mem_header) + MEM_HEADER_PADDING + real_size,
            sizeof(zend_mem_header) + MEM_HEADER_PADDING + ZEND_CAP_BOUND_BYTES(size));
    if (!p) {
        return (void *)0;
    }

    p->cached = 0;
    ADD_POINTER_TO_LIST(p);             /* :200 */
    p->size = size;                     /* :201 -- the TRUE size */

    return (void *)((char *)p + sizeof(zend_mem_header) + MEM_HEADER_PADDING);  /* :216 */
}

/* ---- zend_alloc.c:248-289, _efree ----
 * The header recovery at :250 is kept verbatim. It only works because the
 * capability covers the header; see the bounds note at the top of this file. */
static void _efree(void *ptr)
{
    zend_mem_header *p = (zend_mem_header *) ((char *)ptr - sizeof(zend_mem_header) - MEM_HEADER_PADDING);
    unsigned int real_size   = REAL_SIZE(p->size);
    unsigned int cache_index = real_size >> 3;

    zend_slot *s = zend_find(ptr);
    void *blk = s ? s->blk : (void *)0;

#ifdef ZEND_TEMPORAL
    /* REVOKE-ON-FREE. Every alias derived from this allocation stops
     * dereferencing here -- including one the caller cached before the free.
     *
     * REVOKE RETURNS THE AUTHORITY BACK. That is the point that makes recycling
     * possible: the returned capability is tagged and its bounds still cover the
     * block (probes/revoke-reclaim.c reports tag + base + end intact). So the
     * block is parked in the cache as PHP does, and the next _emalloc re-mints
     * it with a fresh node.
     *
     * Reuse is SAFE, not merely possible: probes/revoke-reuse-safety.c writes
     * through the PREVIOUS tenant's alias after the range has been re-minted and
     * it still faults. Address reuse does not resurrect a stale pointer, which
     * is exactly the hazard it would be in a conventional allocator. */
    if (s) {
#ifndef ZEND_NO_REVOKE
        blk = __builtin_capstone_cap_revoke(s->rev);   /* the control removes ONLY this */
#endif
        s->base = 0;
    }
#else
    if (s) { s->base = 0; }
#endif

    /* zend_alloc.c:271-278, UNPATCHED in both arms: blocks <= 80 bytes never
     * reach a free path at all, they are parked here and handed straight back.
     * This is what ZEND_DISABLE_MEMORY_CACHE exists to defeat for ASan, and it
     * is left ON. */
    if (blk && (cache_index < MAX_CACHED_MEMORY) && (AG_cache_count[cache_index] < MAX_CACHED_ENTRIES)) {
        AG_cache[cache_index][AG_cache_count[cache_index]++] = blk;
        return;
    }

    REMOVE_POINTER_FROM_LIST(p);        /* :281 */
    /* ZEND_DO_FREE(p) -- the arena does not reclaim address space. */
}

#endif /* ZEND_CAPSTONE_ALLOC_H */
