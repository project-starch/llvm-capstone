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

/* ZEND_CAP_TAG_GUARD: diagnostic only, OFF by default so the CRASH-008 and UAF suites are
 * byte-identical without it. When on, _emalloc checks that the pointer it is about to hand
 * out is actually TAGGED, and if not reports the REQUESTED SIZE through the fault channel
 * (php_fault_report mints a cap over [v,v+8) and stores at +64, so the monitor prints
 * badaddr = v+64) and returns NULL instead. An untagged pointer handed to a caller shows up
 * far away as a cause-24 inside memcpy with no indication of which allocation produced it. */
#if defined(ZEND_CAP_TAG_GUARD)
extern void php_fault_report(unsigned long);
#  define ZEND_CAP_CHECK_TAG(ptr, size)                                        \
    do {                                                                       \
        if (!__builtin_capstone_cap_get_tag((void *)(ptr))) {                   \
            php_fault_report((unsigned long)(size));                           \
            return (void *)0;                                                  \
        }                                                                      \
    } while (0)
#else
#  define ZEND_CAP_CHECK_TAG(ptr, size) do { } while (0)
#endif

/* ZEND_CAP_REV_AUDIT: diagnostic only, OFF by default. Finds which revocation invariant breaks,
 * and when, instead of inferring it from where the cause-24 lands.
 *
 * Three invariants, reported through the fault channel with distinct codes so the first one to fire
 * names the cause (php_fault_report halts, so only the first fires):
 *
 *   0xE5 | slot | nfree   _efree found a slot whose `rev` node is UNTAGGED. Either the node was
 *                         already consumed by an earlier revoke, or something cleared it.
 *   0xE6 | bucket| nalloc the cache handed back an UNTAGGED block. Parking stored a dead capability,
 *                         so the park happened after the authority was already gone.
 *   0xE7 | slot | nfree   revoke RETURNED an untagged authority, which is what the cache then parks.
 *                         This is the invariant the comment above _efree asserts from probes.
 */
#if defined(ZEND_CAP_REV_AUDIT)
extern void php_fault_report(unsigned long);
static unsigned long zend_nalloc_ctr, zend_nfree_ctr;
#  define ZEND_REV_NOTE(code, idx, ctr)                                        \
    php_fault_report(((unsigned long)(code) << 56)                             \
                   | (((unsigned long)(idx) & 0xFFFFUL) << 32)                 \
                   | ((unsigned long)(ctr) & 0xFFFFFFFFUL))
#  define ZEND_REV_COUNT_ALLOC() (++zend_nalloc_ctr)
#  define ZEND_REV_COUNT_FREE()  (++zend_nfree_ctr)
#else
#  define ZEND_REV_NOTE(code, idx, ctr) do { } while (0)
#  define ZEND_REV_COUNT_ALLOC() do { } while (0)
#  define ZEND_REV_COUNT_FREE()  do { } while (0)
#endif

/* ZEND_CAP_ARENA_TRACE: diagnostic only, OFF by default. Answers one question -- does the
 * arena actually RUN OUT during a run, or is a downstream fault something else? Both
 * exhaustion modes are reported separately, because they are different bugs:
 *
 *   0x5E000000 | KB-remaining   the BYTE arena is too small to carve the request
 *   0x51000000 | slots-in-use   the SLOT TABLE (ZEND_MAX_SLOTS) is full
 *
 * Reported through php_fault_report, which mints a capability over [v, v+8) and stores at
 * +64, so the monitor prints `badaddr = v + 64`. That channel survives a domain that then
 * wedges or faults, which an ordinary return value does not. */
#if defined(ZEND_CAP_ARENA_TRACE)
extern void php_fault_report(unsigned long);
#  define ZEND_ARENA_NOTE(v) php_fault_report((unsigned long)(v))
#else
#  define ZEND_ARENA_NOTE(v) do { } while (0)
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
    if (end <= base || end - base <= bytes) {
        ZEND_ARENA_NOTE(0x5E000000UL | (((end > base) ? (end - base) : 0UL) >> 10));
        return (void *)0;
    }

    void *hi   = zend_split(&cur, end - bytes); /* [end-bytes, end), fresh node, LIN */
    zend_arena_lin = cur;

    void *rev   = __builtin_capstone_cap_mrev(hi);    /* senior, while still LIN */
    void *alias = __builtin_capstone_cap_delin(hi);   /* NONLIN, what callers hold */

    unsigned i;
    for (i = 0; i < zend_nslots; ++i) { if (zend_slots[i].base == 0) { break; } }
    if (i == zend_nslots) {
        if (zend_nslots >= ZEND_MAX_SLOTS) {
            ZEND_ARENA_NOTE(0x51000000UL | (unsigned long)zend_nslots);
            return (void *)0;
        }
        i = zend_nslots++;
    }
    zend_slots[i].rev  = rev;
    zend_slots[i].blk  = alias;          /* unshrunk: the cache parks THIS */
    zend_slots[i].base = end - bytes;

    unsigned long ab = __builtin_capstone_cap_get_cursor(alias);
    return __builtin_capstone_cap_shrink(alias, ab, ab + bound_bytes);
}

/* base == 0 is the FREE-SLOT marker, so it must never be a lookup key.
 *
 * `__builtin_capstone_cap_get_base` of an UNTAGGED value is 0, so without the `b &&` guard an
 * _efree handed an untagged pointer matched the first free slot and revoked a node that the
 * previous tenant's revoke had already consumed. Revoking an untagged node faults with cause 24,
 * inside the allocator, on a workload containing no lifetime error at all -- which is how the
 * temporal arm came to report CAUGHT for four use-after-free cases that it had not caught. A
 * benign script with revocation on faulted identically.
 *
 * With the guard, an _efree of an untagged pointer finds no slot, so it neither revokes nor parks:
 * the block leaks instead of corrupting the slot table, which is the right failure for a diagnostic
 * allocator. It does not weaken detection -- a use-after-free faults at the USE, through the revoked
 * alias, not here. */
static zend_slot *zend_find(void *p)
{
    unsigned long b = __builtin_capstone_cap_get_base(p);
    if (!b) { return (zend_slot *)0; }
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
#if defined(ZEND_CAP_REV_AUDIT)
        ZEND_REV_COUNT_ALLOC();
        if (!__builtin_capstone_cap_get_tag(blk)) {
            ZEND_REV_NOTE(0xE6, cache_index, zend_nalloc_ctr);
        }
#endif

        unsigned i;
        for (i = 0; i < zend_nslots; ++i) { if (zend_slots[i].base == 0) { break; } }
        if (i == zend_nslots) {
            if (zend_nslots >= ZEND_MAX_SLOTS) {
            ZEND_ARENA_NOTE(0x51000000UL | (unsigned long)zend_nslots);
            return (void *)0;
        }
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
        ZEND_CAP_CHECK_TAG(p, size);
        p->cached = 0;
#ifdef ZEND_TEMPORAL
        /* The other half of the unlink-before-revoke fix, and it is not optional.
         *
         * Stock PHP never removes a cached block from its allocation list, so a cache hit has
         * nothing to re-link. Under revocation every block MUST leave the list before it dies, so
         * every block must also RE-ENTER the list when it is handed out again -- otherwise its
         * header still holds the link fields from its previous lifetime, pointing at blocks that
         * have since been freed and revoked, and the next unlink faults on them.
         *
         * Unlink-on-free and link-on-allocate are one change; adding only the first one moves the
         * fault from the cache-park site to the unlink site without fixing anything. */
        ADD_POINTER_TO_LIST(p);
#endif
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

    ZEND_CAP_CHECK_TAG(p, size);
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

    /* WILL PHP PARK THIS BLOCK? Decided BEFORE the revoke, because the answer decides whether the
     * in-block list links have to be read, and they are only readable while the block is alive.
     * The condition is the same one the park below tests; `blk != 0` is equivalent to `s != 0` at
     * this point, in both arms. */
    int will_cache = (blk != (void *)0) && (cache_index < MAX_CACHED_MEMORY)
                   && (AG_cache_count[cache_index] < MAX_CACHED_ENTRIES);

    /* THE UNLINK MUST PRECEDE THE REVOKE, AND IT MUST BE UNCONDITIONAL UNDER REVOCATION.
     *
     * PHP's allocation list is INTRUSIVE: `REMOVE_POINTER_FROM_LIST(p)` reads `p->pLast` and
     * `p->pNext` out of the block headers, and writes through them into the NEIGHBOURING blocks'
     * headers. Revocation invalidates every capability derived from a block, which includes the link
     * fields other blocks hold pointing AT it. So a revoked block left in the list poisons its
     * neighbours: the next unlink that walks through it loads an untagged capability and faults with
     * cause 24, inside the allocator, on a program with no lifetime error in it.
     *
     * Stock PHP leaves cached blocks in the list -- `_efree` returns early for them (zend_alloc.c:
     * 271-278) -- which is harmless when nothing invalidates memory and fatal once something does.
     * Two orderings therefore have to be fixed at once: unlink BEFORE revoking, and unlink EVERY
     * block rather than only the ones the cache declines. Together they guarantee the list never
     * contains a dead block, so every link the unlink walks is still live.
     *
     * This is a deviation from PHP's `_efree`, and it is confined to `ZEND_TEMPORAL` -- including the
     * `-DZEND_NO_REVOKE` control arm, so the matched pair still differs in revocation ALONE. The
     * spatial arm and the CRASH-008 suite keep PHP's structure byte for byte. What is given up is
     * PHP's leak report at shutdown seeing cached blocks, which this port does not use.
     *
     * The general rule: revocation is a point of no return for everything INSIDE the object,
     * metadata included. An allocator with out-of-band metadata has no such ordering constraint;
     * PHP's is in-band, so the port has to respect it. It took four false CAUGHT verdicts to find,
     * because only the blocks the cache declined reached the unlink, so it needed a long run and
     * looked trigger-specific.
     *
     * The symptom to recognise next time: a cause-24 whose faulting instruction is a capability load
     * through a pointer that was read out of freed memory. */
#ifdef ZEND_TEMPORAL
    REMOVE_POINTER_FROM_LIST(p);            /* :281, hoisted and unconditional */
#else
    if (!will_cache) {
        REMOVE_POINTER_FROM_LIST(p);        /* :281, exactly where PHP has it */
    }
#endif

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
#if defined(ZEND_CAP_REV_AUDIT)
        ZEND_REV_COUNT_FREE();
        if (!__builtin_capstone_cap_get_tag(s->rev)) {
            ZEND_REV_NOTE(0xE5, (unsigned long)(s - zend_slots), zend_nfree_ctr);
        }
#endif
#ifndef ZEND_NO_REVOKE
        blk = __builtin_capstone_cap_revoke(s->rev);   /* the control removes ONLY this */
#endif
#if defined(ZEND_CAP_REV_AUDIT)
        if (!__builtin_capstone_cap_get_tag(blk)) {
            ZEND_REV_NOTE(0xE7, (unsigned long)(s - zend_slots), zend_nfree_ctr);
        }
#endif
        s->base = 0;
    }
#else
    if (s) { s->base = 0; }
#endif

    /* zend_alloc.c:271-278, UNPATCHED in both arms: blocks <= 80 bytes never
     * reach a free path at all, they are parked here and handed straight back.
     * This is what ZEND_DISABLE_MEMORY_CACHE exists to defeat for ASan, and it
     * is left ON. `blk` is re-read here because the revoke above replaced it with the authority it
     * handed back, which is what must be parked -- see the note on recycling. */
    if (will_cache) {
        AG_cache[cache_index][AG_cache_count[cache_index]++] = blk;
        return;
    }
    /* The unlink already happened, above the revoke. */
    /* ZEND_DO_FREE(p) -- the arena does not reclaim address space. */
}

#endif /* ZEND_CAPSTONE_ALLOC_H */
