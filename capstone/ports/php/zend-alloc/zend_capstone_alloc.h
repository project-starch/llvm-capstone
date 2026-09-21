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

static unsigned char zend_arena[ZEND_ARENA_BYTES] __attribute__((aligned(16)));
static unsigned long zend_arena_off;

/* AG(...) globals, flattened: one non-threaded heap. Building non-ZTS is not a
 * simplification but a requirement -- __thread cannot be lowered on this target
 * (ISSUES.md C-47, "Cannot select: c128 = GlobalTLSAddress"). */
static zend_mem_header *AG_head;
static zend_mem_header *AG_cache[MAX_CACHED_MEMORY][MAX_CACHED_ENTRIES];
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

/* Carve `bytes` from the arena and return a capability whose bounds are
 * [block, block + bound_bytes). The arena capability itself stays wide and is
 * never handed out; every block is a narrowed derivation of it. */
static void *zend_arena_carve(unsigned long bytes, unsigned long bound_bytes)
{
    unsigned long off = (zend_arena_off + 15UL) & ~15UL;
    if (off + bytes > (unsigned long)ZEND_ARENA_BYTES) {
        return (void *)0;
    }
    zend_arena_off = off + bytes;

    unsigned char *block = &zend_arena[off];
    /* The CURSOR is the block address. Taking it directly rather than computing
     * arena_base + off matters: -capstone-shrink-globals is on by default, so
     * `zend_arena`'s own capability may already be narrowed and its base is not
     * a fixed reference point. */
    unsigned long base = __builtin_capstone_cap_get_cursor((void *)block);
    return __builtin_capstone_cap_shrink((void *)block, base, base + bound_bytes);
}

/* ---- zend_alloc.c:142-217, _emalloc, structure preserved ---- */
static void *_emalloc(size_t size)
{
    zend_mem_header *p;
    unsigned int real_size;
    unsigned int cache_index;

    real_size   = REAL_SIZE(size);      /* :135 */
    cache_index = real_size >> 3;       /* :136 */

    /* zend_alloc.c:151-168. The cache is ON, exactly as PHP ships it. A cached
     * block is handed back with its ORIGINAL capability bounds, which were set
     * from the size it was first allocated with -- see the note in _efree. */
    if ((cache_index < MAX_CACHED_MEMORY) && (AG_cache_count[cache_index] > 0)) {
        p = AG_cache[cache_index][--AG_cache_count[cache_index]];
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

    /* zend_alloc.c:271-278. Blocks <= 80 bytes never reach the "free" path at
     * all; they are parked here and handed straight back by the next _emalloc.
     * This is what ZEND_DISABLE_MEMORY_CACHE exists to defeat for ASan. It is
     * left ON. A revoke-on-free arm would go here, and is deliberately NOT in
     * this build: this port measures the SPATIAL axis only, and adding a revoke
     * would make a temporal fault indistinguishable from a bounds fault. */
    if ((cache_index < MAX_CACHED_MEMORY) && (AG_cache_count[cache_index] < MAX_CACHED_ENTRIES)) {
        AG_cache[cache_index][AG_cache_count[cache_index]++] = p;
        p->cached = 1;
        return;
    }

    REMOVE_POINTER_FROM_LIST(p);        /* :281 */
    /* ZEND_DO_FREE(p) -- the bump arena does not reclaim. */
}

#endif /* ZEND_CAPSTONE_ALLOC_H */
