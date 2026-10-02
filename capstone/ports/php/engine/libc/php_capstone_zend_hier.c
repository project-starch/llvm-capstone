/* HIERARCHICAL ALLOCATOR: PHP's zend_alloc on top, the capability arena underneath, and BOTH
 * levels bound correctly.
 *
 * This is the structure PHP actually has, and it is kept rather than collapsed:
 *
 *   Zend/zend_alloc.c  _emalloc(size)        upper level: 48-byte zend_mem_header, AG(cache),
 *                        |                   REAL_SIZE rounding -- PHP's own logic, unedited
 *                        v  ZEND_DO_MALLOC
 *   our malloc(48 + REAL_SIZE(size))         lower level: arena carve, cssplit, one capability
 *                        |                   over the RAW BLOCK -- correct for ITS level
 *                        v
 *   the arena (one LINEAR capability over a .bss array)
 *
 * THE RULE A HIERARCHY HAS TO OBEY: every level that hands out an object must re-bound the
 * capability it returns to that level's notion of the object. The lower level bounding its raw
 * block is correct for the lower level and WRONG for the caller, because the caller's object
 * starts 48 bytes in and ends REAL_SIZE(size) - size bytes early. Without a re-bound at the upper
 * level the capability covers PHP's header plus its rounding slack as reachable memory, and a
 * one-byte over-read of a 9-byte string lands inside it -- measured, and the reason both corpus
 * triggers first came back MISSED.
 *
 * So each ALLOCATING entry point of zend_alloc.c is renamed aside (-D on that TU only) and
 * wrapped here. The wrapper calls PHP's real logic, then shrinks the result. PHP's source is not
 * edited. The two entry points that TAKE a pointer -- _efree and _erealloc -- additionally widen it
 * back to the block before descending; see the discipline note above them.
 *
 * WHAT THIS MODE TRADES, stated because it is a real difference and not a free win:
 *   + PHP's allocator is intact and under test -- zend_alloc.c is compiled from the corpus tree;
 *   + bounds are per emalloc OBJECT, so the spatial axis is sound, cache hits included (the
 *     wrapper sits outside _emalloc, so an early return from the cache is re-bounded too);
 *   - the TEMPORAL axis is coarser: AG(cache) absorbs most frees, so an efree does not reach the
 *     arena and revoke-on-free cannot fire per object. For temporal work use PHP_CAP_ALLOC_SEAM,
 *     which replaces PHP's allocator and therefore sees every emalloc/efree pair.
 */
#include <zend.h>
#include "zend_alloc.h"

/* The arm, mirroring REAL_SIZE in zend_alloc.c. The CONTROL reproduces stock PHP by bounding to
 * the rounded size, so the over-read lands in the slack exactly as it does natively; the FAULT
 * arm bounds to the true request. Unlike the collapsed mode these genuinely differ, because the
 * rounding is applied to the CALLER's size here, not to PHP's already-padded block request. */
#ifdef ZEND_CAP_BOUNDS_REAL_SIZE
#  define ZC_BOUND(size) ((unsigned long)(((size) + 7UL) & ~7UL))
#else
#  define ZC_BOUND(size) ((unsigned long)(size))
#endif

/* zend_alloc.c, with its allocating entry points renamed by -D. */
extern void *php_raw_emalloc(size_t size);
extern void *php_raw_ecalloc(size_t nmemb, size_t size);
extern void *php_raw_erealloc(void *ptr, size_t size, int allow_failure);
extern void *php_raw_safe_emalloc(size_t nmemb, size_t size, size_t offset);
extern char *php_raw_estrdup(const char *s);
extern char *php_raw_estrndup(const char *s, unsigned int length);
extern void  php_raw_efree(void *ptr);

/* Defined beside the arena, which is the only place that still knows the block bounds. */
extern void *zend_cap_widen(void *p);

/* Re-bound: KEEP THE BASE, tighten only the END to cursor + bound(size).
 *
 * The base must not move, and getting that wrong is instructive. A hierarchy has TWO headers
 * below the object --
 *
 *     [ our 48-byte header ][ PHP's 48-byte zend_mem_header ][ payload ]
 *
 * -- because the lower level puts its own header under the block it hands PHP, and PHP puts its
 * header under the pointer it hands the caller. Shrinking the base to `cursor - sizeof(php hdr)`
 * therefore excludes OUR header, and the next ZEND_DO_FREE or ZEND_DO_REALLOC subtracts another 48
 * to reach it, lands below the base and faults. Measured: it made the CONTROL arm halt with cause
 * 5 on both triggers, which the harness correctly refused to score.
 *
 * Leaving the base alone keeps every header in the chain reachable however deep the hierarchy
 * goes, and the forward extent -- the only direction an over-read travels -- is still exact. The
 * cost is that an under-read below the object is not caught, which the ported allocator already
 * accepts for the same reason.
 *
 * The shrink is always legal: the incoming capability ends at or past sizeof(php hdr) +
 * REAL_SIZE(size) beyond the cursor, and bound(size) <= REAL_SIZE(size). */
static void *zc_rebound(void *p, unsigned long size)
{
    unsigned long b, c;
    if (!p) { return p; }
    b = __builtin_capstone_cap_get_base(p);
    c = __builtin_capstone_cap_get_cursor(p);
    return __builtin_capstone_cap_shrink(p, b, c + ZC_BOUND(size));
}

/* ATTRIBUTING AN OVER-READ THAT HAPPENS INSIDE A SHARED COPY PRIMITIVE.
 *
 * CRASH-073 is `ret->path = estrndup(s, (ue-s))` at url.c:292 with `ue - s` three bytes too long,
 * and ASAN reports it exactly where we see it -- frame #0 memcpy, frame #1 _estrndup
 * (zend_alloc.c:403), frame #2 php_url_parse -- because the read is performed BY the copy. A
 * capability fault inside memcpy is therefore the correct and expected shape of this catch, which
 * is awkward, because it is also the shape of a bound that the port itself got wrong (see
 * _erealloc above). The faulting pc cannot tell the two apart.
 *
 * So the attribution is made at the boundary instead: before descending, compare the requested
 * length against what the SOURCE capability actually authorises. A caller asking for more than the
 * source holds is the defect, whoever ends up performing the load. The overrun is reported through
 * the fault channel, which encodes the numbers in badaddr, so the amount can be checked against
 * ASAN's "READ of size 3".
 *
 * Default OFF: it halts on the first overrun, so it answers the attribution question and must not
 * be in a scoring image. */
#ifdef ZEND_CAP_ESTRNDUP_AUDIT
static void zc_audit_src(const char *s, unsigned long length, unsigned long who)
{
    extern void php_fault_report(unsigned long);
    unsigned long c, e, avail;
    if (!s) { return; }
    c = __builtin_capstone_cap_get_cursor((void *) s);
    e = __builtin_capstone_cap_get_end((void *) s);
    avail = (e > c) ? (e - c) : 0UL;
    if (length > avail) {
        /* 0xE3 | who | requested | available | overrun -- overrun last so it reads off badaddr. */
        php_fault_report((0xE3UL << 56) | ((who & 0xFFUL) << 48)
                       | ((length & 0xFFFFUL) << 24) | ((avail & 0xFFFFUL) << 8)
                       | ((length - avail) & 0xFFUL));
    }
}
#  define ZC_AUDIT(s, n, who) zc_audit_src((s), (unsigned long)(n), (who))
#else
#  define ZC_AUDIT(s, n, who) ((void)0)
#endif

/* NARROW ON THE WAY UP, WIDEN ON THE WAY DOWN. This is the whole discipline of the wrapper, and
 * both halves are load-bearing. A narrowed capability describes the CALLER's object; the lower
 * level's own bookkeeping is in terms of its BLOCK, and it is entitled to touch all of it. So any
 * pointer that re-enters the lower level must be widened back to the block first.
 *
 * Both entry points that take a pointer were found to need it, each by a fault:
 *
 *   _efree   -- PHP's AG(cache) PARKS the pointer it is handed. Narrowing is monotonic, so a
 *               narrow capability in a size-CLASS cache poisons every later tenant of that class:
 *               a block freed as a 6-byte string, re-served for a 9-byte one, over-reads at the
 *               new object's own NUL write. Surfaced as a false-positive CAUGHT at
 *               zend_alloc.c:404.
 *
 *   _erealloc -- the lower level COPIES the old contents, and its copy length comes from its own
 *               block record, not from the caller's last request. Handed a narrow capability it
 *               reads a whole block's worth through an object-sized bound and faults inside
 *               memcpy. Surfaced as a 16-byte capability-granule load one granule past the end
 *               (cause 5 in beebs_freestanding_string.c, the chunk-copy loop) -- again a fault in
 *               the right arm for entirely the wrong reason.
 *
 * The two look unrelated and are the same mistake: a bound that belongs to one level used at
 * another. zend_cap_widen recovers the block capability from the arena's slot table, which is keyed
 * by base -- and re-bounding deliberately preserves the base, which is what makes the lookup
 * possible at all. */
ZEND_API void _efree(void *ptr)
{
    php_raw_efree(zend_cap_widen(ptr));
}

ZEND_API void *_emalloc(size_t size)
{
    return zc_rebound(php_raw_emalloc(size), (unsigned long)size);
}

ZEND_API void *_ecalloc(size_t nmemb, size_t size)
{
    return zc_rebound(php_raw_ecalloc(nmemb, size), (unsigned long)nmemb * (unsigned long)size);
}

ZEND_API void *_erealloc(void *ptr, size_t size, int allow_failure)
{
    /* Widen going down (the lower level copies the whole block), narrow coming back up. */
    return zc_rebound(php_raw_erealloc(zend_cap_widen(ptr), size, allow_failure),
                      (unsigned long)size);
}

ZEND_API void *_safe_emalloc(size_t nmemb, size_t size, size_t offset)
{
    return zc_rebound(php_raw_safe_emalloc(nmemb, size, offset),
                      (unsigned long)nmemb * (unsigned long)size + (unsigned long)offset);
}

ZEND_API char *_estrndup(const char *s, unsigned int length)
{
    ZC_AUDIT(s, length, 1UL);
    return (char *) zc_rebound(php_raw_estrndup(s, length), (unsigned long)length + 1UL);
}

ZEND_API char *_estrdup(const char *s)
{
    unsigned long n = (unsigned long)strlen(s);
    return (char *) zc_rebound(php_raw_estrdup(s), n + 1UL);
}
