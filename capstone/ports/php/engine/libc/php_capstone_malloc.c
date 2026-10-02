/* malloc/free/realloc/calloc for the PHP engine domain — THE SEAM OF THE WHOLE EXERCISE.
 *
 * ZEND_MM is left UNDEFINED, exactly as the corpus builds it (Zend/zend_mm.h:34 is
 * commented out). That makes Zend/zend_alloc.c:55-57 define
 *
 *     ZEND_DO_MALLOC(size)       malloc(size)
 *     ZEND_DO_FREE(ptr)          free(ptr)
 *     ZEND_DO_REALLOC(ptr, size) realloc(ptr, size)
 *
 * so every single emalloc block in the engine bottoms out HERE, and each one gets its own
 * precisely-bounded capability from the ported allocator.
 *
 * The alternative -- letting PHP's real zend_mm arena serve emalloc -- would put every
 * allocation inside ONE capability and catch NOTHING. The point of the port would be lost.
 * Do not "fix" this by enabling ZEND_MM.
 */
#include <string.h>   /* memcpy -- realloc needs the TAG-PRESERVING copy, see below */

/* THE SEAM MOVES UP ONE LAYER (PHP_CAP_ALLOC_SEAM).
 *
 * Leaving ZEND_MM undefined makes ZEND_DO_MALLOC plain malloc, and the plan concluded that this
 * alone gives "every emalloc block its own precisely-bounded capability". IT DOES NOT. PHP's own
 * _emalloc still sits on top: it asks malloc for sizeof(zend_mem_header) + MEM_HEADER_PADDING +
 * REAL_SIZE(size) -- measured 48 + 0 + REAL_SIZE(size) here -- and returns p + 48. So the
 * capability covers PHP's whole block and the engine's object lives 48 bytes inside it. A 9-byte
 * string got 16 reachable bytes, and both corpus triggers MISSED in that slack.
 *
 * Worse, REAL_SIZE is the IDENTITY on PHP's request (48 + REAL_SIZE(n) is always a multiple of
 * 8), so the fault and control arms computed the SAME bound: rung E was not a matched pair at all.
 *
 * So the ported allocator becomes PHP's allocator rather than sitting under it. The entry points
 * below replace Zend/zend_alloc.c's (that TU is dropped from the build), and each bound is
 * computed from the CALLER's own size. They live in this TU because the arena is a static in
 * zend_capstone_alloc.h -- a second includer would get a second, disjoint arena.
 *
 * The header's own `_emalloc`/`_efree` are renamed out of the way so the PHP-facing names are
 * free; the header itself is not edited, so the CRASH-008 and UAF suites are unaffected. */
#if defined(PHP_CAP_ALLOC_SEAM)
#  define _emalloc zend_cap_emalloc
#  define _efree   zend_cap_efree
#endif
#include "../../zend-alloc/zend_capstone_alloc.h"
#if defined(PHP_CAP_ALLOC_SEAM)
#  undef _emalloc
#  undef _efree
#endif

/* STACK-DEPTH WATCHDOG, sited in malloc.
 *
 * -finstrument-functions is not usable on this target: it crashes clang with
 * "Calling a function with a bad signature!" (llvm/lib/IR/Instructions.cpp:759), because
 * the __cyg_profile hooks' void* parameters do not match the capability pointer type.
 * malloc is the next best sampling point -- the engine calls it throughout startup, and it
 * is in the UNinstrumented support layer so there is no recursion hazard.
 *
 * On crossing the floor it longjmps out to domain_main, turning an unrecoverable stack
 * overflow into a reportable measurement. setjmp/longjmp is verified for this ABI by
 * probes/setjmp-probe.c. */
/* Declared in php_capstone_depth.c. php_depth_trip_fn is a POINTER, not a direct extern:
 * only the rung that arms the watchdog defines a handler, and a direct reference would
 * make every other rung fail to link. */
/* One spelling for the ported allocator whichever way it was compiled. */
#if defined(PHP_CAP_ALLOC_SEAM)
#  define ZCAP_EMALLOC(n) zend_cap_emalloc(n)
#  define ZCAP_EFREE(p)   zend_cap_efree(p)
#else
#  define ZCAP_EMALLOC(n) _emalloc(n)
#  define ZCAP_EFREE(p)   _efree(p)
#endif

extern unsigned long php_depth_floor, php_depth_low, php_depth_mallocs;
extern int           php_depth_armed;
extern void        (*php_depth_trip_fn)(void);
void php_depth_walk(void);

void *malloc(size_t n)
{
    unsigned long c = __builtin_capstone_cap_get_cursor(__builtin_frame_address(0));
    if (c < php_depth_low) { php_depth_low = c; }
    ++php_depth_mallocs;
    if (php_depth_armed && c < php_depth_floor && php_depth_trip_fn) {
        php_depth_armed = 0;
        php_depth_walk();            /* record the chain BEFORE unwinding it */
        php_depth_trip_fn();
    }
    return ZCAP_EMALLOC(n ? n : 1);
}
void  free(void *p)               { if (p) ZCAP_EFREE(p); }

/* The watchdog is sited here too: the stage bisect showed the 2.6 MB excursion begins at
 * the first zend_hash_init_ex, whose only call is calloc(nTableSize, sizeof(Bucket*)). */
void *calloc(size_t n, size_t sz)
{
    {
        unsigned long c = __builtin_capstone_cap_get_cursor(__builtin_frame_address(0));
        if (c < php_depth_low) { php_depth_low = c; }
        if (php_depth_armed && c < php_depth_floor && php_depth_trip_fn) {
            php_depth_armed = 0;
            php_depth_walk();
            php_depth_trip_fn();
        }
    }
    unsigned long total = (unsigned long)n * (unsigned long)sz;
#ifdef PHP_CAPSTONE_PROBE_CALLOC
    /* Unconditional probe on the FIRST calloc: report through the fault channel whether
     * _emalloc even returns, and with what. Every watchdog has failed to fire because they
     * are depth-triggered and calloc is only ~4 frames deep. This asks the direct question. */
    {
        extern void php_fault_report(unsigned long);
        static int once = 0;
        if (!once) {
            once = 1;
            unsigned char *q = (unsigned char *) ZCAP_EMALLOC(total ? total : 1);
            unsigned long tag = q ? __builtin_capstone_cap_get_tag(q) : 0UL;
            unsigned long len = q ? (__builtin_capstone_cap_get_end(q)
                                   - __builtin_capstone_cap_get_base(q)) : 0UL;
            php_fault_report((0xD2UL << 56) | ((total & 0xFFFFUL) << 32)
                           | ((len & 0xFFFFUL) << 8) | (tag & 1UL));
        }
    }
#endif
    unsigned char *p = (unsigned char *) ZCAP_EMALLOC(total ? total : 1);
    if (p) { for (unsigned long i = 0; i < total; i++) { p[i] = 0; } }
    return p;
}

/* _erealloc is not ported yet; realloc is expressed in terms of the two primitives that
 * are. The copy length is the OLD block's size, recovered from the header exactly as
 * zend_alloc.c:320 does. Bounded by the new size so a shrink cannot over-read. */
void *realloc(void *p, size_t n)
{
    if (!p)  { return malloc(n); }
    if (!n)  { free(p); return (void *)0; }
    zend_mem_header *h = (zend_mem_header *)
        ((char *)p - sizeof(zend_mem_header) - MEM_HEADER_PADDING);
    unsigned long old = h->size;
    unsigned long cp  = old < (unsigned long)n ? old : (unsigned long)n;
    unsigned char *q = (unsigned char *) ZCAP_EMALLOC(n);
    if (!q) { return (void *)0; }
    /* MUST BE A TAG-PRESERVING COPY, NOT A BYTE LOOP.
     *
     * This was `for (i = 0; i < cp; i++) q[i] = s[i];`, and that single byte loop was the root
     * cause of the rung E cause-24 fault. Byte stores register no capability tag, so every
     * capability inside a reallocated block came out with its address intact and its tag gone.
     *
     * It is silent by construction, which is why fourteen other candidates were eliminated
     * before it was found: the destination is a FRESH allocation, so there is no live tag to
     * destroy and a tag watch reports nothing; and an stc of the resulting untagged value
     * registers nothing either, so it then propagates through arbitrarily many correct copies
     * with no signal at all.
     *
     * PHP reallocs capability-bearing memory constantly -- get_next_op grows
     * op_array->opcodes, whose znodes hold zvals holding string pointers; zend_hash grows its
     * bucket array; zend_prepare_string_for_scanning does STR_REALLOC on the script text.
     *
     * memcpy is correct here: both blocks come from this allocator at base + header + padding,
     * which is 16-aligned, so beebs' memcpy sees da == sa == 0 and takes its ldc/stc chunk
     * path. Any tail shorter than a granule goes byte-wise, which is safe because a capability
     * is never stored straddling a granule boundary. */
    memcpy(q, p, cp);
    ZCAP_EFREE(p);
    return q;
}

/* WIDEN A RE-BOUNDED POINTER BACK TO ITS BLOCK, for the hierarchical mode.
 *
 * Capability narrowing is MONOTONIC: shrink cannot widen. So if an upper-level allocator parks the
 * pointer it handed the caller into its own size-class cache, that cache now holds a bound that
 * belongs to ONE PAST TENANT. PHP's cache is a size-CLASS cache -- bucket i takes anything that
 * rounded to i*8 -- so a block freed as a 6-byte string gets re-served for a 9-byte one, and the
 * new tenant inherits the old, tighter bound. Measured: _estrndup's `p[length] = 0` at
 * zend_alloc.c:404 stored one past the end, which looked exactly like a caught corpus bug and was
 * not one.
 *
 * The lower level still holds the UNSHRUNK block alias in zend_slots[].blk, keyed by base -- and
 * re-bounding keeps the base, so the key still matches. This returns that wide capability with the
 * caller's cursor, so the upper level can cache something whose bound belongs to the BLOCK rather
 * than to a past request. Re-bounding then happens on each handout, which is exactly the policy
 * the ported allocator's own cache already uses.
 *
 * Unknown pointers pass through unchanged: a block the arena did not hand out has no slot. */
void *zend_cap_widen(void *p)
{
    zend_slot *s;
    unsigned long c;
    if (!p) { return p; }
    s = zend_find(p);
    if (!s || !s->blk) { return p; }
    c = __builtin_capstone_cap_get_cursor(p);
    return (void *)((char *)s->blk
                    + (c - __builtin_capstone_cap_get_base(s->blk)));
}

#if defined(PHP_CAP_ALLOC_SEAM)
/* ============================================================================
 * Zend/zend_alloc.c's PUBLIC API, served directly by the capability-bounding
 * allocator so every bound is computed from the CALLER's size.
 *
 * With ZEND_DEBUG off, ZEND_FILE_LINE_DC and ZEND_FILE_LINE_ORIG_DC expand to nothing, so these
 * signatures match zend_alloc.h:78-86 exactly.
 *
 * WHAT THIS GAINS over wrapping PHP's allocator, which is the whole point:
 *   - bounds are PER EMALLOC OBJECT, not per underlying malloc block, so a one-byte over-read of
 *     a 9-byte string is now outside the capability instead of inside PHP's 48-byte header slack;
 *   - PHP's own AG(cache) is gone with the TU, so an emalloc/efree pair is no longer invisible to
 *     us -- which is what the temporal axis needs, since PHP's cache used to recycle one malloc
 *     block across several object lifetimes without the allocator seeing it.
 * ==========================================================================*/

void *_emalloc(size_t size)              { return ZCAP_EMALLOC(size ? size : 1); }
void  _efree(void *ptr)                  { if (ptr) ZCAP_EFREE(ptr); }

void *_ecalloc(size_t nmemb, size_t size)
{
    size_t n = nmemb * size;
    unsigned char *p;
    if (size && nmemb != n / size) { return (void *)0; }   /* overflow */
    p = (unsigned char *) ZCAP_EMALLOC(n ? n : 1);
    if (p) { size_t i; for (i = 0; i < n; i++) { p[i] = 0; } }
    return (void *)p;
}

void *_safe_emalloc(size_t nmemb, size_t size, size_t offset)
{
    size_t n = nmemb * size;
    if (size && nmemb != n / size) { return (void *)0; }
    if (n + offset < n)            { return (void *)0; }
    return ZCAP_EMALLOC((n + offset) ? (n + offset) : 1);
}

/* The copy MUST be tag-preserving; a byte loop here was the rung E root cause. memcpy takes its
 * ldc/stc chunk path because both blocks start 16-aligned at base + header + padding. */
void *_erealloc(void *ptr, size_t size, int allow_failure)
{
    zend_mem_header *h;
    unsigned long old, cp;
    void *q;
    (void)allow_failure;
    if (!ptr)  { return ZCAP_EMALLOC(size ? size : 1); }
    if (!size) { ZCAP_EFREE(ptr); return (void *)0; }
    h   = (zend_mem_header *)((char *)ptr - sizeof(zend_mem_header) - MEM_HEADER_PADDING);
    old = h->size;
    cp  = old < (unsigned long)size ? old : (unsigned long)size;
    q   = ZCAP_EMALLOC(size);
    if (!q) { return (void *)0; }
    memcpy(q, ptr, cp);
    ZCAP_EFREE(ptr);
    return q;
}

char *_estrndup(const char *s, unsigned int length)
{
    char *p = (char *) ZCAP_EMALLOC((size_t)length + 1);
    if (!p) { return (char *)0; }
    memcpy(p, s, length);
    p[length] = 0;
    return p;
}

char *_estrdup(const char *s)
{
    return _estrndup(s, (unsigned int)strlen(s));
}

/* PERSISTENT allocations in PHP (plain malloc, never efree'd). There is no separate persistent
 * heap in a domain, so they come from the same arena; nothing frees them, which matches how PHP
 * uses them. */
char *zend_strndup(const char *s, unsigned int length)
{
    char *p = (char *) ZCAP_EMALLOC((size_t)length + 1);
    if (!p) { return (char *)0; }
    memcpy(p, s, length);
    p[length] = 0;
    return p;
}

char *zend_strdup(const char *s)
{
    if (!s) { return (char *)0; }
    return zend_strndup(s, (unsigned int)strlen(s));
}
#endif /* PHP_CAP_ALLOC_SEAM */
