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
#include "../../zend-alloc/zend_capstone_alloc.h"

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
    return _emalloc(n ? n : 1);
}
void  free(void *p)               { if (p) _efree(p); }

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
            unsigned char *q = (unsigned char *) _emalloc(total ? total : 1);
            unsigned long tag = q ? __builtin_capstone_cap_get_tag(q) : 0UL;
            unsigned long len = q ? (__builtin_capstone_cap_get_end(q)
                                   - __builtin_capstone_cap_get_base(q)) : 0UL;
            php_fault_report((0xD2UL << 56) | ((total & 0xFFFFUL) << 32)
                           | ((len & 0xFFFFUL) << 8) | (tag & 1UL));
        }
    }
#endif
    unsigned char *p = (unsigned char *) _emalloc(total ? total : 1);
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
    unsigned char *q = (unsigned char *) _emalloc(n);
    if (!q) { return (void *)0; }
    const unsigned char *s = (const unsigned char *)p;
    for (unsigned long i = 0; i < cp; i++) { q[i] = s[i]; }
    _efree(p);
    return q;
}
