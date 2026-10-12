/* The lower allocator level on the virtual profile: what the HIERARCHICAL wrapper needs from it.
 *
 * libc/php_capstone_zend_hier.c is reused UNCHANGED in Phase A -- it is the experiment, so it
 * is the one file from the physical port's support layer that carries over. It narrows every
 * pointer it hands up to the caller's object, and therefore has to widen any pointer that
 * re-enters the level below ("narrow on the way up, widen on the way down"). Physically the
 * widen recovers the block capability from the arena's slot table in
 * libc/php_capstone_malloc.c, and that file records it being found NECESSARY TWICE, each time
 * by a fault: PHP's AG(cache) parks the narrow pointer it is handed and poisons the next
 * tenant of that size class, and the lower level's realloc copies a whole block's worth
 * through whatever bound it is given.
 *
 * On this profile the lower level is musl's mallocng, via runtime/virtual/heap.c, which keeps
 * its own authority over every block -- so the widen is a LOOKUP here rather than a table of
 * our own. Which of the physical port's two reasons still bite, and which does not, is
 * recorded on zend_cap_widen below, along with the measurement that settled it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <malloc_capstone.h>

/* ZEND_CAP_HEADER_BYTES, asserted against PHP's own sizeof in php_virtual_main.c. */
#include "php_virtual_port.h"

/* THE WIDEN IS REAL WORK HERE TOO, and the identity was WRONG. Recorded because the mistake is
 * instructive and the probe that blessed it was blind to the case that matters.
 *
 * The first version of this function returned p unchanged, on the strength of reading
 * runtime/virtual/heap.c: free() and realloc() take the caller's pointer only as an address
 * plus a one-byte probe at the cursor, then work through r->raw -- the capability the heap
 * retained over the block. That reading is correct, and app/probe-hier-widen.c confirmed it on
 * the machine: free and realloc both succeed through a narrowed pointer, with a negative
 * control proving the narrowing was enforced.
 *
 * It is also not what the widen is for. PHP's _efree does not merely hand the pointer DOWN --
 * it PARKS it: `AG(cache)[CACHE_INDEX][count++] = p` (zend_alloc.c:272), and a later _emalloc
 * of the same size class returns that same p + 48 to a new tenant. Capability narrowing is
 * monotonic, so a narrow pointer in a size-CLASS cache poisons every later tenant of that
 * class: CACHE_INDEX is REAL_SIZE(size)>>3, so a 54-byte object and a 55-byte one share a
 * bucket, and the 55-byte tenant inherits a capability that ends after 54. Its own
 * `p[length] = 0` then faults -- zend_alloc.c:404, inside _estrndup, with no defect in sight.
 *
 * That is exactly what the exact-bounds arm did on its first run: all six cases AND the arm's
 * own control faulted at zend_alloc.c:404 during zend_startup, before any trigger ran. The
 * judge refused all six as NO-READING/setup-fault rather than reporting catches, which is the
 * gate working. The hier wrapper's own comment block names this line and this mechanism, as
 * the reason the physical port needed the widen -- the probe tested the half of that comment
 * about descending into the lower level and not the half about the cache.
 *
 * So: recover the block capability, the same way the physical port does
 * (php_capstone_malloc.c:182-192), with the heap's own index in place of the arena's slot
 * table. __capstone_malloc_find is part of the musl-capstone port's declared interface
 * (arch-capstone64/malloc_capstone.h), and the slot's `raw` IS the heap's capability over the
 * block -- no second source of truth and no table of our own to exhaust.
 *
 * THE LOOKUP IS KEYED ON THE BLOCK, NOT THE OBJECT. heap.c indexes on the exact cursor of
 * what malloc returned (heap.c:34-39), and PHP's object pointer sits sizeof(zend_mem_header)
 * bytes into that block, so the object's own cursor misses. Both keys are tried rather than
 * assuming which one the caller holds: _efree and _erealloc pass the OBJECT pointer, and a
 * pointer that is already the block must still widen correctly. Reading the cursor of
 * `p - HEADER` is safe even though it lies below that capability's base -- it is metadata, not
 * a dereference, which app/probe-hier-widen.c stage 4 of the seam probe also measured.
 */
void *zend_cap_widen(void *p)
{
    struct capstone_malloc_slot *r;
    unsigned long c;
    if (!p) { return p; }
    r = __capstone_malloc_find(p);
    if (!r) { r = __capstone_malloc_find((char *)p - ZEND_CAP_HEADER_BYTES); }
    /* Not a block this heap handed out (a static buffer, or a pointer PHP synthesised): hand it
     * back unchanged rather than inventing authority for it. */
    if (!r || !r->raw) { return p; }
    /* Re-base onto r->raw's CURSOR, not its base. The physical widen uses the base
     * (php_capstone_malloc.c:190) because its arena slots hold a capability whose cursor IS
     * its base; heap.c's r->raw is the pointer mallocng returned, so its cursor sits at the
     * block key while its base is lower -- mallocng's slot prefix and size-class slack are
     * below and above the object. Using the base therefore shifted the cursor by exactly that
     * prefix, and _efree's `ptr - 48` would have landed 16 bytes off the header. Caught by
     * probe-hier-widen.c stage 3, which checks the cursor survives the widen rather than only
     * that the extent grew.
     *
     * NOTE WHAT THIS GRANTS. The recovered capability spans the heap's whole block, which
     * includes mallocng's slot prefix below the object. PHP only ever holds it as the parked
     * cache entry and for its own header arithmetic -- the wrapper re-narrows everything it
     * hands up -- but it does mean an under-read below a cached object is not caught. The
     * physical port makes the same trade for the same reason and says so
     * (php_capstone_zend_hier.c, on zc_rebound keeping the base). */
    c = __builtin_capstone_cap_get_cursor(p);
    return (void *)((char *)r->raw + (c - __builtin_capstone_cap_get_cursor(r->raw)));
}

/* For the probe: the block extent the widen recovers, so a test can check the widen did
 * something rather than merely that the program survived. Returns 0 when p is not a known
 * block. */
unsigned long zend_cap_block_end(void *p)
{
    struct capstone_malloc_slot *r = p ? __capstone_malloc_find(p) : 0;
    if (!r) { r = p ? __capstone_malloc_find((char *)p - ZEND_CAP_HEADER_BYTES) : 0; }
    return (r && r->raw) ? __builtin_capstone_cap_get_end(r->raw) : 0UL;
}

/* The diagnostic channel the physical port encodes in a deliberate fault.
 *
 * php_capstone_depth.c:98 mints a capability over [v, v+8) and stores at offset 64, so the
 * word lands in badaddr and the domain halts -- the only way to get a number out of a machine
 * with no stdout. Here there IS a stdout, so the number is printed. The HALT is kept: every
 * caller treats this as terminal (an allocator invariant broken, a copy audit tripped, an
 * attribution point reached), and a diagnostic that let the run continue would turn a reading
 * into a trigger that completed -- i.e. into a control that appears to hold.
 *
 * Reached in Phase A only from capstone_memcpy_phase_note, which the beebs memcpy calls and
 * which is not linked on this profile (musl-capstone's string_bounds_safe.c supplies the
 * tag-preserving memcpy instead). It is defined rather than stubbed because the audit and
 * diagnostic builds of php_capstone_zend_hier.c call it too, and a stub that returned would
 * silently disarm them. */
void php_fault_report(unsigned long v)
{
    fprintf(stderr, "PHP_VIRTUAL FAULT_REPORT 0x%016lx\n", v);
    fflush(stderr);
    fflush(stdout);
    abort();
}
