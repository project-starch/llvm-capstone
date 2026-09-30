/* Is std_object_handlers -- 23 function pointers in a STATIC initializer
 * (Zend/zend_object_handlers.c:951) -- correctly initialised in a domain?
 *
 * Rung B exhausts ~2.6 MB of stack, and the largest frame in the whole image is 2,032
 * bytes, so it is ~1,300+ nested calls: recursion, not a fat frame. A static table of
 * function pointers is the prime suspect, because ports/sqlite/README.md records that
 * capability-bearing static aggregates needed RUNTIME initialisation on this target. A call
 * through a wrong entry gives accidental mutual recursion for free.
 *
 * This rung calls nothing. It reads the table and reports:
 *   bits 0..22  tag of each slot, in declaration order
 *   bit 30      read_property points inside the engine's code region
 *   bit 31      the table's own address is tagged
 *
 * Five slots are legitimately NULL in the initialiser (get, set, call_method, cast_object,
 * count_elements = indices 8, 9, 16, 21, 22), so the CORRECT answer is
 *   bits 0..22 set EXCEPT 8, 9, 16, 21, 22, plus bits 30 and 31
 * so the CORRECT mask over bits 0..22 is 0x1F00FF, plus bits 30 and 31.
 */
#include <zend.h>
#include <zend_object_handlers.h>

extern zend_object_handlers std_object_handlers;

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    unsigned out = 0;

    /* Every field is a function pointer, so the struct is a dense array of them. Walking it
     * as void*const* avoids naming 23 fields and cannot miss one. */
    void *const *slot = (void *const *)&std_object_handlers;
    unsigned n = (unsigned)(sizeof(zend_object_handlers) / sizeof(void *));
    if (n > 23u) { n = 23u; }

    for (unsigned i = 0; i < n; i++) {
        if (__builtin_capstone_cap_get_tag((void *)slot[i])) { out |= (1u << i); }
    }

    /* The functions the initialiser names are all `static` in zend_object_handlers.c, so
     * their identity cannot be checked from here. Instead report whether read_property
     * points INSIDE the same code region as a known engine function -- a tagged pointer
     * into the wrong place would still be wrong, and this catches the gross case. */
    {
        void *got = (void *)std_object_handlers.read_property;
        void *ref = (void *)&zend_startup;
        unsigned long c = __builtin_capstone_cap_get_cursor(got);
        if (c >= __builtin_capstone_cap_get_base(ref) &&
            c <  __builtin_capstone_cap_get_end(ref)) { out |= (1u << 30); }
    }
    if (__builtin_capstone_cap_get_tag((void *)&std_object_handlers)) { out |= (1u << 31); }

    *res = out;
}
