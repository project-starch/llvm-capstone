/* The lower allocator level, on the virtual profile: what the HIERARCHICAL wrapper needs from
 * it, and why that is almost nothing here.
 *
 * libc/php_capstone_zend_hier.c is reused UNCHANGED in Phase A -- it is the experiment, so it
 * is the one file from the physical port's support layer that carries over. It narrows every
 * pointer it hands up to the caller's object, and therefore has to widen any pointer that
 * re-enters the level below ("narrow on the way up, widen on the way down"). Physically that
 * widen is real work: zend_cap_widen recovers the block capability from the arena's slot table
 * in libc/php_capstone_malloc.c, and it was found to be necessary TWICE, each time by a fault
 * -- PHP's AG(cache) parks the narrow pointer it is handed and poisons the next tenant of that
 * size class, and the lower level's realloc copies a whole block's worth through whatever
 * bound it is given.
 *
 * ON THIS PROFILE THE LOWER LEVEL IS MUSL'S MALLOCNG, via runtime/virtual/heap.c, and it keeps
 * its own authority. free() and realloc() go through `checked(p)` (heap.c:203-217), which uses
 * the caller's pointer for exactly two things -- a one-byte validity probe at the cursor
 * (`lbu zero, 0(p)`) and an address to look the slot up by -- and then operates through
 * `r->raw`, the capability the heap itself retained over the block. The caller's BOUNDS are
 * never the bounds the copy or the free is performed with. So a narrowed pointer whose base is
 * intact is already enough, and zend_cap_widen has no work to do.
 *
 * THAT IS A READING OF heap.c, AND A READING IS NOT A RESULT, so it has a directed test:
 * app/probe-hier-widen.c reproduces PHP's exact geometry (malloc(48 + n), hand back p + 48,
 * narrow to [block_base, object_cursor + n)), frees and reallocs through the narrowed pointer,
 * and -- the stage that makes the other two mean anything -- confirms in a NEGATIVE CONTROL
 * that one byte past the narrowed end really does fault. If that probe ever fails, this
 * function is wrong and Phase A needs a block registry keyed on the preserved base; the
 * identity is not an assumption that can be left unexamined.
 */
#include <stdio.h>
#include <stdlib.h>

/* Deliberately not `static`: php_capstone_zend_hier.c declares it extern and calls it from
 * _efree and _erealloc. Keeping the signature identical to the physical definition
 * (php_capstone_malloc.c:182) is what makes that file reusable as-is. */
void *zend_cap_widen(void *p)
{
    return p;
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
