/* A NESTED ALLOCATOR, in the shape MicroPython uses.
 *
 * MicroPython does not call the system allocator. mpy_domain.c declares the heap as
 * one static array and hands that single object to gc_init(); gc_alloc and gc_free
 * then carve it up in software. Every language runtime with its own collector is
 * built this way. This header is the smallest honest model of it: one region, a bump
 * cursor, and a free that only does bookkeeping.
 *
 * WHAT THIS RUNG MEASURES, AND WHY A GREEN RESULT MEANS THE OPPOSITE OF USUAL.
 *
 * The ladder harness compares the domain's return value against a NATIVE build of
 * the same source. The native build has no capability hardware at all, so it is the
 * unprotected baseline by construction. If the domain AGREES with it, Capstone
 * changed nothing and the use-after-free went through untrapped. A passing rung here
 * is therefore a finding, not a reassurance: it is the nested-allocator gap, measured.
 *
 * The mechanism, confirmed in the disassembly (see the corpus folder under
 * benchmarks/micropython/temporal-corpus/evidence/): the backend sets bounds once,
 * over the whole `nest_heap` object, and every sub-allocated pointer inherits them.
 * A pointer to a "freed" block is still inside those bounds with a valid tag, so a
 * store through it is a bare `sb` the hardware has no grounds to reject. Capstone
 * does have revocation, but nothing here invokes it: nest_free never reaches the
 * hardware, because the block never had a capability of its own to revoke.
 *
 * NEST_STALE_OFFSET is the ONLY difference between the two arms:
 *   nestalloc_app.c  offset 0                 -> inside the region, expected UNTRAPPED
 *   nestoob_app.c    offset past the region   -> outside it,        expected TRAPPED
 * That pairing is the point. Without the second arm, "no fault" is indistinguishable
 * from "the domain never ran", and this project has read that as a result before.
 */
#ifndef NESTALLOC_KERNEL_H
#define NESTALLOC_KERNEL_H

#define NEST_HEAP_BYTES 1024u
#define NEST_BLOCK      64u

#ifndef NEST_STALE_OFFSET
#define NEST_STALE_OFFSET 0u
#endif

static unsigned char nest_heap[NEST_HEAP_BYTES] __attribute__((aligned(32)));
static unsigned long nest_bump;

static unsigned char *nest_alloc(unsigned long n) {
    unsigned char *p = &nest_heap[nest_bump];
    nest_bump += n;
    return p;
}

/* The free. Bookkeeping only, exactly like gc_free: it returns storage to the
   allocator's own accounting and tells the hardware nothing. */
static void nest_free(unsigned char *p) { (void)p; }

static unsigned nest_run(void) {
    unsigned char *a = nest_alloc(NEST_BLOCK);
    a[0] = 0x11;

    nest_free(a);          /* a is dead from here on */
    nest_bump = 0;         /* the sweep: its storage goes back to the allocator */

    unsigned char *b = nest_alloc(NEST_BLOCK);  /* b is handed exactly a's storage */
    b[0] = 0x22;

    /* The use-after-free write, through a pointer whose object was freed and whose
       storage now belongs to b. Offset kept in a volatile so -O0 cannot fold the
       two pointers together and reason the access away. */
    volatile unsigned long off = NEST_STALE_OFFSET;
    a[off] = 0xAA;

    /* 0xAA (170) means the stale write landed inside a LIVE object and nothing
       objected. 0x22 (34) would mean it went somewhere else entirely. */
    return (unsigned)b[0];
}

#endif /* NESTALLOC_KERNEL_H */
