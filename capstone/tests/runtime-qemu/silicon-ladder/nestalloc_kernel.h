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


/* ---------------------------------------------------------------------------
 * THE SPATIAL HALF OF THE SAME QUESTION.
 *
 * nest_run above is a lifetime defect. This one has NO lifetime component at all:
 * both blocks are alive the whole time and nothing is freed. It is a plain buffer
 * overflow out of one sub-allocated block into the next.
 *
 * The prediction is that Capstone does not catch it either, and for the SAME
 * reason: bounds are set once over the whole nest_heap object, so a and b are not
 * two objects to the hardware, they are two offsets into one. Sub-object bounds
 * would be needed to tell them apart, and a nested allocator carves its blocks in
 * software where no bounds-setting instruction is ever emitted.
 *
 * If that prediction holds, the nested-allocator gap is not specifically a
 * temporal one. It is that the hardware's notion of "object" is the region the
 * allocator was given, and everything the allocator does inside it is invisible.
 *
 * Parameterised by NEST_SPATIAL_OFFSET, and the two arms differ in that ONE
 * constant, exactly as the temporal pair does:
 *   NEST_BLOCK               -> one past a's end, still inside the region: expect UNTRAPPED
 *   NEST_HEAP_BYTES + BLOCK  -> past the region itself:                    expect TRAPPED
 * The second arm is the positive control. Without it, "no fault" and "the domain
 * never ran" look the same.
 */
#ifndef NEST_SPATIAL_OFFSET
#define NEST_SPATIAL_OFFSET NEST_BLOCK
#endif

static unsigned nest_spatial_run(void) {
    unsigned char *a = nest_alloc(NEST_BLOCK);
    unsigned char *b = nest_alloc(NEST_BLOCK);   /* BOTH live; nest_free is never called */

    a[0] = 0x11;
    b[0] = 0x22;

    /* The overflow. volatile so -O0 cannot fold a and b into one base and reason
       the access away, the same guard the temporal arm uses. */
    volatile unsigned long off = NEST_SPATIAL_OFFSET;
    a[off] = 0xAA;

    /* 0xAA (170) means the write walked out of a and into b with nothing objecting.
       0x22 (34) would mean it landed somewhere else and b is intact. */
    return (unsigned)b[0];
}


/* ---------------------------------------------------------------------------
 * THE THIRD SCOPE: two STATIC GLOBALS, not one heap region.
 *
 * The spatial corpus predicts `trapped` for its static-global rows, on the stated
 * ground that -capstone-gp-captable carves each global separately, so unlike two
 * sub-allocations of one heap array they really are two objects to the hardware.
 * Six rows rest on that and NOT ONE of them is measured -- every one is blocked on
 * a VFS, on .mpy loading, or on absent modules, and MPY-S31 has just shown the
 * stack rows cannot test it either because the port guards its own C stack first.
 *
 * So test the prediction directly instead of waiting for an upstream defect. Same
 * shape as nest_spatial_run, one variable changed: the two objects are file-scope
 * globals rather than blocks carved out of nest_heap.
 *
 * The prediction is NOT safe. This build passes -capstone-shrink-globals=false
 * (build-ladder-domain.sh:85), and the MicroPython build passes it too, so whether
 * a global's bounds are its own or span something larger is exactly the open
 * question. Either answer is a result:
 *   TRAP        -> the corpus's static-global prediction is confirmed, and the
 *                  gc-heap rows are untrapped for the reason claimed rather than
 *                  because checking is off somewhere.
 *   returns 170 -> the prediction is WRONG, globals share bounds too under this
 *                  flag, and six corpus rows need rewriting.
 *
 * NEST_GLOBAL_OFFSET is the only difference between the arms:
 *   0            -> writes glob_a[0], in bounds: expect UNTRAPPED, returns 34
 *   NEST_BLOCK   -> one past glob_a's end and into glob_b: the question
 */
#ifndef NEST_GLOBAL_OFFSET
#define NEST_GLOBAL_OFFSET 0u
#endif

static unsigned char nest_glob_a[NEST_BLOCK] __attribute__((aligned(32)));
static unsigned char nest_glob_b[NEST_BLOCK] __attribute__((aligned(32)));

static unsigned nest_global_run(void) {
    nest_glob_a[0] = 0x11;
    nest_glob_b[0] = 0x22;

    /* volatile so -O0 cannot fold the two symbols into one base and reason the
       access away, the same guard both other kernels use. */
    volatile unsigned long off = NEST_GLOBAL_OFFSET;
    nest_glob_a[off] = 0xAA;

    /* 170 means the write left glob_a and landed in glob_b with nothing objecting.
       34 means glob_b is intact, which for the offset-0 arm is the expected pass. */
    return (unsigned)nest_glob_b[0];
}

#endif /* NESTALLOC_KERNEL_H */
